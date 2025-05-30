//===- OpenMPIRBuilder.cpp - Builder for LLVM-IR for OpenMP directives ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the OpenMPIRBuilder class, which is used as a
/// convenient way to create LLVM instructions for OpenMP directives.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include <sstream>

#define DEBUG_TYPE "openmp-ir-builder"

using namespace llvm;
using namespace omp;

static cl::opt<bool>
    OptimisticAttributes("openmp-ir-builder-optimistic-attributes", cl::Hidden,
                         cl::desc("Use optimistic attributes describing "
                                  "'as-if' properties of runtime calls."),
                         cl::init(false));

void OpenMPIRBuilder::addAttributes(omp::RuntimeFunction FnID, Function &Fn) {
  LLVMContext &Ctx = Fn.getContext();

  // Get the function's current attributes.
  auto Attrs = Fn.getAttributes();
  auto FnAttrs = Attrs.getFnAttributes();
  auto RetAttrs = Attrs.getRetAttributes();
  SmallVector<AttributeSet, 4> ArgAttrs;
  for (size_t ArgNo = 0; ArgNo < Fn.arg_size(); ++ArgNo)
    ArgAttrs.emplace_back(Attrs.getParamAttributes(ArgNo));

#define OMP_ATTRS_SET(VarName, AttrSet) AttributeSet VarName = AttrSet;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  // Add attributes to the function declaration.
  switch (FnID) {
#define OMP_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)                \
  case Enum:                                                                   \
    FnAttrs = FnAttrs.addAttributes(Ctx, FnAttrSet);                           \
    RetAttrs = RetAttrs.addAttributes(Ctx, RetAttrSet);                        \
    for (size_t ArgNo = 0; ArgNo < ArgAttrSets.size(); ++ArgNo)                \
      ArgAttrs[ArgNo] =                                                        \
          ArgAttrs[ArgNo].addAttributes(Ctx, ArgAttrSets[ArgNo]);              \
    Fn.setAttributes(AttributeList::get(Ctx, FnAttrs, RetAttrs, ArgAttrs));    \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  default:
    // Attributes are optional.
    break;
  }
}

FunctionCallee
OpenMPIRBuilder::getOrCreateRuntimeFunction(Module &M, RuntimeFunction FnID) {
  FunctionType *FnTy = nullptr;
  Function *Fn = nullptr;

  // Try to find the declation in the module first.
  switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    FnTy = FunctionType::get(ReturnType, ArrayRef<Type *>{__VA_ARGS__},        \
                             IsVarArg);                                        \
    Fn = M.getFunction(Str);                                                   \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }

  if (!Fn) {
    // Create a new declaration if we need one.
    switch (FnID) {
#define OMP_RTL(Enum, Str, ...)                                                \
  case Enum:                                                                   \
    Fn = Function::Create(FnTy, GlobalValue::ExternalLinkage, Str, M);         \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
    }

    // Add information if the runtime function takes a callback function
    if (FnID == OMPRTL___kmpc_fork_call || FnID == OMPRTL___kmpc_fork_teams) {
      if (!Fn->hasMetadata(LLVMContext::MD_callback)) {
        LLVMContext &Ctx = Fn->getContext();
        MDBuilder MDB(Ctx);
        // Annotate the callback behavior of the runtime function:
        //  - The callback callee is argument number 2 (microtask).
        //  - The first two arguments of the callback callee are unknown (-1).
        //  - All variadic arguments to the runtime function are passed to the
        //    callback callee.
        Fn->addMetadata(
            LLVMContext::MD_callback,
            *MDNode::get(Ctx, {MDB.createCallbackEncoding(
                                  2, {-1, -1}, /* VarArgsArePassed */ true)}));
      }
    }

    LLVM_DEBUG(dbgs() << "Created OpenMP runtime function " << Fn->getName()
                      << " with type " << *Fn->getFunctionType() << "\n");
    addAttributes(FnID, *Fn);

  } else {
    LLVM_DEBUG(dbgs() << "Found OpenMP runtime function " << Fn->getName()
                      << " with type " << *Fn->getFunctionType() << "\n");
  }

  assert(Fn && "Failed to create OpenMP runtime function");

  // Cast the function to the expected type if necessary
  Constant *C = ConstantExpr::getBitCast(Fn, FnTy->getPointerTo());
  return {FnTy, C};
}

Function *OpenMPIRBuilder::getOrCreateRuntimeFunctionPtr(RuntimeFunction FnID) {
  FunctionCallee RTLFn = getOrCreateRuntimeFunction(M, FnID);
  auto *Fn = dyn_cast<llvm::Function>(RTLFn.getCallee());
  assert(Fn && "Failed to create OpenMP runtime function pointer");
  return Fn;
}

void OpenMPIRBuilder::initialize() { initializeTypes(M); }

void OpenMPIRBuilder::finalize(Function *Fn, bool AllowExtractorSinking) {
  SmallPtrSet<BasicBlock *, 32> ParallelRegionBlockSet;
  SmallVector<BasicBlock *, 32> Blocks;
  SmallVector<OutlineInfo, 16> DeferredOutlines;
  for (OutlineInfo &OI : OutlineInfos) {
    // Skip functions that have not finalized yet; may happen with nested
    // function generation.
    if (Fn && OI.getFunction() != Fn) {
      DeferredOutlines.push_back(OI);
      continue;
    }

    ParallelRegionBlockSet.clear();
    Blocks.clear();
    OI.collectBlocks(ParallelRegionBlockSet, Blocks);

    Function *OuterFn = OI.getFunction();
    CodeExtractorAnalysisCache CEAC(*OuterFn);
    CodeExtractor Extractor(Blocks, /* DominatorTree */ nullptr,
                            /* AggregateArgs */ false,
                            /* BlockFrequencyInfo */ nullptr,
                            /* BranchProbabilityInfo */ nullptr,
                            /* AssumptionCache */ nullptr,
                            /* AllowVarArgs */ true,
                            /* AllowAlloca */ true,
                            /* Suffix */ ".omp_par");

    LLVM_DEBUG(dbgs() << "Before     outlining: " << *OuterFn << "\n");
    LLVM_DEBUG(dbgs() << "Entry " << OI.EntryBB->getName()
                      << " Exit: " << OI.ExitBB->getName() << "\n");
    assert(Extractor.isEligible() &&
           "Expected OpenMP outlining to be possible!");

    Function *OutlinedFn = Extractor.extractCodeRegion(CEAC);

    LLVM_DEBUG(dbgs() << "After      outlining: " << *OuterFn << "\n");
    LLVM_DEBUG(dbgs() << "   Outlined function: " << *OutlinedFn << "\n");
    assert(OutlinedFn->getReturnType()->isVoidTy() &&
           "OpenMP outlined functions should not return a value!");

    // For compability with the clang CG we move the outlined function after the
    // one with the parallel region.
    OutlinedFn->removeFromParent();
    M.getFunctionList().insertAfter(OuterFn->getIterator(), OutlinedFn);

    // Remove the artificial entry introduced by the extractor right away, we
    // made our own entry block after all.
    {
      BasicBlock &ArtificialEntry = OutlinedFn->getEntryBlock();
      assert(ArtificialEntry.getUniqueSuccessor() == OI.EntryBB);
      assert(OI.EntryBB->getUniquePredecessor() == &ArtificialEntry);
      if (AllowExtractorSinking) {
        // Move instructions from the to-be-deleted ArtificialEntry to the entry
        // basic block of the parallel region. CodeExtractor may have sunk
        // allocas/bitcasts for values that are solely used in the outlined
        // region and do not escape.
        assert(!ArtificialEntry.empty() &&
               "Expected instructions to sink in the outlined region");
        for (BasicBlock::iterator It = ArtificialEntry.begin(),
                                  End = ArtificialEntry.end();
             It != End;) {
          Instruction &I = *It;
          It++;

          if (I.isTerminator())
            continue;

          I.moveBefore(*OI.EntryBB, OI.EntryBB->getFirstInsertionPt());
        }
      }
      OI.EntryBB->moveBefore(&ArtificialEntry);
      ArtificialEntry.eraseFromParent();
    }
    assert(&OutlinedFn->getEntryBlock() == OI.EntryBB);
    assert(OutlinedFn && OutlinedFn->getNumUses() == 1);

    // Run a user callback, e.g. to add attributes.
    if (OI.PostOutlineCB)
      OI.PostOutlineCB(*OutlinedFn);
  }

  // Remove work items that have been completed.
  OutlineInfos = std::move(DeferredOutlines);
}

OpenMPIRBuilder::~OpenMPIRBuilder() {
  assert(OutlineInfos.empty() && "There must be no outstanding outlinings");
}

Value *OpenMPIRBuilder::getOrCreateIdent(Constant *SrcLocStr,
                                         IdentFlag LocFlags,
                                         unsigned Reserve2Flags) {
  // Enable "C-mode".
  LocFlags |= OMP_IDENT_FLAG_KMPC;

  Value *&Ident =
      IdentMap[{SrcLocStr, uint64_t(LocFlags) << 31 | Reserve2Flags}];
  if (!Ident) {
    Constant *I32Null = ConstantInt::getNullValue(Int32);
    Constant *IdentData[] = {
        I32Null, ConstantInt::get(Int32, uint32_t(LocFlags)),
        ConstantInt::get(Int32, Reserve2Flags), I32Null, SrcLocStr};
    Constant *Initializer = ConstantStruct::get(
        cast<StructType>(IdentPtr->getPointerElementType()), IdentData);

    // Look for existing encoding of the location + flags, not needed but
    // minimizes the difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.getType() == IdentPtr && GV.hasInitializer())
        if (GV.getInitializer() == Initializer)
          return Ident = &GV;

    auto *GV = new GlobalVariable(M, IdentPtr->getPointerElementType(),
                                  /* isConstant = */ true,
                                  GlobalValue::PrivateLinkage, Initializer);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GV->setAlignment(Align(8));
    Ident = GV;
  }
  return Builder.CreatePointerCast(Ident, IdentPtr);
}

Type *OpenMPIRBuilder::getLanemaskType() {
  LLVMContext &Ctx = M.getContext();
  Triple triple(M.getTargetTriple());

  // This test is adequate until deviceRTL has finer grained lane widths
  return triple.isAMDGCN() ? Type::getInt64Ty(Ctx) : Type::getInt32Ty(Ctx);
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef LocStr) {
  Constant *&SrcLocStr = SrcLocStrMap[LocStr];
  if (!SrcLocStr) {
    Constant *Initializer =
        ConstantDataArray::getString(M.getContext(), LocStr);

    // Look for existing encoding of the location, not needed but minimizes the
    // difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.isConstant() && GV.hasInitializer() &&
          GV.getInitializer() == Initializer)
        return SrcLocStr = ConstantExpr::getPointerCast(&GV, Int8Ptr);

    SrcLocStr = Builder.CreateGlobalStringPtr(LocStr, /* Name */ "",
                                              /* AddressSpace */ 0, &M);
  }
  return SrcLocStr;
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef FunctionName,
                                                StringRef FileName,
                                                unsigned Line,
                                                unsigned Column) {
  SmallString<128> Buffer;
  Buffer.push_back(';');
  Buffer.append(FileName);
  Buffer.push_back(';');
  Buffer.append(FunctionName);
  Buffer.push_back(';');
  Buffer.append(std::to_string(Line));
  Buffer.push_back(';');
  Buffer.append(std::to_string(Column));
  Buffer.push_back(';');
  Buffer.push_back(';');
  return getOrCreateSrcLocStr(Buffer.str());
}

Constant *OpenMPIRBuilder::getOrCreateDefaultSrcLocStr() {
  return getOrCreateSrcLocStr(";unknown;unknown;0;0;;");
}

Constant *
OpenMPIRBuilder::getOrCreateSrcLocStr(const LocationDescription &Loc) {
  DILocation *DIL = Loc.DL.get();
  if (!DIL)
    return getOrCreateDefaultSrcLocStr();
  StringRef FileName = M.getName();
  if (DIFile *DIF = DIL->getFile())
    if (Optional<StringRef> Source = DIF->getSource())
      FileName = *Source;
  StringRef Function = DIL->getScope()->getSubprogram()->getName();
  Function =
      !Function.empty() ? Function : Loc.IP.getBlock()->getParent()->getName();
  return getOrCreateSrcLocStr(Function, FileName, DIL->getLine(),
                              DIL->getColumn());
}

Value *OpenMPIRBuilder::getOrCreateThreadID(Value *Ident) {
  return Builder.CreateCall(
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_global_thread_num), Ident,
      "omp_global_thread_num");
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createBarrier(const LocationDescription &Loc, Directive DK,
                               bool ForceSimpleCall, bool CheckCancelFlag) {
  if (!updateToLocation(Loc))
    return Loc.IP;
  return emitBarrierImpl(Loc, DK, ForceSimpleCall, CheckCancelFlag);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::emitBarrierImpl(const LocationDescription &Loc, Directive Kind,
                                 bool ForceSimpleCall, bool CheckCancelFlag) {
  // Build call __kmpc_cancel_barrier(loc, thread_id) or
  //            __kmpc_barrier(loc, thread_id);

  IdentFlag BarrierLocFlags;
  switch (Kind) {
  case OMPD_for:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_FOR;
    break;
  case OMPD_sections:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SECTIONS;
    break;
  case OMPD_single:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SINGLE;
    break;
  case OMPD_barrier:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_EXPL;
    break;
  default:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL;
    break;
  }

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Args[] = {getOrCreateIdent(SrcLocStr, BarrierLocFlags),
                   getOrCreateThreadID(getOrCreateIdent(SrcLocStr))};

  // If we are in a cancellable parallel region, barriers are cancellation
  // points.
  // TODO: Check why we would force simple calls or to ignore the cancel flag.
  bool UseCancelBarrier =
      !ForceSimpleCall && isLastFinalizationInfoCancellable(OMPD_parallel);

  Value *Result =
      Builder.CreateCall(getOrCreateRuntimeFunctionPtr(
                             UseCancelBarrier ? OMPRTL___kmpc_cancel_barrier
                                              : OMPRTL___kmpc_barrier),
                         Args);

  if (UseCancelBarrier && CheckCancelFlag)
    emitCancelationCheckImpl(Result, OMPD_parallel);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createCancel(const LocationDescription &Loc,
                              Value *IfCondition,
                              omp::Directive CanceledDirective) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  // LLVM utilities like blocks with terminators.
  auto *UI = Builder.CreateUnreachable();

  Instruction *ThenTI = UI, *ElseTI = nullptr;
  if (IfCondition)
    SplitBlockAndInsertIfThenElse(IfCondition, UI, &ThenTI, &ElseTI);
  Builder.SetInsertPoint(ThenTI);

  Value *CancelKind = nullptr;
  switch (CanceledDirective) {
#define OMP_CANCEL_KIND(Enum, Str, DirectiveEnum, Value)                       \
  case DirectiveEnum:                                                          \
    CancelKind = Builder.getInt32(Value);                                      \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  default:
    llvm_unreachable("Unknown cancel kind!");
  }

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *Args[] = {Ident, getOrCreateThreadID(Ident), CancelKind};
  Value *Result = Builder.CreateCall(
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_cancel), Args);

  // The actual cancel logic is shared with others, e.g., cancel_barriers.
  emitCancelationCheckImpl(Result, CanceledDirective);

  // Update the insertion point and remove the terminator we introduced.
  Builder.SetInsertPoint(UI->getParent());
  UI->eraseFromParent();

  return Builder.saveIP();
}

void OpenMPIRBuilder::emitCancelationCheckImpl(
    Value *CancelFlag, omp::Directive CanceledDirective) {
  assert(isLastFinalizationInfoCancellable(CanceledDirective) &&
         "Unexpected cancellation!");

  // For a cancel barrier we create two new blocks.
  BasicBlock *BB = Builder.GetInsertBlock();
  BasicBlock *NonCancellationBlock;
  if (Builder.GetInsertPoint() == BB->end()) {
    // TODO: This branch will not be needed once we moved to the
    // OpenMPIRBuilder codegen completely.
    NonCancellationBlock = BasicBlock::Create(
        BB->getContext(), BB->getName() + ".cont", BB->getParent());
  } else {
    NonCancellationBlock = SplitBlock(BB, &*Builder.GetInsertPoint());
    BB->getTerminator()->eraseFromParent();
    Builder.SetInsertPoint(BB);
  }
  BasicBlock *CancellationBlock = BasicBlock::Create(
      BB->getContext(), BB->getName() + ".cncl", BB->getParent());

  // Jump to them based on the return value.
  Value *Cmp = Builder.CreateIsNull(CancelFlag);
  Builder.CreateCondBr(Cmp, NonCancellationBlock, CancellationBlock,
                       /* TODO weight */ nullptr, nullptr);

  // From the cancellation block we finalize all variables and go to the
  // post finalization block that is known to the FiniCB callback.
  Builder.SetInsertPoint(CancellationBlock);
  auto &FI = FinalizationStack.back();
  FI.FiniCB(Builder.saveIP());

  // The continuation block is where code generation continues.
  Builder.SetInsertPoint(NonCancellationBlock, NonCancellationBlock->begin());
}

IRBuilder<>::InsertPoint OpenMPIRBuilder::createParallel(
    const LocationDescription &Loc, InsertPointTy OuterAllocaIP,
    BodyGenCallbackTy BodyGenCB, PrivatizeCallbackTy PrivCB,
    FinalizeCallbackTy FiniCB, Value *IfCondition, Value *NumThreads,
    omp::ProcBindKind ProcBind, bool IsCancellable) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadID = getOrCreateThreadID(Ident);

  if (NumThreads) {
    // Build call __kmpc_push_num_threads(&Ident, global_tid, num_threads)
    Value *Args[] = {
        Ident, ThreadID,
        Builder.CreateIntCast(NumThreads, Int32, /*isSigned*/ false)};
    Builder.CreateCall(
        getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_push_num_threads), Args);
  }

  if (ProcBind != OMP_PROC_BIND_default) {
    // Build call __kmpc_push_proc_bind(&Ident, global_tid, proc_bind)
    Value *Args[] = {
        Ident, ThreadID,
        ConstantInt::get(Int32, unsigned(ProcBind), /*isSigned=*/true)};
    Builder.CreateCall(
        getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_push_proc_bind), Args);
  }

  BasicBlock *InsertBB = Builder.GetInsertBlock();
  Function *OuterFn = InsertBB->getParent();

  // Save the outer alloca block because the insertion iterator may get
  // invalidated and we still need this later.
  BasicBlock *OuterAllocaBlock = OuterAllocaIP.getBlock();

  // Vector to remember instructions we used only during the modeling but which
  // we want to delete at the end.
  SmallVector<Instruction *, 4> ToBeDeleted;

  // Change the location to the outer alloca insertion point to create and
  // initialize the allocas we pass into the parallel region.
  Builder.restoreIP(OuterAllocaIP);
  AllocaInst *TIDAddr = Builder.CreateAlloca(Int32, nullptr, "tid.addr");
  AllocaInst *ZeroAddr = Builder.CreateAlloca(Int32, nullptr, "zero.addr");

  // If there is an if condition we actually use the TIDAddr and ZeroAddr in the
  // program, otherwise we only need them for modeling purposes to get the
  // associated arguments in the outlined function. In the former case,
  // initialize the allocas properly, in the latter case, delete them later.
  if (IfCondition) {
    Builder.CreateStore(Constant::getNullValue(Int32), TIDAddr);
    Builder.CreateStore(Constant::getNullValue(Int32), ZeroAddr);
  } else {
    ToBeDeleted.push_back(TIDAddr);
    ToBeDeleted.push_back(ZeroAddr);
  }

  // Create an artificial insertion point that will also ensure the blocks we
  // are about to split are not degenerated.
  auto *UI = new UnreachableInst(Builder.getContext(), InsertBB);

  Instruction *ThenTI = UI, *ElseTI = nullptr;
  if (IfCondition)
    SplitBlockAndInsertIfThenElse(IfCondition, UI, &ThenTI, &ElseTI);

  BasicBlock *ThenBB = ThenTI->getParent();
  BasicBlock *PRegEntryBB = ThenBB->splitBasicBlock(ThenTI, "omp.par.entry");
  BasicBlock *PRegBodyBB =
      PRegEntryBB->splitBasicBlock(ThenTI, "omp.par.region");
  BasicBlock *PRegPreFiniBB =
      PRegBodyBB->splitBasicBlock(ThenTI, "omp.par.pre_finalize");
  BasicBlock *PRegExitBB =
      PRegPreFiniBB->splitBasicBlock(ThenTI, "omp.par.exit");

  auto FiniCBWrapper = [&](InsertPointTy IP) {
    // Hide "open-ended" blocks from the given FiniCB by setting the right jump
    // target to the region exit block.
    if (IP.getBlock()->end() == IP.getPoint()) {
      IRBuilder<>::InsertPointGuard IPG(Builder);
      Builder.restoreIP(IP);
      Instruction *I = Builder.CreateBr(PRegExitBB);
      IP = InsertPointTy(I->getParent(), I->getIterator());
    }
    assert(IP.getBlock()->getTerminator()->getNumSuccessors() == 1 &&
           IP.getBlock()->getTerminator()->getSuccessor(0) == PRegExitBB &&
           "Unexpected insertion point for finalization call!");
    return FiniCB(IP);
  };

  FinalizationStack.push_back({FiniCBWrapper, OMPD_parallel, IsCancellable});

  // Generate the privatization allocas in the block that will become the entry
  // of the outlined function.
  Builder.SetInsertPoint(PRegEntryBB->getTerminator());
  InsertPointTy InnerAllocaIP = Builder.saveIP();

  AllocaInst *PrivTIDAddr =
      Builder.CreateAlloca(Int32, nullptr, "tid.addr.local");
  Instruction *PrivTID = Builder.CreateLoad(Int32, PrivTIDAddr, "tid");

  // Add some fake uses for OpenMP provided arguments.
  ToBeDeleted.push_back(Builder.CreateLoad(Int32, TIDAddr, "tid.addr.use"));
  Instruction *ZeroAddrUse = Builder.CreateLoad(Int32, ZeroAddr,
                                                "zero.addr.use");
  ToBeDeleted.push_back(ZeroAddrUse);

  // ThenBB
  //   |
  //   V
  // PRegionEntryBB         <- Privatization allocas are placed here.
  //   |
  //   V
  // PRegionBodyBB          <- BodeGen is invoked here.
  //   |
  //   V
  // PRegPreFiniBB          <- The block we will start finalization from.
  //   |
  //   V
  // PRegionExitBB          <- A common exit to simplify block collection.
  //

  LLVM_DEBUG(dbgs() << "Before body codegen: " << *OuterFn << "\n");

  // Let the caller create the body.
  assert(BodyGenCB && "Expected body generation callback!");
  InsertPointTy CodeGenIP(PRegBodyBB, PRegBodyBB->begin());
  BodyGenCB(InnerAllocaIP, CodeGenIP, *PRegPreFiniBB);

  LLVM_DEBUG(dbgs() << "After  body codegen: " << *OuterFn << "\n");

  FunctionCallee RTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_fork_call);
  if (auto *F = dyn_cast<llvm::Function>(RTLFn.getCallee())) {
    if (!F->hasMetadata(llvm::LLVMContext::MD_callback)) {
      llvm::LLVMContext &Ctx = F->getContext();
      MDBuilder MDB(Ctx);
      // Annotate the callback behavior of the __kmpc_fork_call:
      //  - The callback callee is argument number 2 (microtask).
      //  - The first two arguments of the callback callee are unknown (-1).
      //  - All variadic arguments to the __kmpc_fork_call are passed to the
      //    callback callee.
      F->addMetadata(
          llvm::LLVMContext::MD_callback,
          *llvm::MDNode::get(
              Ctx, {MDB.createCallbackEncoding(2, {-1, -1},
                                               /* VarArgsArePassed */ true)}));
    }
  }

  OutlineInfo OI;
  OI.PostOutlineCB = [=](Function &OutlinedFn) {
    // Add some known attributes.
    OutlinedFn.addParamAttr(0, Attribute::NoAlias);
    OutlinedFn.addParamAttr(1, Attribute::NoAlias);
    OutlinedFn.addFnAttr(Attribute::NoUnwind);
    OutlinedFn.addFnAttr(Attribute::NoRecurse);

    assert(OutlinedFn.arg_size() >= 2 &&
           "Expected at least tid and bounded tid as arguments");
    unsigned NumCapturedVars =
        OutlinedFn.arg_size() - /* tid & bounded tid */ 2;

    CallInst *CI = cast<CallInst>(OutlinedFn.user_back());
    CI->getParent()->setName("omp_parallel");
    Builder.SetInsertPoint(CI);

    // Build call __kmpc_fork_call(Ident, n, microtask, var1, .., varn);
    Value *ForkCallArgs[] = {
        Ident, Builder.getInt32(NumCapturedVars),
        Builder.CreateBitCast(&OutlinedFn, ParallelTaskPtr)};

    SmallVector<Value *, 16> RealArgs;
    RealArgs.append(std::begin(ForkCallArgs), std::end(ForkCallArgs));
    RealArgs.append(CI->arg_begin() + /* tid & bound tid */ 2, CI->arg_end());

    Builder.CreateCall(RTLFn, RealArgs);

    LLVM_DEBUG(dbgs() << "With fork_call placed: "
                      << *Builder.GetInsertBlock()->getParent() << "\n");

    InsertPointTy ExitIP(PRegExitBB, PRegExitBB->end());

    // Initialize the local TID stack location with the argument value.
    Builder.SetInsertPoint(PrivTID);
    Function::arg_iterator OutlinedAI = OutlinedFn.arg_begin();
    Builder.CreateStore(Builder.CreateLoad(Int32, OutlinedAI), PrivTIDAddr);

    // If no "if" clause was present we do not need the call created during
    // outlining, otherwise we reuse it in the serialized parallel region.
    if (!ElseTI) {
      CI->eraseFromParent();
    } else {

      // If an "if" clause was present we are now generating the serialized
      // version into the "else" branch.
      Builder.SetInsertPoint(ElseTI);

      // Build calls __kmpc_serialized_parallel(&Ident, GTid);
      Value *SerializedParallelCallArgs[] = {Ident, ThreadID};
      Builder.CreateCall(
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_serialized_parallel),
          SerializedParallelCallArgs);

      // OutlinedFn(&GTid, &zero, CapturedStruct);
      CI->removeFromParent();
      Builder.Insert(CI);

      // __kmpc_end_serialized_parallel(&Ident, GTid);
      Value *EndArgs[] = {Ident, ThreadID};
      Builder.CreateCall(
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_serialized_parallel),
          EndArgs);

      LLVM_DEBUG(dbgs() << "With serialized parallel region: "
                        << *Builder.GetInsertBlock()->getParent() << "\n");
    }

    for (Instruction *I : ToBeDeleted)
      I->eraseFromParent();
  };

  // Adjust the finalization stack, verify the adjustment, and call the
  // finalize function a last time to finalize values between the pre-fini
  // block and the exit block if we left the parallel "the normal way".
  auto FiniInfo = FinalizationStack.pop_back_val();
  (void)FiniInfo;
  assert(FiniInfo.DK == OMPD_parallel &&
         "Unexpected finalization stack state!");

  Instruction *PRegPreFiniTI = PRegPreFiniBB->getTerminator();

  InsertPointTy PreFiniIP(PRegPreFiniBB, PRegPreFiniTI->getIterator());
  FiniCB(PreFiniIP);

  OI.EntryBB = PRegEntryBB;
  OI.ExitBB = PRegExitBB;

  SmallPtrSet<BasicBlock *, 32> ParallelRegionBlockSet;
  SmallVector<BasicBlock *, 32> Blocks;
  OI.collectBlocks(ParallelRegionBlockSet, Blocks);

  // Ensure a single exit node for the outlined region by creating one.
  // We might have multiple incoming edges to the exit now due to finalizations,
  // e.g., cancel calls that cause the control flow to leave the region.
  BasicBlock *PRegOutlinedExitBB = PRegExitBB;
  PRegExitBB = SplitBlock(PRegExitBB, &*PRegExitBB->getFirstInsertionPt());
  PRegOutlinedExitBB->setName("omp.par.outlined.exit");
  Blocks.push_back(PRegOutlinedExitBB);

  CodeExtractorAnalysisCache CEAC(*OuterFn);
  CodeExtractor Extractor(Blocks, /* DominatorTree */ nullptr,
                          /* AggregateArgs */ false,
                          /* BlockFrequencyInfo */ nullptr,
                          /* BranchProbabilityInfo */ nullptr,
                          /* AssumptionCache */ nullptr,
                          /* AllowVarArgs */ true,
                          /* AllowAlloca */ true,
                          /* Suffix */ ".omp_par");

  // Find inputs to, outputs from the code region.
  BasicBlock *CommonExit = nullptr;
  SetVector<Value *> Inputs, Outputs, SinkingCands, HoistingCands;
  Extractor.findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  Extractor.findInputsOutputs(Inputs, Outputs, SinkingCands);

  LLVM_DEBUG(dbgs() << "Before privatization: " << *OuterFn << "\n");

  FunctionCallee TIDRTLFn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_global_thread_num);

  auto PrivHelper = [&](Value &V) {
    if (&V == TIDAddr || &V == ZeroAddr)
      return;

    SetVector<Use *> Uses;
    for (Use &U : V.uses())
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (ParallelRegionBlockSet.count(UserI->getParent()))
          Uses.insert(&U);

    // __kmpc_fork_call expects extra arguments as pointers. If the input
    // already has a pointer type, everything is fine. Otherwise, store the
    // value onto stack and load it back inside the to-be-outlined region. This
    // will ensure only the pointer will be passed to the function.
    // FIXME: if there are more than 15 trailing arguments, they must be
    // additionally packed in a struct.
    Value *Inner = &V;
    if (!V.getType()->isPointerTy()) {
      IRBuilder<>::InsertPointGuard Guard(Builder);
      LLVM_DEBUG(llvm::dbgs() << "Forwarding input as pointer: " << V << "\n");

      Builder.restoreIP(OuterAllocaIP);
      Value *Ptr =
          Builder.CreateAlloca(V.getType(), nullptr, V.getName() + ".reloaded");

      // Store to stack at end of the block that currently branches to the entry
      // block of the to-be-outlined region.
      Builder.SetInsertPoint(InsertBB,
                             InsertBB->getTerminator()->getIterator());
      Builder.CreateStore(&V, Ptr);

      // Load back next to allocations in the to-be-outlined region.
      Builder.restoreIP(InnerAllocaIP);
      Inner = Builder.CreateLoad(V.getType(), Ptr);
    }

    Value *ReplacementValue = nullptr;
    CallInst *CI = dyn_cast<CallInst>(&V);
    if (CI && CI->getCalledFunction() == TIDRTLFn.getCallee()) {
      ReplacementValue = PrivTID;
    } else {
      Builder.restoreIP(
          PrivCB(InnerAllocaIP, Builder.saveIP(), V, *Inner, ReplacementValue));
      assert(ReplacementValue &&
             "Expected copy/create callback to set replacement value!");
      if (ReplacementValue == &V)
        return;
    }

    for (Use *UPtr : Uses)
      UPtr->set(ReplacementValue);
  };

  // Reset the inner alloca insertion as it will be used for loading the values
  // wrapped into pointers before passing them into the to-be-outlined region.
  // Configure it to insert immediately after the fake use of zero address so
  // that they are available in the generated body and so that the
  // OpenMP-related values (thread ID and zero address pointers) remain leading
  // in the argument list.
  InnerAllocaIP = IRBuilder<>::InsertPoint(
      ZeroAddrUse->getParent(), ZeroAddrUse->getNextNode()->getIterator());

  // Reset the outer alloca insertion point to the entry of the relevant block
  // in case it was invalidated.
  OuterAllocaIP = IRBuilder<>::InsertPoint(
      OuterAllocaBlock, OuterAllocaBlock->getFirstInsertionPt());

  for (Value *Input : Inputs) {
    LLVM_DEBUG(dbgs() << "Captured input: " << *Input << "\n");
    PrivHelper(*Input);
  }
  LLVM_DEBUG({
    for (Value *Output : Outputs)
      LLVM_DEBUG(dbgs() << "Captured output: " << *Output << "\n");
  });
  assert(Outputs.empty() &&
         "OpenMP outlining should not produce live-out values!");

  LLVM_DEBUG(dbgs() << "After  privatization: " << *OuterFn << "\n");
  LLVM_DEBUG({
    for (auto *BB : Blocks)
      dbgs() << " PBR: " << BB->getName() << "\n";
  });

  // Register the outlined info.
  addOutlineInfo(std::move(OI));

  InsertPointTy AfterIP(UI->getParent(), UI->getParent()->end());
  UI->eraseFromParent();

  return AfterIP;
}

void OpenMPIRBuilder::emitFlush(const LocationDescription &Loc) {
  // Build call void __kmpc_flush(ident_t *loc)
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Args[] = {getOrCreateIdent(SrcLocStr)};

  Builder.CreateCall(getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_flush), Args);
}

void OpenMPIRBuilder::createFlush(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitFlush(Loc);
}

void OpenMPIRBuilder::emitTaskwaitImpl(const LocationDescription &Loc) {
  // Build call kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32
  // global_tid);
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *Args[] = {Ident, getOrCreateThreadID(Ident)};

  // Ignore return result until untied tasks are supported.
  Builder.CreateCall(getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_taskwait),
                     Args);
}

void OpenMPIRBuilder::createTaskwait(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitTaskwaitImpl(Loc);
}

void OpenMPIRBuilder::emitTaskyieldImpl(const LocationDescription &Loc) {
  // Build call __kmpc_omp_taskyield(loc, thread_id, 0);
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Constant *I32Null = ConstantInt::getNullValue(Int32);
  Value *Args[] = {Ident, getOrCreateThreadID(Ident), I32Null};

  Builder.CreateCall(getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_taskyield),
                     Args);
}

void OpenMPIRBuilder::createTaskyield(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitTaskyieldImpl(Loc);
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createReductions(
    const LocationDescription &Loc, InsertPointTy AllocaIP,
    ArrayRef<Value *> Variables, ArrayRef<Value *> PrivateVariables,
    ArrayRef<ReductionGenTy> ReductionGen,
    ArrayRef<ReductionGenTy> AtomicReductionGen, bool IsNoWait) {
  assert(Variables.size() == PrivateVariables.size());
  for (auto pair : zip(Variables, PrivateVariables)) {
    assert(std::get<0>(pair)->getType() == std::get<1>(pair)->getType() &&
           "expected variables and their private equivalents to have the same "
           "type");
    assert(std::get<0>(pair)->getType()->isPointerTy() &&
           "expected variables to be pointers");
  }
  assert(ReductionGen.size() == Variables.size());
  assert(AtomicReductionGen.size() == Variables.size() ||
         AtomicReductionGen.empty());

  if (!updateToLocation(Loc))
    return InsertPointTy();

  Type *RedArrayTy = ArrayType::get(Builder.getInt8PtrTy(), Variables.size());
  Value *RedArray;
  {
    IRBuilderBase::InsertPointGuard guard(Builder);
    Builder.restoreIP(AllocaIP);
    RedArray = Builder.CreateAlloca(RedArrayTy, nullptr, "red.array");
  }

  for (unsigned i = 0, e = Variables.size(); i < e; ++i) {
    Value *RedArrayElemPtr = Builder.CreateConstInBoundsGEP2_64(
        RedArray, 0, i, "red.array.elem." + Twine(i));
    Value *Casted =
        Builder.CreateBitCast(PrivateVariables[i], Builder.getInt8PtrTy(),
                              "private.red.var." + Twine(i) + ".casted");
    Builder.CreateStore(Casted, RedArrayElemPtr);
  }

  Function *function = Builder.GetInsertBlock()->getParent();
  Module *module = function->getParent();
  Value *RedArrayPtr =
      Builder.CreateBitCast(RedArray, Builder.getInt8PtrTy(), "red.array.ptr");
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident =
      getOrCreateIdent(SrcLocStr, !AtomicReductionGen.empty()
                                      ? IdentFlag::OMP_IDENT_FLAG_ATOMIC_REDUCE
                                      : IdentFlag(0));
  Value *ThreadId = getOrCreateThreadID(Ident);
  Constant *NumVariables = Builder.getInt32(Variables.size());
  const DataLayout &DL = module->getDataLayout();
  unsigned RedArrayByteSize = divideCeil(DL.getTypeSizeInBits(RedArrayTy), 8);
  Constant *RedArraySize = Builder.getInt64(RedArrayByteSize);
  FunctionCallee ReductionFunc = module->getOrInsertFunction(
      ".omp.reduction.func", Builder.getVoidTy(), Builder.getInt8PtrTy(),
      Builder.getInt8PtrTy());
  Value *Lock = getOMPCriticalRegionLock(".reduction");
  Function *ReduceFunc = getOrCreateRuntimeFunctionPtr(
      IsNoWait ? RuntimeFunction::OMPRTL___kmpc_reduce_nowait
               : RuntimeFunction::OMPRTL___kmpc_reduce);
  CallInst *ReduceCall =
      Builder.CreateCall(ReduceFunc,
                         {Ident, ThreadId, NumVariables, RedArraySize,
                          RedArrayPtr, ReductionFunc.getCallee(), Lock},
                         "reduce");

  BasicBlock *NonAtomicRedBlock = BasicBlock::Create(
      module->getContext(), "reduce.switch.nonatomic", function);
  BasicBlock *AtomicRedBlock = BasicBlock::Create(
      module->getContext(), "reduce.switch.atomic", function);
  BasicBlock *ContinuationBlock =
      BasicBlock::Create(module->getContext(), "reduce.switch.cont", function);
  SwitchInst *Switch =
      Builder.CreateSwitch(ReduceCall, ContinuationBlock, /* NumCases */ 2);
  Switch->addCase(Builder.getInt32(1), NonAtomicRedBlock);
  Switch->addCase(Builder.getInt32(2), AtomicRedBlock);

  Builder.SetInsertPoint(NonAtomicRedBlock);
  for (unsigned i = 0, e = Variables.size(); i < e; ++i) {
    Value *RedValue = Builder.CreateLoad(Variables[i], "red.value." + Twine(i));
    Value *PrivateRedValue = Builder.CreateLoad(
        PrivateVariables[i], "red.private.value." + Twine(i));
    Value *Reduced;
    Builder.restoreIP(
        ReductionGen[i](Builder.saveIP(), RedValue, PrivateRedValue, Reduced));
    if (!Builder.GetInsertBlock())
      return InsertPointTy();
    Builder.CreateStore(Reduced, Variables[i]);
  }
  Function *EndReduceFunc = getOrCreateRuntimeFunctionPtr(
      IsNoWait ? RuntimeFunction::OMPRTL___kmpc_end_reduce_nowait
               : RuntimeFunction::OMPRTL___kmpc_end_reduce);
  Builder.CreateCall(EndReduceFunc, {Ident, ThreadId, Lock});
  Builder.CreateBr(ContinuationBlock);

  Builder.SetInsertPoint(AtomicRedBlock);
  if (AtomicReductionGen.empty()) {
    Builder.CreateUnreachable();
  } else {
    for (unsigned i = 0, e = Variables.size(); i < e; ++i) {
      Value *unused;
      Builder.restoreIP(AtomicReductionGen[i](Builder.saveIP(), Variables[i],
                                              PrivateVariables[i], unused));
      if (!Builder.GetInsertBlock())
        return InsertPointTy();
    }
    Builder.CreateBr(ContinuationBlock);
  }

  // Populate the outlined reduction function.
  Function *ReductionFunction = cast<Function>(ReductionFunc.getCallee());
  BasicBlock *ReductionFuncBlock =
      BasicBlock::Create(module->getContext(), "", ReductionFunction);
  Builder.SetInsertPoint(ReductionFuncBlock);
  Value *LHSArrayPtr = Builder.CreateBitCast(ReductionFunction->getArg(0),
                                             RedArrayTy->getPointerTo());
  Value *RHSArrayPtr = Builder.CreateBitCast(ReductionFunction->getArg(1),
                                             RedArrayTy->getPointerTo());
  for (unsigned i = 0, e = Variables.size(); i < e; ++i) {
    Value *LHSI8PtrPtr = Builder.CreateConstInBoundsGEP2_64(LHSArrayPtr, 0, i);
    Value *LHSI8Ptr = Builder.CreateLoad(LHSI8PtrPtr);
    Value *LHSPtr = Builder.CreateBitCast(LHSI8Ptr, Variables[i]->getType());
    Value *LHS = Builder.CreateLoad(LHSPtr);
    Value *RHSI8PtrPtr = Builder.CreateConstInBoundsGEP2_64(RHSArrayPtr, 0, i);
    Value *RHSI8Ptr = Builder.CreateLoad(RHSI8PtrPtr);
    Value *RHSPtr =
        Builder.CreateBitCast(RHSI8Ptr, PrivateVariables[i]->getType());
    Value *RHS = Builder.CreateLoad(RHSPtr);
    Value *Reduced;
    Builder.restoreIP(ReductionGen[i](Builder.saveIP(), LHS, RHS, Reduced));
    if (!Builder.GetInsertBlock())
      return InsertPointTy();
    Builder.CreateStore(Reduced, LHSPtr);
  }
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(ContinuationBlock);
  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createMaster(const LocationDescription &Loc,
                              BodyGenCallbackTy BodyGenCB,
                              FinalizeCallbackTy FiniCB) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_master;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {Ident, ThreadId};

  Function *EntryRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_master);
  Instruction *EntryCall = Builder.CreateCall(EntryRTLFn, Args);

  Function *ExitRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_master);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, Args);

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ true, /*hasFinalize*/ true);
}

CanonicalLoopInfo *OpenMPIRBuilder::createLoopSkeleton(
    DebugLoc DL, Value *TripCount, Function *F, BasicBlock *PreInsertBefore,
    BasicBlock *PostInsertBefore, const Twine &Name) {
  Module *M = F->getParent();
  LLVMContext &Ctx = M->getContext();
  Type *IndVarTy = TripCount->getType();

  // Create the basic block structure.
  BasicBlock *Preheader =
      BasicBlock::Create(Ctx, "omp_" + Name + ".preheader", F, PreInsertBefore);
  BasicBlock *Header =
      BasicBlock::Create(Ctx, "omp_" + Name + ".header", F, PreInsertBefore);
  BasicBlock *Cond =
      BasicBlock::Create(Ctx, "omp_" + Name + ".cond", F, PreInsertBefore);
  BasicBlock *Body =
      BasicBlock::Create(Ctx, "omp_" + Name + ".body", F, PreInsertBefore);
  BasicBlock *Latch =
      BasicBlock::Create(Ctx, "omp_" + Name + ".inc", F, PostInsertBefore);
  BasicBlock *Exit =
      BasicBlock::Create(Ctx, "omp_" + Name + ".exit", F, PostInsertBefore);
  BasicBlock *After =
      BasicBlock::Create(Ctx, "omp_" + Name + ".after", F, PostInsertBefore);

  // Use specified DebugLoc for new instructions.
  Builder.SetCurrentDebugLocation(DL);

  Builder.SetInsertPoint(Preheader);
  Builder.CreateBr(Header);

  Builder.SetInsertPoint(Header);
  PHINode *IndVarPHI = Builder.CreatePHI(IndVarTy, 2, "omp_" + Name + ".iv");
  IndVarPHI->addIncoming(ConstantInt::get(IndVarTy, 0), Preheader);
  Builder.CreateBr(Cond);

  Builder.SetInsertPoint(Cond);
  Value *Cmp =
      Builder.CreateICmpULT(IndVarPHI, TripCount, "omp_" + Name + ".cmp");
  Builder.CreateCondBr(Cmp, Body, Exit);

  Builder.SetInsertPoint(Body);
  Builder.CreateBr(Latch);

  Builder.SetInsertPoint(Latch);
  Value *Next = Builder.CreateAdd(IndVarPHI, ConstantInt::get(IndVarTy, 1),
                                  "omp_" + Name + ".next", /*HasNUW=*/true);
  Builder.CreateBr(Header);
  IndVarPHI->addIncoming(Next, Latch);

  Builder.SetInsertPoint(Exit);
  Builder.CreateBr(After);

  // Remember and return the canonical control flow.
  LoopInfos.emplace_front();
  CanonicalLoopInfo *CL = &LoopInfos.front();

  CL->Preheader = Preheader;
  CL->Header = Header;
  CL->Cond = Cond;
  CL->Body = Body;
  CL->Latch = Latch;
  CL->Exit = Exit;
  CL->After = After;

  CL->IsValid = true;

#ifndef NDEBUG
  CL->assertOK();
#endif
  return CL;
}

CanonicalLoopInfo *
OpenMPIRBuilder::createCanonicalLoop(const LocationDescription &Loc,
                                     LoopBodyGenCallbackTy BodyGenCB,
                                     Value *TripCount, const Twine &Name) {
  BasicBlock *BB = Loc.IP.getBlock();
  BasicBlock *NextBB = BB->getNextNode();

  CanonicalLoopInfo *CL = createLoopSkeleton(Loc.DL, TripCount, BB->getParent(),
                                             NextBB, NextBB, Name);
  BasicBlock *After = CL->getAfter();

  // If location is not set, don't connect the loop.
  if (updateToLocation(Loc)) {
    // Split the loop at the insertion point: Branch to the preheader and move
    // every following instruction to after the loop (the After BB). Also, the
    // new successor is the loop's after block.
    Builder.CreateBr(CL->Preheader);
    After->getInstList().splice(After->begin(), BB->getInstList(),
                                Builder.GetInsertPoint(), BB->end());
    After->replaceSuccessorsPhiUsesWith(BB, After);
  }

  // Emit the body content. We do it after connecting the loop to the CFG to
  // avoid that the callback encounters degenerate BBs.
  BodyGenCB(CL->getBodyIP(), CL->getIndVar());

#ifndef NDEBUG
  CL->assertOK();
#endif
  return CL;
}

CanonicalLoopInfo *OpenMPIRBuilder::createCanonicalLoop(
    const LocationDescription &Loc, LoopBodyGenCallbackTy BodyGenCB,
    Value *Start, Value *Stop, Value *Step, bool IsSigned, bool InclusiveStop,
    InsertPointTy ComputeIP, const Twine &Name) {

  // Consider the following difficulties (assuming 8-bit signed integers):
  //  * Adding \p Step to the loop counter which passes \p Stop may overflow:
  //      DO I = 1, 100, 50
  ///  * A \p Step of INT_MIN cannot not be normalized to a positive direction:
  //      DO I = 100, 0, -128

  // Start, Stop and Step must be of the same integer type.
  auto *IndVarTy = cast<IntegerType>(Start->getType());
  assert(IndVarTy == Stop->getType() && "Stop type mismatch");
  assert(IndVarTy == Step->getType() && "Step type mismatch");

  LocationDescription ComputeLoc =
      ComputeIP.isSet() ? LocationDescription(ComputeIP, Loc.DL) : Loc;
  updateToLocation(ComputeLoc);

  ConstantInt *Zero = ConstantInt::get(IndVarTy, 0);
  ConstantInt *One = ConstantInt::get(IndVarTy, 1);

  // Like Step, but always positive.
  Value *Incr = Step;

  // Distance between Start and Stop; always positive.
  Value *Span;

  // Condition whether there are no iterations are executed at all, e.g. because
  // UB < LB.
  Value *ZeroCmp;

  if (IsSigned) {
    // Ensure that increment is positive. If not, negate and invert LB and UB.
    Value *IsNeg = Builder.CreateICmpSLT(Step, Zero);
    Incr = Builder.CreateSelect(IsNeg, Builder.CreateNeg(Step), Step);
    Value *LB = Builder.CreateSelect(IsNeg, Stop, Start);
    Value *UB = Builder.CreateSelect(IsNeg, Start, Stop);
    Span = Builder.CreateSub(UB, LB, "", false, true);
    ZeroCmp = Builder.CreateICmp(
        InclusiveStop ? CmpInst::ICMP_SLT : CmpInst::ICMP_SLE, UB, LB);
  } else {
    Span = Builder.CreateSub(Stop, Start, "", true);
    ZeroCmp = Builder.CreateICmp(
        InclusiveStop ? CmpInst::ICMP_ULT : CmpInst::ICMP_ULE, Stop, Start);
  }

  Value *CountIfLooping;
  if (InclusiveStop) {
    CountIfLooping = Builder.CreateAdd(Builder.CreateUDiv(Span, Incr), One);
  } else {
    // Avoid incrementing past stop since it could overflow.
    Value *CountIfTwo = Builder.CreateAdd(
        Builder.CreateUDiv(Builder.CreateSub(Span, One), Incr), One);
    Value *OneCmp = Builder.CreateICmp(
        InclusiveStop ? CmpInst::ICMP_ULT : CmpInst::ICMP_ULE, Span, Incr);
    CountIfLooping = Builder.CreateSelect(OneCmp, One, CountIfTwo);
  }
  Value *TripCount = Builder.CreateSelect(ZeroCmp, Zero, CountIfLooping,
                                          "omp_" + Name + ".tripcount");

  auto BodyGen = [=](InsertPointTy CodeGenIP, Value *IV) {
    Builder.restoreIP(CodeGenIP);
    Value *Span = Builder.CreateMul(IV, Step);
    Value *IndVar = Builder.CreateAdd(Span, Start);
    BodyGenCB(Builder.saveIP(), IndVar);
  };
  LocationDescription LoopLoc = ComputeIP.isSet() ? Loc.IP : Builder.saveIP();
  return createCanonicalLoop(LoopLoc, BodyGen, TripCount, Name);
}

// Returns an LLVM function to call for initializing loop bounds using OpenMP
// static scheduling depending on `type`. Only i32 and i64 are supported by the
// runtime. Always interpret integers as unsigned similarly to
// CanonicalLoopInfo.
static FunctionCallee getKmpcForStaticInitForType(Type *Ty, Module &M,
                                                  OpenMPIRBuilder &OMPBuilder) {
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_for_static_init_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_for_static_init_8u);
  llvm_unreachable("unknown OpenMP loop iterator bitwidth");
}

// Sets the number of loop iterations to the given value. This value must be
// valid in the condition block (i.e., defined in the preheader) and is
// interpreted as an unsigned integer.
void setCanonicalLoopTripCount(CanonicalLoopInfo *CLI, Value *TripCount) {
  Instruction *CmpI = &CLI->getCond()->front();
  assert(isa<CmpInst>(CmpI) && "First inst must compare IV with TripCount");
  CmpI->setOperand(1, TripCount);
  CLI->assertOK();
}

CanonicalLoopInfo *OpenMPIRBuilder::createStaticWorkshareLoop(
    const LocationDescription &Loc, CanonicalLoopInfo *CLI,
    InsertPointTy AllocaIP, bool NeedsBarrier, Value *Chunk) {
  // Set up the source location value for OpenMP runtime.
  if (!updateToLocation(Loc))
    return nullptr;

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *SrcLoc = getOrCreateIdent(SrcLocStr);

  // Declare useful OpenMP runtime functions.
  Value *IV = CLI->getIndVar();
  Type *IVTy = IV->getType();
  FunctionCallee StaticInit = getKmpcForStaticInitForType(IVTy, M, *this);
  FunctionCallee StaticFini =
      getOrCreateRuntimeFunction(M, omp::OMPRTL___kmpc_for_static_fini);

  // Allocate space for computed loop bounds as expected by the "init" function.
  Builder.restoreIP(AllocaIP);
  Type *I32Type = Type::getInt32Ty(M.getContext());
  Value *PLastIter = Builder.CreateAlloca(I32Type, nullptr, "p.lastiter");
  Value *PLowerBound = Builder.CreateAlloca(IVTy, nullptr, "p.lowerbound");
  Value *PUpperBound = Builder.CreateAlloca(IVTy, nullptr, "p.upperbound");
  Value *PStride = Builder.CreateAlloca(IVTy, nullptr, "p.stride");

  // At the end of the preheader, prepare for calling the "init" function by
  // storing the current loop bounds into the allocated space. A canonical loop
  // always iterates from 0 to trip-count with step 1. Note that "init" expects
  // and produces an inclusive upper bound.
  Builder.SetInsertPoint(CLI->getPreheader()->getTerminator());
  Constant *Zero = ConstantInt::get(IVTy, 0);
  Constant *One = ConstantInt::get(IVTy, 1);
  Builder.CreateStore(Zero, PLowerBound);
  Value *UpperBound = Builder.CreateSub(CLI->getTripCount(), One);
  Builder.CreateStore(UpperBound, PUpperBound);
  Builder.CreateStore(One, PStride);

  if (!Chunk)
    Chunk = One;

  Value *ThreadNum = getOrCreateThreadID(SrcLoc);

  // TODO: extract scheduling type and map it to OMP constant. This is curently
  // happening in kmp.h and its ilk and needs to be moved to OpenMP.td first.
  constexpr int StaticSchedType = 34;
  Constant *SchedulingType = ConstantInt::get(I32Type, StaticSchedType);

  // Call the "init" function and update the trip count of the loop with the
  // value it produced.
  Builder.CreateCall(StaticInit,
                     {SrcLoc, ThreadNum, SchedulingType, PLastIter, PLowerBound,
                      PUpperBound, PStride, One, Chunk});
  Value *LowerBound = Builder.CreateLoad(IVTy, PLowerBound);
  Value *InclusiveUpperBound = Builder.CreateLoad(IVTy, PUpperBound);
  Value *TripCountMinusOne = Builder.CreateSub(InclusiveUpperBound, LowerBound);
  Value *TripCount = Builder.CreateAdd(TripCountMinusOne, One);
  setCanonicalLoopTripCount(CLI, TripCount);

  // Update all uses of the induction variable except the one in the condition
  // block that compares it with the actual upper bound, and the increment in
  // the latch block.
  // TODO: this can eventually move to CanonicalLoopInfo or to a new
  // CanonicalLoopInfoUpdater interface.
  Builder.SetInsertPoint(CLI->getBody(), CLI->getBody()->getFirstInsertionPt());
  Value *UpdatedIV = Builder.CreateAdd(IV, LowerBound);
  IV->replaceUsesWithIf(UpdatedIV, [&](Use &U) {
    auto *Instr = dyn_cast<Instruction>(U.getUser());
    return !Instr ||
           (Instr->getParent() != CLI->getCond() &&
            Instr->getParent() != CLI->getLatch() && Instr != UpdatedIV);
  });

  // In the "exit" block, call the "fini" function.
  Builder.SetInsertPoint(CLI->getExit(),
                         CLI->getExit()->getTerminator()->getIterator());
  Builder.CreateCall(StaticFini, {SrcLoc, ThreadNum});

  // Add the barrier if requested.
  if (NeedsBarrier)
    createBarrier(LocationDescription(Builder.saveIP(), Loc.DL),
                  omp::Directive::OMPD_for, /* ForceSimpleCall */ false,
                  /* CheckCancelFlag */ false);

  CLI->assertOK();
  return CLI;
}

CanonicalLoopInfo *OpenMPIRBuilder::createWorkshareLoop(
    const LocationDescription &Loc, CanonicalLoopInfo *CLI,
    InsertPointTy AllocaIP, bool NeedsBarrier) {
  // Currently only supports static schedules.
  return createStaticWorkshareLoop(Loc, CLI, AllocaIP, NeedsBarrier);
}

/// Make \p Source branch to \p Target.
///
/// Handles two situations:
/// * \p Source already has an unconditional branch.
/// * \p Source is a degenerate block (no terminator because the BB is
///             the current head of the IR construction).
static void redirectTo(BasicBlock *Source, BasicBlock *Target, DebugLoc DL) {
  if (Instruction *Term = Source->getTerminator()) {
    auto *Br = cast<BranchInst>(Term);
    assert(!Br->isConditional() &&
           "BB's terminator must be an unconditional branch (or degenerate)");
    BasicBlock *Succ = Br->getSuccessor(0);
    Succ->removePredecessor(Source, /*KeepOneInputPHIs=*/true);
    Br->setSuccessor(0, Target);
    return;
  }

  auto *NewBr = BranchInst::Create(Target, Source);
  NewBr->setDebugLoc(DL);
}

/// Redirect all edges that branch to \p OldTarget to \p NewTarget. That is,
/// after this \p OldTarget will be orphaned.
static void redirectAllPredecessorsTo(BasicBlock *OldTarget,
                                      BasicBlock *NewTarget, DebugLoc DL) {
  for (BasicBlock *Pred : make_early_inc_range(predecessors(OldTarget)))
    redirectTo(Pred, NewTarget, DL);
}

/// Determine which blocks in \p BBs are reachable from outside and remove the
/// ones that are not reachable from the function.
static void removeUnusedBlocksFromParent(ArrayRef<BasicBlock *> BBs) {
  SmallPtrSet<BasicBlock *, 6> BBsToErase{BBs.begin(), BBs.end()};
  auto HasRemainingUses = [&BBsToErase](BasicBlock *BB) {
    for (Use &U : BB->uses()) {
      auto *UseInst = dyn_cast<Instruction>(U.getUser());
      if (!UseInst)
        continue;
      if (BBsToErase.count(UseInst->getParent()))
        continue;
      return true;
    }
    return false;
  };

  while (true) {
    bool Changed = false;
    for (BasicBlock *BB : make_early_inc_range(BBsToErase)) {
      if (HasRemainingUses(BB)) {
        BBsToErase.erase(BB);
        Changed = true;
      }
    }
    if (!Changed)
      break;
  }

  SmallVector<BasicBlock *, 7> BBVec(BBsToErase.begin(), BBsToErase.end());
  DeleteDeadBlocks(BBVec);
}

CanonicalLoopInfo *
OpenMPIRBuilder::collapseLoops(DebugLoc DL, ArrayRef<CanonicalLoopInfo *> Loops,
                               InsertPointTy ComputeIP) {
  assert(Loops.size() >= 1 && "At least one loop required");
  size_t NumLoops = Loops.size();

  // Nothing to do if there is already just one loop.
  if (NumLoops == 1)
    return Loops.front();

  CanonicalLoopInfo *Outermost = Loops.front();
  CanonicalLoopInfo *Innermost = Loops.back();
  BasicBlock *OrigPreheader = Outermost->getPreheader();
  BasicBlock *OrigAfter = Outermost->getAfter();
  Function *F = OrigPreheader->getParent();

  // Setup the IRBuilder for inserting the trip count computation.
  Builder.SetCurrentDebugLocation(DL);
  if (ComputeIP.isSet())
    Builder.restoreIP(ComputeIP);
  else
    Builder.restoreIP(Outermost->getPreheaderIP());

  // Derive the collapsed' loop trip count.
  // TODO: Find common/largest indvar type.
  Value *CollapsedTripCount = nullptr;
  for (CanonicalLoopInfo *L : Loops) {
    Value *OrigTripCount = L->getTripCount();
    if (!CollapsedTripCount) {
      CollapsedTripCount = OrigTripCount;
      continue;
    }

    // TODO: Enable UndefinedSanitizer to diagnose an overflow here.
    CollapsedTripCount = Builder.CreateMul(CollapsedTripCount, OrigTripCount,
                                           {}, /*HasNUW=*/true);
  }

  // Create the collapsed loop control flow.
  CanonicalLoopInfo *Result =
      createLoopSkeleton(DL, CollapsedTripCount, F,
                         OrigPreheader->getNextNode(), OrigAfter, "collapsed");

  // Build the collapsed loop body code.
  // Start with deriving the input loop induction variables from the collapsed
  // one, using a divmod scheme. To preserve the original loops' order, the
  // innermost loop use the least significant bits.
  Builder.restoreIP(Result->getBodyIP());

  Value *Leftover = Result->getIndVar();
  SmallVector<Value *> NewIndVars;
  NewIndVars.set_size(NumLoops);
  for (int i = NumLoops - 1; i >= 1; --i) {
    Value *OrigTripCount = Loops[i]->getTripCount();

    Value *NewIndVar = Builder.CreateURem(Leftover, OrigTripCount);
    NewIndVars[i] = NewIndVar;

    Leftover = Builder.CreateUDiv(Leftover, OrigTripCount);
  }
  // Outermost loop gets all the remaining bits.
  NewIndVars[0] = Leftover;

  // Construct the loop body control flow.
  // We progressively construct the branch structure following in direction of
  // the control flow, from the leading in-between code, the loop nest body, the
  // trailing in-between code, and rejoining the collapsed loop's latch.
  // ContinueBlock and ContinuePred keep track of the source(s) of next edge. If
  // the ContinueBlock is set, continue with that block. If ContinuePred, use
  // its predecessors as sources.
  BasicBlock *ContinueBlock = Result->getBody();
  BasicBlock *ContinuePred = nullptr;
  auto ContinueWith = [&ContinueBlock, &ContinuePred, DL](BasicBlock *Dest,
                                                          BasicBlock *NextSrc) {
    if (ContinueBlock)
      redirectTo(ContinueBlock, Dest, DL);
    else
      redirectAllPredecessorsTo(ContinuePred, Dest, DL);

    ContinueBlock = nullptr;
    ContinuePred = NextSrc;
  };

  // The code before the nested loop of each level.
  // Because we are sinking it into the nest, it will be executed more often
  // that the original loop. More sophisticated schemes could keep track of what
  // the in-between code is and instantiate it only once per thread.
  for (size_t i = 0; i < NumLoops - 1; ++i)
    ContinueWith(Loops[i]->getBody(), Loops[i + 1]->getHeader());

  // Connect the loop nest body.
  ContinueWith(Innermost->getBody(), Innermost->getLatch());

  // The code after the nested loop at each level.
  for (size_t i = NumLoops - 1; i > 0; --i)
    ContinueWith(Loops[i]->getAfter(), Loops[i - 1]->getLatch());

  // Connect the finished loop to the collapsed loop latch.
  ContinueWith(Result->getLatch(), nullptr);

  // Replace the input loops with the new collapsed loop.
  redirectTo(Outermost->getPreheader(), Result->getPreheader(), DL);
  redirectTo(Result->getAfter(), Outermost->getAfter(), DL);

  // Replace the input loop indvars with the derived ones.
  for (size_t i = 0; i < NumLoops; ++i)
    Loops[i]->getIndVar()->replaceAllUsesWith(NewIndVars[i]);

  // Remove unused parts of the input loops.
  SmallVector<BasicBlock *, 12> OldControlBBs;
  OldControlBBs.reserve(6 * Loops.size());
  for (CanonicalLoopInfo *Loop : Loops)
    Loop->collectControlBlocks(OldControlBBs);
  removeUnusedBlocksFromParent(OldControlBBs);

#ifndef NDEBUG
  Result->assertOK();
#endif
  return Result;
}

std::vector<CanonicalLoopInfo *>
OpenMPIRBuilder::tileLoops(DebugLoc DL, ArrayRef<CanonicalLoopInfo *> Loops,
                           ArrayRef<Value *> TileSizes) {
  assert(TileSizes.size() == Loops.size() &&
         "Must pass as many tile sizes as there are loops");
  int NumLoops = Loops.size();
  assert(NumLoops >= 1 && "At least one loop to tile required");

  CanonicalLoopInfo *OutermostLoop = Loops.front();
  CanonicalLoopInfo *InnermostLoop = Loops.back();
  Function *F = OutermostLoop->getBody()->getParent();
  BasicBlock *InnerEnter = InnermostLoop->getBody();
  BasicBlock *InnerLatch = InnermostLoop->getLatch();

  // Collect original trip counts and induction variable to be accessible by
  // index. Also, the structure of the original loops is not preserved during
  // the construction of the tiled loops, so do it before we scavenge the BBs of
  // any original CanonicalLoopInfo.
  SmallVector<Value *, 4> OrigTripCounts, OrigIndVars;
  for (CanonicalLoopInfo *L : Loops) {
    OrigTripCounts.push_back(L->getTripCount());
    OrigIndVars.push_back(L->getIndVar());
  }

  // Collect the code between loop headers. These may contain SSA definitions
  // that are used in the loop nest body. To be usable with in the innermost
  // body, these BasicBlocks will be sunk into the loop nest body. That is,
  // these instructions may be executed more often than before the tiling.
  // TODO: It would be sufficient to only sink them into body of the
  // corresponding tile loop.
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> InbetweenCode;
  for (int i = 0; i < NumLoops - 1; ++i) {
    CanonicalLoopInfo *Surrounding = Loops[i];
    CanonicalLoopInfo *Nested = Loops[i + 1];

    BasicBlock *EnterBB = Surrounding->getBody();
    BasicBlock *ExitBB = Nested->getHeader();
    InbetweenCode.emplace_back(EnterBB, ExitBB);
  }

  // Compute the trip counts of the floor loops.
  Builder.SetCurrentDebugLocation(DL);
  Builder.restoreIP(OutermostLoop->getPreheaderIP());
  SmallVector<Value *, 4> FloorCount, FloorRems;
  for (int i = 0; i < NumLoops; ++i) {
    Value *TileSize = TileSizes[i];
    Value *OrigTripCount = OrigTripCounts[i];
    Type *IVType = OrigTripCount->getType();

    Value *FloorTripCount = Builder.CreateUDiv(OrigTripCount, TileSize);
    Value *FloorTripRem = Builder.CreateURem(OrigTripCount, TileSize);

    // 0 if tripcount divides the tilesize, 1 otherwise.
    // 1 means we need an additional iteration for a partial tile.
    //
    // Unfortunately we cannot just use the roundup-formula
    //   (tripcount + tilesize - 1)/tilesize
    // because the summation might overflow. We do not want introduce undefined
    // behavior when the untiled loop nest did not.
    Value *FloorTripOverflow =
        Builder.CreateICmpNE(FloorTripRem, ConstantInt::get(IVType, 0));

    FloorTripOverflow = Builder.CreateZExt(FloorTripOverflow, IVType);
    FloorTripCount =
        Builder.CreateAdd(FloorTripCount, FloorTripOverflow,
                          "omp_floor" + Twine(i) + ".tripcount", true);

    // Remember some values for later use.
    FloorCount.push_back(FloorTripCount);
    FloorRems.push_back(FloorTripRem);
  }

  // Generate the new loop nest, from the outermost to the innermost.
  std::vector<CanonicalLoopInfo *> Result;
  Result.reserve(NumLoops * 2);

  // The basic block of the surrounding loop that enters the nest generated
  // loop.
  BasicBlock *Enter = OutermostLoop->getPreheader();

  // The basic block of the surrounding loop where the inner code should
  // continue.
  BasicBlock *Continue = OutermostLoop->getAfter();

  // Where the next loop basic block should be inserted.
  BasicBlock *OutroInsertBefore = InnermostLoop->getExit();

  auto EmbeddNewLoop =
      [this, DL, F, InnerEnter, &Enter, &Continue, &OutroInsertBefore](
          Value *TripCount, const Twine &Name) -> CanonicalLoopInfo * {
    CanonicalLoopInfo *EmbeddedLoop = createLoopSkeleton(
        DL, TripCount, F, InnerEnter, OutroInsertBefore, Name);
    redirectTo(Enter, EmbeddedLoop->getPreheader(), DL);
    redirectTo(EmbeddedLoop->getAfter(), Continue, DL);

    // Setup the position where the next embedded loop connects to this loop.
    Enter = EmbeddedLoop->getBody();
    Continue = EmbeddedLoop->getLatch();
    OutroInsertBefore = EmbeddedLoop->getLatch();
    return EmbeddedLoop;
  };

  auto EmbeddNewLoops = [&Result, &EmbeddNewLoop](ArrayRef<Value *> TripCounts,
                                                  const Twine &NameBase) {
    for (auto P : enumerate(TripCounts)) {
      CanonicalLoopInfo *EmbeddedLoop =
          EmbeddNewLoop(P.value(), NameBase + Twine(P.index()));
      Result.push_back(EmbeddedLoop);
    }
  };

  EmbeddNewLoops(FloorCount, "floor");

  // Within the innermost floor loop, emit the code that computes the tile
  // sizes.
  Builder.SetInsertPoint(Enter->getTerminator());
  SmallVector<Value *, 4> TileCounts;
  for (int i = 0; i < NumLoops; ++i) {
    CanonicalLoopInfo *FloorLoop = Result[i];
    Value *TileSize = TileSizes[i];

    Value *FloorIsEpilogue =
        Builder.CreateICmpEQ(FloorLoop->getIndVar(), FloorCount[i]);
    Value *TileTripCount =
        Builder.CreateSelect(FloorIsEpilogue, FloorRems[i], TileSize);

    TileCounts.push_back(TileTripCount);
  }

  // Create the tile loops.
  EmbeddNewLoops(TileCounts, "tile");

  // Insert the inbetween code into the body.
  BasicBlock *BodyEnter = Enter;
  BasicBlock *BodyEntered = nullptr;
  for (std::pair<BasicBlock *, BasicBlock *> P : InbetweenCode) {
    BasicBlock *EnterBB = P.first;
    BasicBlock *ExitBB = P.second;

    if (BodyEnter)
      redirectTo(BodyEnter, EnterBB, DL);
    else
      redirectAllPredecessorsTo(BodyEntered, EnterBB, DL);

    BodyEnter = nullptr;
    BodyEntered = ExitBB;
  }

  // Append the original loop nest body into the generated loop nest body.
  if (BodyEnter)
    redirectTo(BodyEnter, InnerEnter, DL);
  else
    redirectAllPredecessorsTo(BodyEntered, InnerEnter, DL);
  redirectAllPredecessorsTo(InnerLatch, Continue, DL);

  // Replace the original induction variable with an induction variable computed
  // from the tile and floor induction variables.
  Builder.restoreIP(Result.back()->getBodyIP());
  for (int i = 0; i < NumLoops; ++i) {
    CanonicalLoopInfo *FloorLoop = Result[i];
    CanonicalLoopInfo *TileLoop = Result[NumLoops + i];
    Value *OrigIndVar = OrigIndVars[i];
    Value *Size = TileSizes[i];

    Value *Scale =
        Builder.CreateMul(Size, FloorLoop->getIndVar(), {}, /*HasNUW=*/true);
    Value *Shift =
        Builder.CreateAdd(Scale, TileLoop->getIndVar(), {}, /*HasNUW=*/true);
    OrigIndVar->replaceAllUsesWith(Shift);
  }

  // Remove unused parts of the original loops.
  SmallVector<BasicBlock *, 12> OldControlBBs;
  OldControlBBs.reserve(6 * Loops.size());
  for (CanonicalLoopInfo *Loop : Loops)
    Loop->collectControlBlocks(OldControlBBs);
  removeUnusedBlocksFromParent(OldControlBBs);

#ifndef NDEBUG
  for (CanonicalLoopInfo *GenL : Result)
    GenL->assertOK();
#endif
  return Result;
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createCopyPrivate(const LocationDescription &Loc,
                                   llvm::Value *BufSize, llvm::Value *CpyBuf,
                                   llvm::Value *CpyFn, llvm::Value *DidIt) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);

  llvm::Value *DidItLD = Builder.CreateLoad(Builder.getInt32Ty(), DidIt);

  Value *Args[] = {Ident, ThreadId, BufSize, CpyBuf, CpyFn, DidItLD};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_copyprivate);
  Builder.CreateCall(Fn, Args);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createSingle(const LocationDescription &Loc,
                              BodyGenCallbackTy BodyGenCB,
                              FinalizeCallbackTy FiniCB, llvm::Value *DidIt) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  // If needed (i.e. not null), initialize `DidIt` with 0
  if (DidIt) {
    Builder.CreateStore(Builder.getInt32(0), DidIt);
  }

  Directive OMPD = Directive::OMPD_single;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {Ident, ThreadId};

  Function *EntryRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_single);
  Instruction *EntryCall = Builder.CreateCall(EntryRTLFn, Args);

  Function *ExitRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_single);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, Args);

  // generates the following:
  // if (__kmpc_single()) {
  //		.... single region ...
  // 		__kmpc_end_single
  // }

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ true, /*hasFinalize*/ true);
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createCritical(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    FinalizeCallbackTy FiniCB, StringRef CriticalName, Value *HintInst) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_critical;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *LockVar = getOMPCriticalRegionLock(CriticalName);
  Value *Args[] = {Ident, ThreadId, LockVar};

  SmallVector<llvm::Value *, 4> EnterArgs(std::begin(Args), std::end(Args));
  Function *RTFn = nullptr;
  if (HintInst) {
    // Add Hint to entry Args and create call
    EnterArgs.push_back(HintInst);
    RTFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_critical_with_hint);
  } else {
    RTFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_critical);
  }
  Instruction *EntryCall = Builder.CreateCall(RTFn, EnterArgs);

  Function *ExitRTLFn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_critical);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, Args);

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ false, /*hasFinalize*/ true);
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::EmitOMPInlinedRegion(
    Directive OMPD, Instruction *EntryCall, Instruction *ExitCall,
    BodyGenCallbackTy BodyGenCB, FinalizeCallbackTy FiniCB, bool Conditional,
    bool HasFinalize) {

  if (HasFinalize)
    FinalizationStack.push_back({FiniCB, OMPD, /*IsCancellable*/ false});

  // Create inlined region's entry and body blocks, in preparation
  // for conditional creation
  BasicBlock *EntryBB = Builder.GetInsertBlock();
  Instruction *SplitPos = EntryBB->getTerminator();
  if (!isa_and_nonnull<BranchInst>(SplitPos))
    SplitPos = new UnreachableInst(Builder.getContext(), EntryBB);
  BasicBlock *ExitBB = EntryBB->splitBasicBlock(SplitPos, "omp_region.end");
  BasicBlock *FiniBB =
      EntryBB->splitBasicBlock(EntryBB->getTerminator(), "omp_region.finalize");

  Builder.SetInsertPoint(EntryBB->getTerminator());
  emitCommonDirectiveEntry(OMPD, EntryCall, ExitBB, Conditional);

  // generate body
  BodyGenCB(/* AllocaIP */ InsertPointTy(),
            /* CodeGenIP */ Builder.saveIP(), *FiniBB);

  // If we didn't emit a branch to FiniBB during body generation, it means
  // FiniBB is unreachable (e.g. while(1);). stop generating all the
  // unreachable blocks, and remove anything we are not going to use.
  auto SkipEmittingRegion = FiniBB->hasNPredecessors(0);
  if (SkipEmittingRegion) {
    FiniBB->eraseFromParent();
    ExitCall->eraseFromParent();
    // Discard finalization if we have it.
    if (HasFinalize) {
      assert(!FinalizationStack.empty() &&
             "Unexpected finalization stack state!");
      FinalizationStack.pop_back();
    }
  } else {
    // emit exit call and do any needed finalization.
    auto FinIP = InsertPointTy(FiniBB, FiniBB->getFirstInsertionPt());
    assert(FiniBB->getTerminator()->getNumSuccessors() == 1 &&
           FiniBB->getTerminator()->getSuccessor(0) == ExitBB &&
           "Unexpected control flow graph state!!");
    emitCommonDirectiveExit(OMPD, FinIP, ExitCall, HasFinalize);
    assert(FiniBB->getUniquePredecessor()->getUniqueSuccessor() == FiniBB &&
           "Unexpected Control Flow State!");
    MergeBlockIntoPredecessor(FiniBB);
  }

  // If we are skipping the region of a non conditional, remove the exit
  // block, and clear the builder's insertion point.
  assert(SplitPos->getParent() == ExitBB &&
         "Unexpected Insertion point location!");
  if (!Conditional && SkipEmittingRegion) {
    ExitBB->eraseFromParent();
    Builder.ClearInsertionPoint();
  } else {
    auto merged = MergeBlockIntoPredecessor(ExitBB);
    BasicBlock *ExitPredBB = SplitPos->getParent();
    auto InsertBB = merged ? ExitPredBB : ExitBB;
    if (!isa_and_nonnull<BranchInst>(SplitPos))
      SplitPos->eraseFromParent();
    Builder.SetInsertPoint(InsertBB);
  }

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::emitCommonDirectiveEntry(
    Directive OMPD, Value *EntryCall, BasicBlock *ExitBB, bool Conditional) {

  // if nothing to do, Return current insertion point.
  if (!Conditional)
    return Builder.saveIP();

  BasicBlock *EntryBB = Builder.GetInsertBlock();
  Value *CallBool = Builder.CreateIsNotNull(EntryCall);
  auto *ThenBB = BasicBlock::Create(M.getContext(), "omp_region.body");
  auto *UI = new UnreachableInst(Builder.getContext(), ThenBB);

  // Emit thenBB and set the Builder's insertion point there for
  // body generation next. Place the block after the current block.
  Function *CurFn = EntryBB->getParent();
  CurFn->getBasicBlockList().insertAfter(EntryBB->getIterator(), ThenBB);

  // Move Entry branch to end of ThenBB, and replace with conditional
  // branch (If-stmt)
  Instruction *EntryBBTI = EntryBB->getTerminator();
  Builder.CreateCondBr(CallBool, ThenBB, ExitBB);
  EntryBBTI->removeFromParent();
  Builder.SetInsertPoint(UI);
  Builder.Insert(EntryBBTI);
  UI->eraseFromParent();
  Builder.SetInsertPoint(ThenBB->getTerminator());

  // return an insertion point to ExitBB.
  return IRBuilder<>::InsertPoint(ExitBB, ExitBB->getFirstInsertionPt());
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::emitCommonDirectiveExit(
    omp::Directive OMPD, InsertPointTy FinIP, Instruction *ExitCall,
    bool HasFinalize) {

  Builder.restoreIP(FinIP);

  // If there is finalization to do, emit it before the exit call
  if (HasFinalize) {
    assert(!FinalizationStack.empty() &&
           "Unexpected finalization stack state!");

    FinalizationInfo Fi = FinalizationStack.pop_back_val();
    assert(Fi.DK == OMPD && "Unexpected Directive for Finalization call!");

    Fi.FiniCB(FinIP);

    BasicBlock *FiniBB = FinIP.getBlock();
    Instruction *FiniBBTI = FiniBB->getTerminator();

    // set Builder IP for call creation
    Builder.SetInsertPoint(FiniBBTI);
  }

  // place the Exitcall as last instruction before Finalization block terminator
  ExitCall->removeFromParent();
  Builder.Insert(ExitCall);

  return IRBuilder<>::InsertPoint(ExitCall->getParent(),
                                  ExitCall->getIterator());
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createCopyinClauseBlocks(
    InsertPointTy IP, Value *MasterAddr, Value *PrivateAddr,
    llvm::IntegerType *IntPtrTy, bool BranchtoEnd) {
  if (!IP.isSet())
    return IP;

  IRBuilder<>::InsertPointGuard IPG(Builder);

  // creates the following CFG structure
  //	   OMP_Entry : (MasterAddr != PrivateAddr)?
  //       F     T
  //       |      \
  //       |     copin.not.master
  //       |      /
  //       v     /
  //   copyin.not.master.end
  //		     |
  //         v
  //   OMP.Entry.Next

  BasicBlock *OMP_Entry = IP.getBlock();
  Function *CurFn = OMP_Entry->getParent();
  BasicBlock *CopyBegin =
      BasicBlock::Create(M.getContext(), "copyin.not.master", CurFn);
  BasicBlock *CopyEnd = nullptr;

  // If entry block is terminated, split to preserve the branch to following
  // basic block (i.e. OMP.Entry.Next), otherwise, leave everything as is.
  if (isa_and_nonnull<BranchInst>(OMP_Entry->getTerminator())) {
    CopyEnd = OMP_Entry->splitBasicBlock(OMP_Entry->getTerminator(),
                                         "copyin.not.master.end");
    OMP_Entry->getTerminator()->eraseFromParent();
  } else {
    CopyEnd =
        BasicBlock::Create(M.getContext(), "copyin.not.master.end", CurFn);
  }

  Builder.SetInsertPoint(OMP_Entry);
  Value *MasterPtr = Builder.CreatePtrToInt(MasterAddr, IntPtrTy);
  Value *PrivatePtr = Builder.CreatePtrToInt(PrivateAddr, IntPtrTy);
  Value *cmp = Builder.CreateICmpNE(MasterPtr, PrivatePtr);
  Builder.CreateCondBr(cmp, CopyBegin, CopyEnd);

  Builder.SetInsertPoint(CopyBegin);
  if (BranchtoEnd)
    Builder.SetInsertPoint(Builder.CreateBr(CopyEnd));

  return Builder.saveIP();
}

CallInst *OpenMPIRBuilder::createOMPAlloc(const LocationDescription &Loc,
                                          Value *Size, Value *Allocator,
                                          std::string Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {ThreadId, Size, Allocator};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_alloc);

  return Builder.CreateCall(Fn, Args, Name);
}

CallInst *OpenMPIRBuilder::createOMPFree(const LocationDescription &Loc,
                                         Value *Addr, Value *Allocator,
                                         std::string Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {ThreadId, Addr, Allocator};
  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_free);
  return Builder.CreateCall(Fn, Args, Name);
}

CallInst *OpenMPIRBuilder::createCachedThreadPrivate(
    const LocationDescription &Loc, llvm::Value *Pointer,
    llvm::ConstantInt *Size, const llvm::Twine &Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Constant *ThreadPrivateCache =
      getOrCreateOMPInternalVariable(Int8PtrPtr, Name);
  llvm::Value *Args[] = {Ident, ThreadId, Pointer, Size, ThreadPrivateCache};

  Function *Fn =
  		getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_threadprivate_cached);

  return Builder.CreateCall(Fn, Args);
}

std::string OpenMPIRBuilder::getNameWithSeparators(ArrayRef<StringRef> Parts,
                                                   StringRef FirstSeparator,
                                                   StringRef Separator) {
  SmallString<128> Buffer;
  llvm::raw_svector_ostream OS(Buffer);
  StringRef Sep = FirstSeparator;
  for (StringRef Part : Parts) {
    OS << Sep << Part;
    Sep = Separator;
  }
  return OS.str().str();
}

Constant *OpenMPIRBuilder::getOrCreateOMPInternalVariable(
    llvm::Type *Ty, const llvm::Twine &Name, unsigned AddressSpace) {
  // TODO: Replace the twine arg with stringref to get rid of the conversion
  // logic. However This is taken from current implementation in clang as is.
  // Since this method is used in many places exclusively for OMP internal use
  // we will keep it as is for temporarily until we move all users to the
  // builder and then, if possible, fix it everywhere in one go.
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Name;
  StringRef RuntimeName = Out.str();
  auto &Elem = *InternalVars.try_emplace(RuntimeName, nullptr).first;
  if (Elem.second) {
    assert(Elem.second->getType()->getPointerElementType() == Ty &&
           "OMP internal variable has different type than requested");
  } else {
    // TODO: investigate the appropriate linkage type used for the global
    // variable for possibly changing that to internal or private, or maybe
    // create different versions of the function for different OMP internal
    // variables.
    Elem.second = new llvm::GlobalVariable(
        M, Ty, /*IsConstant*/ false, llvm::GlobalValue::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Elem.first(),
        /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
        AddressSpace);
  }

  return Elem.second;
}

Value *OpenMPIRBuilder::getOMPCriticalRegionLock(StringRef CriticalName) {
  std::string Prefix = Twine("gomp_critical_user_", CriticalName).str();
  std::string Name = getNameWithSeparators({Prefix, "var"}, ".", ".");
  return getOrCreateOMPInternalVariable(KmpCriticalNameTy, Name);
}

// Create all simple and struct types exposed by the runtime and remember
// the llvm::PointerTypes of them for easy access later.
void OpenMPIRBuilder::initializeTypes(Module &M) {
  LLVMContext &Ctx = M.getContext();
  StructType *T;
#define OMP_TYPE(VarName, InitValue) VarName = InitValue;
#define OMP_ARRAY_TYPE(VarName, ElemTy, ArraySize)                             \
  VarName##Ty = ArrayType::get(ElemTy, ArraySize);                             \
  VarName##PtrTy = PointerType::getUnqual(VarName##Ty);
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  VarName = FunctionType::get(ReturnType, {__VA_ARGS__}, IsVarArg);            \
  VarName##Ptr = PointerType::getUnqual(VarName);
#define OMP_STRUCT_TYPE(VarName, StructName, ...)                              \
  T = StructType::getTypeByName(Ctx, StructName);                              \
  if (!T)                                                                      \
    T = StructType::create(Ctx, {__VA_ARGS__}, StructName);                    \
  VarName = T;                                                                 \
  VarName##Ptr = PointerType::getUnqual(T);
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}

void OpenMPIRBuilder::OutlineInfo::collectBlocks(
    SmallPtrSetImpl<BasicBlock *> &BlockSet,
    SmallVectorImpl<BasicBlock *> &BlockVector) {
  SmallVector<BasicBlock *, 32> Worklist;
  BlockSet.insert(EntryBB);
  BlockSet.insert(ExitBB);

  Worklist.push_back(EntryBB);
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    BlockVector.push_back(BB);
    for (BasicBlock *SuccBB : successors(BB))
      if (BlockSet.insert(SuccBB).second)
        Worklist.push_back(SuccBB);
  }
}

void CanonicalLoopInfo::collectControlBlocks(
    SmallVectorImpl<BasicBlock *> &BBs) {
  // We only count those BBs as control block for which we do not need to
  // reverse the CFG, i.e. not the loop body which can contain arbitrary control
  // flow. For consistency, this also means we do not add the Body block, which
  // is just the entry to the body code.
  BBs.reserve(BBs.size() + 6);
  BBs.append({Preheader, Header, Cond, Latch, Exit, After});
}

void CanonicalLoopInfo::assertOK() const {
#ifndef NDEBUG
  if (!IsValid)
    return;

  // Verify standard control-flow we use for OpenMP loops.
  assert(Preheader);
  assert(isa<BranchInst>(Preheader->getTerminator()) &&
         "Preheader must terminate with unconditional branch");
  assert(Preheader->getSingleSuccessor() == Header &&
         "Preheader must jump to header");

  assert(Header);
  assert(isa<BranchInst>(Header->getTerminator()) &&
         "Header must terminate with unconditional branch");
  assert(Header->getSingleSuccessor() == Cond &&
         "Header must jump to exiting block");

  assert(Cond);
  assert(Cond->getSinglePredecessor() == Header &&
         "Exiting block only reachable from header");

  assert(isa<BranchInst>(Cond->getTerminator()) &&
         "Exiting block must terminate with conditional branch");
  assert(size(successors(Cond)) == 2 &&
         "Exiting block must have two successors");
  assert(cast<BranchInst>(Cond->getTerminator())->getSuccessor(0) == Body &&
         "Exiting block's first successor jump to the body");
  assert(cast<BranchInst>(Cond->getTerminator())->getSuccessor(1) == Exit &&
         "Exiting block's second successor must exit the loop");

  assert(Body);
  assert(Body->getSinglePredecessor() == Cond &&
         "Body only reachable from exiting block");
  assert(!isa<PHINode>(Body->front()));

  assert(Latch);
  assert(isa<BranchInst>(Latch->getTerminator()) &&
         "Latch must terminate with unconditional branch");
  assert(Latch->getSingleSuccessor() == Header && "Latch must jump to header");
  // TODO: To support simple redirecting of the end of the body code that has
  // multiple; introduce another auxiliary basic block like preheader and after.
  assert(Latch->getSinglePredecessor() != nullptr);
  assert(!isa<PHINode>(Latch->front()));

  assert(Exit);
  assert(isa<BranchInst>(Exit->getTerminator()) &&
         "Exit block must terminate with unconditional branch");
  assert(Exit->getSingleSuccessor() == After &&
         "Exit block must jump to after block");

  assert(After);
  assert(After->getSinglePredecessor() == Exit &&
         "After block only reachable from exit block");
  assert(After->empty() || !isa<PHINode>(After->front()));

  Instruction *IndVar = getIndVar();
  assert(IndVar && "Canonical induction variable not found?");
  assert(isa<IntegerType>(IndVar->getType()) &&
         "Induction variable must be an integer");
  assert(cast<PHINode>(IndVar)->getParent() == Header &&
         "Induction variable must be a PHI in the loop header");
  assert(cast<PHINode>(IndVar)->getIncomingBlock(0) == Preheader);
  assert(
      cast<ConstantInt>(cast<PHINode>(IndVar)->getIncomingValue(0))->isZero());
  assert(cast<PHINode>(IndVar)->getIncomingBlock(1) == Latch);

  auto *NextIndVar = cast<PHINode>(IndVar)->getIncomingValue(1);
  assert(cast<Instruction>(NextIndVar)->getParent() == Latch);
  assert(cast<BinaryOperator>(NextIndVar)->getOpcode() == BinaryOperator::Add);
  assert(cast<BinaryOperator>(NextIndVar)->getOperand(0) == IndVar);
  assert(cast<ConstantInt>(cast<BinaryOperator>(NextIndVar)->getOperand(1))
             ->isOne());

  Value *TripCount = getTripCount();
  assert(TripCount && "Loop trip count not found?");
  assert(IndVar->getType() == TripCount->getType() &&
         "Trip count and induction variable must have the same type");

  auto *CmpI = cast<CmpInst>(&Cond->front());
  assert(CmpI->getPredicate() == CmpInst::ICMP_ULT &&
         "Exit condition must be a signed less-than comparison");
  assert(CmpI->getOperand(0) == IndVar &&
         "Exit condition must compare the induction variable");
  assert(CmpI->getOperand(1) == TripCount &&
         "Exit condition must compare with the trip count");
#endif
}
