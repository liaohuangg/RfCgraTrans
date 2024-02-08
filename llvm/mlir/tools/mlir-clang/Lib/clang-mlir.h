#ifndef CLANG_MLIR_H
#define CLANG_MLIR_H

#include "clang/AST/StmtVisitor.h"
#include <clang/AST/ASTConsumer.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/HeaderSearchOptions.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeTranslation.h"
#include "llvm/IR/DerivedTypes.h"

#include "../../../../clang/lib/CodeGen/CGRecordLayout.h"
#include "../../../../clang/lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Mangle.h"

using namespace clang;
using namespace mlir;

struct LoopContext {
  mlir::Value keepRunning;
  mlir::Value noBreak;
};

struct AffineLoopDescriptor {
private:
  mlir::Value upperBound;
  mlir::Value lowerBound;
  int64_t step;
  mlir::Type indVarType;
  std::string indVar;
  bool forwardMode;

public:
  AffineLoopDescriptor()
      : upperBound(nullptr), lowerBound(nullptr),
        step(std::numeric_limits<int64_t>::max()), indVarType(nullptr),
        indVar("nullptr"), forwardMode(true){};
  AffineLoopDescriptor(const AffineLoopDescriptor &) = delete;

  void setLowerBound(mlir::Value value) { lowerBound = value; }
  void setUpperBound(mlir::Value value) { upperBound = value; }

  void setStep(int value) { step = value; };
  void setType(mlir::Type type) { indVarType = type; }
  void setName(std::string value) { indVar = value; }

  std::string getName() const { return indVar; }
  mlir::Type getType() const { return indVarType; }
  int getStep() const { return step; }

  auto getLowerBound() const { return lowerBound; }

  auto getUpperBound() const { return upperBound; }

  void setForwardMode(bool value) { forwardMode = value; };
  bool getForwardMode() const { return forwardMode; }
};

struct ValueWithOffsets {
  mlir::Value val;
  bool isReference;
  ValueWithOffsets() : val(nullptr), isReference(false){};
  ValueWithOffsets(std::nullptr_t) : val(nullptr), isReference(false){};
  ValueWithOffsets(mlir::Value val, bool isReference)
      : val(val), isReference(isReference) {
    if (isReference) {
      if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {

      } else if (val.getType().isa<mlir::MemRefType>()) {

      } else {
        llvm::errs() << val << "\n";
        assert(val.getType().isa<mlir::MemRefType>());
      }
    }
  };

  mlir::Value getValue(OpBuilder &builder) const {
    assert(val);
    if (!isReference)
      return val;
    auto loc = builder.getUnknownLoc();
    if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return builder.create<mlir::LLVM::LoadOp>(loc, val);
    }
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    if (!val.getType().isa<mlir::MemRefType>()) {
      llvm::errs() << val << "\n";
    }
    assert(val.getType().isa<mlir::MemRefType>());
    return builder.create<memref::LoadOp>(loc, val,
                                          std::vector<mlir::Value>({c0}));
  }

  ValueWithOffsets dereference(OpBuilder &builder) const {
    assert(val);
    if (!isReference)
      return ValueWithOffsets(val, /*isReference*/ true);
    auto loc = builder.getUnknownLoc();
    auto c0 = builder.create<mlir::ConstantIndexOp>(loc, 0);
    if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueWithOffsets(builder.create<mlir::LLVM::LoadOp>(loc, val),
                              /*isReference*/ true);
    }
    auto mt = val.getType().cast<mlir::MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    if (shape.size() > 1) {
      shape.erase(shape.begin());
    } else {
      shape[0] = -1;
      // builder.create<LoadOp>(loc, val, std::vector<mlir::Value>({c0}))
    }
    auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                     mt.getAffineMaps(), mt.getMemorySpace());
    auto post = builder.create<memref::SubIndexOp>(loc, mt0, val, c0);

    return ValueWithOffsets(post,
                            /*isReference*/ true);
  }
};

/// The location of the scop, as delimited by scop and endscop
/// pragmas by the user.
/// "scop" and "endscop" are the source locations of the scop and
/// endscop pragmas.
/// "start_line" is the line number of the start position.
struct ScopLoc {
  ScopLoc() : end(0) {}

  clang::SourceLocation scop;
  clang::SourceLocation endscop;
  unsigned startLine;
  unsigned start;
  unsigned end;
};

/// Taken from pet.cc
/// List of pairs of #pragma scop and #pragma endscop locations.
struct ScopLocList {
  std::vector<ScopLoc> list;

  // Add a new start (#pragma scop) location to the list.
  // If the last #pragma scop did not have a matching
  // #pragma endscop then overwrite it.
  // "start" points to the location of the scop pragma.

  void addStart(SourceManager &SM, SourceLocation start) {
    ScopLoc loc;

    loc.scop = start;
    int line = SM.getExpansionLineNumber(start);
    start = SM.translateLineCol(SM.getFileID(start), line, 1);
    loc.startLine = line;
    loc.start = SM.getFileOffset(start);
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      list.push_back(loc);
    else
      list[list.size() - 1] = loc;
  }

  // Set the end location (#pragma endscop) of the last pair
  // in the list.
  // If there is no such pair of if the end of that pair
  // is already set, then ignore the spurious #pragma endscop.
  // "end" points to the location of the endscop pragma.

  void addEnd(SourceManager &SM, SourceLocation end) {
    if (list.size() == 0 || list[list.size() - 1].end != 0)
      return;
    list[list.size() - 1].endscop = end;
    int line = SM.getExpansionLineNumber(end);
    end = SM.translateLineCol(SM.getFileID(end), line + 1, 1);
    list[list.size() - 1].end = SM.getFileOffset(end);
  }

  // Check if the current location is in the scop.
  bool isInScop(SourceLocation target) {
    // If the user selects the raise-scf-to-affine we ignore pragmas and try to
    // raise all we can. Similar behavior to pet --autodetect. This allow us to
    // test the raising.
    if (RaiseToAffine)
      return false;

    if (!list.size())
      return false;
    for (auto &scopLoc : list)
      if ((target >= scopLoc.scop) && (target <= scopLoc.endscop))
        return true;
    return false;
  }
};

struct PragmaScopHandler : public PragmaHandler {
  ScopLocList &scops;

  PragmaScopHandler(ScopLocList &scops) : PragmaHandler("scop"), scops(scops) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                            Token &scopTok) {
    auto &SM = PP.getSourceManager();
    auto loc = scopTok.getLocation();
    scops.addStart(SM, loc);
  }
};

struct PragmaEndScopHandler : public PragmaHandler {
  ScopLocList &scops;

  PragmaEndScopHandler(ScopLocList &scops)
      : PragmaHandler("endscop"), scops(scops) {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducer introducer,
                            Token &endScopTok) {
    auto &SM = PP.getSourceManager();
    auto loc = endScopTok.getLocation();
    scops.addEnd(SM, loc);
  }
};

struct MLIRASTConsumer : public ASTConsumer {
  std::set<std::string> &emitIfFound;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals;
  std::map<std::string, mlir::FuncOp> &functions;
  Preprocessor &PP;
  ASTContext &astContext;
  mlir::ModuleOp &module;
  clang::SourceManager &SM;
  LLVMContext lcontext;
  llvm::Module llvmMod;
  CodeGenOptions codegenops;
  CodeGen::CodeGenModule CGM;
  bool error;
  ScopLocList scopLocList;

  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
  LLVM::TypeToLLVMIRTranslator reverseTypeTranslator;

  MLIRASTConsumer(
      std::set<std::string> &emitIfFound,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals,
      std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals,
      std::map<std::string, mlir::FuncOp> &functions, Preprocessor &PP,
      ASTContext &astContext, mlir::ModuleOp &module, clang::SourceManager &SM)
      : emitIfFound(emitIfFound), llvmStringGlobals(llvmStringGlobals),
        globals(globals), functions(functions), PP(PP), astContext(astContext),
        module(module), SM(SM), lcontext(), llvmMod("tmp", lcontext),
        codegenops(),
        CGM(astContext, PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        error(false), typeTranslator(*module.getContext()),
        reverseTypeTranslator(lcontext) {
    PP.AddPragmaHandler(new PragmaScopHandler(scopLocList));
    PP.AddPragmaHandler(new PragmaEndScopHandler(scopLocList));
  }

  ~MLIRASTConsumer() {}

  mlir::FuncOp GetOrCreateMLIRFunction(const FunctionDecl *FD);

  std::map<const FunctionDecl *, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  mlir::LLVM::LLVMFuncOp GetOrCreateLLVMFunction(const FunctionDecl *FD);

  std::map<const VarDecl *, mlir::LLVM::GlobalOp> llvmGlobals;
  mlir::LLVM::GlobalOp GetOrCreateLLVMGlobal(const VarDecl *VD);

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  mlir::Value GetOrCreateGlobalLLVMString(mlir::Location loc,
                                          mlir::OpBuilder &builder,
                                          StringRef value);

  std::map<std::string, clang::VarDecl *> globalVariables;
  std::map<std::string, clang::FunctionDecl *> globalFunctions;
  std::pair<mlir::memref::GlobalOp, bool> GetOrCreateGlobal(const VarDecl *VD);

  std::deque<const FunctionDecl *> functionsToEmit;
  std::set<const FunctionDecl *> done;

  void run();

  virtual bool HandleTopLevelDecl(DeclGroupRef dg);

  mlir::Type getMLIRType(clang::QualType t);

  llvm::Type *getLLVMType(clang::QualType t);

  mlir::Type getMLIRType(llvm::Type *t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);
};

struct MLIRScanner : public StmtVisitor<MLIRScanner, ValueWithOffsets> {
private:
  MLIRASTConsumer &Glob;
  mlir::FuncOp function;
  mlir::ModuleOp &module;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::Block *entryBlock;
  std::vector<std::map<std::string, ValueWithOffsets>> scopes;
  std::vector<LoopContext> loops;

  void setValue(std::string name, ValueWithOffsets &&val);

  ValueWithOffsets getValue(std::string name);

  std::map<const void *, std::vector<mlir::LLVM::AllocaOp>> bufs;
  mlir::LLVM::AllocaOp allocateBuffer(size_t i, mlir::LLVM::LLVMPointerType t) {
    auto &vec = bufs[t.getAsOpaquePointer()];
    if (i < vec.size())
      return vec[i];

    mlir::OpBuilder subbuilder(builder.getContext());
    subbuilder.setInsertionPointToStart(entryBlock);

    auto indexType = subbuilder.getIntegerType(64);
    auto one = subbuilder.create<mlir::ConstantOp>(
        loc, indexType,
        subbuilder.getIntegerAttr(subbuilder.getIntegerType(64), 1));
    auto rs = subbuilder.create<mlir::LLVM::AllocaOp>(loc, t, one, 0);
    vec.push_back(rs);
    return rs;
  }

  mlir::Type getLLVMTypeFromMLIRType(mlir::Type t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

  mlir::Type getMLIRType(clang::QualType t);

  llvm::Type *getLLVMType(clang::QualType t);

  size_t getTypeSize(clang::QualType t);

  mlir::Value createAllocOp(mlir::Type t, std::string name, uint64_t memspace,
                            bool isArray);

  mlir::Value createAndSetAllocOp(std::string name, mlir::Value v,
                                  uint64_t memspace);

  const clang::FunctionDecl *EmitCallee(const Expr *E);

  mlir::FuncOp EmitDirectCallee(GlobalDecl GD);

  std::map<int, mlir::Value> constants;
  mlir::Value getConstantIndex(int x);

  mlir::Value castToIndex(mlir::Location loc, mlir::Value val);

  bool isTrivialAffineLoop(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  bool getUpperBound(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  bool getLowerBound(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  bool getConstantStep(clang::ForStmt *fors, AffineLoopDescriptor &descr);

  void buildAffineLoop(clang::ForStmt *fors, mlir::Location loc,
                       const AffineLoopDescriptor &descr);

  void buildAffineLoopImpl(clang::ForStmt *fors, mlir::Location loc,
                           mlir::Value lb, mlir::Value ub,
                           const AffineLoopDescriptor &descr);
  std::vector<Block *> prevBlock;
  std::vector<Block::iterator> prevIterator;

public:
  void pushLoopIf();
  void popLoopIf();

public:
  MLIRScanner(MLIRASTConsumer &Glob, mlir::FuncOp function,
              const FunctionDecl *fd, mlir::ModuleOp &module)
      : Glob(Glob), function(function), module(module),
        builder(module.getContext()), loc(builder.getUnknownLoc()) {
    // llvm::errs() << *fd << "\n";
    // fd->dump();

    scopes.emplace_back();

    entryBlock = function.addEntryBlock();

    builder.setInsertionPointToStart(entryBlock);

    unsigned i = 0;
    for (auto parm : fd->parameters()) {
      assert(i != function.getNumArguments());
      auto name = parm->getName().str();
      // function.getArgument(i).setName(name);
      createAndSetAllocOp(name, function.getArgument(i), 0);
      i++;
    }
    scopes.emplace_back();

    Stmt *stmt = fd->getBody();
    // stmt->dump();
    Visit(stmt);

    auto endBlock = builder.getInsertionBlock();
    if (endBlock->empty() ||
        !endBlock->back().mightHaveTrait<OpTrait::IsTerminator>()) {
      if (function.getType().getResults().size()) {
        auto ty = function.getType().getResults()[0].cast<mlir::IntegerType>();
        auto val = (mlir::Value)builder.create<mlir::ConstantOp>(
            loc, ty, builder.getIntegerAttr(ty, 0));
        builder.create<mlir::ReturnOp>(loc, val);
      } else
        builder.create<mlir::ReturnOp>(loc);
    }
    // function.dump();
  }

  ValueWithOffsets VisitDeclStmt(clang::DeclStmt *decl);

  ValueWithOffsets
  VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl);

  ValueWithOffsets VisitIntegerLiteral(clang::IntegerLiteral *expr);

  ValueWithOffsets VisitFloatingLiteral(clang::FloatingLiteral *expr);

  ValueWithOffsets VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr);

  ValueWithOffsets VisitStringLiteral(clang::StringLiteral *expr);

  ValueWithOffsets VisitParenExpr(clang::ParenExpr *expr);

  ValueWithOffsets VisitVarDecl(clang::VarDecl *decl);

  ValueWithOffsets VisitForStmt(clang::ForStmt *fors);

  ValueWithOffsets VisitWhileStmt(clang::WhileStmt *fors);

  ValueWithOffsets VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  ValueWithOffsets VisitCallExpr(clang::CallExpr *expr);

  ValueWithOffsets VisitCXXConstructExpr(clang::CXXConstructExpr *expr);

  ValueWithOffsets VisitMSPropertyRefExpr(MSPropertyRefExpr *expr);

  ValueWithOffsets VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);

  ValueWithOffsets VisitUnaryOperator(clang::UnaryOperator *U);

  ValueWithOffsets
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *expr);

  ValueWithOffsets VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop);

  ValueWithOffsets VisitBinaryOperator(clang::BinaryOperator *BO);

  ValueWithOffsets VisitAttributedStmt(AttributedStmt *AS);

  ValueWithOffsets VisitExprWithCleanups(ExprWithCleanups *E);

  ValueWithOffsets VisitDeclRefExpr(DeclRefExpr *E);

  ValueWithOffsets VisitOpaqueValueExpr(OpaqueValueExpr *E);

  ValueWithOffsets VisitMemberExpr(MemberExpr *ME);

  ValueWithOffsets VisitCastExpr(CastExpr *E);

  ValueWithOffsets VisitIfStmt(clang::IfStmt *stmt);

  ValueWithOffsets VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueWithOffsets VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueWithOffsets VisitBreakStmt(clang::BreakStmt *stmt);

  ValueWithOffsets VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueWithOffsets VisitReturnStmt(clang::ReturnStmt *stmt);
};

#endif
