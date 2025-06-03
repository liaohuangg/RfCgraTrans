//===- MemRefDataFlowOpt.cpp - MemRef DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely.
// TODO: In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'affine.for'/'affine.if' statements is available.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <set>

#define DEBUG_TYPE "memref-dataflow-opt"

using namespace mlir;
using namespace memref;

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct MemRefDataFlowOpt : public MemRefDataFlowOptBase<MemRefDataFlowOpt> {
  void runOnFunction() override;

  void forwardStoreToLoad(AffineReadOpInterface loadOp,
                          SmallVectorImpl<Operation *> &loadOpsToErase,
                          SmallPtrSetImpl<Value> &memrefsToErase,
                          DominanceInfo *domInfo,
                          PostDominanceInfo *postDominanceInfo);
  void removeUnusedStore(AffineWriteOpInterface loadOp,
                         SmallVectorImpl<Operation *> &loadOpsToErase,
                         SmallPtrSetImpl<Value> &memrefsToErase,
                         DominanceInfo *domInfo,
                         PostDominanceInfo *postDominanceInfo);
  void forwardLoadToLoad(AffineReadOpInterface loadOp,
                         SmallVectorImpl<Operation *> &loadOpsToErase,
                         DominanceInfo *domInfo);
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> mlir::createMemRefDataFlowOptPass() {
  return std::make_unique<MemRefDataFlowOpt>();
}

bool hasNoInterveningStore(Operation *start, AffineReadOpInterface loadOp) {
  bool legal = true;
  std::function<void(Operation *)> check = [&](Operation *op) {
    if (!legal)
      return;

    if (auto store = dyn_cast<AffineStoreOp>(op)) {
      if (loadOp.getMemRef().getDefiningOp<memref::AllocaOp>() ||
          loadOp.getMemRef().getDefiningOp<memref::AllocOp>()) {
        if (store.getMemRef().getDefiningOp<memref::AllocaOp>() ||
            store.getMemRef().getDefiningOp<memref::AllocOp>()) {
          if (loadOp.getMemRef() == store.getMemRef())
            legal = false;
          return;
        }
      }
    }

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Collect all memory effects on `v`.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffectsOnValue(loadOp.getMemRef(), effects);

      if (llvm::any_of(effects,
                       [](const MemoryEffects::EffectInstance &instance) {
                         return isa<MemoryEffects::Write>(instance.getEffect());
                       })) {
        legal = false;
        return;
      }
    }

    if (op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // Recurse into the regions for this op and check whether the contained
      // ops can be hoisted.
      for (auto &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &innerOp : block) {
            check(&innerOp);
            if (!legal)
              return;
          }
        }
      }
    }
  };

  auto until = [&](Operation *parent, Operation *to) {
    // todo perhaps recur
    check(parent);
  };

  std::function<void(Operation *, Operation *)> recur = [&](Operation *from,
                                                            Operation *to) {
    if (from->getParentRegion() != to->getParentRegion()) {
      recur(from, to->getParentOp());
      until(to->getParentOp(), to);
      return;
    }
    std::deque<Block *> todo;
    {
      bool seen = false;
      for (auto &op : *from->getBlock()) {
        if (&op == from) {
          seen = true;
          continue;
        }
        if (!seen) {
          continue;
        }
        if (&op == to) {
          break;
        }
        check(&op);
        if (&op == from->getBlock()->getTerminator()) {
          for (auto succ : from->getBlock()->getSuccessors()) {
            todo.push_back(succ);
          }
        }
      }
    }
    SmallPtrSet<Block *, 4> done;
    while (todo.size()) {
      auto blk = todo.front();
      todo.pop_front();
      if (done.count(blk))
        continue;
      done.insert(blk);
      for (auto &op : *blk) {
        if (&op == to) {
          break;
        }
        check(&op);
        if (&op == blk->getTerminator()) {
          for (auto succ : blk->getSuccessors()) {
            todo.push_back(succ);
          }
        }
      }
    }
  };
  recur(start, loadOp.getOperation());
  return legal;
}

bool hasNoInterveningLoad(AffineWriteOpInterface start,
                          AffineWriteOpInterface loadOp) {
  bool legal = true;
  std::function<void(Operation *)> check = [&](Operation *op) {
    if (!legal)
      return;

    if (auto store = dyn_cast<AffineLoadOp>(op)) {
      if (loadOp.getMemRef().getDefiningOp<memref::AllocaOp>() ||
          loadOp.getMemRef().getDefiningOp<AllocOp>()) {
        if (store.getMemRef().getDefiningOp<memref::AllocaOp>() ||
            store.getMemRef().getDefiningOp<AllocOp>()) {
          if (loadOp.getMemRef() == store.getMemRef())
            legal = false;
          return;
        }
      }
    }

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Collect all memory effects on `v`.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffectsOnValue(start.getMemRef(), effects);
      //	llvm::errs() << " -----  potential use " << *op << " -- ";
      // for(auto e : effects) llvm::errs() << e.getEffect() << " ";
      // llvm::errs() << "\n";
      if (llvm::any_of(effects,
                       [](const MemoryEffects::EffectInstance &instance) {
                         return isa<MemoryEffects::Read>(instance.getEffect());
                       })) {
        legal = false;
        return;
      }
    }

    if (op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // Recurse into the regions for this op and check whether the contained
      // ops can be hoisted.
      for (auto &region : op->getRegions()) {
        for (auto &block : region) {
          for (auto &innerOp : block) {
            check(&innerOp);
            if (!legal)
              return;
          }
        }
      }
    }
  };

  auto until = [&](Operation *parent, Operation *to) {
    // todo perhaps recur
    check(parent);
  };

  std::function<void(Operation *, Operation *)> recur = [&](Operation *from,
                                                            Operation *to) {
    if (from->getParentRegion() != to->getParentRegion()) {
      recur(from, to->getParentOp());
      until(to->getParentOp(), to);
      return;
    }
    std::deque<Block *> todo;
    {
      bool seen = false;
      for (auto &op : *from->getBlock()) {
        if (&op == from) {
          seen = true;
          continue;
        }
        if (!seen) {
          continue;
        }
        if (&op == to) {
          break;
        }
        check(&op);
        if (&op == from->getBlock()->getTerminator()) {
          for (auto succ : from->getBlock()->getSuccessors()) {
            todo.push_back(succ);
          }
        }
      }
    }
    SmallPtrSet<Block *, 4> done;
    while (todo.size()) {
      auto blk = todo.front();
      todo.pop_front();
      if (done.count(blk))
        continue;
      done.insert(blk);
      for (auto &op : *blk) {
        if (&op == to) {
          break;
        }
        check(&op);
        if (&op == blk->getTerminator()) {
          for (auto succ : blk->getSuccessors()) {
            todo.push_back(succ);
          }
        }
      }
    }
  };
  recur(start.getOperation(), loadOp.getOperation());
  return legal;
}

void MemRefDataFlowOpt::removeUnusedStore(
    AffineWriteOpInterface writeOp,
    SmallVectorImpl<Operation *> &loadOpsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo *domInfo,
    PostDominanceInfo *postDominanceInfo) {
  // First pass over the use list to get the minimum number of surrounding
  // loops common between the load op and the store op, with min taken across
  // all store ops.
  SmallVector<Operation *, 8> storeOps;
  for (auto *user : writeOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    storeOps.push_back(storeOp);
  }

  // The list of store op candidates for forwarding that satisfy conditions
  // (1) and (2) above - they will be filtered later when checking (3).
  SmallVector<Operation *, 8> fwdingCandidates;

  // Store ops that have a dependence into the load (even if they aren't
  // forwarding candidates). Each forwarding candidate will be checked for a
  // post-dominance on these. 'fwdingCandidates' are a subset of depSrcStores.
  // llvm::errs() << " considering potential erasable store: " << writeOp <<
  // "\n";
  for (auto *storeOp : storeOps) {
    if (storeOp == writeOp)
      continue;
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(writeOp);

    if (storeOp->getParentRegion() != writeOp->getParentRegion())
      continue;
    // Stores that *may* be reaching the load.

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess) {
      continue;
    }
    // llvm::errs() << " + -- " << *storeOp << "\n";
    if (!postDominanceInfo->postDominates(storeOp, writeOp)) {
      continue;
    }

    bool legal = hasNoInterveningLoad(writeOp, storeOp);
    // llvm::errs() << " + " << *storeOp << " legal: " << legal << "\n";
    if (!legal)
      continue;

    loadOpsToErase.push_back(writeOp);
    break;
  }
}

// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
void MemRefDataFlowOpt::forwardStoreToLoad(
    AffineReadOpInterface loadOp, SmallVectorImpl<Operation *> &loadOpsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo *domInfo,
    PostDominanceInfo *postDominanceInfo) {
  // First pass over the use list to get the minimum number of surrounding
  // loops common between the load op and the store op, with min taken across
  // all store ops.
  SmallVector<Operation *, 8> storeOps;
  unsigned minSurroundingLoops = getNestingDepth(loadOp);
  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    unsigned nsLoops = getNumCommonSurroundingLoops(*loadOp, *storeOp);
    minSurroundingLoops = std::min(nsLoops, minSurroundingLoops);
    storeOps.push_back(storeOp);
  }

  // The list of store op candidates for forwarding that satisfy conditions
  // (1) and (2) above - they will be filtered later when checking (3).
  SmallVector<Operation *, 8> fwdingCandidates;

  // Store ops that have a dependence into the load (even if they aren't
  // forwarding candidates). Each forwarding candidate will be checked for a
  // post-dominance on these. 'fwdingCandidates' are a subset of depSrcStores.
  SmallVector<Operation *, 8> depSrcStores;
  // llvm::errs() << " considering load: " << loadOp << "\n";
  for (auto *storeOp : storeOps) {
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);

    // Stores that *may* be reaching the load.
    depSrcStores.push_back(storeOp);

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess) {
      continue;
    }

    if (!domInfo->dominates(storeOp, loadOp)) {
      continue;
    }

    bool legal = hasNoInterveningStore(storeOp, loadOp);
    // llvm::errs() << " + " << *storeOp << " legal: " << legal << "\n";
    if (!legal)
      continue;

    // We now have a candidate for forwarding.
    fwdingCandidates.push_back(storeOp);
  }

  // 3. Of all the store op's that meet the above criteria, the store that
  // postdominates all 'depSrcStores' (if one exists) is the unique store
  // providing the value to the load, i.e., provably the last writer to that
  // memref loc.
  // Note: this can be implemented in a cleaner way with postdominator tree
  // traversals. Consider this for the future if needed.
  Operation *lastWriteStoreOp = nullptr;
  for (auto *storeOp : fwdingCandidates) {
    assert(!lastWriteStoreOp);
    lastWriteStoreOp = storeOp;
  }
  if (!lastWriteStoreOp) {
    return;
  }

  // Perform the actual store to load forwarding.
  Value storeVal =
      cast<AffineWriteOpInterface>(lastWriteStoreOp).getValueToStore();
  loadOp.getValue().replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp.getMemRef());
  // Record this to erase later.
  loadOpsToErase.push_back(loadOp);
}

// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
void MemRefDataFlowOpt::forwardLoadToLoad(
    AffineReadOpInterface loadOp, SmallVectorImpl<Operation *> &loadOpsToErase,
    DominanceInfo *domInfo) {
  SmallVector<AffineReadOpInterface, 4> LoadOptions;
  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto loadOp2 = dyn_cast<AffineReadOpInterface>(user);
    if (!loadOp2 || loadOp2 == loadOp)
      continue;

    MemRefAccess srcAccess(loadOp2);
    MemRefAccess destAccess(loadOp);

    if (srcAccess != destAccess) {
      continue;
    }

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo->dominates(loadOp2, loadOp)) {
      // llvm::errs() << " - not fone from dominating load\n";
      continue;
    }

    bool legal = hasNoInterveningStore(loadOp2.getOperation(), loadOp);
    if (!legal)
      continue;

    LoadOptions.push_back(loadOp2);
  }

  Value lastOp = nullptr;
  for (auto option : LoadOptions) {
    if (llvm::all_of(LoadOptions, [&](AffineReadOpInterface depStore) {
          return depStore == option ||
                 domInfo->dominates(option.getOperation(),
                                    depStore.getOperation());
        })) {
      lastOp = option.getValue();
      break;
    }
  }

  if (lastOp) {
    // llvm::errs() << "replacing: " << loadOp << " with " << lastOp << "\n";
    loadOp.getValue().replaceAllUsesWith(lastOp);
    // Record this to erase later.
    loadOpsToErase.push_back(loadOp);
  }
}

void MemRefDataFlowOpt::runOnFunction() {
  // Only supports single block functions at the moment.
  FuncOp f = getFunction();
  // f.dump();
  // if (!llvm::hasSingleElement(f)) {
  //  markAllAnalysesPreserved();
  //  return;
  //}

  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> loadOpsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  auto domInfo = &getAnalysis<DominanceInfo>();
  auto postDominanceInfo = &getAnalysis<PostDominanceInfo>();

  // f.dump();

  // Walk all load's and perform store to load forwarding.
  f.walk([&](AffineReadOpInterface loadOp) {
    forwardStoreToLoad(loadOp, loadOpsToErase, memrefsToErase, domInfo,
                       postDominanceInfo);
  });

  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *loadOp : loadOpsToErase)
    loadOp->erase();
  loadOpsToErase.clear();

  // f.dump();
  f.walk([&](AffineReadOpInterface loadOp) {
    forwardLoadToLoad(loadOp, loadOpsToErase, domInfo);
  });
  for (auto *loadOp : loadOpsToErase)
    loadOp->erase();
  loadOpsToErase.clear();

  // f.dump();
  f.walk([&](AffineWriteOpInterface loadOp) {
    removeUnusedStore(loadOp, loadOpsToErase, memrefsToErase, domInfo,
                      postDominanceInfo);
  });

  // f.dump();
  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *loadOp : loadOpsToErase)
    loadOp->erase();

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canonicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been alloc'ed in this function, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<AllocOp>(defOp))
      // TODO: if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<AffineWriteOpInterface, DeallocOp>(ownerOp);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
  // f.dump();
}
