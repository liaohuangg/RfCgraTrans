//===- Utils.cpp ---- Utilities for affine dialect transformation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous transformation utilities for the Affine
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

/// Promotes the `then` or the `else` block of `ifOp` (depending on whether
/// `elseBlock` is false or true) into `ifOp`'s containing block, and discards
/// the rest of the op.
static void promoteIfBlock(AffineIfOp ifOp, bool elseBlock) {
  if (elseBlock)
    assert(ifOp.hasElse() && "else block expected");

  Block *destBlock = ifOp->getBlock();
  Block *srcBlock = elseBlock ? ifOp.getElseBlock() : ifOp.getThenBlock();
  destBlock->getOperations().splice(
      Block::iterator(ifOp), srcBlock->getOperations(), srcBlock->begin(),
      std::prev(srcBlock->end()));
  ifOp.erase();
}

/// Returns the outermost affine.for/parallel op that the `ifOp` is invariant
/// on. The `ifOp` could be hoisted and placed right before such an operation.
/// This method assumes that the ifOp has been canonicalized (to be correct and
/// effective).
static Operation *getOutermostInvariantForOp(AffineIfOp ifOp) {
  // Walk up the parents past all for op that this conditional is invariant on.
  auto ifOperands = ifOp.getOperands();
  auto *res = ifOp.getOperation();
  while (!isa<FuncOp>(res->getParentOp())) {
    auto *parentOp = res->getParentOp();
    if (auto forOp = dyn_cast<AffineForOp>(parentOp)) {
      if (llvm::is_contained(ifOperands, forOp.getInductionVar()))
        break;
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(parentOp)) {
      for (auto iv : parallelOp.getIVs())
        if (llvm::is_contained(ifOperands, iv))
          break;
    } else if (!isa<AffineIfOp>(parentOp)) {
      // Won't walk up past anything other than affine.for/if ops.
      break;
    }
    // You can always hoist up past any affine.if ops.
    res = parentOp;
  }
  return res;
}

/// A helper for the mechanics of mlir::hoistAffineIfOp. Hoists `ifOp` just over
/// `hoistOverOp`. Returns the new hoisted op if any hoisting happened,
/// otherwise the same `ifOp`.
static AffineIfOp hoistAffineIfOp(AffineIfOp ifOp, Operation *hoistOverOp) {
  // No hoisting to do.
  if (hoistOverOp == ifOp)
    return ifOp;

  // Create the hoisted 'if' first. Then, clone the op we are hoisting over for
  // the else block. Then drop the else block of the original 'if' in the 'then'
  // branch while promoting its then block, and analogously drop the 'then'
  // block of the original 'if' from the 'else' branch while promoting its else
  // block.
  BlockAndValueMapping operandMap;
  OpBuilder b(hoistOverOp);
  auto hoistedIfOp = b.create<AffineIfOp>(ifOp.getLoc(), ifOp.getIntegerSet(),
                                          ifOp.getOperands(),
                                          /*elseBlock=*/true);

  // Create a clone of hoistOverOp to use for the else branch of the hoisted
  // conditional. The else block may get optimized away if empty.
  Operation *hoistOverOpClone = nullptr;
  // We use this unique name to identify/find  `ifOp`'s clone in the else
  // version.
  Identifier idForIfOp = b.getIdentifier("__mlir_if_hoisting");
  operandMap.clear();
  b.setInsertionPointAfter(hoistOverOp);
  // We'll set an attribute to identify this op in a clone of this sub-tree.
  ifOp->setAttr(idForIfOp, b.getBoolAttr(true));
  hoistOverOpClone = b.clone(*hoistOverOp, operandMap);

  // Promote the 'then' block of the original affine.if in the then version.
  promoteIfBlock(ifOp, /*elseBlock=*/false);

  // Move the then version to the hoisted if op's 'then' block.
  auto *thenBlock = hoistedIfOp.getThenBlock();
  thenBlock->getOperations().splice(thenBlock->begin(),
                                    hoistOverOp->getBlock()->getOperations(),
                                    Block::iterator(hoistOverOp));

  // Find the clone of the original affine.if op in the else version.
  AffineIfOp ifCloneInElse;
  hoistOverOpClone->walk([&](AffineIfOp ifClone) {
    if (!ifClone->getAttr(idForIfOp))
      return WalkResult::advance();
    ifCloneInElse = ifClone;
    return WalkResult::interrupt();
  });
  assert(ifCloneInElse && "if op clone should exist");
  // For the else block, promote the else block of the original 'if' if it had
  // one; otherwise, the op itself is to be erased.
  if (!ifCloneInElse.hasElse())
    ifCloneInElse.erase();
  else
    promoteIfBlock(ifCloneInElse, /*elseBlock=*/true);

  // Move the else version into the else block of the hoisted if op.
  auto *elseBlock = hoistedIfOp.getElseBlock();
  elseBlock->getOperations().splice(
      elseBlock->begin(), hoistOverOpClone->getBlock()->getOperations(),
      Block::iterator(hoistOverOpClone));

  return hoistedIfOp;
}

/// Get the AtomicRMWKind that corresponds to the class of the operation. This
/// list of cases covered must be kept in sync with `getSupportedReduction` and
/// `affineParallelize`.
static AtomicRMWKind getReductionOperationKind(Operation *op) {
  return TypeSwitch<Operation *, AtomicRMWKind>(op)
      .Case<AddFOp>([](Operation *) { return AtomicRMWKind::addf; })
      .Case<MulFOp>([](Operation *) { return AtomicRMWKind::mulf; })
      .Case<AddIOp>([](Operation *) { return AtomicRMWKind::addi; })
      .Case<MulIOp>([](Operation *) { return AtomicRMWKind::muli; })
      .Default([](Operation *) -> AtomicRMWKind {
        llvm_unreachable("unsupported reduction op");
      });
}

/// Get the value that is being reduced by `pos`-th reduction in the loop if
/// such a reduction can be performed by affine parallel loops. This assumes
/// floating-point operations are commutative. On success, `kind` will be the
/// reduction kind suitable for use in affine parallel loop builder. The list
/// of supported cases must be kept in sync with `affineParallelize`. If the
/// reduction is not supported, returns null.
static Value getSupportedReduction(AffineForOp forOp, unsigned pos,
                                   AtomicRMWKind &kind) {
  auto yieldOp = cast<AffineYieldOp>(forOp.getBody()->back());
  Value yielded = yieldOp.operands()[pos];
  Operation *definition = yielded.getDefiningOp();
  if (!isa_and_nonnull<AddFOp, MulFOp, AddIOp, MulIOp>(definition))
    return nullptr;

  if (!forOp.getRegionIterArgs()[pos].hasOneUse())
    return nullptr;

  kind = getReductionOperationKind(definition);
  if (definition->getOperand(0) == forOp.getRegionIterArgs()[pos]) {
    return definition->getOperand(1);
  }
  if (definition->getOperand(1) == forOp.getRegionIterArgs()[pos]) {
    return definition->getOperand(0);
  }

  return nullptr;
}

/// Replace affine.for with a 1-d affine.parallel and clone the former's body
/// into the latter while remapping values. Also parallelize reductions if
/// supported.
LogicalResult mlir::affineParallelize(AffineForOp forOp) {
  unsigned numReductions = forOp.getNumRegionIterArgs();
  SmallVector<AtomicRMWKind> reductionKinds;
  SmallVector<Value> reducedValues;
  reductionKinds.reserve(numReductions);
  reducedValues.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    reducedValues.push_back(
        getSupportedReduction(forOp, i, reductionKinds.emplace_back()));
    if (!reducedValues.back())
      return failure();
  }

  Location loc = forOp.getLoc();
  OpBuilder outsideBuilder(forOp);
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  // Creating empty 1-D affine.parallel op.
  AffineParallelOp newPloop = outsideBuilder.create<AffineParallelOp>(
      loc, ValueRange(reducedValues).getTypes(), reductionKinds,
      llvm::makeArrayRef(lowerBoundMap), lowerBoundOperands,
      llvm::makeArrayRef(upperBoundMap), upperBoundOperands,
      llvm::makeArrayRef(forOp.getStep()));
  // Steal the body of the old affine for op.
  newPloop.region().takeBody(forOp.region());
  Operation *yieldOp = &newPloop.getBody()->back();

  // Handle the initial values of reductions because the parallel loop always
  // starts from the neutral value.
  SmallVector<Value> newResults;
  newResults.reserve(numReductions);
  for (unsigned i = 0; i < numReductions; ++i) {
    Value init = forOp.getIterOperands()[i];
    // This works because we are only handling single-op reductions at the
    // moment. A switch on reduction kind or a mechanism to collect operations
    // participating in the reduction will be necessary for multi-op reductions.
    Operation *reductionOp = yieldOp->getOperand(i).getDefiningOp();
    outsideBuilder.getInsertionBlock()->getOperations().splice(
        outsideBuilder.getInsertionPoint(), newPloop.getBody()->getOperations(),
        reductionOp);
    reductionOp->setOperands({init, newPloop->getResult(i)});
    forOp->getResult(i).replaceAllUsesWith(reductionOp->getResult(0));
  }

  // Update the loop terminator to yield reduced values bypassing the reduction
  // operation itself (now moved outside of the loop) and erase the block
  // arguments that correspond to reductions.
  yieldOp->setOperands(reducedValues);
  newPloop.getBody()->eraseArguments(
      llvm::to_vector<4>(llvm::seq<unsigned>(1, numReductions + 1)));

  forOp.erase();
  return success();
}

// Returns success if any hoisting happened.
LogicalResult mlir::hoistAffineIfOp(AffineIfOp ifOp, bool *folded) {
  // Bail out early if the ifOp returns a result.  TODO: Consider how to
  // properly support this case.
  if (ifOp.getNumResults() != 0)
    return failure();

  // Apply canonicalization patterns and folding - this is necessary for the
  // hoisting check to be correct (operands should be composed), and to be more
  // effective (no unused operands). Since the pattern rewriter's folding is
  // entangled with application of patterns, we may fold/end up erasing the op,
  // in which case we return with `folded` being set.
  RewritePatternSet patterns(ifOp.getContext());
  AffineIfOp::getCanonicalizationPatterns(patterns, ifOp.getContext());
  bool erased;
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyOpPatternsAndFold(ifOp, frozenPatterns, &erased);
  if (erased) {
    if (folded)
      *folded = true;
    return failure();
  }
  if (folded)
    *folded = false;

  // The folding above should have ensured this, but the affine.if's
  // canonicalization is missing composition of affine.applys into it.
  assert(llvm::all_of(ifOp.getOperands(),
                      [](Value v) {
                        return isTopLevelValue(v) || isForInductionVar(v);
                      }) &&
         "operands not composed");

  // We are going hoist as high as possible.
  // TODO: this could be customized in the future.
  auto *hoistOverOp = getOutermostInvariantForOp(ifOp);

  AffineIfOp hoistedIfOp = ::hoistAffineIfOp(ifOp, hoistOverOp);
  // Nothing to hoist over.
  if (hoistedIfOp == ifOp)
    return failure();

  // Canonicalize to remove dead else blocks (happens whenever an 'if' moves up
  // a sequence of affine.fors that are all perfectly nested).
  (void)applyPatternsAndFoldGreedily(
      hoistedIfOp->getParentWithTrait<OpTrait::IsIsolatedFromAbove>(),
      frozenPatterns);

  return success();
}

// Return the min expr after replacing the given dim.
AffineExpr mlir::substWithMin(AffineExpr e, AffineExpr dim, AffineExpr min,
                              AffineExpr max, bool positivePath) {
  if (e == dim)
    return positivePath ? min : max;
  if (auto bin = e.dyn_cast<AffineBinaryOpExpr>()) {
    AffineExpr lhs = bin.getLHS();
    AffineExpr rhs = bin.getRHS();
    if (bin.getKind() == mlir::AffineExprKind::Add)
      return substWithMin(lhs, dim, min, max, positivePath) +
             substWithMin(rhs, dim, min, max, positivePath);

    auto c1 = bin.getLHS().dyn_cast<AffineConstantExpr>();
    auto c2 = bin.getRHS().dyn_cast<AffineConstantExpr>();
    if (c1 && c1.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), c1, substWithMin(rhs, dim, min, max, !positivePath));
    if (c2 && c2.getValue() < 0)
      return getAffineBinaryOpExpr(
          bin.getKind(), substWithMin(lhs, dim, min, max, !positivePath), c2);
    return getAffineBinaryOpExpr(
        bin.getKind(), substWithMin(lhs, dim, min, max, positivePath),
        substWithMin(rhs, dim, min, max, positivePath));
  }
  return e;
}
