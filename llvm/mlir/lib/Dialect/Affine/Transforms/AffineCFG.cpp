#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "affine-cfg"

using namespace mlir;

namespace {
struct AffineCFGPass : public AffineCFGBase<AffineCFGPass> {
  void runOnFunction() override;
};
} // namespace

static bool inAffine(Operation *op) {
  auto *curOp = op;
  while (auto *parentOp = curOp->getParentOp()) {
    if (isa<AffineForOp, AffineParallelOp>(parentOp))
      return true;
    curOp = parentOp;
  }
  return false;
}

static void setLocationAfter(OpBuilder &b, mlir::Value val) {
  if (val.getDefiningOp()) {
    auto it = val.getDefiningOp()->getIterator();
    it++;
    b.setInsertionPoint(val.getDefiningOp()->getBlock(), it);
  }
  if (auto bop = val.dyn_cast<mlir::BlockArgument>())
    b.setInsertionPoint(bop.getOwner(), bop.getOwner()->begin());
}

struct IndexCastMovement : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    mlir::Value val = op.getOperand();
    if (auto bop = val.dyn_cast<mlir::BlockArgument>()) {
      if (op.getOperation()->getBlock() != bop.getOwner()) {
        op.getOperation()->moveAfter(bop.getOwner(), bop.getOwner()->begin());
        return success();
      }
      return failure();
    }

    if (val.getDefiningOp()) {
      if (op.getOperation()->getBlock() != val.getDefiningOp()->getBlock()) {
        auto it = val.getDefiningOp()->getIterator();
        it++;
        op.getOperation()->moveAfter(val.getDefiningOp()->getBlock(), it);
      }
      return failure();
    }
    return failure();
  }
};

struct SimplfyIntegerCastMath : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<AddIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<AddIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0), op.getType()),
          b2.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                 op.getType()));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<SubIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<SubIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0), op.getType()),
          b2.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                 op.getType()));
      return success();
    }
    if (auto iadd = op.getOperand().getDefiningOp<MulIOp>()) {
      OpBuilder b(rewriter);
      setLocationAfter(b, iadd.getOperand(0));
      OpBuilder b2(rewriter);
      setLocationAfter(b2, iadd.getOperand(1));
      rewriter.replaceOpWithNewOp<MulIOp>(
          op,
          b.create<IndexCastOp>(op.getLoc(), iadd.getOperand(0), op.getType()),
          b2.create<IndexCastOp>(op.getLoc(), iadd.getOperand(1),
                                 op.getType()));
      return success();
    }
    return failure();
  }
};

struct CanonicalizeAffineApply : public OpRewritePattern<AffineApplyOp> {
  using OpRewritePattern<AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineApplyOp affineOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 4> mapOperands(affineOp.mapOperands());
    auto map = affineOp.map();
    auto prevMap = map;

    fullyComposeAffineMapAndOperands(&map, &mapOperands);
    canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);

    if (map == prevMap)
      return failure();

    rewriter.replaceOpWithNewOp<AffineApplyOp>(affineOp, map, mapOperands);
    return success();
  }
};

struct CanonicalizeIndexCast : public OpRewritePattern<IndexCastOp> {
  using OpRewritePattern<IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IndexCastOp indexcastOp,
                                PatternRewriter &rewriter) const override {

    // Fold IndexCast(IndexCast(x)) -> x
    auto cast = indexcastOp.getOperand().getDefiningOp<IndexCastOp>();
    if (cast && cast.getOperand().getType() == indexcastOp.getType()) {
      mlir::Value vals[] = {cast.getOperand()};
      rewriter.replaceOp(indexcastOp, vals);
      return success();
    }

    // Fold IndexCast(constant) -> constant
    // A little hack because we go through int.  Otherwise, the size
    // of the constant might need to change.
    if (auto cst = indexcastOp.getOperand().getDefiningOp<ConstantOp>()) {
      rewriter.replaceOpWithNewOp<ConstantIndexOp>(
          indexcastOp, cst.getValue().cast<IntegerAttr>().getInt());
      return success();
    }
    return failure();
  }
};

/*
struct CanonicalizeAffineIf : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineIfOp affineOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 4> mapOperands(affineOp.mapOperands());
    auto map = affineOp.map();
    auto prevMap = map;

    fullyComposeAffineMapAndOperands(&map, &mapOperands);
    canonicalizeMapAndOperands(&map, &mapOperands);
    map = removeDuplicateExprs(map);

    if (map == prevMap)
      return failure();

    rewriter.replaceOpWithNewOp<AffineApplyOp>(affineOp, map, mapOperands);
    return success();
  }
};
*/

static bool isValidIndex(Value val) {
  if (mlir::isValidSymbol(val))
    return true;

  if (auto cast = val.getDefiningOp<IndexCastOp>())
    return isValidIndex(cast.getOperand());

  if (auto bop = val.getDefiningOp<AddIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (auto bop = val.getDefiningOp<MulIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (auto bop = val.getDefiningOp<SubIOp>())
    return isValidIndex(bop.getOperand(0)) && isValidIndex(bop.getOperand(1));

  if (val.getDefiningOp<ConstantIndexOp>())
    return true;

  if (val.getDefiningOp<ConstantOp>())
    return true;

  if (auto ba = val.dyn_cast<BlockArgument>()) {
    if (isa<AffineForOp>(ba.getOwner()->getParentOp()))
      return true;

    if (isa<AffineParallelOp>(ba.getOwner()->getParentOp()))
      return true;

    if (isa<FuncOp>(ba.getOwner()->getParentOp()))
      return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "illegal isValidIndex: " << val << "\n");
  return false;
}

bool handle(OpBuilder &b, CmpIOp cmpi, SmallVectorImpl<AffineExpr> &exprs,
            SmallVectorImpl<bool> &eqflags, SmallVectorImpl<Value> &applies) {
  AffineMap lhsmap =
      AffineMap::get(0, 1, getAffineSymbolExpr(0, cmpi.getContext()));
  if (!isValidIndex(cmpi.lhs())) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal lhs: " << cmpi.lhs() << " - " << cmpi << "\n");
    return false;
  }
  if (!isValidIndex(cmpi.rhs())) {
    LLVM_DEBUG(llvm::dbgs()
               << "illegal rhs: " << cmpi.rhs() << " - " << cmpi << "\n");
    return false;
  }
  SmallVector<Value, 4> lhspack = {cmpi.lhs()};
  if (!lhspack[0].getType().isa<IndexType>()) {
    auto op = b.create<mlir::IndexCastOp>(
        cmpi.getLoc(), lhspack[0], mlir::IndexType::get(cmpi.getContext()));
    lhspack[0] = op;
  }

  AffineMap rhsmap =
      AffineMap::get(0, 1, getAffineSymbolExpr(0, cmpi.getContext()));
  SmallVector<Value, 4> rhspack = {cmpi.rhs()};
  if (!rhspack[0].getType().isa<IndexType>()) {
    auto op = b.create<mlir::IndexCastOp>(
        cmpi.getLoc(), rhspack[0], mlir::IndexType::get(cmpi.getContext()));
    rhspack[0] = op;
  }

  applies.push_back(
      b.create<mlir::AffineApplyOp>(cmpi.getLoc(), lhsmap, lhspack));
  applies.push_back(
      b.create<mlir::AffineApplyOp>(cmpi.getLoc(), rhsmap, rhspack));
  AffineExpr dims[2] = {b.getAffineDimExpr(2 * exprs.size() + 0),
                        b.getAffineDimExpr(2 * exprs.size() + 1)};
  switch (cmpi.getPredicate()) {
  case CmpIPredicate::eq:
    exprs.push_back(dims[0] - dims[1]);
    eqflags.push_back(true);
    break;

  case CmpIPredicate::sge:
    exprs.push_back(dims[0] - dims[1]);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::sle:
    exprs.push_back(dims[1] - dims[0]);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::sgt:
    exprs.push_back(dims[0] - dims[1] + 1);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::slt:
    exprs.push_back(dims[1] - dims[0] - 1);
    eqflags.push_back(false);
    break;

  case CmpIPredicate::ne:
  case CmpIPredicate::ult:
  case CmpIPredicate::ule:
  case CmpIPredicate::ugt:
  case CmpIPredicate::uge:
    llvm_unreachable("unhandled icmp");
  }
  return true;
}

static void replaceStore(memref::StoreOp store,
                         const SmallVector<Value, 2> &newIndexes) {
  OpBuilder builder(store);
  Location loc = store.getLoc();
  builder.create<AffineStoreOp>(loc, store.getValueToStore(), store.getMemRef(),
                                newIndexes);
  store.erase();
}

static void replaceLoad(memref::LoadOp load, const SmallVector<Value, 2> &newIndexes) {
  OpBuilder builder(load);
  Location loc = load.getLoc();
  AffineLoadOp affineLoad =
      builder.create<AffineLoadOp>(loc, load.getMemRef(), newIndexes);
  load.getResult().replaceAllUsesWith(affineLoad.getResult());
  load.erase();
}

struct MoveLoadToAffine : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern<memref::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!inAffine(load))
      return failure();

    if (!llvm::all_of(load.getIndices(),
                      [&](Value index) { return isValidIndex(index); }))
      return failure();

    AffineLoadOp affineLoad = rewriter.create<AffineLoadOp>(
        load.getLoc(), load.getMemRef(), load.getIndices());
    load.getResult().replaceAllUsesWith(affineLoad.getResult());
    rewriter.eraseOp(load);
    return success();
  }
};

struct MoveStoreToAffine : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern<memref::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!inAffine(store))
      return failure();

    if (!llvm::all_of(store.getIndices(),
                      [&](Value index) { return isValidIndex(index); }))
      return failure();

    rewriter.create<AffineStoreOp>(store.getLoc(), store.getValueToStore(),
                                   store.getMemRef(), store.getIndices());
    rewriter.eraseOp(store);
    return success();
  }
};

void AffineCFGPass::runOnFunction() {
  getFunction().walk([&](scf::IfOp ifOp) {
    if (inAffine(ifOp)) {
      OpBuilder b(ifOp);
      AffineIfOp affineIfOp;
      std::vector<mlir::Type> types;
      for (auto v : ifOp.results()) {
        types.push_back(v.getType());
      }

      SmallVector<AffineExpr, 2> exprs;
      SmallVector<bool, 2> eqflags;
      SmallVector<Value, 4> applies;

      std::deque<Value> todo = {ifOp.condition()};
      while (todo.size()) {
        auto cur = todo.front();
        todo.pop_front();
        if (auto cmpi = cur.getDefiningOp<CmpIOp>()) {
          if (!handle(b, cmpi, exprs, eqflags, applies)) {
            return;
          }
          continue;
        }
        if (auto andi = cur.getDefiningOp<AndOp>()) {
          todo.push_back(andi.getOperand(0));
          todo.push_back(andi.getOperand(1));
          continue;
        }
        return;
      }

      auto iset = IntegerSet::get(/*dim*/ 2 * exprs.size(), /*symbol*/ 0, exprs,
                                  eqflags);
      affineIfOp = b.create<AffineIfOp>(ifOp.getLoc(), types, iset, applies,
                                        /*elseBlock=*/true);
      affineIfOp.thenRegion().takeBody(ifOp.thenRegion());
      affineIfOp.elseRegion().takeBody(ifOp.elseRegion());

      for (auto &blk : affineIfOp.thenRegion()) {
        if (auto yop = dyn_cast<scf::YieldOp>(blk.getTerminator())) {
          OpBuilder b(yop);
          b.create<AffineYieldOp>(yop.getLoc(), yop.results());
          yop.erase();
        }
      }
      for (auto &blk : affineIfOp.elseRegion()) {
        if (auto yop = dyn_cast<scf::YieldOp>(blk.getTerminator())) {
          OpBuilder b(yop);
          b.create<AffineYieldOp>(yop.getLoc(), yop.results());
          yop.erase();
        }
      }
      ifOp.replaceAllUsesWith(affineIfOp);
      ifOp.erase();
    }
  });

  getFunction().walk([](memref::StoreOp store) {
    if (!inAffine(store) && !llvm::all_of(store.getIndices(), [](mlir::Value V) {
      return V.getDefiningOp<mlir::ConstantOp>() != nullptr;
    }))
      return;
    if (!llvm::all_of(store.getIndices(),
                      [](Value index) { return isValidIndex(index); }))
      return;

    LLVM_DEBUG(llvm::dbgs() << "  affine store checks -> ok\n");
    SmallVector<Value, 2> newIndices;
    newIndices.reserve(store.getIndices().size());

    OpBuilder b(store);
    for (auto idx : store.getIndices()) {
      AffineMap idxmap =
          AffineMap::get(0, 1, getAffineSymbolExpr(0, idx.getContext()));
      if (!idx.getType().isa<IndexType>())
        idx = b.create<mlir::IndexCastOp>(
            idx.getLoc(), idx, mlir::IndexType::get(idx.getContext()));

      Value idxpack[1] = {idx};
      newIndices.push_back(
          b.create<mlir::AffineApplyOp>(idx.getLoc(), idxmap, idxpack));
    }

    replaceStore(store, newIndices);
  });

  getFunction().walk([](memref::LoadOp load) {
    if (!inAffine(load) && !llvm::all_of(load.getIndices(), [](mlir::Value V) {
      return V.getDefiningOp<mlir::ConstantOp>() != nullptr;
    }))
      return;
    if (!llvm::all_of(load.getIndices(),
                      [](Value index) { return isValidIndex(index); }))
      return;

    LLVM_DEBUG(llvm::dbgs() << "  affine load checks -> ok\n");
    SmallVector<Value, 2> newIndices;
    newIndices.reserve(load.getIndices().size());

    OpBuilder b(load);
    for (auto idx : load.getIndices()) {
      AffineMap idxmap =
          AffineMap::get(0, 1, getAffineSymbolExpr(0, idx.getContext()));
      if (!idx.getType().isa<IndexType>())
        idx = b.create<mlir::IndexCastOp>(
            idx.getLoc(), idx, mlir::IndexType::get(idx.getContext()));

      Value idxpack[1] = {idx};
      newIndices.push_back(
          b.create<mlir::AffineApplyOp>(idx.getLoc(), idxmap, idxpack));
    }

    replaceLoad(load, newIndices);
  });

  // getFunction().dump();
  {
    
    mlir::RewritePatternSet rpl(getFunction().getContext());
    rpl.add<SimplfyIntegerCastMath, CanonicalizeAffineApply, 
            CanonicalizeIndexCast, IndexCastMovement,
            MoveStoreToAffine, MoveLoadToAffine>(getFunction().getContext());
    applyPatternsAndFoldGreedily(getFunction().getOperation(), std::move(rpl),
                                 /*fold*/ false);
  }
  // getFunction().dump();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::replaceAffineCFGPass() {
  return std::make_unique<AffineCFGPass>();
}
