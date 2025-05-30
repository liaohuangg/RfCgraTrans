//===- ScopStmtOpt.cc - Optimise SCoP statement extraction ------------C++-===//

#include "RfCgraTrans/Transforms/ScopStmtOpt.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include <queue>
#include <utility>

#define DEBUG_TYPE "scop-stmt-opt"

using namespace mlir;
using namespace llvm;
using namespace RfCgraTrans;

static void replace(ValueRange srcValues,
                    SmallVectorImpl<mlir::Value> &dstValues,
                    BlockAndValueMapping &mapping) {
  for (Value src : srcValues) {
    // src could come from an index_cast.
    if (IndexCastOp op = src.getDefiningOp<IndexCastOp>())
      src = op.getOperand();

    dstValues.push_back(mapping.lookup(src));
  }
}

static Value findLastDefined(ValueRange values) {
  assert(!values.empty());

  Operation *parentOp = values[0].getParentBlock()->getParentOp();
  assert(isa<FuncOp>(parentOp));

  FuncOp f = cast<FuncOp>(parentOp);
  DominanceInfo dom(f);

  // TODO: should do a proper topological sort here.
  for (Value value : values) {
    if (value.isa<BlockArgument>())
      continue;
    bool dominatedByAll = true;
    for (Value other : values) {
      if (other == value || other.isa<BlockArgument>())
        continue;

      if (!dom.dominates(other.getDefiningOp(), value.getDefiningOp())) {
        dominatedByAll = false;
        break;
      }
    }

    if (dominatedByAll)
      return value;
  }

  assert(false);
}

static Operation *apply(mlir::AffineMap affMap, ValueRange operands,
                        BlockAndValueMapping &mapping, mlir::CallOp call,
                        OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);

  SmallVector<mlir::Value, 8> newOperands;
  replace(operands, newOperands, mapping);

  if (newOperands.size() > 0) {
    Value insertAfter = findLastDefined(newOperands);
    b.setInsertionPointAfterValue(insertAfter);
  } else
    b.setInsertionPointToStart(
        &(*(call->getParentOfType<FuncOp>().body().begin())));

  // TODO: properly handle these index casting cases.
  for (size_t i = 0; i < newOperands.size(); i++)
    if (newOperands[i].getType() != b.getIndexType())
      newOperands[i] = b.create<IndexCastOp>(call.getLoc(), newOperands[i],
                                             b.getIndexType());

  return b.create<mlir::AffineApplyOp>(call.getLoc(), affMap, newOperands);
}

static void projectAllOutExcept(FlatAffineConstraints &cst,
                                SetVector<mlir::Value> ids) {
  SmallVector<Value, 4> dims;
  cst.getIdValues(0, cst.getNumDimIds(), &dims);

  for (Value dim : dims)
    if (!ids.contains(dim))
      cst.projectOut(dim);
}

/// Calculate the lower bound and upper bound through affine apply, before the
/// function is being called.
static void getMemRefSize(MutableArrayRef<mlir::AffineForOp> forOps, FuncOp f,
                          CallOp call, int numDims,
                          SmallVectorImpl<Value> &dims, OpBuilder &b) {
  OpBuilder::InsertionGuard guard(b);

  assert(static_cast<int>(forOps.size()) >= numDims);

  SetVector<mlir::Value> indices;
  for (size_t i = forOps.size() - numDims; i < forOps.size(); i++)
    indices.insert(forOps[i].getInductionVar());

  SmallVector<Operation *, 4> enclosingOps{forOps.begin(), forOps.end()};

  FlatAffineConstraints cst;
  assert(succeeded(getIndexSet(enclosingOps, &cst)));

  // Project out indices other than the innermost.
  projectAllOutExcept(cst, indices);

  BlockAndValueMapping mapping;
  mapping.map(f.getArguments(), call.getOperands());

  SmallVector<mlir::Value, 4> mapOperands;
  cst.getIdValues(cst.getNumDimIds(), cst.getNumDimAndSymbolIds(),
                  &mapOperands);

  for (int dim = 0; dim < numDims; dim++) {
    mlir::AffineMap lbMap, ubMap;
    std::tie(lbMap, ubMap) = cst.getLowerAndUpperBound(
        dim, 0, numDims, numDims, llvm::None, b.getContext());

    // mlir::AffineMap lbMap = forOp.getLowerBoundMap();
    // mlir::AffineMap ubMap = forOp.getUpperBoundMap();

    assert(lbMap.getNumResults() == 1 &&
           "The given loop should have a single lower bound.");
    assert(ubMap.getNumResults() == 1 &&
           "The given loop should have a single upper bound.");
    Operation *lbOp = apply(lbMap, mapOperands, mapping, call, b);
    Operation *ubOp = apply(ubMap, mapOperands, mapping, call, b);

    b.setInsertionPointAfterValue(
        findLastDefined(ValueRange({lbOp->getResult(0), ubOp->getResult(0)})));

    Value size = b.create<mlir::AffineApplyOp>(
        forOps.back().getLoc(),
        mlir::AffineMap::get(
            0, 2, b.getAffineSymbolExpr(1) - b.getAffineSymbolExpr(0)),
        ValueRange{lbOp->getResult(0), ubOp->getResult(0)});
    dims.push_back(size);
  }
}

/// Append the given argument to the end of the argument list for both the
/// function and the caller.
static Value appendArgument(Value arg, FuncOp func, CallOp call, OpBuilder &b) {
  SmallVector<Type, 4> argTypes;
  SmallVector<Value, 4> operands;
  for (Type type : func.getArgumentTypes())
    argTypes.push_back(type);
  for (Value operand : call.getOperands())
    operands.push_back(operand);

  argTypes.push_back(arg.getType());
  operands.push_back(arg);

  call->setOperands(operands);
  func.setType(b.getFunctionType(argTypes, TypeRange(call.getResults())));
  Block &entryBlock = *(func.body().begin());
  entryBlock.addArgument(arg.getType());

  return entryBlock.getArguments().back();
}

static void scopStmtSplit(ModuleOp m, OpBuilder &b, FuncOp f, mlir::CallOp call,
                          Operation *op) {
  // op->dump();

  SmallVector<mlir::AffineForOp, 4> forOps;
  getLoopIVs(*op, &forOps);

  assert(
      forOps.size() >= 3 &&
      "The given op to split should be enclosed in at least three affine.for");
  int numDims = forOps.size() - 2;

  // For now we focus on the innermost for loop.
  mlir::AffineForOp forOp = forOps.back();
  // forOp.dump();

  SmallVector<mlir::Value, 4> memSizes;
  getMemRefSize(forOps, f, call, numDims, memSizes, b);
  // Since there is only one loop depth.
  MemRefType memType = MemRefType::get(SmallVector<int64_t>(numDims, -1),
                                       op->getResult(0).getType());

  b.setInsertionPointAfterValue(findLastDefined(ValueRange(memSizes)));
  // Allocation of the scratchpad memory.
  Operation *scrAlloc =
      b.create<memref::AllocaOp>(forOp.getLoc(), memType, memSizes);
  scrAlloc->setAttr("scop.scratchpad", b.getUnitAttr());

  // Pass it into the target function.
  Value scrInFunc = appendArgument(scrAlloc->getResult(0), f, call, b);
  // scrInFunc.dump();

  // Insert scratchpad read and write.
  b.setInsertionPointAfter(op);

  SmallVector<Value, 4> addrs;
  for (int dim = 0; dim < numDims; dim++) {
    mlir::AffineForOp curr = forOps[forOps.size() - numDims + dim];
    Value lb = b.create<mlir::AffineApplyOp>(
        op->getLoc(), curr.getLowerBoundMap(), curr.getLowerBoundOperands());
    Value addr = b.create<mlir::AffineApplyOp>(
        op->getLoc(),
        AffineMap::get(2, 0, b.getAffineDimExpr(0) - b.getAffineDimExpr(1)),
        ValueRange({curr.getInductionVar(), lb}));
    addrs.push_back(addr);
  }

  Operation *loadOp =
      b.create<mlir::AffineLoadOp>(op->getLoc(), scrInFunc, addrs);
  op->replaceAllUsesWith(loadOp);

  b.setInsertionPointAfterValue(addrs.back());
  b.create<mlir::AffineStoreOp>(op->getLoc(), op->getResult(0), scrInFunc,
                                addrs);
}

static void scopStmtSplit(ModuleOp m, OpBuilder &b, int toSplit) {
  FuncOp func;
  CallOp call;
  Operation *opToSplit;

  // Find the target function and the op to split within it.
  bool found = false;
  m.walk([&](FuncOp f) {
    if (found)
      return;

    f.walk([&](Operation *op) {
      if (found)
        return;
      if (op->hasAttr("scop.splittable") &&
          op->getAttrOfType<mlir::IntegerAttr>("scop.splittable").getInt() ==
              toSplit) {
        func = f;
        opToSplit = op;
        found = true;
      }
    });
  });

  assert(found && "Given split ID cannot be found");

  // Find the corresponding call op.
  m.walk([&](CallOp callOp) {
    if (callOp.callee() == func.getName())
      call = callOp;
  });

  scopStmtSplit(m, b, func, call, opToSplit);
}

namespace {
struct ScopStmtSplitPass
    : public mlir::PassWrapper<ScopStmtSplitPass, OperationPass<ModuleOp>> {
  ScopStmtSplitPass() = default;
  ScopStmtSplitPass(const ScopStmtSplitPass &pass) {}

  ListOption<int> toSplit{
      *this, "to-split",
      llvm::cl::desc(
          "A list of integer IDs describing the selected split points.")};

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    if (toSplit.empty()) {
      m.walk([&](Operation *op) {
        if (op->hasAttr("scop.splittable")) {
          toSplit.push_back(
              op->getAttrOfType<mlir::IntegerAttr>("scop.splittable").getInt());
        }
      });
    }

    for (auto id : toSplit)
      scopStmtSplit(m, b, id);
  }
};
} // namespace

static bool isSplittable(Operation *op) {
  // NOTE: some ops cannot be annotated in textual format. We skip them for now.
  if (isa<mlir::SIToFPOp>(op)) {
    return false;
  }

  SmallVector<mlir::AffineForOp, 4> forOps;
  getLoopIVs(*op, &forOps);

  if (forOps.size() < 1)
    return false;

  // For now we focus on the innermost for loop.
  mlir::AffineForOp forOp = forOps.back();
  mlir::AffineMap lbMap = forOp.getLowerBoundMap();
  mlir::AffineMap ubMap = forOp.getUpperBoundMap();
  if (lbMap.getNumDims() != 0 && lbMap.getNumResults() == 1)
    return false;
  if (ubMap.getNumDims() != 0 && ubMap.getNumResults() == 1)
    return false;

  return true;
}

static int annotateSplittable(FuncOp f, OpBuilder &b, int startId) {
  int id = startId;

  f.walk([&](mlir::AffineStoreOp storeOp) {
    // Breadth first search to find the splittable operations.

    // Stores operation and depth pairs.
    std::queue<std::pair<Operation *, int>> worklist;
    worklist.push(std::make_pair(storeOp, 0));

    while (!worklist.empty()) {
      std::pair<Operation *, int> front = worklist.front();
      worklist.pop();

      Operation *op = front.first;
      int depth = front.second;

      // Annotation.
      if (depth > 1 && !op->hasAttr("scop.splittable") && isSplittable(op)) {
        op->setAttr("scop.splittable", b.getIndexAttr(id));
        id++;
      }

      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        // Filter out block arguments.
        if (defOp == nullptr || operand.isa<BlockArgument>())
          continue;
        // Filter out operations out of the current region.
        if (defOp->getParentRegion() != storeOp->getParentRegion())
          continue;
        // Filter out defining operations of specific types.
        if (isa<mlir::AffineReadOpInterface, mlir::ConstantOp>(defOp))
          continue;

        worklist.push(std::make_pair(defOp, depth + 1));
      }
    }
  });

  return id - startId;
}

namespace {

struct AnnotateSplittablePass
    : public mlir::PassWrapper<AnnotateSplittablePass,
                               OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    int numSplittable = 0;
    m.walk([&](FuncOp f) {
      numSplittable += annotateSplittable(f, b, numSplittable);
    });
  }
};

} // namespace

static int64_t findOperand(Value value, Operation *op) {
  for (auto operand : enumerate(op->getOperands()))
    if (operand.value() == value)
      return operand.index();
  return -1;
}

static void unifyScratchpad(FuncOp f, ModuleOp m, OpBuilder &b) {
  // First find all the scratchpads generated.
  SmallVector<Value, 4> scratchpads;
  f.getBody().walk([&](memref::AllocaOp op) {
    if (op->hasAttr("scop.scratchpad"))
      scratchpads.push_back(op.getResult());
  });

  // No need to unify.
  if (scratchpads.size() <= 1)
    return;

  // Let's assume they all have the same dimensionality and the same element
  // type.
  size_t numDim;
  Type elemType;
  SmallVector<size_t, 4> numDims;
  for (size_t i = 0; i < scratchpads.size(); i++) {
    Value scr = scratchpads[i];
    MemRefType memType = scr.getType().cast<MemRefType>();
    if (i == 0) {
      numDim = memType.getShape().size();
      elemType = memType.getElementType();
    }
    if (memType.getShape().size() != numDim ||
        elemType != memType.getElementType()) { // Just exit, no effect.
      return;
    }
  }

  // Create a new scratchpad by taking the max dim size.
  SmallVector<int64_t, 4> shape(numDim, -1);
  MemRefType newMemType = MemRefType::get(shape, elemType);

  // Insert after the last scratchpad discovered.
  b.setInsertionPointAfterValue(scratchpads.back());

  SmallVector<Value, 4> maxDims;
  for (size_t d = 0; d < numDim; d++) {
    SmallVector<Value, 4> dims;
    for (Value scr : scratchpads) {
      memref::AllocaOp op = scr.getDefiningOp<memref::AllocaOp>();
      dims.push_back(op.getOperand(d));
    }

    Value maxDim = b.create<mlir::AffineMaxOp>(
        dims.front().getLoc(), dims.front().getType(),
        b.getMultiDimIdentityMap(dims.size()), ValueRange(dims));
    maxDims.push_back(maxDim);
  }

  Value newScr =
      b.create<memref::AllocaOp>(scratchpads.back().getDefiningOp()->getLoc(),
                                 newMemType, ValueRange(maxDims));

  // Then, replace the use of the first scratchpads with this one.
  scratchpads.front().replaceAllUsesWith(newScr);

  for (Operation *op : newScr.getUsers()) {
    if (mlir::CallOp caller = dyn_cast<mlir::CallOp>(op)) {
      FuncOp callee = m.lookupSymbol<FuncOp>(caller.getCallee());

      int64_t newMemIdx = findOperand(newScr, caller);

      // Replace scratchpad uses.
      for (Value scr : scratchpads)
        for (auto operand : enumerate(ValueRange(caller->getOperands())))
          if (operand.value() == scr)
            callee.getArgument(operand.index())
                .replaceAllUsesWith(callee.getArgument(newMemIdx));
    }
  }
}

static bool hasScratchpadDefined(FuncOp f) {
  bool result = false;

  f.getBody().walk([&](memref::AllocaOp op) {
    if (!result && op->hasAttr("scop.scratchpad")) {
      result = true;
      return;
    }
  });

  return result;
}

namespace {

/// Find scratchpads created by statement split and unify them into a single
/// one.
struct UnifyScratchpadPass
    : public mlir::PassWrapper<UnifyScratchpadPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // First find the main function that has those scratchpad declared.
    FuncOp f = nullptr;
    m.walk([&](FuncOp fun) {
      if (!f && hasScratchpadDefined(fun)) {
        f = fun;
        return;
      }
    });

    if (!f)
      return;
    unifyScratchpad(f, m, b);
  }
};
} // namespace

static void findAccessPatterns(Operation *op,
                               SmallVectorImpl<mlir::MemRefAccess> &accesses) {
  std::queue<Operation *> worklist;
  SmallPtrSet<Operation *, 8> visited;
  worklist.push(op);
  visited.insert(op);
  while (!worklist.empty()) {
    Operation *curr = worklist.front();
    worklist.pop();

    if (mlir::AffineLoadOp loadOp = dyn_cast<mlir::AffineLoadOp>(curr)) {
      std::vector<Value> ivs;
      OperandRange mapOperands = loadOp.getMapOperands();
      std::copy(mapOperands.begin(), mapOperands.end(),
                std::back_inserter(ivs));

      accesses.push_back(MemRefAccess(loadOp));
      continue;
    }

    for (Value operand : curr->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || visited.contains(defOp))
        continue;

      worklist.push(defOp);
      visited.insert(defOp);
    }
  }
}

static bool hasAdjacentInnermostLoop(ArrayRef<mlir::AffineForOp> forOps) {
  if (forOps.size() < 2)
    return false;

  mlir::AffineForOp innermost = forOps.back();
  mlir::AffineForOp parent = forOps[forOps.size() - 2];

  bool hasAdjacent = false;
  parent.getBody()->walk([&](mlir::AffineForOp op) {
    if (op != innermost) {
      hasAdjacent = true;
      return;
    }
  });

  return hasAdjacent;
}

static bool setEqual(ValueRange a, ValueRange b) {
  for (Value v : a) {
    if (std::find(b.begin(), b.end(), v) == b.end())
      return false;
  }
  for (Value v : b) {
    if (std::find(a.begin(), a.end(), v) == a.end())
      return false;
  }

  return true;
}

static bool satisfySplitHeuristic(mlir::AffineStoreOp op) {
  // Get the enclosing loop IVs.
  SmallVector<mlir::AffineForOp, 4> forOps;
  getLoopIVs(*op.getOperation(), &forOps);

  if (forOps.size() < 3)
    return false;
  if (hasAdjacentInnermostLoop(forOps))
    return false;

  SmallVector<mlir::Value, 4> ivs(forOps.size());
  std::transform(forOps.begin(), forOps.end(), ivs.begin(),
                 [](mlir::AffineForOp op) { return op.getInductionVar(); });

  // Check if the innermost loop index is being accessed by the store op (LHS).
  for (Value idx : op.getMapOperands())
    if (idx == ivs.back())
      return false;

  // All indvars except the innermost should appear on the LHS.
  if (!setEqual(op.getMapOperands(),
                ValueRange(ArrayRef<Value>(ivs.begin(), std::prev(ivs.end())))))
    return false;

  // Find if there are at least two different access patterns on the RHS.
  SmallVector<MemRefAccess, 4> accesses;
  findAccessPatterns(op, accesses);

  // Check if all memref access are to the same memref.
  bool allAccessToSame = true;
  for (MemRefAccess access : accesses)
    if (access.memref != op.getMemRef()) {
      allAccessToSame = false;
      break;
    }
  if (allAccessToSame)
    return false;
  if (accesses.size() <= 1)
    return false;

  // Examine all patterns. Each pattern is the list of indices being accessed.
  // We want to find the number of disjoint sets among all patterns, where in
  // the same set all patterns should access to the same indices in the same
  // order. We have a simple algo here, check each pair of patterns, and
  // determine whether they should be in the same set.
  SmallSet<size_t, 4> visited;
  int64_t numSets = 0;
  for (size_t i = 0; i < accesses.size(); i++) {
    if (visited.contains(i))
      continue;

    visited.insert(i);
    numSets++;
    for (size_t j = i + 1; j < accesses.size(); j++) {
      if (visited.contains(j) ||
          accesses[i].indices.size() != accesses[j].indices.size())
        continue;

      AffineValueMap vmap1, vmap2;
      accesses[i].getAccessMap(&vmap1);
      accesses[j].getAccessMap(&vmap2);

      AffineValueMap diff;
      AffineValueMap::difference(vmap1, vmap2, &diff);

      bool isSame = true;
      for (unsigned k = 0; isSame && k < diff.getNumResults(); k++) {
        if (!diff.getResult(k).isSymbolicOrConstant()) {
          isSame = false;
        }
      }
      if (isSame)
        visited.insert(j);
    }
  }
  return numSets >= 2;
}

int64_t annotateSplitId(mlir::AffineStoreOp op, int64_t startId, OpBuilder &b,
                        int targetDepth = 2) {
  std::queue<std::pair<Operation *, int>> worklist;
  SmallPtrSet<Operation *, 4> visited;

  int64_t currId = startId;
  worklist.emplace(op, 0);
  visited.insert(op);
  while (!worklist.empty()) {
    Operation *curr;
    int depth;
    std::tie(curr, depth) = worklist.front();
    worklist.pop();

    if (depth == targetDepth) {
      // if (!isSplittable(curr))
      //   continue;
      curr->setAttr("scop.splittable", b.getIndexAttr(currId));
      currId++;
      continue;
    }

    for (Value operand : curr->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || visited.contains(defOp))
        continue;
      if (isa<mlir::AffineLoadOp, ConstantOp>(defOp))
        continue;

      visited.insert(defOp);
      worklist.emplace(defOp, depth + 1);
    }
  }

  return currId;
}

int64_t annotateHeuristic(FuncOp f, int64_t startId, OpBuilder &b) {
  int64_t currId = startId;
  f.walk([&](mlir::AffineStoreOp op) {
    if (satisfySplitHeuristic(op)) {
      currId = annotateSplitId(op, currId, b);
    }
  });
  return currId;
}

namespace {

struct AnnotateHeuristicPass
    : public mlir::PassWrapper<AnnotateHeuristicPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    int64_t splitId = 0;
    m.walk([&](FuncOp f) { splitId = annotateHeuristic(f, splitId, b); });
  }
};

} // namespace

void RfCgraTrans::registerScopStmtOptPasses() {
  PassRegistration<AnnotateSplittablePass>(
      "annotate-splittable",
      "Give operations that are splittable in its expression tree.");
  PassRegistration<AnnotateHeuristicPass>(
      "annotate-heuristic",
      "Using the split heuristic to find split statements.");
  PassRegistration<ScopStmtSplitPass>(
      "scop-stmt-split", "Split a given set of splittable operations.");
  PassRegistration<UnifyScratchpadPass>(
      "unify-scratchpad", "Unify multiple scratchpads into a single one.");

  PassPipelineRegistration<>(
      "heuristic-split", "Split by heuristics", [](OpPassManager &pm) {
        pm.addPass(std::make_unique<AnnotateHeuristicPass>());
        pm.addPass(std::make_unique<ScopStmtSplitPass>());
        pm.addPass(std::make_unique<UnifyScratchpadPass>());
        pm.addPass(createCanonicalizerPass());
      });
}
