//===- OslScopStmtOpSet.h ---------------------------------------*- C++ -*-===//
//
// This file declares the class OslScopStmtOpSet.
//
//===----------------------------------------------------------------------===//
#ifndef RfCgraTrans_SUPPORT_OSLSCOPSTMTOPSET_H
#define RfCgraTrans_SUPPORT_OSLSCOPSTMTOPSET_H

#include "llvm/ADT/SetVector.h"

using namespace llvm;

namespace mlir {
class Operation;
struct LogicalResult;
class FlatAffineConstraints;
} // namespace mlir

namespace RfCgraTrans {

/// This class contains a set of operations that will correspond to a single
/// OpenScop statement body. The underlying data structure is SetVector.
class OslScopStmtOpSet {
public:
  using Set = SetVector<mlir::Operation *>;
  using iterator = Set::iterator;
  using reverse_iterator = Set::reverse_iterator;

  OslScopStmtOpSet() {}

  /// The core store op. There should be only one of it.
  mlir::Operation *getStoreOp() { return storeOp; }

  /// Insert.
  void insert(mlir::Operation *op);

  /// Count.
  unsigned count(mlir::Operation *op) { return opSet.count(op); };

  /// Size.
  unsigned size() { return opSet.size(); }

  /// Iterators.
  iterator begin() { return opSet.begin(); }
  iterator end() { return opSet.end(); }
  reverse_iterator rbegin() { return opSet.rbegin(); }
  reverse_iterator rend() { return opSet.rend(); }

  mlir::Operation *get(unsigned i) { return opSet[i]; }

  /// The domain of a stmtOpSet is the union of all load/store operations in
  /// that set. We calculate such a union by concatenating the constraints of
  /// domain defined by FlatAffineConstraints.
  /// TODO: improve the interface.
  mlir::LogicalResult getDomain(mlir::FlatAffineConstraints &domain);
  mlir::LogicalResult
  getDomain(mlir::FlatAffineConstraints &domain,
            SmallVectorImpl<mlir::Operation *> &enclosingOps);

  /// Get the enclosing operations for the opSet.
  mlir::LogicalResult
  getEnclosingOps(SmallVectorImpl<mlir::Operation *> &enclosingOps);

private:
  Set opSet;

  mlir::Operation *storeOp = nullptr;
};

} // namespace RfCgraTrans

#endif
