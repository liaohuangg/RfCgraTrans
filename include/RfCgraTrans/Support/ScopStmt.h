//===- ScopStmt.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the class ScopStmt.
//
//===----------------------------------------------------------------------===//

#ifndef RfCgraTrans_SUPPORT_SCOPSTMT_H
#define RfCgraTrans_SUPPORT_SCOPSTMT_H

#include <memory>

#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class FlatAffineConstraints;
class AffineValueMap;
class FuncOp;
class CallOp;
class Value;
} // namespace mlir

namespace RfCgraTrans {

class ScopStmtImpl;

/// Class that stores all the essential information for a Scop statement,
/// including the MLIR operations and Scop relations, and handles the processing
/// of them.
class ScopStmt {
public:
  ScopStmt(mlir::Operation *caller, mlir::Operation *callee);
  ~ScopStmt();

  ScopStmt(ScopStmt &&);
  ScopStmt(const ScopStmt &) = delete;
  ScopStmt &operator=(ScopStmt &&);
  ScopStmt &operator=(const ScopStmt &&) = delete;

  mlir::FlatAffineConstraints *getDomain() const;

  /// Get a copy of the enclosing operations.
  void getEnclosingOps(llvm::SmallVectorImpl<mlir::Operation *> &ops,
                       bool forOnly = false) const;
  /// Get the callee of this scop stmt.
  mlir::FuncOp getCallee() const;
  /// Get the caller of this scop stmt.
  mlir::CallOp getCaller() const;
  /// Get the access AffineValueMap of an op in the callee and the memref in the
  /// caller scope that this op is using.
  void getAccessMapAndMemRef(mlir::Operation *op, mlir::AffineValueMap *vMap,
                             mlir::Value *memref) const;

private:
  std::unique_ptr<ScopStmtImpl> impl;
};
} // namespace RfCgraTrans

#endif
