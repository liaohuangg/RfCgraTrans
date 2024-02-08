//===- OpenScop.h -----------------------------------------------*- C++ -*-===//
//
// This file declares the interfaces for converting OpenScop representation to
// MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef RfCgraTrans_TARGET_OPENSCOP_H
#define RfCgraTrans_TARGET_OPENSCOP_H

#include <memory>

#include "pluto/internal/pluto.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class OwningModuleRef;
class MLIRContext;
class ModuleOp;
class FuncOp;
struct LogicalResult;
class Operation;
class Value;
} // namespace mlir

namespace RfCgraTrans {

class OslScop;
class OslSymbolTable;

std::unique_ptr<OslScop> createOpenScopFromFuncOp(mlir::FuncOp funcOp,
                                                  OslSymbolTable &symTable);

/// Create a function (FuncOp) from the given OpenScop object in the given
/// module (ModuleOp).

mlir::Operation *
judgeFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, mlir::ModuleOp module,
                         OslSymbolTable &symTable, mlir::MLIRContext *context,
                         PlutoProg *prog = nullptr,
                         const char *dumpClastAfterPluto = nullptr,int times = 0);

mlir::Operation *
finalFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, mlir::ModuleOp module,
                         OslSymbolTable &symTable, mlir::MLIRContext *context,
                         PlutoProg *prog = nullptr,
                         const char *dumpClastAfterPluto = nullptr,int times = 0);

mlir::Operation *
createFuncOpFromOpenScop(std::unique_ptr<OslScop> scop, mlir::ModuleOp module,
                         OslSymbolTable &symTable, mlir::MLIRContext *context,
                         PlutoProg *prog = nullptr,
                         const char *dumpClastAfterPluto = nullptr);

mlir::OwningModuleRef translateOpenScopToModule(std::unique_ptr<OslScop> scop,
                                                mlir::MLIRContext *context);

mlir::LogicalResult translateModuleToOpenScop(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<std::unique_ptr<OslScop>> &scops,
    llvm::raw_ostream &os);

void registerToOpenScopTranslation();
void registerFromOpenScopTranslation();

} // namespace RfCgraTrans

#endif
