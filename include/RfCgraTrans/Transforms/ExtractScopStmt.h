//===- ExtractScopStmt.h - Extract scop stmt to func ------------------C++-===//
//
// This file declares the transformation that extracts scop statements into MLIR
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef RfCgraTrans_TRANSFORMS_EXTRACTSCOPSTMT_H
#define RfCgraTrans_TRANSFORMS_EXTRACTSCOPSTMT_H

/// TODO: place this macro at the right position.
#define SCOP_STMT_ATTR_NAME "scop.stmt"

namespace RfCgraTrans {

void registerExtractScopStmtPass();

}

#endif
