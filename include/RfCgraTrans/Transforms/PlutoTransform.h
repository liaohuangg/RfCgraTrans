//===- PlutoTransform.h - Transform MLIR code by PLUTO --------------------===//
//
// This file declares the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

#ifndef RfCgraTrans_TRANSFORMS_PLUTOTRANSFORM_H
#define RfCgraTrans_TRANSFORMS_PLUTOTRANSFORM_H
#include "llvm/ADT/SetVector.h"

namespace RfCgraTrans {
void registerPlutoTransformPass();
}

#endif
