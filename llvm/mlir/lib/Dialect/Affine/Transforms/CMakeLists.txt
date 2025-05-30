add_mlir_dialect_library(MLIRAffineTransforms
  AffineDataCopyGeneration.cpp
  AffineLoopInvariantCodeMotion.cpp
  AffineLoopNormalize.cpp
  AffineParallelize.cpp
  AffineCFG.cpp
  AffineReduction.cpp
  LoopTiling.cpp
  LoopUnroll.cpp
  LoopUnrollAndJam.cpp
  SuperVectorize.cpp
  SimplifyAffineStructures.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineOpsIncGen
  MLIRAffinePassIncGen
  MLIRLoopLikeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineUtils
  MLIREDSC
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRStandard
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToLLVM
  )

