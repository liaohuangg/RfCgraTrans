add_mlir_dialect_library(MLIRSCFTransforms
  Bufferize.cpp
  LoopSpecialization.cpp
  ParallelLoopFusion.cpp
  ParallelLoopTiling.cpp
	RaiseToAffine.cpp
  StructuralTypeConversions.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SCF

  DEPENDS
  MLIRSCFPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
