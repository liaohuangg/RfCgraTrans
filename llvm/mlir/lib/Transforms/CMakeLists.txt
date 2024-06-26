add_subdirectory(Utils)

add_mlir_library(MLIRTransforms
  BufferDeallocation.cpp
  BufferOptimizations.cpp
  BufferResultsToOutParams.cpp
  BufferUtils.cpp
  Bufferize.cpp
  Canonicalizer.cpp
  CopyRemoval.cpp
  CSE.cpp
  Inliner.cpp
  LocationSnapshot.cpp
  LoopCoalescing.cpp
  LoopFusion.cpp
  LoopInvariantCodeMotion.cpp
  LoopRestructure.cpp
  Mem2Reg.cpp
  MemRefDataFlowOpt.cpp
  NormalizeMemRefs.cpp
  OpStats.cpp
  ParallelLower.cpp
  ParallelLoopCollapsing.cpp
  PipelineDataTransfer.cpp
  SCCP.cpp
  StripDebugInfo.cpp
  SymbolDCE.cpp
  ViewOpGraph.cpp
  ViewRegionGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRCopyOpInterface
  MLIRGPU
  MLIRLLVMIR
  MLIRLinalg
  MLIRLoopLikeInterface
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRTransformUtils
  MLIRVector
  )
