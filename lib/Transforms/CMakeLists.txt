add_mlir_library(RfCgraTransTransforms
  PlutoTransform.cc
  Reg2Mem.cc
  ExtractScopStmt.cc
  ScopStmtOpt.cc
  LoopAnnotate.cc
  LoopExtract.cc
  DfgCreate.cc
  Schedule.cc
  SearchUnroll.cc 
  RfCgraMap.cc

  ADDITIONAL_HEADER_DIRS
  "${RfCgraTrans_MAIN_INCLUDE_DIR}/RfCgraTrans/Transforms"

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRStandard
  MLIRSupport
  MLIRAffineToStandard
  libglpk
  libRfCgraMap
  RfCgraTransSupport
  RfCgraTransTargetOpenScop
)
