set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(RfCgraTrans-translate
  RfCgraTrans-translate.cc
)
llvm_update_compile_flags(RfCgraTrans-translate)
target_link_libraries(RfCgraTrans-translate
  PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  RfCgraTransTargetOpenScop
  )
