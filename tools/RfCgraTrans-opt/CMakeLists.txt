set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(RfCgraTrans-opt
  RfCgraTrans-opt.cc
)
llvm_update_compile_flags(RfCgraTrans-opt)
target_link_libraries(RfCgraTrans-opt
  PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIREDSC
  MLIRMath
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  RfCgraTransTransforms
  )
