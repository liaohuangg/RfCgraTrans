add_mlir_translation_library(RfCgraTransTargetOpenScop
  OpenScop/ConvertFromOpenScop.cc
  OpenScop/ConvertToOpenScop.cc

  ADDITIONAL_HEADER_DIRS
  ${RfCgraTrans_MAIN_INCLUDE_DIR}/RfCgraTrans/Target/OpenScop

  LINK_COMPONENTS
  Core
  TransformUtils

  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRAffine
  MLIRAffineUtils
  MLIRTranslation
  
  RfCgraTransSupport
)
