add_mlir_library(RfCgraTransSupport
  OslScop.cc
  OslScopStmtOpSet.cc
  OslSymbolTable.cc
  ScopStmt.cc
  ScatteringUtils.cc

  ADDITIONAL_HEADER_DIRS
  ${RfCgraTrans_MAIN_INCLUDE_DIR}/RfCgraTrans/Support

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLoopAnalysis

  # libosl
  # libcloog
  # libisl
  libplutoisl
  libplutoosl
  libplutopip
  libplutopolylib
  libplutocloog
  libplutocandl
  libpluto
  libglpk
)
target_link_libraries(RfCgraTransSupport PUBLIC ${GMP_LIBRARY})
