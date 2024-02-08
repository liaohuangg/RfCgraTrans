configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
  )

set(RfCgraTrans_TEST_DEPENDS
  FileCheck count not
  RfCgraTrans-translate
  RfCgraTrans-opt
  )

add_lit_testsuite(check-RfCgraTrans "Running the RfCgraTrans regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${RfCgraTrans_TEST_DEPENDS}
  )
set_target_properties(check-RfCgraTrans PROPERTIES FOLDER "Tests")

add_lit_testsuites(RfCgraTrans ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${RfCgraTrans_TEST_DEPS}
)
