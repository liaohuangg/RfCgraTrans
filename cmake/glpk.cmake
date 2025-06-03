# Install glpk as an external project.

include(ExternalProject)
set(GLPK_INCLUDE_DIR "${CMAKE_BINARY_DIR}/glpk/include")
set(GLPK_LIB_DIR "${CMAKE_BINARY_DIR}/glpk/lib")

ExternalProject_Add(
  glpk 
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/glpk"
  CONFIGURE_COMMAND "${CMAKE_SOURCE_DIR}/glpk/autogen.sh" && "${CMAKE_SOURCE_DIR}/glpk/configure"  --prefix=${CMAKE_BINARY_DIR}/glpk
  PREFIX ${CMAKE_BINARY_DIR}/glpk
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${GLPK_LIB_DIR}/libglpk.a"
)
add_library(libglpk STATIC IMPORTED)
set_target_properties(libglpk PROPERTIES IMPORTED_LOCATION "${GLPK_LIB_DIR}/libglpk.a")
add_dependencies(libglpk glpk)
