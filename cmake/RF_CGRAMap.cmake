# Install RF_CGRAMap as an external project.

include(ExternalProject)
set(RF_CGRAMap_INCLUDE_DIR "${CMAKE_BINARY_DIR}/RF_CGRAMap/include")
set(RF_CGRAMap_LIB_DIR "${CMAKE_BINARY_DIR}/RF_CGRAMap/lib")

ExternalProject_Add(
  RF_CGRAMap 
  SOURCE_DIR "${CMAKE_SOURCE_DIR}/RF_CGRAMap"
  CONFIGURE_COMMAND "${CMAKE_SOURCE_DIR}/RF_CGRAMap/build.sh"  --prefix=${CMAKE_BINARY_DIR}/RF_CGRAMap
  PREFIX ${CMAKE_BINARY_DIR}/RF_CGRAMap
  # BUILD_COMMAND make
  # INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  BUILD_BYPRODUCTS "${RF_CGRAMap_LIB_DIR}/libRfCgraMap.a"
)
add_library(libRfCgraMap STATIC IMPORTED)
set_target_properties(libRfCgraMap PROPERTIES IMPORTED_LOCATION "${RF_CGRAMap_LIB_DIR}/libRfCgraMap.a")
add_dependencies(libRfCgraMap RF_CGRAMap)

