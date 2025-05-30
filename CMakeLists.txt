cmake_minimum_required(VERSION 3.10)

# Policies requried by LLVM.
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()


project(RfCgraTrans LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_SOURCE_DIR}/cmake")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

# ------------------------------------------------- Dependencies
# Find whether the GMP package exists.
find_package(GMP REQUIRED)

# Configure and build OpenScop to ${CMAKE_BUILD_DIR}/openscop 
# include("${CMAKE_SOURCE_DIR}/cmake/OpenScop.cmake")
# include_directories("${OSL_INCLUDE_DIR}")

# Configure and build CLooG/ISL to ${CMAKE_BUILD_DIR}/cloog
# include("${CMAKE_SOURCE_DIR}/cmake/CLooG.cmake")
# include_directories("${CLOOG_INCLUDE_DIR}")

# Configure and build PLUTO to ${CMAKE_BUILD_DIR}/pluto
include("${CMAKE_SOURCE_DIR}/cmake/PLUTO.cmake")
include_directories("${PLUTO_INCLUDE_DIR}")

# Configure and build glpk to ${CMAKE_BUILD_DIR}/glpk
include("${CMAKE_SOURCE_DIR}/cmake/glpk.cmake")
include_directories("${GLPK_INCLUDE_DIR}")

# Configure and build RF_CGRAMap to ${CMAKE_BUILD_DIR}/RF_CGRAMap
include("${CMAKE_SOURCE_DIR}/cmake/RF_CGRAMap.cmake")
include_directories("${RF_CGRAMap_INCLUDE_DIR}")


# ------------------------------------------------- This project
set(RfCgraTrans_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" ) # --src-root
set(RfCgraTrans_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ) # --includedir

set(RfCgraTrans_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(RfCgraTrans_BINARY_DIR "${CMAKE_BINARY_DIR}/bin")
set(RfCgraTrans_TOOLS_DIR  "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories("${LLVM_INCLUDE_DIRS}")
include_directories("${MLIR_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")


add_subdirectory(include/RfCgraTrans)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
