# otherwise those tools added by add_llvm_tool won't be installed.
set(LLVM_BUILD_TOOLS ON)

add_subdirectory(RfCgraTrans-translate)
add_subdirectory(RfCgraTrans-opt)
