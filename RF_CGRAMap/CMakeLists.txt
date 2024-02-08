cmake_minimum_required(VERSION 3.5)
project(RfCgraMap)

#Generate the static library from the library sources
add_library(RfCgraMap STATIC
        src/CGRA.cpp  
        src/Register.cpp  
        src/DFG.cpp 
        src/tool.cpp 
        src/GraphRegister.cpp  
        src/Path.cpp
        )

target_include_directories(RfCgraMap
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        )

set(rfCGRAH include/CGRA.h  
        include/Register.h  
        include/DFG.h 
        include/tool.h 
        include/GraphRegister.h  
        include/Path.h)

#set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/RF_CGRAMap)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
install(TARGETS RfCgraMap
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib  # 静态库安装路径
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include  # 头文件安装路径
        )