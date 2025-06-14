# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspace/RfCgraTrans/RF_CGRAMap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspace/RfCgraTrans/RF_CGRAMap

# Include any dependencies generated for this target.
include CMakeFiles/RfCgraMap.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/RfCgraMap.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/RfCgraMap.dir/flags.make

CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.o: CMakeFiles/RfCgraMap.dir/flags.make
CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.o: src/CGRA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.o -c /root/workspace/RfCgraTrans/RF_CGRAMap/src/CGRA.cpp

CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/RfCgraTrans/RF_CGRAMap/src/CGRA.cpp > CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.i

CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/RfCgraTrans/RF_CGRAMap/src/CGRA.cpp -o CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.s

CMakeFiles/RfCgraMap.dir/src/Register.cpp.o: CMakeFiles/RfCgraMap.dir/flags.make
CMakeFiles/RfCgraMap.dir/src/Register.cpp.o: src/Register.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/RfCgraMap.dir/src/Register.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RfCgraMap.dir/src/Register.cpp.o -c /root/workspace/RfCgraTrans/RF_CGRAMap/src/Register.cpp

CMakeFiles/RfCgraMap.dir/src/Register.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RfCgraMap.dir/src/Register.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/RfCgraTrans/RF_CGRAMap/src/Register.cpp > CMakeFiles/RfCgraMap.dir/src/Register.cpp.i

CMakeFiles/RfCgraMap.dir/src/Register.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RfCgraMap.dir/src/Register.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/RfCgraTrans/RF_CGRAMap/src/Register.cpp -o CMakeFiles/RfCgraMap.dir/src/Register.cpp.s

CMakeFiles/RfCgraMap.dir/src/DFG.cpp.o: CMakeFiles/RfCgraMap.dir/flags.make
CMakeFiles/RfCgraMap.dir/src/DFG.cpp.o: src/DFG.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/RfCgraMap.dir/src/DFG.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RfCgraMap.dir/src/DFG.cpp.o -c /root/workspace/RfCgraTrans/RF_CGRAMap/src/DFG.cpp

CMakeFiles/RfCgraMap.dir/src/DFG.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RfCgraMap.dir/src/DFG.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/RfCgraTrans/RF_CGRAMap/src/DFG.cpp > CMakeFiles/RfCgraMap.dir/src/DFG.cpp.i

CMakeFiles/RfCgraMap.dir/src/DFG.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RfCgraMap.dir/src/DFG.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/RfCgraTrans/RF_CGRAMap/src/DFG.cpp -o CMakeFiles/RfCgraMap.dir/src/DFG.cpp.s

CMakeFiles/RfCgraMap.dir/src/tool.cpp.o: CMakeFiles/RfCgraMap.dir/flags.make
CMakeFiles/RfCgraMap.dir/src/tool.cpp.o: src/tool.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/RfCgraMap.dir/src/tool.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RfCgraMap.dir/src/tool.cpp.o -c /root/workspace/RfCgraTrans/RF_CGRAMap/src/tool.cpp

CMakeFiles/RfCgraMap.dir/src/tool.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RfCgraMap.dir/src/tool.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/RfCgraTrans/RF_CGRAMap/src/tool.cpp > CMakeFiles/RfCgraMap.dir/src/tool.cpp.i

CMakeFiles/RfCgraMap.dir/src/tool.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RfCgraMap.dir/src/tool.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/RfCgraTrans/RF_CGRAMap/src/tool.cpp -o CMakeFiles/RfCgraMap.dir/src/tool.cpp.s

CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.o: CMakeFiles/RfCgraMap.dir/flags.make
CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.o: src/GraphRegister.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.o -c /root/workspace/RfCgraTrans/RF_CGRAMap/src/GraphRegister.cpp

CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/RfCgraTrans/RF_CGRAMap/src/GraphRegister.cpp > CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.i

CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/RfCgraTrans/RF_CGRAMap/src/GraphRegister.cpp -o CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.s

CMakeFiles/RfCgraMap.dir/src/Path.cpp.o: CMakeFiles/RfCgraMap.dir/flags.make
CMakeFiles/RfCgraMap.dir/src/Path.cpp.o: src/Path.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/RfCgraMap.dir/src/Path.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/RfCgraMap.dir/src/Path.cpp.o -c /root/workspace/RfCgraTrans/RF_CGRAMap/src/Path.cpp

CMakeFiles/RfCgraMap.dir/src/Path.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/RfCgraMap.dir/src/Path.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/RfCgraTrans/RF_CGRAMap/src/Path.cpp > CMakeFiles/RfCgraMap.dir/src/Path.cpp.i

CMakeFiles/RfCgraMap.dir/src/Path.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/RfCgraMap.dir/src/Path.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/RfCgraTrans/RF_CGRAMap/src/Path.cpp -o CMakeFiles/RfCgraMap.dir/src/Path.cpp.s

# Object files for target RfCgraMap
RfCgraMap_OBJECTS = \
"CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.o" \
"CMakeFiles/RfCgraMap.dir/src/Register.cpp.o" \
"CMakeFiles/RfCgraMap.dir/src/DFG.cpp.o" \
"CMakeFiles/RfCgraMap.dir/src/tool.cpp.o" \
"CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.o" \
"CMakeFiles/RfCgraMap.dir/src/Path.cpp.o"

# External object files for target RfCgraMap
RfCgraMap_EXTERNAL_OBJECTS =

libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/src/CGRA.cpp.o
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/src/Register.cpp.o
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/src/DFG.cpp.o
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/src/tool.cpp.o
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/src/GraphRegister.cpp.o
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/src/Path.cpp.o
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/build.make
libRfCgraMap.a: CMakeFiles/RfCgraMap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libRfCgraMap.a"
	$(CMAKE_COMMAND) -P CMakeFiles/RfCgraMap.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/RfCgraMap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/RfCgraMap.dir/build: libRfCgraMap.a

.PHONY : CMakeFiles/RfCgraMap.dir/build

CMakeFiles/RfCgraMap.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/RfCgraMap.dir/cmake_clean.cmake
.PHONY : CMakeFiles/RfCgraMap.dir/clean

CMakeFiles/RfCgraMap.dir/depend:
	cd /root/workspace/RfCgraTrans/RF_CGRAMap && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/RfCgraTrans/RF_CGRAMap /root/workspace/RfCgraTrans/RF_CGRAMap /root/workspace/RfCgraTrans/RF_CGRAMap /root/workspace/RfCgraTrans/RF_CGRAMap /root/workspace/RfCgraTrans/RF_CGRAMap/CMakeFiles/RfCgraMap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/RfCgraMap.dir/depend

