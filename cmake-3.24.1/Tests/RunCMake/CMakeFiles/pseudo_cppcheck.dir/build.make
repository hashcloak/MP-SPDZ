# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Bootstrap.cmk/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1

# Include any dependencies generated for this target.
include Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/compiler_depend.make

# Include the progress variables for this target.
include Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/progress.make

# Include the compile flags for this target's objects.
include Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/flags.make

Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o: Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/flags.make
Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o: Tests/RunCMake/pseudo_cppcheck.c
Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o: Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o"
	cd /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake && /usr/lib64/ccache/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o -MF CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o.d -o CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o -c /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake/pseudo_cppcheck.c

Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.i"
	cd /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake && /usr/lib64/ccache/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake/pseudo_cppcheck.c > CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.i

Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.s"
	cd /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake && /usr/lib64/ccache/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake/pseudo_cppcheck.c -o CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.s

# Object files for target pseudo_cppcheck
pseudo_cppcheck_OBJECTS = \
"CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o"

# External object files for target pseudo_cppcheck
pseudo_cppcheck_EXTERNAL_OBJECTS =

Tests/RunCMake/pseudo_cppcheck: Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/pseudo_cppcheck.c.o
Tests/RunCMake/pseudo_cppcheck: Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/build.make
Tests/RunCMake/pseudo_cppcheck: Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable pseudo_cppcheck"
	cd /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pseudo_cppcheck.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/build: Tests/RunCMake/pseudo_cppcheck
.PHONY : Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/build

Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/clean:
	cd /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake && $(CMAKE_COMMAND) -P CMakeFiles/pseudo_cppcheck.dir/cmake_clean.cmake
.PHONY : Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/clean

Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/depend:
	cd /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1 /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1 /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake /home/mikerah/Documents/HashCloak/Projects/MP-SPDZ/cmake-3.24.1/Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Tests/RunCMake/CMakeFiles/pseudo_cppcheck.dir/depend

