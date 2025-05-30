# -*- Python -*-

# Configuration file for the 'lit' test runner.

import os
import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'RfCgraTrans'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files. This is overriden
# by individual lit.local.cfg files in the test subdirectories.
config.suffixes = ['.mlir', '.scop']

# excludes: A list of directories or files to exclude from the testsuite even
# if they match the suffixes pattern.
config.excludes = ['archive']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.RfCgraTrans_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Propagate some variables from the host environment.
llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])


llvm_config.use_default_substitutions()

# For each occurrence of a tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.

tool_dirs = [config.RfCgraTrans_tools_dir, config.llvm_tools_dir]
tools = [
    'RfCgraTrans-opt',
    'RfCgraTrans-translate'
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
