# RfCgraTrans
Artifact Evaluation Reproduction for "Polyhedral-based Data Reuse Optimization for Imperfectly-Nested Loop Mapping on CGRAs" 

## Table of contents

# Directory Structure
```
RfCgraTrans
│   README.md
│   CMakeLists.txt
│   LICENSE
│   Makefile
│───build
│───cmake
│     │───CLooG.cmake
│     │───FindGMP.cmake
│     │───glpk.cmake
│     │───OpenScop.cmake
│     │───PLUTO.cmake
│     └───RF_CGRAMap.cmake
│───example (Test case)
│───glpk (Integer Linear Programming (ILP) toolkits)
│───include
│───lib
│───llvm
│───pluto (Tools supporting polyhedral model loop optimization)
│───RF_DATE (Tools supporting RF-CGRA mapping)
│───RF_CGRAMap
│───test
└───tools
```

Getting start
Hardware pre-requisities
--
* Ubuntu 20.04.5
* Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
Software pre-requisities
--
*PLuTo
*LLVM

Installation
--
Prepare dependency packages
'''Bash
apt-get update
apt-get install apt-utils
sudo apt install tzdata build-essential \
libtool autoconf pkg-config flex bison \
libgmp-dev clang-9 libclang-9-dev texinfo \
cmake ninja-build git texlive-full numactl
'''


