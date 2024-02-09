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

# Getting start
## Hardware pre-requisities
* Ubuntu 20.04.5
* Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
## Software pre-requisities
* PLuTo
* LLVM

## Installation
Prepare dependency packages
```
apt-get update
apt-get install apt-utils
sudo apt install tzdata build-essential \
libtool autoconf pkg-config flex bison \
libgmp-dev clang-9 libclang-9-dev texinfo \
cmake ninja-build git texlive-full numactl
```
After downloading, perform version control for better support of PLuTo.
```
update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 100
update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 100
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 100
```
First, complite the pluto tool
```
cd pluto
make clean
./autogen.sh
./configure
make -j`nproc`
```
Second, compile the clang-mlir front-end, and thanks to the Polygeist project for providing us with the front-end.
```
cd
git clone -b main-042621 --single-branch \
https://github.com/wsmoses/Polygeist \
mlir-clang
cd mlir-clang/
mkdir build
cd build/
cmake -G Ninja ../llvm \
-DLLVM_ENABLE_PROJECTS="mlir;
polly;clang;openmp" \
-DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="host" \
-DCMAKE_BUILD_TYPE=Release \
-DLLVM_ENABLE_ASSERTIONS=ON
ninja -j $(nproc) 
```
Third, compile this project.
```
mkdir llvm/build
cmake ../llvm \
      -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
      -DLLVM_TARGETS_TO_BUILD="host" \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DCMAKE_BUILD_TYPE=DEBUG \
      -DLLVM_INSTALL_UTILS=ON \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -G Ninja 
ninja -j$(nproc)

mkdir build
cd build
export BUILD=$PWD/../llvm/build
cmake .. \
	-DCMAKE_BUILD_TYPE=DEBUG \
	-DMLIR_DIR=$BUILD/lib/cmake/mlir \
	-DLLVM_DIR=$BUILD/lib/cmake/llvm \
	-DLLVM_ENABLE_ASSERTIONS=ON \
	-DLLVM_EXTERNAL_LIT=$BUILD/bin/llvm-lit \
-G Ninja
ninja -j`nproc`
export LD_LIBRARY_PATH=~/RfCgraTrans/build/pluto/lib:$LD_LIBRARY_PATH
```

```
```

