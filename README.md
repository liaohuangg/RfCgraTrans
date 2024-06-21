# RfCgraTrans
Artifact Evaluation Reproduction for "Polyhedral-based Data Reuse Optimization for Imperfectly-Nested Loop Mapping on CGRAs" 

## Table of contents
1. [Directory Structure](#directory-structure)
2. [Getting Started](#getting-started)
    1. [Hardware pre-requisities](#hardware-pre-requisities)
    2. [Software pre-requisites](#software-pre-requisites)
    3. [Installation](#installation)
    4. [Running example](#running-example)
    5. [Modify the parameters](#modify-the-parameters)
    6. [Data formats](#data-formats)
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
* CMAKE 3.10

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
## Running Example
You can run examples
```
cd example
./run.sh
```
In the example directory of Example, you will find this information：
* example.RfCgraTrans.in.mlir (The program transformed from C to MLIR representation.)
* example.RfCgraTrans.out.mlir (The final result of loop transformations.)
* example.RfCgraTrans*.cloog (Intermediate results of multiple loop transformations.)
* example.RfCgraTransFinal.cloog (Intermediate results of loop transformations.)
* Schedule*_solu*.out (DFG scheduling result)
* map*_solu*.txt (Scheduling format of back-end mapping)
* DFGInformation.out (Information about the DFG of the final loop transformation result.)

## Modify the parameters
If you wish to modify the parameters, you can open RfCgraTrans/include/RfCgraTrans/Transforms/DfgCreate.h. The meanings of each parameter will be explained next.
* PERow (The size of the Processing Element Array.)
* Tf (Cost of switching loop pipelines)
* Experimental_option (You can choose different methods for loop transformation. 0  local transformation, 1  global transformation, 2  maxfuse transformation, 3  nofuse transformation, 4  original loop)
* PBPMethod (You can toggle this switch to obtain the transformation results of PBP. 0 means off, and 1 means on.)
* search_trans_unroll_Switch (During loop transformation search, explore unroll factors of loop unrolling. 0 means off, and 1 means on.)
* final_unroll_Switch (After searching loop transformations, explore unroll factors of loop unrolling. 0 means off, and 1 means on.)
* schedule_Switch (You can choose whether to search for sub-scheduling of the DFG of final result of loop transformation. 0 means off, and 1 means on.)
* DFGLength (The range of activities of operators in DFG.)
* AfterUnrollDFGLength (The range of activities of operators in unrolled DFG.)
* searchScheduleNum (The total number of sub-schedules) 
* subScheduleNum (The number of sub-schedules searched after each relaxation of constraints.)

## Data Formats
For each loop, the intermediate result of the loop transformation is as follows:
```
if ((P0 >= 2) && (P1 >= 2)) {
  for (t2=0;t2<=P0-1;t2++) {
    S0(t2)
  }
  for (t2=0;t2<=P1-1;t2++) {
    for (t3=1;t3<=P0-1;t3++) {
      S2(t2, t3)
    }
  }
  for (t2=0;t2<=P0-2;t2++) {
    for (t3=1;t3<=P1-1;t3++) {
      S1(t3, t2)
      S3(t3-1, t2)
    }
  }
  for (t3=1;t3<=P1-1;t3++) {
    S1(t3, P0-1)
  }
}

```
After converting to mlir, it looks like this:
```
affine.if #set()[%1, %0] {
      affine.for %arg7 = 0 to %1 {
        call @S0(%arg4, %arg7, %arg6) : (memref<?x2600xf64>, index, memref<?xf64>) -> ()
      }
      affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = 1 to %1 {
          call @S2(%arg3, %arg7, %arg8, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
        }
      }
      affine.for %arg7 = 0 to #map0()[%1] {
        affine.for %arg8 = 1 to %0 {
          call @S1(%arg4, %arg8, %arg7, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
          %2 = affine.apply #map1(%arg8)
          call @S3(%arg5, %2, %arg7, %arg4, %arg3) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, memref<?x2600xf64>) -> ()
        }
      }
      affine.for %arg7 = 1 to %0 {
        %2 = affine.apply #map0()[%1]
        call @S1(%arg4, %arg7, %2, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
      }
    }
    return
  }
```
The information in the DFG is as follows：
```
dfg_id
0
II of DFG
1
dfg_dim
1

dfg_node_info
————————————loadNode————————————

NodeShift
0 
iterFlag
0 
iterOrder
-1 
nodeID
0
NodeType
ArrayLoad
Array
%3 = memref.alloc() : memref<1000xf64>
earlist 0
lastest 3
timeStep 3
————————————————————————————————

dfg_edge_info
Edge1: begin 2 end 4 min 0 dif 0 type Normal
```
Format of DFG scheduling
```
=======schedule=====
 II =  1
timeStep0  LSU                 PE 
timeStep1  LSU N1T1_4 N2T1_10  PE 
timeStep2  LSU                 PE 
timeStep3  LSU                 PE 
timeStep4  LSU                 PE N4T4_16 
timeStep5  LSU                 PE N5T5_20 
timeStep6  LSU                 PE N6T6_24 
timeStep7  LSU N7T7_28         PE 

The total number of register
begin 2 end 4 dif 0 startTime 1 endTime 4 useRe 3
begin 4 end 5 dif 0 startTime 4 endTime 5 useRe 1
begin 5 end 6 dif 0 startTime 5 endTime 6 useRe 1
begin 1 end 6 dif 0 startTime 1 endTime 6 useRe 5
begin 6 end 7 dif 0 startTime 6 endTime 7 useRe 1
begin 2 end 4 dif 1 startTime 1 endTime 4 useRe 4
```
Format of DFG scheduling that try to map
```
|----------|------------|-------------|------------|------------|-------------|-------------|
|node index|  time step |node's type  |child node 1|child node 2|edge 1's dif |edge 2's dif |
|----------|------------|-------------|------------|------------|-------------|-------------|
```
An example
```
0,0,1,2,2,0,1
1,1,1,4,-1,0,-1
2,2,0,3,-1,0,-1
3,4,0,4,-1,0,-1
4,6,0,5,-1,0,-1
5,7,1,-1,-1,-1,-1
```

Project [RF-Map](https://github.com/coralabo/RF-Map ) is able to perform back-end mapping of DFG in such a format.

# Reference

```
@inproceedings{huang2023optimizing,
  title={Optimizing Data Reuse for CGRA Mapping Using Polyhedral-based Loop Transformations},
  author={Huang, Liao and Liu, Dajiang},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
```
