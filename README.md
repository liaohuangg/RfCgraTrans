RfCgraTrans
==
Artifact Evaluation Reproduction for "Polyhedral-based Data Reuse Optimization for Imperfectly-Nested Loop Mapping on CGRAs" 
Table of contents
--
Getting start
==
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


