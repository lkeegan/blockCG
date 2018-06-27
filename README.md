# blockCG [![Build Status](https://travis-ci.org/lkeegan/blockCG.svg?branch=master)](https://travis-ci.org/lkeegan/blockCG) [![codecov](https://codecov.io/gh/lkeegan/blockCG/branch/master/graph/badge.svg)](https://codecov.io/gh/lkeegan/blockCG)
Reference implementations of several Block Conjugate-Gradient iterative solvers in C++, including the SBCGrQ  multi-shift block CG solver. The block solvers are implemented in [inc/block_solvers.hpp](inc/block_solvers.hpp), with some simple examples of their use in [test/block_solvers.cpp](test/block_solvers.cpp). For full details of the solvers and references see the [documentation](doc/blockCG.pdf). This code uses the [Eigen](http://eigen.tuxfamily.org) C++ template library for the small dense matrix operations, and the [Catch2](https://github.com/catchorg/Catch2) test framework, both included.

## Solvers

Standard | Block | Multishift Block
-------- | ----- | ----------------
 CG      | BCG   | SBCGrQ
 SCG     | BCGrQ |
         | ...   |

## Use
To compile and run the tests in debug mode to check everything is working:
```
mkdir Debug
cd Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make tests
```
To compile the benchmark code that compares the SCG and SBCGrQ solvers:
```
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make benchmark
./benchmark 500 0.01
```
Where 500 is the "lattice volume", proportional to the size of the sparse matrix, and 0.01 is the "mass" - the smaller this number is the more ill-conditioned the matrix should be.