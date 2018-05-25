# blockCG [![Build Status](https://travis-ci.org/lkeegan/blockCG.svg?branch=master)](https://travis-ci.org/lkeegan/blockCG) [![codecov](https://codecov.io/gh/lkeegan/blockCG/branch/master/graph/badge.svg)](https://codecov.io/gh/lkeegan/blockCG)
Reference implementations of several Block Conjugate-Gradient iterative solvers in C++, including the SBCGrQ  multi-shift block CG solver. The block solvers are implemented in [inc/block_solvers.hpp](inc/block_solvers.hpp), with some simple examples of their use in [test/block_solvers.cpp](test/block_solvers.cpp). For full details of the solvers and references see the [documentation](doc/blockCG.pdf). This code uses the [Eigen](http://eigen.tuxfamily.org) C++ template library for matrix operations, and the [Catch2](https://github.com/catchorg/Catch2) test framework, both included.

## Standard Solvers
- CG
- SCG

## Block Solvers
- BCG
- BCGrQ [todo]
- ...

## Multishift Block Solvers
- SBCGrQ [todo]

## Use
To compile the code in debug mode and run the tests:
```
mkdir Debug
cd Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```
Alternatively to compile an optimised version:
```
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
