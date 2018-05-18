# blockCG [![Build Status](https://travis-ci.org/lkeegan/blockCG.svg?branch=master)](https://travis-ci.org/lkeegan/blockCG) [![codecov](https://codecov.io/gh/lkeegan/blockCG/branch/master/graph/badge.svg)](https://codecov.io/gh/lkeegan/blockCG)
Reference implementations of several Block Conjugate-Gradient iterative solvers in C++, using the [Eigen](http://eigen.tuxfamily.org) C++ template library for matrix operations.

## Block Solvers
- BCG
- BCGrQ
- ...

## Multishift Block Solvers
- SBCGrQ

## Use
To compile the code in debug mode and run the tests:
```
mkdir Debug
cd Debug
cmake ..
make
```
Alternatively to compile an optimised version:
```
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
