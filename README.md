# blockCG [![Build Status](https://travis-ci.org/lkeegan/blockCG.svg?branch=master)](https://travis-ci.org/lkeegan/blockCG) [![codecov](https://codecov.io/gh/lkeegan/blockCG/branch/master/graph/badge.svg)](https://codecov.io/gh/lkeegan/blockCG)
Reference implementations of several Block Conjugate-Gradient iterative solvers in C++, using the [Eigen](http://eigen.tuxfamily.org) C++ template library for matrix operations.

## Standard Solvers
- CG
- SCG

## Block Solvers
- BCG [todo]
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
