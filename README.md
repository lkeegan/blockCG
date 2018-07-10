# blockCG [![Build Status](https://travis-ci.org/lkeegan/blockCG.svg?branch=master)](https://travis-ci.org/lkeegan/blockCG) [![codecov](https://codecov.io/gh/lkeegan/blockCG/branch/master/graph/badge.svg)](https://codecov.io/gh/lkeegan/blockCG)
Reference implementations of several Block Conjugate-Gradient iterative solvers in C++, including the SBCGrQ multi-shift block CG solver. The block solvers are implemented in [inc/block_solvers.hpp](inc/block_solvers.hpp), with some simple examples of their use in [test/solvers.cpp](test/solvers.cpp) and [benchmark.cpp](benchmark.cpp).

Block solvers are a variant of iterative Krylov solvers that act on multiple RHS vectors, and can converge significantly faster than standard solvers. For more details of the solvers implemented here, along with references to the original formulations, see the [documentation](doc/blockCG.pdf).

This code uses the [Eigen](http://eigen.tuxfamily.org) C++ template library for the small dense matrix operations, and the [Catch2](https://github.com/catchorg/Catch2) test framework, both included.

## Implemented solvers

Block     | Multishift Block | Standard 
--------- | ---------------- | --------
 BCG      | SBCGrQ           | CG
 BCGrQ    |                  | SCG

## Use
To compile and run the tests in debug mode to check everything is working:
```
mkdir Debug
cd Debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make tests
```
To compile and run the example benchmark code that compares the SCG and SBCGrQ solvers:
```
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release
make benchmark
./benchmark 1e3 1e-3 1e-10
```
where
- 1e3 is the "lattice volume": proportional to the size of the sparse matrix
- 1e-3 is the "mass": inversely related to the condition number of the matrix
- 1e-10 is the stopping criterion for the relative residual.