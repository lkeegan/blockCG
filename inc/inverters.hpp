#ifndef LKEEGAN_BLOCKCG_INVERTERS_H
#define LKEEGAN_BLOCKCG_INVERTERS_H
#include <vector>
#include <complex>
#ifdef EIGEN_USE_MKL_ALL
  // ugly hack to get mkl to work with c++ std::complex type 
  #define MKL_Complex16 std::complex<double>
  #include "mkl.h"
#endif
#include "Eigen3/Eigen/Dense"
#include "Eigen3/Eigen/StdVector"

int cg ();
 
#endif //LKEEGAN_BLOCKCG_INVERTERS_H