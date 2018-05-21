#ifndef LKEEGAN_BLOCKCG_INVERTERS_H
#define LKEEGAN_BLOCKCG_INVERTERS_H

#include "fields.hpp"
#include "dirac_op.hpp"

// CG inversion of D x = b
// stops iterating when |Dx - b| / | b | < eps
// returns number of times Dirac operator D was called
int CG (field<fermion>& x, const field<fermion>& b, double eps = 1.e-15);

// Multishift-CG inversion of (D + sigma_i) x_i = b 
int SCG (std::vector<field<fermion>>& x, const field<fermion>& b, std::vector<double>& sigma, double eps = 1.e-15, double eps_shifts = 1.e-15);

#endif //LKEEGAN_BLOCKCG_INVERTERS_H