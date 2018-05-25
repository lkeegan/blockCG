#ifndef LKEEGAN_BLOCKCG_STANDARD_SOLVERS_H
#define LKEEGAN_BLOCKCG_STANDARD_SOLVERS_H

#include "fields.hpp"
#include "dirac_op.hpp"

// CG inversion of D x = b
// stops iterating when |Dx - b| / | b | < eps
// returns number of times Dirac operator D was called
int CG (fermion_field& x, const fermion_field& b, double eps = 1.e-15);

// Multishift-CG inversion of (D + sigma_i) x_i = b 
// stops iterating when |(D + sigma_0)x - b| / | b | < eps
// and stops updating shifted solution i when |(D + sigma_i)x - b| / | b | < eps_shifts
// returns number of times Dirac operator D was called
int SCG (std::vector<fermion_field>& x, const fermion_field& b, std::vector<double>& sigma, double eps = 1.e-15, double eps_shifts = 1.e-15);

#endif //LKEEGAN_BLOCKCG_STANDARD_SOLVERS_H