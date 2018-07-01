#ifndef LKEEGAN_BLOCKCG_STANDARD_SOLVERS_H
#define LKEEGAN_BLOCKCG_STANDARD_SOLVERS_H
#include <algorithm>  // only used to check input SCG shifts
#include "dirac_op.hpp"
#include "fields.hpp"

// CG inversion of D x = b
// stops iterating when |Dx - b| / | b | < eps
// returns number of times Dirac operator D was called
int CG(fermion_field& x, const fermion_field& b, const dirac_op& D,
       double eps = 1.e-15, int max_iterations = 1e6);

// SCG inversion of (D + sigma_i) x^sigma_i = b
// stops iterating when |(D + sigma_0) x^sigma_0 - b| / | b | < eps
// and updates each shifted solution while |(D + sigma_i)x^sigma_i - b| / | b |
// > eps_shifts returns number of times Dirac operator D was called NOTE: shifts
// sigma must be non-negative and in ascending order
int SCG(std::vector<fermion_field>& x, const fermion_field& b,
        const dirac_op& D, std::vector<double>& sigma, double eps = 1.e-15,
        double eps_shifts = 1.e-15, int max_iterations = 1e6);

#endif  // LKEEGAN_BLOCKCG_STANDARD_SOLVERS_H