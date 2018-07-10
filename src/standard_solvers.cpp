#include "standard_solvers.hpp"

int CG(fermion_field& x, const fermion_field& b, const dirac_op& D, double eps,
       int max_iterations) {
  // initial guess x = 0
  x.setZero();
  fermion_field t(x);
  fermion_field p(b);
  fermion_field r(b);
  double r2 = r.real_dot(r);
  int iter = 0;
  eps *= sqrt(r2);
  // do while |Ax - b| > |b| eps
  while (sqrt(r2) > eps && iter < max_iterations) {
    // t = A p
    D.op(t, p);
    ++iter;
    // alpha = r.r / p.t
    double alpha = r2 / p.real_dot(t);
    // r -= t alpha
    r.add(t, -alpha);
    // beta = r.r / r_old.r_old
    double r2_old = r2;
    r2 = r.real_dot(r);
    double beta = r2 / r2_old;
    // x += p alpha
    x.add(p, alpha);
    // p = p beta + r
    p.rescale_add(beta, r, 1.0);
  }
  return iter;
}

int SCG(std::vector<fermion_field>& x, const fermion_field& b,
        const dirac_op& D, std::vector<double>& sigma, double eps,
        double eps_shifts, int max_iterations) {
  // some sanity checks on supplied parameters:
  assert(sigma.size() == x.size() &&
         "number of shifts does not match number of solution vectors");
  assert(sigma[0] >= 0.0 && "shifts must be zero or positive");
  assert(std::is_sorted(sigma.begin(), sigma.end()) &&
         "shifts must be in ascending order");

  int n_shifts = x.size();
  int n_unconverged_shifts = n_shifts;
  double alpha = 1.0;
  double beta = 0.0;
  std::vector<double> zeta(n_shifts, 1.0);
  std::vector<double> theta(n_shifts, 1.0);  // theta_k == zeta_k/zeta_{k-1}
  for (int i_shift = 0; i_shift < n_shifts; ++i_shift) {
    x[i_shift].setZero();
  }
  std::vector<fermion_field> p(n_shifts, b);
  fermion_field t(x[0]), r(b);
  double r2 = r.real_dot(r);
  int iter = 0;
  eps *= sqrt(r2);
  while (sqrt(r2) > eps && iter < max_iterations) {
    // t = (A + sigma_0) p_0
    D.op(t, p[0]);
    t.add(p[0], sigma[0]);
    ++iter;
    // alpha = r.r/p_0.t
    double alpha_old = alpha;
    alpha = r2 / p[0].real_dot(t);
    // r -= t alpha
    r.add(t, -alpha);
    double r2_old = r2;
    r2 = r.real_dot(r);
    double beta_old = beta;
    beta = r2 / r2_old;
    // x_0 += p_0 alpha
    x[0].add(p[0], alpha);
    // p_0 = p_0 beta + r
    p[0].rescale_add(beta, r, 1.0);
    for (int i_shift = n_unconverged_shifts - 1; i_shift > 0; --i_shift) {
      // calculate alpha, beta and zeta coefficients for shifted vectors
      double inv_theta = 1.0 + (sigma[i_shift] - sigma[0]) * alpha;
      inv_theta += beta_old * (alpha / alpha_old) * (1.0 - theta[i_shift]);
      theta[i_shift] = 1.0 / inv_theta;
      zeta[i_shift] *= theta[i_shift];
      double alpha_shift = alpha * theta[i_shift];
      double beta_shift = beta * theta[i_shift] * theta[i_shift];
      // x^i += p^0 alpha^i
      x[i_shift].add(p[i_shift], alpha_shift);
      // p^i = p^i beta^i + r zeta^i
      p[i_shift].rescale_add(beta_shift, r, zeta[i_shift]);
    }
    // if normalised residual of largest shift < eps_shifts, stop updating it
    if (sqrt(r2) * zeta[n_unconverged_shifts - 1] < eps_shifts) {
      --n_unconverged_shifts;
    }
  }
  return iter;
}