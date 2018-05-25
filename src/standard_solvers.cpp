#include "standard_solvers.hpp"

// CG inversion of A x = b
int CG (fermion_field& x, const fermion_field& b, double eps) {	
	// initial guess x = 0
	x.setZero();
	fermion_field t (x);
	fermion_field p (x);
	fermion_field r (b);
	double r2 = r.dot(r);
	double r2_old;
	double alpha = r2;
	double beta;
	int iter = 0;
	eps *= sqrt(r2);
	// do while |Ax - b| > |b| eps
	while (sqrt(r2) > eps)
	{
		// p = p alpha + r
		p.rescale_add(alpha, r, 1.0);
		// t = A p
		dirac_op (t, p);
		++iter;
		// beta = r.r / p.t
		beta = r2 / p.dot(t);
		// r -= t beta
		r.add(t, -beta);
		// x += p beta
		x.add(p, beta);
		r2_old = r2;
		r2 = r.dot(r);
		alpha = r2 / r2_old;
	}
	return iter;
}

// SCG inversion of (A + sigma_i) x_i = b
int SCG (std::vector<fermion_field>& x, const fermion_field& b, std::vector<double>& sigma, double eps, double eps_shifts) {
	int n_shifts = x.size();
	int n_unconverged_shifts = n_shifts;
	std::vector<std::complex<double>> beta(n_shifts, 0.0), beta_m1(n_shifts, 1.0);
	std::vector<std::complex<double>> zeta_m1(n_shifts, 1.0), zeta(n_shifts, 1.0), zeta_p1(n_shifts, 1.0);
	std::vector<std::complex<double>> alpha(n_shifts, 0.0);
	// initial guess x=0
	for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
		x[i_shift].setZero();
	}
	std::vector<fermion_field> p(n_shifts, b);
	fermion_field ap(x[0]), r(b);

	double r2 = r.dot(r);
	double r2_old;
	int iter = 0;
	eps *= sqrt(r2);
	while (sqrt(r2) > eps)
	{
		// ap = (A + sigma_0) p_0
		dirac_op (ap, p[0]);
		ap.add(p[0], sigma[0]);
		++iter;
		// beta = -<r|r>/<p_0|(A+sigma_0)p_0>
		beta[0] = -r2 / p[0].dot(ap);
		// r += beta_0 a
		r.add(ap, beta[0]);
		// r2_new = <r|r>
		r2_old = r2;
		r2 = r.dot(r);
		// calculate alpha, zeta and beta coefficients for shifted vectors
		// see arXiv:hep-lat/9612014 for derivation
		// TODO rewrite in similar as possible form to SBCGrQ
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			zeta_p1[i_shift] = zeta[i_shift] * zeta_m1[i_shift] * beta_m1[0];
			zeta_p1[i_shift] /= (beta[0]*alpha[0]*(zeta_m1[i_shift]-zeta[i_shift]) + zeta_m1[i_shift]*beta_m1[0]*(1.0 - (sigma[i_shift]-sigma[0])*beta[0]));
			beta[i_shift] = beta[0] * zeta_p1[i_shift] / zeta[i_shift];
			zeta_m1[i_shift] = zeta[i_shift];
			zeta[i_shift] = zeta_p1[i_shift];
			beta_m1[i_shift] = beta[i_shift];
		}
		alpha[0] = r2 / r2_old;
		beta_m1[0] = beta[0];
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			alpha[i_shift] = alpha[0] * (zeta[i_shift] * beta_m1[i_shift]) / (zeta_m1[i_shift] * beta_m1[0]);
		}
		// x_i -= beta_i * p_i
		// p_i = alpha_i p_i + zeta_i r
		for(int i_shift=0; i_shift<n_unconverged_shifts; ++i_shift) {
			x[i_shift].add(p[i_shift], -beta[i_shift]);		
			p[i_shift].rescale_add(alpha[i_shift], r, zeta[i_shift]);
		}
		// if largest shift has converged i.e. normalised residual < eps_shifts, stop updating it
		if(sqrt(norm(zeta[n_unconverged_shifts-1])*r2) < eps_shifts) {
			--n_unconverged_shifts;
		}		
	}
	return iter;		
}