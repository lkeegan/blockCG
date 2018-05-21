#include "inverters.hpp"
#include <iostream>

// CG inversion of A x = b
int CG (field<fermion>& x, const field<fermion>& b, double eps) {	
	std::complex<double> beta;
	double alpha;
	// initial guess x=0
	x.setZero();
	field<fermion> ax (x);
	field<fermion> r (b);
	field<fermion> p (b);
	double r2 = r.squaredNorm();
	double r2_old;
	int iter = 0;
	eps *= sqrt(r2);
	// do while | Ax - b| > |b| eps
	while (sqrt(r2) > eps)
	{
		// ax = A p
		dirac_op (ax, p);
		++iter;
		// beta = -<r|r> / <p|ax>
		beta = -r2 / p.dot(ax);
		// r += beta ax
		r.add(ax, beta);
		r2_old = r2;
		r2 = r.squaredNorm();
		alpha = r2 / r2_old;
		// x += -beta * p
		x.add(p, -beta);
		// p = alpha p + r
		p.rescale_add(alpha, r, 1.0);
	}
	return iter;
}

// SCG inversion of (A + sigma_i) x_i = b
int SCG (std::vector<field<fermion>>& x, const field<fermion>& b, std::vector<double>& sigma, double eps, double eps_shifts) {
	int n_shifts = x.size();
	int n_unconverged_shifts = n_shifts;
	std::vector<std::complex<double>> beta(n_shifts, 0.0), beta_m1(n_shifts, 1.0);
	std::vector<std::complex<double>> zeta_m1(n_shifts, 1.0), zeta(n_shifts, 1.0), zeta_p1(n_shifts, 1.0);
	std::vector<std::complex<double>> alpha(n_shifts, 0.0);
	// initial guess x=0
	for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
		x[i_shift].setZero();
	}
	std::vector<field<fermion>> p(n_shifts, b);
	field<fermion> ax(x[0]), r(b);

	double r2 = r.squaredNorm();
	double r2_old;
	int iter = 0;
	eps *= sqrt(r2);
	while (sqrt(r2) > eps)
	{
		// ax = (A + sigma_0) p_0
		dirac_op (ax, p[0]);
		ax.add(p[0], sigma[0]);
		++iter;
		// beta = -<r|r>/<p|a>
		beta[0] = -r2 / p[0].dot(ax);
		// r += beta_0 a
		r.add(ax, beta[0]);
		// r2_new = <r|r>
		r2_old = r2;
		r2 = r.squaredNorm();
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