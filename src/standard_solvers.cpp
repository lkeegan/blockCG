#include "standard_solvers.hpp"

// CG inversion of A x = b
int CG (fermion_field& x, const fermion_field& b, const dirac_op& D, 
		double eps, int max_iterations) {	
	// initial guess x = 0
	x.setZero();
	fermion_field t (x);
	fermion_field p (b);
	fermion_field r (b);
	double r2 = r.dot(r);
	double alpha = r2;
	int iter = 0;
	eps *= sqrt(r2);
	// do while |Ax - b| > |b| eps
	while (sqrt(r2) > eps && iter < max_iterations)
	{
		// t = A p
		D.op (t, p);
		++iter;
		// beta = r.r / p.t
		double beta = r2 / p.dot(t);
		// r -= t beta
		r.add(t, -beta);
		// alpha = r.r / r_old.r_old
		double r2_old = r2;
		r2 = r.dot(r);
		alpha = r2 / r2_old;
		// x += p beta
		x.add(p, beta);
		// p = p alpha + r
		p.rescale_add(alpha, r, 1.0);
	}
	return iter;
}

// SCG inversion of (A + sigma_i) x_i = b
int SCG (std::vector<fermion_field>& x, const fermion_field& b, const dirac_op& D,
		 std::vector<double>& sigma, double eps, double eps_shifts, int max_iterations) {
	int n_shifts = x.size();
	int n_unconverged_shifts = n_shifts;
	double beta = 1;
	double alpha = 0;
	std::vector<double> zeta(n_shifts, 1.0);
	// zeta ratio == zeta_k/zeta_{k-1}
	std::vector<double> zeta_ratio(n_shifts, 1.0);
	// initial guess x=0
	for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
		x[i_shift].setZero();
	}
	std::vector<fermion_field> p(n_shifts, b);
	fermion_field t(x[0]), r(b);
	double r2 = r.dot(r);
	int iter = 0;
	eps *= sqrt(r2);
	while (sqrt(r2) > eps && iter < max_iterations)
	{
		// t = (A + sigma_0) p_0
		D.op (t, p[0]);
		t.add(p[0], sigma[0]);
		++iter;
		// beta = r.r/p_0.t
		double beta_m1 = beta;
		beta = r2 / p[0].dot(t);
		// r -= t beta
		r.add(t, -beta);
		double r2_old = r2;
		r2 = r.dot(r);
		double alpha_m1 = alpha;
		alpha = r2 / r2_old;
		// x_0 += p_0 beta
		x[0].add(p[0], beta);
		// p_0 = p_0 alpha + r
		p[0].rescale_add(alpha, r, 1.0);
		for(int i_shift=1; i_shift<n_unconverged_shifts; ++i_shift) {
			// calculate alpha, beta and zeta coefficients for shifted vectors
			double inv_zeta_ratio = alpha_m1*(beta/beta_m1)*(1.0 - zeta_ratio[i_shift]);
			inv_zeta_ratio += 1.0 + (sigma[i_shift]-sigma[0])*beta;
			zeta_ratio[i_shift] = 1.0 / inv_zeta_ratio;
			zeta[i_shift] *= zeta_ratio[i_shift];
			double beta_shift = beta * zeta_ratio[i_shift];
			double alpha_shift = alpha * zeta_ratio[i_shift] * zeta_ratio[i_shift];
			// x_i += p_0 beta_i
			x[i_shift].add(p[i_shift], beta_shift);		
			// p_i = p_i alpha_i + r zeta_i
			p[i_shift].rescale_add(alpha_shift, r, zeta[i_shift]);
		}
		// if largest shift has converged i.e. normalised residual < eps_shifts, stop updating it
		if(sqrt(r2)*zeta[n_unconverged_shifts-1] < eps_shifts) {
			--n_unconverged_shifts;
		}		
	}
	return iter;		
}