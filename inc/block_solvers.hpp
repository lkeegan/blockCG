#ifndef LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H
#define LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H

#include "fields.hpp"
#include "dirac_op.hpp"

// BCG inversion of D X = B
// stops iterating when | D X - B |_i / | B |_i < eps for all vectors i
// returns number of times Dirac operator D was called
template<int N_rhs>
int BCG (block_fermion_field<N_rhs>& X, const block_fermion_field<N_rhs>& B, const dirac_op& D,
		 double eps = 1.e-15, int max_iterations = 1e6) {
	// initial guess x = 0
	X.setZero();
	block_fermion_field<N_rhs> T (X);
	block_fermion_field<N_rhs> P (B);
	block_fermion_field<N_rhs> R (B);
	block_matrix<N_rhs> r2 = R.hermitian_dot(R);
	block_matrix<N_rhs> alpha = r2;
	block_matrix<N_rhs> beta;
	block_matrix<N_rhs> r2_old;
	Eigen::Array<double,N_rhs,1> residual_norms = r2.diagonal().real().array().sqrt();
	double residual = 1.0;

	int iter = 0;
	// do while |Ax - b|_i/|b|_i } > eps for all i
	while (residual > eps && iter < max_iterations)
	{
		// T = A P
		D.op (T, P);
		++iter;
		// beta = (P.T)^-1 (R.R)
		// use fullPivLu decomposition to invert the hermitian matrix (P.T)^-1
		beta = (P.hermitian_dot(T)).fullPivLu().solve(r2);
		// R -= T beta
		R.add(T, -beta);
		r2_old = r2;
		r2 = R.hermitian_dot(R);
		alpha = r2_old.fullPivLu().solve(r2);
		// X += P beta
		X.add(P, beta);
		// P = P alpha + R
		P.rescale_add(alpha, R, 1.0);
		residual = (r2.diagonal().real().array().sqrt()/residual_norms).maxCoeff();
	}
	return iter;
}

// BCGrQ inversion of D X = B
// stops iterating when | D X_i - B_i | / | B_i | < eps for all vectors i
// returns number of times Dirac operator D was called
template<int N_rhs>
int BCGrQ (block_fermion_field<N_rhs>& X, const block_fermion_field<N_rhs>& B, const dirac_op& D,
		   double eps = 1.e-15, int max_iterations = 1e6) {
	// initial guess x = 0
	X.setZero();
	block_fermion_field<N_rhs> T (X), R (B);
	block_matrix<N_rhs> alpha, beta, delta;
	// {R, delta} <- QR(R)
	R.thinQR(delta);
	block_fermion_field<N_rhs> P (R);
	// |residual_i| = sqrt(\sum_j R_i^dag R_j) = sqrt(\sum_j delta_ij)
	Eigen::Array<double,N_rhs,1> residual_norms = delta.rowwise().norm().array();
	int iter = 0;
	// do while | AX - B |_j / | B |_j > eps for all j
	while ((delta.rowwise().norm().array()/residual_norms).maxCoeff() > eps && iter < max_iterations)
	{
		// T = A P
		D.op (T, P);
		++iter;
		// beta = (P.T)^-1 (R.R)
		// use fullPivLu decomposition to invert hermitian matrix (P.T)^-1
		beta = (P.hermitian_dot(T)).fullPivLu().solve(block_matrix<N_rhs>::Identity());
		// R -= T beta
		R.add(T, -beta);
		//{R, alpha} <- QR(R)
		R.thinQR(alpha);
		// X += P beta delta
		X.add(P, beta * delta);
		// P = P alpha^{\dagger} + R [where alpha^{\dagger} is upper triangular]
		P.rescale_add(alpha.adjoint().eval(), R, 1.0);
		delta = alpha * delta;
	}
	return iter;
}

// SBCGrQ inversion of (D + sigma_j) X^{sigma_j} = B
// stops iterating when | (D + sigma_0) X^{sigma_0}_i - B_i| / | B_i | < eps for all vectors i
// returns number of times Dirac operator D was called
template<int N_rhs>
int SBCGrQ(std::vector< block_fermion_field<N_rhs> >& X, const block_fermion_field<N_rhs>& B,
		   const dirac_op& D, std::vector<double>& sigma, double eps = 1.e-15, 
		   double eps_shifts = 1.e-15, int max_iterations = 1e6) {
	// some sanity checks on supplied parameters:
	assert(sigma.size()==X.size() && "number of shifts does not match number of solution vectors");
	assert(sigma[0] >= 0.0 && "shifts must be zero or positive");
	assert(std::is_sorted(sigma.begin(), sigma.end()) && "shifts must be in ascending order");

	int n_shifts = static_cast<int>(sigma.size());
	int n_unconverged_shifts = n_shifts;

	block_matrix<N_rhs> Identity = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> alpha, beta, delta;
	block_matrix<N_rhs> alpha_old, beta_inv_old, beta_inv (Identity);
	block_fermion_field<N_rhs> T (B), R (B);
	// start from X=0 initial guess
	for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
		X[i_shift].setZero();
	}
	// {R, delta} <- QR(R)
	R.thinQR(delta);
	alpha = delta;
	std::vector< block_fermion_field<N_rhs> > P(n_shifts, R);

	// matrices for shifted residuals
	block_matrix<N_rhs> theta_s_inv;
	using bm_alloc = Eigen::aligned_allocator<block_matrix<N_rhs>>;
	std::vector< block_matrix<N_rhs>, bm_alloc > theta_s (n_shifts, Identity),
												 beta_s (n_shifts, Identity); 

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j delta_ij)
	Eigen::ArrayXd b_norm = delta.rowwise().norm().array();
	double residual = 1.0;
	while(residual > eps && iter < max_iterations) {
		// Apply dirac op (with lowest shift) to P[0]:
		D.op (T, P[0]);
		// do lowest shift as part of dirac op
		T.add (P[0], sigma[0]);
		++iter;

		beta_inv_old = beta_inv;
		beta_inv = P[0].hermitian_dot(T);
		// Find inverse of beta_inv via fullPivLu decomposition
		beta = beta_inv.fullPivLu().solve(Identity);

		// X[0] = X[0] + P[0] beta delta
		X[0].add(P[0], beta*delta);

		//R -= T beta
		R.add(T, -beta);

		alpha_old = alpha;
		// in-place thinQR decomposition of residuals matrix Q
		R.thinQR(alpha);
		delta = alpha * delta;
		// use maximum over relative residual of lowest shift as residual
		residual = (delta.rowwise().norm().array()/b_norm).maxCoeff();

		// P <- P S^dag + R
		P[0].rescale_add(alpha.adjoint(), R, 1.0);

		// calculate shifted X and P
		for(int i_shift=n_unconverged_shifts-1; i_shift>0; --i_shift) {
			// calculate shifted coefficients
			theta_s_inv = beta_inv + (sigma[i_shift]-sigma[0])*Identity + alpha_old*(Identity - beta_inv_old * theta_s[i_shift]) * beta_inv_old * alpha_old.adjoint();
			theta_s[i_shift] = theta_s_inv.fullPivLu().solve(Identity);
			beta_s[i_shift] = theta_s[i_shift] * alpha_old * beta_inv_old * beta_s[i_shift];
			double residual_shift = ((alpha * beta_inv * beta_s[i_shift]).rowwise().norm().array()/b_norm).maxCoeff();
			// update shifted X and P
			// X_s = X_s + P_s tmp_betaC
			X[i_shift].add(P[i_shift], beta_s[i_shift]);
			// P_s <- P_s tmp_Sdag + R
			P[i_shift].rescale_add(theta_s[i_shift] * beta_inv * alpha.adjoint(), R, 1.0);
			// if shift has converged stop updating it
			if(residual_shift < eps_shifts) {
				--n_unconverged_shifts;
			}			
		}
	}
	return iter;
}

#endif //LKEEGAN_BLOCKCG_INVERTERS_H