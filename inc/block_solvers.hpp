#ifndef LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H
#define LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H

#include "fields.hpp"
#include "dirac_op.hpp"

// BCG inversion of D X = B
// stops iterating when | D X_i - B_i | / | B_i | < eps for all vectors i
// returns number of times Dirac operator D was called
template<int N_rhs>
int BCG (block_fermion_field<N_rhs>& X, const block_fermion_field<N_rhs>& B, const dirac_op& D,
		 double eps = 1.e-15, int max_iterations = 1e6) {
	// initial guess x = 0
	X.setZero();
	block_fermion_field<N_rhs> T (X);
	block_fermion_field<N_rhs> P (X);
	block_fermion_field<N_rhs> R (B);
	block_matrix<N_rhs> r2 = R.hermitian_dot(R);
	block_matrix<N_rhs> alpha = r2;
	block_matrix<N_rhs> beta;
	block_matrix<N_rhs> r2_old;
	Eigen::Array<double,N_rhs,1> residual_norms = r2.diagonal().real().array().sqrt();
	double residual = 1.0;

	int iter = 0;
	// do while Max_j{ |Ax_j - b_j|/|b_j| } > eps
	while (residual > eps && iter < max_iterations)
	{
		// P = P alpha + R
		P.rescale_add(alpha, R, 1.0);
		// T = A P
		D.op (T, P);
		++iter;
		// beta = (P.T)^-1 (R.R)
		// use fullPivLu decomposition to invert the hermitian matrix (P.T)^-1
		beta = (P.hermitian_dot(T)).fullPivLu().solve(r2);
		// R -= T beta
		R.add(T, -beta);
		// X += P beta
		X.add(P, beta);
		r2_old = r2;
		r2 = R.hermitian_dot(R);
		alpha = r2_old.fullPivLu().solve(r2);
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
	block_fermion_field<N_rhs> T (X);
	block_fermion_field<N_rhs> P (X);
	block_fermion_field<N_rhs> R (B);
	block_matrix<N_rhs> delta;
	// {R, delta} <- QR(R)
	R.thinQR(delta);
	block_matrix<N_rhs> alpha = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> beta;
	// |residual_i| = sqrt(\sum_j R_i^dag R_j) = sqrt(\sum_j delta_ij)
	Eigen::Array<double,N_rhs,1> residual_norms = delta.rowwise().norm().array();
	int iter = 0;
	// do while Max_j{ | A x_j - b_j | / | b_j | } > eps
	while ((delta.rowwise().norm().array()/residual_norms).maxCoeff() > eps && iter < max_iterations)
	{
		// P = P alpha^{\dagger} + R [where alpha^{\dagger} is upper triangular]
		P.rescale_add(alpha.adjoint().eval(), R, block_matrix<N_rhs>::Identity());
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
		beta = beta * delta;
		X.add(P, beta);
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

	// Unshifted matrices:
	block_matrix<N_rhs> Identity = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> S = Identity;
	block_matrix<N_rhs> delta = Identity;
	block_matrix<N_rhs> alpha = Identity;
	// AP, Q are [NxVOL]
	block_fermion_field<N_rhs> AP(B), Q(B);
	// start from X=0 initial guess, so residual Q = B [NxVOL]
	// X = 0 for unshifted + all shifts
	for(int i_shift=0; i_shift<n_shifts; ++i_shift) {
		X[i_shift].setZero();
	}
	//Q = B;
	// in place thinQR decomposition of residual Q[N][VOL] into orthonormal Q[N][VOL] and triangular C[NxN]
	Q.thinQR(delta);
	S = delta;
	AP = Q;
	// P has one NxVOL block per shift (+unshifted)
	// P = Q for all shifts
	std::vector< block_fermion_field<N_rhs> > P(n_shifts, Q);

	// Shifted matrices:
	// previous / inverted versions of beta
	block_matrix<N_rhs> alpha_inv = Identity;
	block_matrix<N_rhs> alpha_inv_m1 = Identity;
	block_matrix<N_rhs> S_m1 = Identity;
	// These are temporary matrices used for each shift
	block_matrix<N_rhs> tmp_Sdag;
	using bm_alloc = Eigen::aligned_allocator<block_matrix<N_rhs>>;
	std::vector< block_matrix<N_rhs>, bm_alloc > zeta_tilde_s(n_shifts, Identity),
												 alpha_s(n_shifts, Identity); 

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j delta_ij)
	Eigen::ArrayXd b_norm = delta.rowwise().norm().array();
	double residual = 1.0;
	while(residual > eps && iter < max_iterations) {
		// Apply dirac op (with lowest shift) to P[0]:
		D.op (AP, P[0]);
		// do lowest shift as part of dirac op
		AP.add(P[0], sigma[0]);
		++iter;

		alpha_inv_m1 = alpha_inv;
		alpha_inv = P[0].hermitian_dot(AP);
		// Find inverse of alpha_inv via fullPivLu decomposition
		// and solving beta beta_inv = I
		alpha = alpha_inv.fullPivLu().solve(Identity);

		// X[0] = X[0] + P[0] alpha delta
		X[0].add(P[0], alpha*delta);

		//Q -= AP alpha
		Q.add(AP, -alpha);

		S_m1 = S;
		// in-place thinQR decomposition of residuals matrix Q
		Q.thinQR(S);
		delta = S * delta;
		// use maximum over relative residual of lowest shift as residual
		residual = (delta.rowwise().norm().array()/b_norm).maxCoeff();

		// P <- P S^dag + Q
		P[0].rescale_add(S.adjoint().eval(), Q, 1.0);

		// calculate shifted X and P
		for(int i_shift=n_unconverged_shifts-1; i_shift>0; --i_shift) {
			// calculate shifted coefficients
			tmp_Sdag = alpha_inv + (sigma[i_shift]-sigma[0])*Identity + S_m1*(Identity - alpha_inv_m1 * zeta_tilde_s[i_shift]) * alpha_inv_m1 * S_m1.adjoint();
			alpha_s[i_shift] = tmp_Sdag.fullPivLu().solve(S_m1*alpha_inv_m1)*alpha_s[i_shift];
			zeta_tilde_s[i_shift] = tmp_Sdag.fullPivLu().solve(Identity);
			double residual_shift = ((tmp_Sdag * alpha_s[i_shift]).rowwise().norm().array()/b_norm).maxCoeff();
			tmp_Sdag = zeta_tilde_s[i_shift]*alpha_inv*S.adjoint();
			// update shifted X and P
			// X_s = X_s + P_s tmp_betaC
			X[i_shift].add(P[i_shift], alpha_s[i_shift]);
			// P_s <- P_s tmp_Sdag + Q
			P[i_shift].rescale_add(tmp_Sdag, Q, 1.0);
			// if shift has converged stop updating it
			if(residual_shift < eps_shifts) {
				--n_unconverged_shifts;
			}			
		}
	}
	return iter;
}

#endif //LKEEGAN_BLOCKCG_INVERTERS_H