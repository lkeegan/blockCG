#ifndef LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H
#define LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H

#include "fields.hpp"
#include "dirac_op.hpp"
#include <iostream>

// BCG inversion of D X = B
// stops iterating when |DX^i - B^i| / | B^i | < eps for all vectors i
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
		// use LDL^T Cholesky decomposition to invert the hermitian matrix (P.T)^-1
		beta = (P.hermitian_dot(T)).ldlt().solve(r2);
		// R -= T beta
		R.add(T, -beta);
		// X += P beta
		X.add(P, beta);
		r2_old = r2;
		r2 = R.hermitian_dot(R);
		alpha = r2_old.ldlt().solve(r2);
		residual = (r2.diagonal().real().array().sqrt()/residual_norms).maxCoeff();
	}
	return iter;
}

// BCGrQ inversion of D X = B
// stops iterating when |DX^i - B^i| / | B^i | < eps for all vectors i
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
	// do while Max_j{ |Ax_j - b_j|/|b_j| } > eps
	while ((delta.rowwise().norm().array()/residual_norms).maxCoeff() > eps && iter < max_iterations)
	{
		// P = P alpha^{\dagger} + R [where alpha^{\dagger} is upper triangular]
		P.rescale_add(alpha.adjoint().eval(), R, block_matrix<N_rhs>::Identity());
		// T = A P
		D.op (T, P);
		++iter;
		// beta = (P.T)^-1 (R.R)
		// use LDL^T Cholesky decomposition to invert hermitian matrix (P.T)^-1
		beta = (P.hermitian_dot(T)).ldlt().solve(block_matrix<N_rhs>::Identity());
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
// stops iterating when |(D + sigma_0) X^{sigma_j}^i - B^i| / | B^i | < eps for all vectors i
// returns number of times Dirac operator D was called
template<int N_rhs>
int SBCGrQ(std::vector< block_fermion_field<N_rhs> >& X, const block_fermion_field<N_rhs>& B,
		   const dirac_op& D, std::vector<double>& input_shifts, double eps = 1.e-15, 
		   double eps_shifts = 1.e-15, int max_iterations = 1e6) {
	// count shifts (not including first one that is included in the dirac op)
	int N_shifts = static_cast<int>(input_shifts.size()) - 1;
	int N_unconverged_shifts = N_shifts;

	// subtract first shift from remaining set of shifts
	std::vector<double> shifts(N_shifts);
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		shifts[i_shift] = input_shifts[i_shift+1] - input_shifts[0];
	}

	// Unshifted matrices:
	block_matrix<N_rhs> S = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> C = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> beta = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> betaC = block_matrix<N_rhs>::Zero();
	// AP, Q are [NxVOL]
	block_fermion_field<N_rhs> AP(B), Q(B);
	// start from X=0 initial guess, so residual Q = B [NxVOL]
	// X = 0 for unshifted + all shifts
	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
		X[i_shift].setZero();
	}
	//Q = B;
	// in place thinQR decomposition of residual Q[N][VOL] into orthonormal Q[N][VOL] and triangular C[NxN]
	Q.thinQR(C);
	S = C;
	AP = Q;
	// P has one NxVOL block per shift (+unshifted)
	// P = Q for all shifts
	std::vector< block_fermion_field<N_rhs> > P(N_shifts+1, Q);
//	for(int i_shift=0; i_shift<N_shifts+1; ++i_shift) {
//		P.push_back(Q);
//	}

	// Shifted matrices:
	// previous / inverted versions of beta
	block_matrix<N_rhs> beta_inv = block_matrix<N_rhs>::Identity();
	block_matrix<N_rhs> beta_inv_m1 = block_matrix<N_rhs>::Identity();
	// Construct decomposition of S: can then do S^-1 M using S_inv.solve(M)
	Eigen::FullPivLU< block_matrix<N_rhs> > S_inv (S);
	block_matrix<N_rhs> S_m1 = block_matrix<N_rhs>::Zero();
	// These are temporary matrices used for each shift
	block_matrix<N_rhs> tmp_betaC, tmp_Sdag; 
	// ksi_s are the scale factors related shifted and unshifted residuals
	// 3-term recurrence so need _k, _k-1, _k-2 for each shift
	// initially ksi_s_m2 = I, ksi_s_m1 = S
	std::vector< block_matrix<N_rhs>, Eigen::aligned_allocator<block_matrix<N_rhs>> > ksi_s, ksi_s_m1; 
	std::vector< Eigen::FullPivLU<block_matrix<N_rhs>>, Eigen::aligned_allocator<Eigen::FullPivLU<block_matrix<N_rhs>>> > ksi_s_inv_m1, ksi_s_inv_m2;
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
 		ksi_s.push_back(block_matrix<N_rhs>::Identity());
 		ksi_s_m1.push_back(S);
 		ksi_s_inv_m1.push_back(S_inv);
 		ksi_s_inv_m2.push_back(block_matrix<N_rhs>::Identity().fullPivLu());
	}

	// main loop
	int iter = 0;
	// get norm of each vector b in matrix B
	// NB: v.norm() == sqrt(\sum_i v_i v_i^*) = l2 norm of vector v
	// residual_i = sqrt(\sum_j Q_i^dag Q_j) = sqrt(\sum_j C_ij)
	Eigen::ArrayXd b_norm = C.rowwise().norm().array();
	double residual = 1.0;

	while(residual > eps && iter < max_iterations) {
		// Apply dirac op (with lowest shift) to P[0]:
		D.op (AP, P[0]);
		// do lowest shift as part of dirac op
		AP.add(P[0], input_shifts[0]);
		++iter;

		beta_inv_m1 = beta_inv;
		beta_inv = P[0].hermitian_dot(AP);
		// Find inverse of beta_inv via LDLT cholesky decomposition
		// and solving beta beta_inv = I
		beta = beta_inv.fullPivLu().solve(block_matrix<N_rhs>::Identity());
		betaC = beta * C;

		// X[0] = X[0] + P[0] beta C
		X[0].add(P[0], betaC);

		//Q -= AP beta
		//betaC = -beta;
		Q.add(AP, -beta);

		S_m1 = S;
		// in-place thinQR decomposition of residuals matrix Q
		Q.thinQR(S);
		C = S * C;
		if(N_unconverged_shifts>0) {
			// update decomposition of S for S_inv operations
			S_inv.compute(S);
		}

		// P <- P S^dag + Q
		P[0].rescale_add(S.adjoint().eval(), Q, 1.0);

		// calculate shifted X and P
		// note X[0] and P[0] are the unshifted ones, so first shift has index 1 in X and P
		for(int i_shift=0; i_shift<N_unconverged_shifts; ++i_shift) {
			// calculate shifted coefficients
			// ksi_s:
			tmp_betaC = S_m1 * beta_inv_m1 - ksi_s_m1[i_shift] * ksi_s_inv_m2[i_shift].solve(beta_inv_m1);
			tmp_Sdag = block_matrix<N_rhs>::Identity() + shifts[i_shift] * beta + tmp_betaC * S_m1.adjoint() * beta;
			ksi_s[i_shift] = S * tmp_Sdag.fullPivLu().solve(ksi_s_m1[i_shift]);
			// tmp_betaC == "alpha^sigma" in paper:
			tmp_betaC = beta * S_inv.solve(ksi_s[i_shift]);
			// tmp_Sdag == "beta^sigma" in paper:
			//tmp_Sdag = tmp_betaC * ksi_s_m1[i_shift].fullPivLu().solve(beta_inv); //* S.adjoint();
			//std::cout << "X " << ksi_s_m1[i_shift] << " " << beta_inv << " " << S.adjoint().eval() << std::endl;
			tmp_Sdag = tmp_betaC * ksi_s_inv_m1[i_shift].solve(beta_inv * S.adjoint());
			// update shifted X and P
			// X_s = X_s + P_s tmp_betaC
			X[i_shift+1].add(P[i_shift+1], tmp_betaC);
			// P_s <- P_s tmp_Sdag + Q
			P[i_shift+1].rescale_add(tmp_Sdag, Q, 1.0);
/*
std::cout << "C: " << (C - ksi_s[i_shift]).norm() << std::endl;
std::cout << "betaC: " << (tmp_betaC - betaC).norm() << std::endl;
std::cout << "Sdag: " << (tmp_Sdag - S.adjoint().eval()).norm() << std::endl;
std::cout << beta * S_inv.solve(C)* ksi_s_m1[i_shift].fullPivLu().solve(beta_inv) << std::endl;
*/
			// update inverse ksi's for next iteration
			ksi_s_inv_m2[i_shift] = ksi_s_inv_m1[i_shift];
			ksi_s_m1[i_shift] = ksi_s[i_shift];
			ksi_s_inv_m1[i_shift].compute(ksi_s[i_shift]);
			// check if largest unconverged shift has converged
			if((ksi_s[N_unconverged_shifts-1].rowwise().norm().array()/b_norm).maxCoeff() < eps_shifts) {
				--N_unconverged_shifts;
			}			
		}
		// use maximum over vectors of residuals/b_norm for unshifted solution as stopping crit
		// assumes that all shifts are positive i.e. better conditioned and converge faster
		// worst vector should be equal to CG with same eps on that vector, others will be better
		residual = (C.rowwise().norm().array()/b_norm).maxCoeff();

/*
		std::cout << "#SBCGrQ " << iter << "\t" << residual;
		for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
			std::cout << "\t" << (ksi_s[i_shift].rowwise().norm().array()/b_norm).maxCoeff();
		}
		std::cout << std::endl;

		// check shifted residual for X[1] explicitly
		// Apply dirac op to P[0]:
		D.op(AP, X[0]);
		// do input shift
		AP.add(X[0], input_shifts[0]);
		AP -= B;
		std::cout << "RES0 " << AP.hermitian_dot(AP) << std::endl;

		// check shifted residual for X[1] explicitly
		// Apply dirac op to P[0]:
		D.op(AP, X[1]);
		// do input shift
		AP.add(X[1], input_shifts[1]);
		AP -= B;
		std::cout << "RES1 " << AP.hermitian_dot(AP) << std::endl;
*/
	}
	return iter;
}

#endif //LKEEGAN_BLOCKCG_INVERTERS_H