#ifndef LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H
#define LKEEGAN_BLOCKCG_BLOCK_SOLVERS_H

#include "fields.hpp"
#include "dirac_op.hpp"

// BCG inversion of D X = b
// stops iterating when |DX - B| / | B | < eps
// returns number of times Dirac operator D was called
template<int N_rhs>
int BCG (block_fermion_field<N_rhs>& X, const block_fermion_field<N_rhs>& B, double eps = 1.e-15) {
	// initial guess x = 0
	X.setZero();
	block_fermion_field<N_rhs> T (X);
	block_fermion_field<N_rhs> P (X);
	block_fermion_field<N_rhs> R (B);
	block_matrix<N_rhs> r2 = R.hermitian_dot(R);
	block_matrix<N_rhs> alpha = r2;
	block_matrix<N_rhs> beta;
	block_matrix<N_rhs> r2_old;
	Eigen::Array<double,N_rhs,1> b_norms = r2.diagonal().real().array().sqrt();

	int iter = 0;
	// do while Max_j{ |Ax_j - b_j|/|b_j| } > eps
	while ((r2.diagonal().real().array().sqrt()/b_norms).maxCoeff() > eps)
	{
		// P = P alpha + R
		P.rescale_add(alpha, R, 1.0);
		// T = A P
		dirac_op (T, P);
		++iter;
		// beta = (P.T)^-1 (R.R)
		// use LDL^T Cholesky decomposition to invert hermitian matrix (P.T)^-1
		beta = (P.hermitian_dot(T)).ldlt().solve(r2);
		// R -= T beta
		R.add(T, -beta);
		// X += P beta
		X.add(P, beta);
		r2_old = r2;
		r2 = R.hermitian_dot(R);
		alpha = r2_old.ldlt().solve(r2);
	}
	return iter;

}

#endif //LKEEGAN_BLOCKCG_INVERTERS_H