#ifndef LKEEGAN_BLOCKCG_INVERTERS_H
#define LKEEGAN_BLOCKCG_INVERTERS_H

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

// BCG inversion of D X = b
// stops iterating when |DX - B| / | B | < eps
// returns number of times Dirac operator D was called
template<int N_rhs>
int BCG (block_fermion_field<N_rhs>& X, const block_fermion_field<N_rhs>& B, double eps = 1.e-15) {
	block_matrix<N_rhs> alpha, beta, pap;
	// initial guess x = 0
	X.setZero();
	block_fermion_field<N_rhs> AP (X);
	block_fermion_field<N_rhs> R (B);
	block_fermion_field<N_rhs> P (B);
	block_matrix<N_rhs> r2 = R.hermitian_dot(R);
	block_matrix<N_rhs> r2_old = r2;
	Eigen::Array<double,N_rhs,1> b_norms = r2.diagonal().real().array().sqrt();

	int iter = 0;
	// do while Max_j{ |Ax_j - b_j|/|b_j| } > eps
	while ((r2.diagonal().real().array().sqrt()/b_norms).maxCoeff() > eps)
	{
		// AP = A P
		dirac_op (AP, P);
		++iter;
		// beta = -r.r / p.ap
		pap = P.hermitian_dot(AP);
		beta = -pap.ldlt().solve(r2);
		// R += AP beta
		R.add(AP, beta);
		r2_old = r2;
		r2 = R.hermitian_dot(R);
		alpha = r2_old.ldlt().solve(r2);
		// X -= P beta
		X.add(P, -beta);
		// P = P alpha + R
		P.rescale_add(alpha, R, 1.0);
	}
	return iter;

}

#endif //LKEEGAN_BLOCKCG_INVERTERS_H