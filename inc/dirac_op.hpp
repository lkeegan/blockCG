#ifndef LKEEGAN_BLOCKCG_DIRAC_OP_H
#define LKEEGAN_BLOCKCG_DIRAC_OP_H

#include "fields.hpp"

// Dummy "Dirac operator" D that acts on block_fermion fields:
// lhs = D rhs
// with adjustable condition number
// https://www.researchgate.net/publication/230538536_Tridiagonal_Toeplitz_matrices_Properties_and_novel_applications
template <int N_rhs>
void dirac_op(field< block_fermion<N_rhs> >& lhs, const field< block_fermion<N_rhs> >& rhs, double condition_number = 1e2, double phase = 0.86) {
	// tri-diagonal Hermitian Toeplitz matrix (tau*, 0.5, tau)
	// with tau = (0.25 - 0.5/condition_number) exp (i phase)
	// eigenvalues lie in range [0.5 - 2|tau|, 0.5 + 2|tau|] ~ [1/condition_number, 1]
	std::complex<double> tau = std::polar(0.25 - 0.5/condition_number, phase);
	lhs[0] = 0.5 * rhs[0];
	for(int ix=1; ix<lhs.V-1; ++ix) {
		lhs[ix] = 0.5 * rhs[ix] + tau * rhs[ix+1] + std::conj(tau) * rhs[ix-1];
	}
	lhs[lhs.V-1] = 0.5 * rhs[lhs.V-1];
}

#endif //LKEEGAN_BLOCKCG_DIRAC_OP_H