#ifndef LKEEGAN_BLOCKCG_FIELDS_H
#define LKEEGAN_BLOCKCG_FIELDS_H

// Define data structure for block fermion fields:

// "fermion" - N_f-component vector of complex doubles
// "block_fermion<N_rhs>" - (N_f x N_rhs) complex matrix, i.e. N_rhs x "fermion" vectors 
// "block_matrix<N_rhs>" - (N_rhs x N_rhs) complex matrix

// "fermion_field(V)" - a V-component field of "fermion"
// "block_fermion_field<N_rhs>(V)" - a V-component field of "block_fermion<N_rhs>"

#include <vector>
#include <complex>
#include "Eigen3/Eigen/Dense"
#include "Eigen3/Eigen/StdVector"

constexpr int N_f = 3;
template <int N_rhs>
using block_fermion = Eigen::Matrix<std::complex<double>, N_f, N_rhs>;
typedef block_fermion<1> fermion;
template <int N_rhs>
using block_matrix = Eigen::Matrix<std::complex<double>, N_rhs, N_rhs>;

template<int N_rhs> class block_fermion_field {
protected:
	std::vector<block_fermion<N_rhs>, Eigen::aligned_allocator<block_fermion<N_rhs>>> data_;

public:
	int V;

	explicit block_fermion_field (int V) : V(V) {
		data_.resize(V);
	}

	block_fermion_field<N_rhs>& operator=(const block_fermion_field<N_rhs>& rhs) {
		V = rhs.V;
		data_.resize(V);
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = rhs[ix];
		}
		return *this;
	}
	block_fermion_field<N_rhs>& operator-=(const block_fermion_field<N_rhs>& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] -= rhs[ix];
		}
	    return *this;
	}

	// [i] operator returns data with index i
	block_fermion<N_rhs>& operator[](int i) { return data_[i]; }
	const block_fermion<N_rhs>& operator[](int i) const { return data_[i]; }

	// these are some eigen routines for individual vectors or matrices
	// trivially extended by applying them to each element in the field in turn
	// this += rhs * rhs_multiplier
	template<typename Targ>
	block_fermion_field<N_rhs>& add(const block_fermion_field<N_rhs>& rhs, const Targ& rhs_multiplier)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix].noalias() += rhs[ix] * rhs_multiplier;
		}
	    return *this;
	}
	template<typename T_lhs_arg, typename T_rhs_arg>
	block_fermion_field<N_rhs>& rescale_add(const T_lhs_arg& lhs_multiplier, const block_fermion_field<N_rhs>& rhs, const T_rhs_arg& rhs_multiplier)
	{
		block_fermion<N_rhs> tmp;
		for(int ix=0; ix<V; ++ix) {
			tmp.noalias() = data_[ix] * lhs_multiplier;
			tmp.noalias() += rhs[ix] * rhs_multiplier;
			data_[ix] = tmp;
		}
	    return *this;
	}
	void setZero() {
		for(int ix=0; ix<V; ++ix) {
			data_[ix].setZero();
		}
	}
	void setRandom() {
		for(int ix=0; ix<V; ++ix) {
			data_[ix].setRandom();
		}
	}
	//real part of complex conjugate of this dotted with rhs
	double dot (const block_fermion_field<1>& rhs) const {
		double sum = 0.0;
		for(int ix=0; ix<V; ++ix) {
			sum += data_[ix].dot(rhs[ix]).real();
		}
		return sum;
	}
	block_matrix<N_rhs> hermitian_dot(const block_fermion_field<N_rhs>& rhs) const {
		// construct lower-triangular part of matrix
		block_matrix<N_rhs> R;
		R.setZero();
		for(int ix=0; ix<V; ++ix) {
			for(int i=0; i<N_rhs; ++i) {
				for(int j=0; j<=i; ++j) {
					R(i, j) += data_[ix].col(i).dot(rhs[ix].col(j));
				}
			}
		}
		// reconstruct upper triangular part from conjugate of lower triangular elements
		for(int i=1; i<N_rhs; ++i) {
			for(int j=0; j<i; ++j) {
				R(j, i) = std::conj(R(i, j));
			}
		}
		return R;
	}
	// In-place Multiply field X on RHS by inverse of triangular matrix R, i.e.
	// X <- X R^{-1}
	block_fermion_field<N_rhs>& multiply_triangular_inverse_RHS(const block_matrix<N_rhs>& R) {
		for(int ix=0; ix<V; ++ix) {
			for(int i=0; i<N_rhs; ++i) {
				for(int j=0; j<i; ++j) {
					data_[ix].col(i) -= R(j,i) * data_[ix].col(j);
				}
				data_[ix].col(i) /= R(i,i);
			}
		}
		return *this;
	}
	block_fermion_field<N_rhs>& thinQR(block_matrix<N_rhs>& R) {
		// Construct R_ij = Q_i^dag Q_j = hermitian, 
		R = hermitian_dot(*this);
		// Find upper triangular R such that R^dag R = H (i.e. previous contents of R) = Q^dag Q
		// i.e. adjoint of cholesky decomposition L matrix: L L^dag = H
		R = R.llt().matrixL().adjoint();
		// Q <- Q R^-1
		multiply_triangular_inverse_RHS(R);
		return *this;
	}

};
typedef block_fermion_field<1> fermion_field;

#endif //LKEEGAN_BLOCKCG_FIELDS_H