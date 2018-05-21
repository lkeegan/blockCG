#ifndef LKEEGAN_BLOCKCG_FIELDS_H
#define LKEEGAN_BLOCKCG_FIELDS_H

#include <vector>
#include <complex>
#include "Eigen3/Eigen/Dense"
#include "Eigen3/Eigen/StdVector"

// Define data structure for block fermion fields:
// "fermion" - 12-component complex vector
// "block_fermion<N_rhs>" - (12 x N_rhs) complex matrix, i.e. N_rhs x "fermion" vectors 
// "field<T>(V)" - a V-component field of type T, where T could be e.g. block_matrix<4>
// "block_matrix<N_rhs>" - (N_rhs x N_rhs) complex matrix

constexpr int N_fermion_dof = 12;
template <int N_rhs>
using block_fermion = Eigen::Matrix<std::complex<double>, N_fermion_dof, N_rhs>;
typedef block_fermion<1> fermion;
template <int N_rhs>
using block_matrix = Eigen::Matrix<std::complex<double>, N_rhs, N_rhs>;

template<typename T> class field {
protected:
	std::vector<T, Eigen::aligned_allocator<T>> data_;

public:
	int V;

	explicit field (int V) : V(V) {
		data_.resize(V);
	}

	field& operator=(const field& rhs) {
		V = rhs.V;
		data_.resize(V);
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = rhs[ix];
		}
		return *this;
	}
	field& operator-=(const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] -= rhs[ix];
		}
	    return *this;
	}

	// [i] operator returns data with index i
	T& operator[](int i) { return data_[i]; }
	const T& operator[](int i) const { return data_[i]; }

	// these are some eigen routines for individual vectors or matrices
	// trivially extended by applying them to each element in the field in turn
	// this += rhs * rhs_multiplier
	template<typename Targ>
	field& add(const field& rhs, const Targ& rhs_multiplier)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix].noalias() += rhs[ix] * rhs_multiplier;
		}
	    return *this;
	}
	template<typename Targ>
	field& rescale_add(const Targ& lhs_multiplier, const field& rhs, const Targ& rhs_multiplier)
	{
		T tmp;
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
	// equivalent to real part of dot product with itself i.e. l2-norm squared
	double squaredNorm() const {
		double norm = 0.0;
		for(int ix=0; ix<V; ++ix) {
			norm += data_[ix].squaredNorm();
		}
		return norm;		
	}
	//complex conjugate of this dotted with rhs
	std::complex<double> dot (const field& rhs) const {
		std::complex<double> sum (0.0, 0.0);
		for(int ix=0; ix<V; ++ix) {
			sum += data_[ix].dot(rhs[ix]);
		}
		return sum;		
	}
};

#endif //LKEEGAN_BLOCKCG_FIELDS_H