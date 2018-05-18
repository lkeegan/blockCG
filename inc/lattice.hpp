#ifndef LKEEGAN_BLOCKCG_LATTICE_H
#define LKEEGAN_BLOCKCG_LATTICE_H

#include <vector>
#include <complex>
// for mkl versions of some matrix ops
#ifdef EIGEN_USE_MKL_ALL
  // ugly hack to get mkl to work with c++ std::complex type 
  #define MKL_Complex16 std::complex<double>
  #include "mkl.h"
#endif
#include "Eigen3/Eigen/Dense"
#include "Eigen3/Eigen/StdVector"

// define fermion type as complex vector
constexpr int N_fermion_dof = 12;
typedef Eigen::Matrix<std::complex<double>, N_fermion_dof, 1> fermion;

// simple class to store a field of V fermions
template<typename T> class field {
protected:
	// data
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

	field& operator+=(const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs[ix];
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

	template<typename Targ>
	field& operator*=(Targ multiplier)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] *= multiplier;
		}
	    return *this;
	}

	field& operator/=(double scalar)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] /= scalar;
		}
	    return *this;
	}

	field& operator/=(std::complex<double> scalar)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] /= scalar;
		}
	    return *this;
	}

	template<typename Targ>
	field& add(const Targ& rhs_multiplier, const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix].noalias() += rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	// *thisField += rhsField * rhs_multiplier
	template<typename Targ>
	field& add(const field& rhs, const Targ& rhs_multiplier)
	{
		//#pragma omp parallel for
		for(int ix=0; ix<V; ++ix) {
			data_[ix].noalias() += rhs[ix] * rhs_multiplier;
		}
	    return *this;
	}
  #ifdef EIGEN_USE_MKL_ALL
	// mkl zgemm call at each site 
	template<typename Targ>
	field& add_mkl(const field& rhs, const Targ& rhs_multiplier)
	{
		std::complex<double> one (1.0, 0.0);
		const int N_rhs = rhs_multiplier.cols();
		for(int ix=0; ix<V; ++ix) {
		    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
       			 3, N_rhs, N_rhs, &one, rhs[ix].data(), 3, rhs_multiplier.data(), N_rhs, &one, data_[ix].data(), 3);
			//data_[ix].noalias() += rhs[ix] * rhs_multiplier;
		}
	    return *this;
	}
	// single mkl zgemm call for (3xvolume)xN_rhs fermion matrices [data must be mkl_malloced contiguous array]
	template<typename Targ>
	field& add_mkl_bigmat(const field& rhs, const Targ& rhs_multiplier)
	{
		std::complex<double> one (1.0, 0.0);
		const int N_rhs = rhs_multiplier.cols();
	    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
   			 3*V, N_rhs, N_rhs, &one, rhs[0].data(), 3*V, rhs_multiplier.data(), N_rhs, &one, data_[0].data(), 3*V);
			//data_[ix].noalias() += rhs[ix] * rhs_multiplier;
	    return *this;
	}
  #endif
/*	field& add(double rhs_multiplier, const field& rhs)
	{
		for(int ix=0; ix<V; ++ix) {
			data_[ix] += rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
*/	// *this = scale * (*this) + rhs_multiplier * rhs
	field& scale_add(double scale, double rhs_multiplier, const field& rhs)
	{
		////#pragma omp parallel for
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = scale * data_[ix] + rhs_multiplier * rhs[ix];
		}
	    return *this;
	}
	field& scale_add(std::complex<double> scale, std::complex<double> rhs_multiplier, const field& rhs)
	{
		////#pragma omp parallel for
		for(int ix=0; ix<V; ++ix) {
			data_[ix] = scale * data_[ix] + rhs_multiplier * rhs[ix];
		}
	    return *this;
	}

	// these are some eigen routines for individual vectors or matrices
	// trivially extended by applying them to each element in the vector in turn

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
	// equivalent to real part of dot with itself i.e. l2-norm squared
	double squaredNorm() const {
		double norm = 0.0;
		for(int ix=0; ix<V; ++ix) {
			norm += data_[ix].squaredNorm();
		}
		return norm;		
	}
	// returns square root of squaredNorm() above i.e. l2-norm
	double norm() const {
		return sqrt(squaredNorm());		
	}

	//complex conjugate of this dotted with rhs
	std::complex<double> dot (const field& rhs) const {
		std::complex<double> sum (0.0, 0.0);
		for(int ix=0; ix<V; ++ix) {
			sum += data_[ix].dot(rhs[ix]);
		}
		return sum;		
	}

	// [i] operator returns data with index i
	T& operator[](int i) { return data_[i]; }
	const T& operator[](int i) const { return data_[i]; }
};

#endif //LKEEGAN_BLOCKCG_LATTICE_H