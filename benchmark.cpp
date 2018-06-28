#include "standard_solvers.hpp"
#include "block_solvers.hpp"
#include <iostream>
#include <chrono>

// Simple Benchmark code

// number of RHS vectors for block solvers
constexpr int N_rhs = 12;
// shifts for shifted solver
std::vector<double> shifts = {0, 0.0001, 0.001, 0.01, 0.1};
int N_shifts = static_cast<int>(shifts.size());

int main(int argc, char *argv[]) {

	constexpr int n_args = 3;
    if (argc-1 != n_args) {
        std::cout << "This program requires " << n_args << " arguments:" << std::endl;
        std::cout << "Lattice volume, Dirac operator mass, solver stopping criterion" << std::endl;
        std::cout << "e.g. ./benchmark 1024 0.01 1e-12" << std::endl;
        return 1;
    }

    // initialise dirac op
	int V = static_cast<int>(atof(argv[1]));
	double mass = static_cast<double>(atof(argv[2]));
	double stopping_criterion = static_cast<double>(atof(argv[3]));
	dirac_op D(V, mass);

	// make random block fermion source vector for SBCGrQ 
	block_fermion_field<N_rhs> B(V);
	B.setRandom();

	std::cout << "#Benchmark of SBCGrQ vs SCG solvers - N_rhs = " << N_rhs << ", eps=" << stopping_criterion << std::endl;

	// do SCG solve for each RHS separately
	std::vector<double> resSCG(N_shifts, 0.0);
	fermion_field b(V), Ax(V);
	std::vector<fermion_field> x (N_shifts, b);
	int iterSCG = 0;
	for(int i_rhs=0; i_rhs<N_rhs; ++i_rhs) {
		// get i'th RHS column of block vector B
		for(int i_x=0; i_x<V; ++i_x) {
			b[i_x] = B[i_x].col(i_rhs);
		}
		iterSCG += SCG(x, b, D, shifts, stopping_criterion);
		for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
			double shift = shifts[i_shift];
			D.op(Ax, x[i_shift]);
			Ax.add(x[i_shift], shift);
			Ax -= b;
			double residual = sqrt( Ax.dot(Ax) / b.dot(b) );
			if(residual > resSCG[i_shift]) {
				resSCG[i_shift] = residual;
			}
		}
	}
	// measure and output worst residual for each shift
	std::cout << "#SCG residuals:\t\t";
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		std::cout << resSCG[i_shift] << "\t";
	}
	std::cout << std::endl;

	// do SBCGrQ solve
	block_fermion_field<N_rhs> AX(V);
	std::vector< block_fermion_field<N_rhs> > X (N_shifts, B);
	int iterSBCGrQ = N_rhs * SBCGrQ(X, B, D, shifts, stopping_criterion);
	// measure and output worst residual for each shift
	std::cout << "#SBCGrQ residuals:\t";
	block_matrix<N_rhs> b2 = B.hermitian_dot(B);
	for(int i_shift=0; i_shift<N_shifts; ++i_shift) {
		double shift = shifts[i_shift];
		D.op(AX, X[i_shift]);
		AX.add(X[i_shift], shift);
		AX -= B;
		block_matrix<N_rhs> r2 = AX.hermitian_dot(AX);
		std::cout << sqrt((r2.diagonal().real().array()/b2.diagonal().array().real()).maxCoeff()) << "\t";
	}
	std::cout << std::endl;

	std::cout << "#V\tmass\tSCG_iterations\tSBCGrQ_iterations" << std::endl;
	std::cout << V << "\t" << mass << "\t" << iterSCG << "\t\t" << iterSBCGrQ << std::endl;

	return 0;
}