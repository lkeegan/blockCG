#include "catch.hpp"
#include "inverters.hpp"

int V = 512;
double stopping_criterion = 1.e-10;

TEST_CASE( "CG", "[standard_solvers]") {
	fermion_field x(V), b(V), Ax(V);
	b.setRandom();
	int iterations = CG(x, b, stopping_criterion);
	dirac_op(Ax, x);
	Ax -= b;
	double residual = sqrt( Ax.dot(Ax) / b.dot(b) );
	CAPTURE(stopping_criterion);
	CAPTURE(iterations);
	CAPTURE(residual);
	REQUIRE( residual < 2 * stopping_criterion );
}

TEST_CASE( "SCG", "[standard_solvers]") {
	std::vector<double> shifts = {0.01, 0.05, 0.20};
	int N_shifts = shifts.size();
	fermion_field b(V), Ax(V);
	std::vector<fermion_field> x (N_shifts, b);
	b.setRandom();
	int iterations = SCG(x, b, shifts, stopping_criterion);
	for(int i=0; i<N_shifts; ++i) {
		double shift = shifts[i];
		dirac_op(Ax, x[i]);
		Ax.add(x[i], shift);
		Ax -= b;
		double residual = sqrt( Ax.dot(Ax) / b.dot(b) );
		CAPTURE(shift);
		CAPTURE(stopping_criterion);
		CAPTURE(iterations);
		CAPTURE(residual);
		REQUIRE( residual < 2 * stopping_criterion );		
	}
}

TEST_CASE( "BCG", "[block_solvers]") {
	constexpr int N_rhs = 3;
	block_fermion_field<N_rhs> X(V), B(V), AX(V);
	B.setRandom();
	int iterations = BCG(X, B, stopping_criterion);
	dirac_op(AX, X);
	AX -= B;
	block_matrix<N_rhs> r2 = AX.hermitian_dot(AX);
	block_matrix<N_rhs> b2 = B.hermitian_dot(B);
	CAPTURE(N_rhs);
	CAPTURE(stopping_criterion);
	CAPTURE(iterations);
	for(int i=0; i<N_rhs; ++i) {
		double residual = sqrt(r2(i,i).real()/b2(i,i).real());
		CAPTURE(i);
		CAPTURE(residual);
		REQUIRE( residual < 2 * stopping_criterion );		
	}
}
