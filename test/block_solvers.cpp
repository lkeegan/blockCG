#include "catch.hpp"
#include "block_solvers.hpp"

TEST_CASE( "BCG", "[block_solvers]") {
	int V = 128;
	double stopping_criterion = 1.e-10;
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

TEST_CASE( "BCGrQ", "[block_solvers]") {
	int V = 128;
	double stopping_criterion = 1.e-10;
	constexpr int N_rhs = 3;
	block_fermion_field<N_rhs> X(V), B(V), AX(V);
	B.setRandom();
	int iterations = BCGrQ(X, B, stopping_criterion);
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
