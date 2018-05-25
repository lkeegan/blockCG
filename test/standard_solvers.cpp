#include "catch.hpp"
#include "standard_solvers.hpp"

TEST_CASE( "CG", "[standard_solvers]") {
	int V = 128;
	double stopping_criterion = 1.e-10;
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
	int V = 128;
	double stopping_criterion = 1.e-10;
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