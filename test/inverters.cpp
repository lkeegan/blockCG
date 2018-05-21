#include "catch.hpp"
#include "inverters.hpp"

int V = 512;
double stopping_criterion = 1.e-10;

TEST_CASE( "CG", "[solvers]") {
	field<fermion> x(V), b(V), Ax(V);
	b.setRandom();
	int iterations = CG(x, b, stopping_criterion);
	dirac_op(Ax, x);
	Ax -= b;
	double residual = sqrt( Ax.squaredNorm() / b.squaredNorm() );
	CAPTURE(stopping_criterion);
	CAPTURE(iterations);
	CAPTURE(residual);
	REQUIRE( residual < 2 * stopping_criterion );
}

TEST_CASE( "SCG", "[solvers]") {
	std::vector<double> shifts = {0.01, 0.05, 0.20};
	int N_shifts = shifts.size();
	field<fermion> b(V), Ax(V);
	std::vector<field<fermion>> x (N_shifts, b);
	b.setRandom();
	int iterations = SCG(x, b, shifts, stopping_criterion);
	for(int i=0; i<N_shifts; ++i) {
		double shift = shifts[i];
		dirac_op(Ax, x[i]);
		Ax.add(x[i], shift);
		Ax -= b;
		double residual = sqrt( Ax.squaredNorm() / b.squaredNorm() );
		CAPTURE(shift);
		CAPTURE(stopping_criterion);
		CAPTURE(iterations);
		CAPTURE(residual);
		REQUIRE( residual < 2 * stopping_criterion );		
	}
}
