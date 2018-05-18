#include "catch.hpp"
#include "inverters.hpp"

constexpr int V = 4096;
constexpr double EPS = 1.e-14;

TEST_CASE( "Sample test", "[inverters]") {
	field<fermion> x (V);
	x.setRandom();

		SECTION( "CG" ) {
			REQUIRE( x.dot(x).real() - x.squaredNorm() < EPS );
		}
}