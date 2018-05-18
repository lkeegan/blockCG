#include "catch.hpp"
#include "inverters.hpp"

TEST_CASE( "Sample test", "[inverters]") {
		SECTION( "CG" ) {
			REQUIRE( cg() == 1 );
		}
}