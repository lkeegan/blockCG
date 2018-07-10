#include "block_solvers.hpp"
#include "catch.hpp"
#include "standard_solvers.hpp"

// Unit tests

// size of "lattice"
int V = 128;
// mass is inversely related to condition number of "Dirac operator"
double mass = 0.5;
// stopping criterion for solvers
double stopping_criterion = 1.e-10;
// number of RHS vectors for block solverss
constexpr int N_rhs = 3;
// shifts for shifted solvers
std::vector<double> shifts = {0.0, 0.01, 0.10, 0.20, 0.9};
int N_shifts = static_cast<int>(shifts.size());

TEST_CASE("CG", "[standard_solvers]") {
  fermion_field x(V), b(V), Ax(V);
  dirac_op D(V, mass);
  b.setRandom();
  int iterations = CG(x, b, D, stopping_criterion);
  D.op(Ax, x);
  Ax -= b;
  double residual = sqrt(Ax.real_dot(Ax) / b.real_dot(b));
  CAPTURE(stopping_criterion);
  CAPTURE(iterations);
  CAPTURE(residual);
  REQUIRE(residual < 2 * stopping_criterion);
}

TEST_CASE("SCG", "[standard_solvers]") {
  fermion_field b(V), Ax(V);
  dirac_op D(V, mass);
  std::vector<fermion_field> x(N_shifts, b);
  b.setRandom();
  int iterations = SCG(x, b, D, shifts, stopping_criterion);
  for (int i = 0; i < N_shifts; ++i) {
    double shift = shifts[i];
    D.op(Ax, x[i]);
    Ax.add(x[i], shift);
    Ax -= b;
    double residual = sqrt(Ax.real_dot(Ax) / b.real_dot(b));
    CAPTURE(shift);
    CAPTURE(stopping_criterion);
    CAPTURE(iterations);
    CAPTURE(residual);
    REQUIRE(residual < 2 * stopping_criterion);
  }
}

TEST_CASE("BCG", "[block_solvers]") {
  block_fermion_field<N_rhs> X(V), B(V), AX(V);
  dirac_op D(V, mass);
  B.setRandom();
  int iterations = BCG(X, B, D, stopping_criterion);
  D.op(AX, X);
  AX -= B;
  block_matrix<N_rhs> r2 = AX.hermitian_dot(AX);
  block_matrix<N_rhs> b2 = B.hermitian_dot(B);
  CAPTURE(N_rhs);
  CAPTURE(stopping_criterion);
  CAPTURE(iterations);
  for (int i_rhs = 0; i_rhs < N_rhs; ++i_rhs) {
    double residual = sqrt(r2(i_rhs, i_rhs).real() / b2(i_rhs, i_rhs).real());
    CAPTURE(i_rhs);
    CAPTURE(residual);
    REQUIRE(residual < 2 * stopping_criterion);
  }
}

TEST_CASE("BCGrQ", "[block_solvers]") {
  block_fermion_field<N_rhs> X(V), B(V), AX(V);
  dirac_op D(V, mass);
  B.setRandom();
  int iterations = BCGrQ(X, B, D, stopping_criterion);
  D.op(AX, X);
  AX -= B;
  block_matrix<N_rhs> r2 = AX.hermitian_dot(AX);
  block_matrix<N_rhs> b2 = B.hermitian_dot(B);
  CAPTURE(N_rhs);
  CAPTURE(stopping_criterion);
  CAPTURE(iterations);
  for (int i_rhs = 0; i_rhs < N_rhs; ++i_rhs) {
    double residual = sqrt(r2(i_rhs, i_rhs).real() / b2(i_rhs, i_rhs).real());
    CAPTURE(i_rhs);
    CAPTURE(residual);
    REQUIRE(residual < 2 * stopping_criterion);
  }
}

TEST_CASE("SBCGrQ", "[block_solvers]") {
  block_fermion_field<N_rhs> B(V), AX(V);
  dirac_op D(V, mass);
  std::vector<block_fermion_field<N_rhs> > X(N_shifts, B);
  B.setRandom();
  int iterations = SBCGrQ(X, B, D, shifts, stopping_criterion);
  block_matrix<N_rhs> b2 = B.hermitian_dot(B);
  CAPTURE(N_rhs);
  CAPTURE(N_shifts);
  CAPTURE(stopping_criterion);
  CAPTURE(iterations);
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    double shift = shifts[i_shift];
    D.op(AX, X[i_shift]);
    AX.add(X[i_shift], shift);
    AX -= B;
    block_matrix<N_rhs> r2 = AX.hermitian_dot(AX);
    CAPTURE(i_shift);
    CAPTURE(shift);
    for (int i_rhs = 0; i_rhs < N_rhs; ++i_rhs) {
      double residual = sqrt(r2(i_rhs, i_rhs).real() / b2(i_rhs, i_rhs).real());
      CAPTURE(i_rhs);
      CAPTURE(residual);
      REQUIRE(residual < 2 * stopping_criterion);
    }
  }
}
