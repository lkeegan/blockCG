#ifndef LKEEGAN_BLOCKCG_DIRAC_OP_H
#define LKEEGAN_BLOCKCG_DIRAC_OP_H
#include "fields.hpp"

// D.op(ls, rhs) is our dummy "Dirac operator" - a simple sparse
// VxV hermitian positive definite matrix with condition number ~ 1/m that acts
// on block_fermion fields
class dirac_op {
 private:
  using gauge = Eigen::Matrix<std::complex<double>, N_f, N_f>;
  std::vector<gauge, Eigen::aligned_allocator<gauge>> U;

  // 1-d nearest-neighbour staggered fermion Dirac operator with periodic bcs
  template <int N_rhs>
  void D(block_fermion_field<N_rhs>& lhs,
         const block_fermion_field<N_rhs>& rhs) const {
    for (int ix = 0; ix < V; ++ix) {
      lhs[ix] = 0.5 * U[ix] * rhs[(ix + 1) % V] -
                0.5 * U[(ix - 1 + V) % V].adjoint() * rhs[(ix - 1 + V) % V];
    }
  }

 public:
  int V;
  double mass;

  explicit dirac_op(int V, double mass = 0.1) : U(V), V(V), mass(mass) {
    // generate random N_fxN_f complex matrices as "gauge links" U
    for (int ix = 0; ix < V; ++ix) {
      U[ix].setRandom();
    }
  }

  // "Dirac operator"
  // lhs = (m^2 - D^2) rhs
  template <int N_rhs>
  void op(block_fermion_field<N_rhs>& lhs,
          const block_fermion_field<N_rhs>& rhs) const {
    block_fermion_field<N_rhs> tmp(lhs.V);
    D(tmp, rhs);
    D(lhs, tmp);
    lhs.rescale_add(-1.0, rhs, mass * mass);
  }
};

#endif  // LKEEGAN_BLOCKCG_DIRAC_OP_H