// lasso-fork: c=1 (unstructured) Lasso for the Softmax exponentiation table.
//
// Per Bing-Jyue's guidance: tables that are not "structured" — i.e. tables
// whose MLE has no closed-form O(log N) evaluation — must be handled with
// c=1 in the Lasso prover. There is no subtable decomposition; instead the
// table itself is committed as a polynomial and opened at a single random
// point at the end of the protocol.
//
// In zkGPT this applies to `neuralNetwork::table[]` of size 655360 ≈ 2^20,
// holding round(exp(-St·i)/Se) for the (t, E) lookup inside Softmax. The
// committed table is independent of any individual proof and can in principle
// be reused across runs (we re-commit per call here for simplicity; adding a
// cache is a future optimisation).
//
// Protocol (c=1 Lasso, single subtable equal to the full table):
//
//   1. Caller has assembled M lookup positions, each (t_idx[k], E[k]) where
//      the prover claims E[k] = T[t_idx[k]]. The caller passes:
//        * dim       = vector of t indices in [0, N)        — length M
//        * E         = vector of claimed lookup outputs     — length M
//        * T         = the static table                     — length N
//      and commits to T separately (or reuses an existing commitment).
//   2. We Hyrax-commit to dim, E (over log_M variables) and to T (over log_N).
//   3. Run Spice memory checking on (T, dim, E) — this is the same machinery
//      as Surge for c=1 (single subtable α=0). Memory checking proves
//      E[k] = T[dim[k]] for all k.
//   4. Surge main sumcheck for c=1:
//        E_committed(z_z) = Σ_k eq(z_z, k) · E[k]
//      where z_z ∈ F^{log_M} is the random opening point. This is a degree-2
//      sumcheck. The final claim is opened against C_E.
//   5. Discharge memory-checking final claims using the same scheme as Surge
//      but with T̃(point) coming from a Hyrax opening of C_T at the relevant
//      grand-product final point (instead of an analytic identity).
//
// The exp-lookup driver (neuralNetwork.cpp) builds dim/E from the Softmax
// witness positions and calls run_exp_lookup() once per inference.

#ifndef ZKCNN_LASSO_UNSTRUCTURED_HPP
#define ZKCNN_LASSO_UNSTRUCTURED_HPP

#include <vector>

#include "lasso_types.hpp"
#include "prover.hpp"

namespace lasso_unstructured {

// One (t_idx, E_idx) pair to be checked.
//   t_idx : index into prover->val[0]   for the lookup key   t = T-index
//   E_idx : index into prover->val[0]   for the looked-up value E = T[t]
//
// We alias the prover-side struct (declared in prover.hpp) so that the
// neuralNetwork collector can populate prover::exp_lookup_pairs directly
// without an extra copy at the verifier callsite.
using LookupPair = prover::ExpLookupPair;

// Caller-supplied table. We do not copy or own this storage; the static
// `table[]` lives in `neuralNetwork`.
struct StaticTable {
    const int *data;     // borrowed
    u32        size;     // power-of-two-extended outside; the actual entries
                         //   beyond `size` are read as 0
};

// Run c=1 Lasso for the exp table.
//
// `pairs` : list of (t_idx, E_idx) inside p->val[0] that the prover claims
//           obey  val[0][E_idx] == T[ val[0][t_idx] ].
// `T`     : the static lookup table, size N (padded by caller to 2^log_N).
//
// Updates p->proof_size and p->accumulated_lasso_time.
lasso_core::LassoBenchmark run_exp_lookup(prover *p,
                                          const std::vector<LookupPair> &pairs,
                                          const std::vector<F> &T_padded,
                                          int log_N);

}  // namespace lasso_unstructured

#endif  // ZKCNN_LASSO_UNSTRUCTURED_HPP
