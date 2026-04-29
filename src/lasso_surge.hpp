// lasso-fork: Surge protocol for "structured + decomposable" tables.
//
// Specialised for the 16-bit identity table T[j] = j over j ∈ [0, 2^16) used
// for the GPT-2 quantization range checks (d1, d2, d3, s, dt2). The table
// decomposes as
//
//     T[j] = T_1[j_high] · 256 + T_2[j_low],
//     T_α[u] = u                for α ∈ {1, 2}, u ∈ [0, 2^8)
//
// with combining function g(t_1, t_2) = t_1 · 256 + t_2. Each subtable is the
// identity of size 2^8, whose MLE evaluates analytically in O(log √N) = 8
// field operations:
//
//     T̃_α(z) = Σ_{i=0}^{7} 2^i · z_i.
//
// Protocol (Surge with c = 2):
//
//   1. Prover collects the lookup vector v ∈ F^M from the witness positions
//      `lasso_range_indices`, validates each v[k] ∈ [0, 2^16), and decomposes
//      every v[k] into (dim_1[k], dim_2[k]) ∈ [0, 256)².
//   2. Prover builds the "subtable evaluation columns" E_α[k] = T_α[dim_α[k]].
//      Sanity: g(E_1[k], E_2[k]) == v[k].
//   3. Prover Hyrax-commits to dim_1, dim_2, E_1, E_2 (and to v itself; the
//      caller may already have it inside p->val[0], but for self-contained
//      verification we commit again here).
//   4. For each subtable α ∈ {1, 2}, run Spice memory checking on (T_α,
//      dim_α, E_α). The four grand-product products must satisfy the multiset
//      identity, witnessed by Hyrax openings of the relevant polynomials
//      (handled inside lasso_memcheck and the discharge step below).
//   5. Surge main sumcheck — pick z_z ← F^{log M}, prove
//
//          v(z_z)  =  Σ_k eq(z_z, k) · g(E_1(k), E_2(k))
//                  =  Σ_k eq(z_z, k) · (256·E_1(k) + E_2(k))
//
//      This is a linear combination of two degree-2 sumchecks; we run it as
//      a single degree-2 sumcheck whose round polynomial collects both terms.
//      Final point w; prover opens E_1, E_2 at w. Verifier checks the
//      identity using opened E_α(w).
//   6. Discharge memory-checking final claims: each grand-product produced a
//      `final_point` and `final_claim`. The verifier opens
//        - the appropriate `dim_α`,
//        - the appropriate `E_α` (for read/write multisets),
//        - `read_cts_α` and `final_cts_α`,
//        - the analytic T̃_α
//      so that hash_leaf(γ, τ; address, value, timestamp) matches each
//      `final_claim`. (For the identity subtable we substitute the analytic
//      T̃_α(point) into the value column rather than opening a committed T.)
//
// Soundness gap: each memory-check identity carries Schwartz-Zippel error
// ≤ 3·max(N_α, M)/|F|; the Surge sumcheck contributes ≤ 2·log M/|F|. For
// |F| ≈ 2^254 these are negligible.

#ifndef ZKCNN_LASSO_SURGE_HPP
#define ZKCNN_LASSO_SURGE_HPP

#include <vector>

#include "lasso_memcheck.hpp"
#include "lasso_types.hpp"
#include "prover.hpp"

namespace lasso_surge {

// Run Surge for the 16-bit identity range table over the prover's
// `lasso_range_indices`. The validation of v[k] ∈ [0, 2^16) is performed
// using the signed `convert(Fr)` helper rather than any masking.
//
// Updates p->proof_size and p->accumulated_lasso_time so the existing
// reporting in `verifier::prove` continues to work.
lasso_core::LassoBenchmark run_range_check(prover *p);

}  // namespace lasso_surge

#endif  // ZKCNN_LASSO_SURGE_HPP
