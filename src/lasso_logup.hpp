// Sound LogUp range-check lookup module.
//
// Implements the lookup argument from
//   Habök, "Multivariate Lookups Based on Logarithmic Derivatives",
//   eprint 2022/1530
// over the table T[j] = j for j in [0, 2^16), used to enforce that the
// auxiliary witness values collected in `prover::lasso_range_indices`
// are within the 16-bit signed range used by the GPT-2 quantization.
//
// Why this module exists:
//   The original zkGPT codebase did NOT implement Lasso (Setty-Thaler-Wahby,
//   eprint 2023/1216) as the paper claims; it only performed a GKR-style
//   batched opening of the input MLE under a misleading `verifyLasso` name,
//   and provided no actual lookup argument or range-check soundness.
//   Earlier yedidel additions sketched the LogUp math but were not sound:
//     - no commitment to the multiplicities polynomial,
//     - the "verifier" recomputed both sides locally instead of receiving
//       openings,
//     - a `& 0xFFFF` mask silently let out-of-range values pass,
//     - the Fiat-Shamir order was reversed (challenge before commit).
//   This module replaces that placeholder with a real, sound LogUp protocol:
//     (1) Pedersen-Hyrax commit to f_vec and m before any challenge.
//     (2) Sample r AFTER the commits.
//     (3) Compute reciprocals g_i = 1/(r+f_i), h_j = m_j/(r+j); commit both.
//     (4) Sample sumcheck challenges r_A, r_B and Schwartz-Zippel points
//         z_A, z_B AFTER the reciprocal commits.
//     (5) Two degree-1 sumchecks prove Σ g = S_g and Σ h = S_h
//         (using the existing `sum_check_product` infrastructure).
//     (6) Verifier checks S_g == S_h (the LogUp identity).
//     (7) Two pointwise consistency checks at random z_A, z_B verify
//         g(z_A)·(r+f(z_A)) == 1  and  h(z_B)·(r+J(z_B)) == m(z_B),
//         with all four MLEs opened via Hyrax.
//   The 16-bit range is enforced explicitly: every f_i is validated via the
//   signed `convert(Fr)` helper and the protocol aborts on out-of-range
//   values rather than masking them away.

#ifndef ZKCNN_LASSO_LOGUP_HPP
#define ZKCNN_LASSO_LOGUP_HPP

#include "global_var.hpp"
#include "prover.hpp"

namespace lasso_logup {

struct LookupBenchmark {
    bool   sound;            // true iff every f_i was in [0, 2^16) and all
                             //   sumcheck / opening / identity checks passed.
    u32    num_lookups_raw;  // |lasso_range_indices|.
    u32    log_M;            // log2 of the padded lookup-vector length.
    double prover_time_s;    // wall-clock time spent on the prover side.
    double verifier_time_s;  // wall-clock time spent on the verifier side.
    u64    proof_size_bytes; // bytes added to the proof by this protocol.
};

// Run the full sound LogUp range-check protocol against the values stored at
// `p->val[0][p->lasso_range_indices[i]]` for i in [0, |lasso_range_indices|).
// Updates `p->proof_size` and `p->accumulated_lasso_time` so the existing
// reporting in `verifier::prove` continues to work, and returns a detailed
// benchmark for the new server-side run.
LookupBenchmark run(prover *p);

}  // namespace lasso_logup

#endif  // ZKCNN_LASSO_LOGUP_HPP
