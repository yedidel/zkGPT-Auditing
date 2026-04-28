// yedidel-lasso: Grand-Product GKR.
//
// For a vector v of length 2^L (committed via Hyrax as a multilinear
// polynomial), this module proves the product P = ∏_i v[i] without exposing
// v itself. Used by the Spice memory-checking layer of Lasso to prove
// equality of two multiset hashes
//
//     H(init) · H(write)  ?=  H(read) · H(final)
//
// where each multiset hash is itself a product over its elements.
//
// Construction (the Thaler13 / Setty-Thaler-Wahby style):
//   * View v as the leaves of a binary product-tree of depth L.
//   * Layer k from the root has 2^k entries. Layer 0 is the single root P.
//   * Each layer satisfies V_k(z) = V_{k+1}(0, z) · V_{k+1}(1, z), where
//     V_{k+1} is the MLE of layer k+1 with the new variable prepended in
//     LSB position to match our existing little-endian convention.
//   * Because the "wiring" at every gate is trivial — the two children of
//     gate j live at indices (2j, 2j+1) of the next layer — there is no
//     sumcheck inside the protocol; each layer reduces a single claim to
//     a new claim in O(1) rounds and 2 field elements of transcript.
//
// Per-layer round, with current claim V_k(z_k) = c_k:
//     1. Prover sends a = V_{k+1}(0, z_k) and b = V_{k+1}(1, z_k).
//     2. Verifier checks  a · b == c_k.
//     3. Verifier samples γ_k ← F (Fiat-Shamir / CSPRNG).
//     4. New claim: V_{k+1}((γ_k, z_k)) = (1-γ_k)·a + γ_k·b.
//
// After L layers, the claim is V_L(z_L) = c_L which the caller must
// discharge with an opening of commit(v) at z_L.
//
// Total transcript: L · (2 field elements) = 2L · F_BYTE_SIZE.
// Prover work:      O(2^L) to build the tree, O(2^L) folding per layer.
// Verifier work:    O(L) field operations.

#ifndef ZKCNN_LASSO_GRAND_PRODUCT_HPP
#define ZKCNN_LASSO_GRAND_PRODUCT_HPP

#include <vector>

#include "lasso_types.hpp"

namespace lasso_core {

// Output of one grand-product run; the caller must (a) accept the value of
// `claimed_product` against whatever the surrounding protocol expects, and
// (b) discharge `final_claim` against an opening of v at `final_point`.
struct GrandProductProof {
    F                claimed_product;  // P = ∏ v[i], announced by prover.
    std::vector<F>   final_point;      // z_L ∈ F^L, length L = log2(|v|).
    F                final_claim;      // V_L(z_L), to be opened against commit(v).
    bool             sound;            // true iff every per-layer check passed.
    u64              proof_size_bytes; // L * 2 * F_BYTE_SIZE.
    double           prover_time_s;
    double           verifier_time_s;
};

// Run grand-product GKR on the leaves vector v (size = 1 << log_len).
// Both prover and verifier sides are simulated in-process; the proof object
// records every value an honest prover would have sent, plus the random
// coins the verifier sampled.
//
// Cost: O(2^log_len) prover, O(log_len) verifier, O(log_len) transcript.
GrandProductProof prove_grand_product(const std::vector<F> &v);

// Recompute the verifier's view of one grand-product run from the prover's
// transcript. Used by the surrounding Lasso verifier to re-derive
// final_point and final_claim deterministically (same CSPRNG seeding) and
// confirm soundness without trusting the prover's `sound` flag.
//
// Returns true iff every per-layer check `a·b == c_k` holds and the final
// claim matches the opened evaluation `final_claim_check`.
bool verify_grand_product(const GrandProductProof &proof,
                          F final_claim_check);

}  // namespace lasso_core

#endif  // ZKCNN_LASSO_GRAND_PRODUCT_HPP
