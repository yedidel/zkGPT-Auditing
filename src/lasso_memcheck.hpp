// lasso-fork: Spice-style offline memory checking.
//
// Given:
//   * A static "memory" T_α of size N_α (either an MLE-evaluable subtable for
//     the Surge variant or a Pedersen-committed table for the c=1 variant).
//   * A sequence of M reads via dereference indices dim_α[k] ∈ [N_α], with
//     values E_α[k] = T_α[dim_α[k]].
//
// Spice (Setty et al., 2018) proves consistency of every read against T_α
// using four multisets:
//
//   init  = {(j,            T_α[j],     0                  ) : j ∈ [N_α]}
//   read  = {(dim_α[k],     E_α[k],     read_cts[k]        ) : k ∈ [M]   }
//   write = {(dim_α[k],     E_α[k],     read_cts[k] + 1    ) : k ∈ [M]   }
//   final = {(j,            T_α[j],     final_cts[j]       ) : j ∈ [N_α]}
//
// where read_cts[k] is the timestamp the k-th read sees on its address (the
// number of times that address has been written before, tracked online by the
// honest prover) and final_cts[j] is the total number of reads from address j.
//
// The memory-checking lemma (Spice §3.2) says: the four multisets satisfy
// init ⊎ write = read ⊎ final iff every read returned the value the memory
// genuinely held at that timestamp. We enforce equality via multiset-hash
// equality
//
//   ∏_(a,v,t)∈init  (γ - h(a,v,t)) ·
//   ∏_(a,v,t)∈write (γ - h(a,v,t))
//     ?=
//   ∏_(a,v,t)∈read  (γ - h(a,v,t)) ·
//   ∏_(a,v,t)∈final (γ - h(a,v,t))
//
// where h(a,v,t) = a + τ·v + τ²·t for verifier-chosen (γ, τ) ∈ F². The four
// products are realised as four grand-product GKR runs (lasso_grand_product).
//
// Soundness rests on Schwartz-Zippel over (γ, τ): for distinct multisets the
// hashes collide with probability ≤ 3·max(N_α, M)/|F|.
//
// This module is variant-agnostic: it accepts pre-computed dim_α, E_α and
// the table T_α and produces the grand-product proofs plus the witness
// vectors (read_cts, final_cts) the surrounding driver must commit to.

#ifndef ZKCNN_LASSO_MEMCHECK_HPP
#define ZKCNN_LASSO_MEMCHECK_HPP

#include <vector>

#include "lasso_grand_product.hpp"
#include "lasso_types.hpp"

namespace lasso_core {

// Inputs to one memory-checking run for a single subtable α.
struct MemCheckInputs {
    // The static memory T_α as a length-N_α vector. Used by the prover to
    // construct init / final multisets and by the verifier (only at MLE
    // evaluation points, never as a whole) to discharge final claims.
    const std::vector<F> *T;          // length N_α, borrowed
    // The dereference indices dim_α[k] for k ∈ [M]. Required to be in [0, N_α).
    const std::vector<u32> *dim;      // length M, borrowed
    // The values E_α[k] = T_α[dim_α[k]] the prover claims (and which the
    // surrounding Surge / c=1 sumcheck has already constrained).
    const std::vector<F> *E;          // length M, borrowed
    int log_N;                        // log2(N_α)
    int log_M;                        // log2(M), with M padded if needed
};

// Witnesses generated as a side effect of running the protocol; the caller
// commits them via Hyrax to support the grand-product final claims.
struct MemCheckWitness {
    std::vector<F> read_cts;          // length M (in F), per-read timestamp
    std::vector<F> final_cts;         // length N_α (in F), per-address total
};

// Aggregated proof bundle for one memory-checking run.
struct MemCheckProof {
    GrandProductProof  gp_init;
    GrandProductProof  gp_write;
    GrandProductProof  gp_read;
    GrandProductProof  gp_final;
    F                  gamma;         // verifier's hash-row challenge
    F                  tau;           // verifier's hash-column challenge
    bool               sound;
    u64                proof_size_bytes;
    double             prover_time_s;
    double             verifier_time_s;
};

// Run the four-multiset memory check, populating `out_witness` and returning
// the bundle of grand-product proofs. The four GKR claims must subsequently
// be discharged against committed openings of:
//     T (init / final value column),
//     dim (read / write address column),
//     E (read / write value column),
//     read_cts, final_cts.
// That discharge happens in the variant driver (Surge or c=1).
MemCheckProof run_memory_check(const MemCheckInputs &inp,
                               MemCheckWitness &out_witness);

}  // namespace lasso_core

#endif  // ZKCNN_LASSO_MEMCHECK_HPP
