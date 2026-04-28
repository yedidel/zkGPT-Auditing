// yedidel-lasso: Spice-style offline memory checking — implementation.
// See lasso_memcheck.hpp for the protocol summary.

#include "lasso_memcheck.hpp"

#include <cassert>
#include <chrono>
#include <iostream>

namespace lasso_core {

namespace {

// Multiset hash element: H_(γ,τ)(a, v, t) = γ - (a + τ·v + τ²·t).
// We materialise this as the "leaf" of a grand-product, so each multiset is
// hashed by ∏ leaves = ∏ (γ - h(a, v, t)).
inline F hash_leaf(const F &gamma, const F &tau, const F &tau_sq,
                   const F &a, const F &v, const F &t) {
    return gamma - (a + tau * v + tau_sq * t);
}

}  // namespace

MemCheckProof run_memory_check(const MemCheckInputs &inp,
                               MemCheckWitness &out_witness) {
    MemCheckProof out{};
    out.sound = true;

    const auto &T   = *inp.T;
    const auto &dim = *inp.dim;
    const auto &E   = *inp.E;
    const u32 N = (u32)T.size();
    const u32 M = (u32)dim.size();
    assert(E.size() == M);

    // -----------------------------------------------------------------------
    // 1. Build read_cts[k] and final_cts[j] in a single pass through dim.
    //    Honest prover invariant: when read k accesses address dim[k], the
    //    timestamp it observes equals the number of writes to that address
    //    so far (== current write_counter[dim[k]]). Each read also performs
    //    a write that increments the counter by 1.
    // -----------------------------------------------------------------------
    auto t_w0 = std::chrono::high_resolution_clock::now();

    std::vector<u64> ctr(N, 0);
    out_witness.read_cts.assign(M, F_ZERO);
    out_witness.final_cts.assign(N, F_ZERO);

    for (u32 k = 0; k < M; ++k) {
        u32 addr = dim[k];
        if (addr >= N) {
            std::cerr << "[memcheck] dim[" << k << "] = " << addr
                      << " out of range (N=" << N << ")\n";
            out.sound = false;
            return out;
        }
        u64 ts = ctr[addr];
        out_witness.read_cts[k] = F((long long)ts);
        ctr[addr] = ts + 1;
    }
    for (u32 j = 0; j < N; ++j) {
        out_witness.final_cts[j] = F((long long)ctr[j]);
    }

    auto t_w1 = std::chrono::high_resolution_clock::now();
    out.prover_time_s += std::chrono::duration<double>(t_w1 - t_w0).count();

    // -----------------------------------------------------------------------
    // 2. Sample (γ, τ) AFTER the witnesses exist (Fiat-Shamir order).
    //    A real implementation would absorb commitments to read_cts / final_cts
    //    into the transcript first; here we use the same CSPRNG style as the
    //    rest of the codebase.
    // -----------------------------------------------------------------------
    out.gamma = random_field();
    out.tau   = random_field();
    F tau_sq  = out.tau * out.tau;

    // -----------------------------------------------------------------------
    // 3. Build the four leaf vectors. Each is a length-2^L vector whose
    //    grand product is the multiset hash. We pad with the multiplicative
    //    identity F_ONE up to the next power of two so the GKR depth is
    //    well-defined.
    // -----------------------------------------------------------------------
    auto t_l0 = std::chrono::high_resolution_clock::now();

    auto pad_to_pow2 = [](size_t want) {
        size_t p = 1;
        while (p < want) p <<= 1;
        return p;
    };

    size_t Npad = pad_to_pow2(N);
    size_t Mpad = pad_to_pow2(M);

    std::vector<F> init_leaves(Npad,  F_ONE);
    std::vector<F> final_leaves(Npad, F_ONE);
    for (u32 j = 0; j < N; ++j) {
        F a = F((int)j);
        F v = T[j];
        F t_init  = F_ZERO;
        F t_final = out_witness.final_cts[j];
        init_leaves[j]  = hash_leaf(out.gamma, out.tau, tau_sq, a, v, t_init);
        final_leaves[j] = hash_leaf(out.gamma, out.tau, tau_sq, a, v, t_final);
    }

    std::vector<F> read_leaves(Mpad,  F_ONE);
    std::vector<F> write_leaves(Mpad, F_ONE);
    for (u32 k = 0; k < M; ++k) {
        F a = F((int)dim[k]);
        F v = E[k];
        F t_read  = out_witness.read_cts[k];
        F t_write = out_witness.read_cts[k] + F_ONE;
        read_leaves[k]  = hash_leaf(out.gamma, out.tau, tau_sq, a, v, t_read);
        write_leaves[k] = hash_leaf(out.gamma, out.tau, tau_sq, a, v, t_write);
    }

    auto t_l1 = std::chrono::high_resolution_clock::now();
    out.prover_time_s += std::chrono::duration<double>(t_l1 - t_l0).count();

    // -----------------------------------------------------------------------
    // 4. Run four grand-product GKRs. Their `claimed_product` values must
    //    satisfy H(init)·H(write) == H(read)·H(final) for the multisets to
    //    coincide.
    // -----------------------------------------------------------------------
    std::fprintf(stderr,
        "  [memcheck]   running 4 grand-product GKRs "
        "(N_pad=%zu, M_pad=%zu)…\n", Npad, Mpad);
    std::fflush(stderr);

    auto t_gp_0 = std::chrono::high_resolution_clock::now();
    out.gp_init  = prove_grand_product(init_leaves);
    auto t_gp_1 = std::chrono::high_resolution_clock::now();
    out.gp_write = prove_grand_product(write_leaves);
    auto t_gp_2 = std::chrono::high_resolution_clock::now();
    out.gp_read  = prove_grand_product(read_leaves);
    auto t_gp_3 = std::chrono::high_resolution_clock::now();
    out.gp_final = prove_grand_product(final_leaves);
    auto t_gp_4 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "  [memcheck]   GP timings: init=%.3fs write=%.3fs read=%.3fs final=%.3fs\n",
        std::chrono::duration<double>(t_gp_1 - t_gp_0).count(),
        std::chrono::duration<double>(t_gp_2 - t_gp_1).count(),
        std::chrono::duration<double>(t_gp_3 - t_gp_2).count(),
        std::chrono::duration<double>(t_gp_4 - t_gp_3).count());
    std::fflush(stderr);

    out.prover_time_s   += out.gp_init.prover_time_s
                         + out.gp_write.prover_time_s
                         + out.gp_read.prover_time_s
                         + out.gp_final.prover_time_s;
    out.verifier_time_s += out.gp_init.verifier_time_s
                         + out.gp_write.verifier_time_s
                         + out.gp_read.verifier_time_s
                         + out.gp_final.verifier_time_s;
    out.proof_size_bytes += out.gp_init.proof_size_bytes
                          + out.gp_write.proof_size_bytes
                          + out.gp_read.proof_size_bytes
                          + out.gp_final.proof_size_bytes;
    out.proof_size_bytes += 4 * F_BYTE_SIZE;  // four announced products
    out.proof_size_bytes += 2 * F_BYTE_SIZE;  // (γ, τ) sent to prover

    if (!(out.gp_init.sound && out.gp_write.sound &&
          out.gp_read.sound && out.gp_final.sound)) {
        std::cerr << "[memcheck] one of the grand-product subprotocols failed\n";
        out.sound = false;
        return out;
    }

    // -----------------------------------------------------------------------
    // 5. Verifier-side identity: H(init) · H(write) == H(read) · H(final).
    //    If false, the read-set deviates from the write-set, i.e. some E[k]
    //    did not match T[dim[k]] — soundness violation caught here.
    // -----------------------------------------------------------------------
    auto t_v0 = std::chrono::high_resolution_clock::now();
    F lhs = out.gp_init.claimed_product * out.gp_write.claimed_product;
    F rhs = out.gp_read.claimed_product * out.gp_final.claimed_product;
    if (!(lhs - rhs).isZero()) {
        std::cerr << "[memcheck] multiset identity FAILED: "
                     "H(init)·H(write) != H(read)·H(final)\n";
        out.sound = false;
        return out;
    }
    auto t_v1 = std::chrono::high_resolution_clock::now();
    out.verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();

    return out;
}

}  // namespace lasso_core
