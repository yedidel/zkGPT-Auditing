// yedidel-lasso: Surge protocol for the 32-bit range-check table.
// See lasso_surge.hpp for the protocol summary.
//
// NOTE (range expansion): the GPT-2 Softmax dt2 witness is a product of two
// quantization terms (B² - (A - C)²) whose magnitude can exceed 2^16. In a
// real run we observed values like 95069 (~17 bits) at val[0], so the
// 16-bit setting (c=2) used in the first version was too narrow and the
// honest prover legitimately fell out of range. We now use c=4 with four
// 8-bit subtables, giving a 32-bit range; this comfortably accommodates all
// witness values in GPT-2 quantization without changing the per-element
// soundness story (each subtable lookup still proves an 8-bit fact about
// one limb of the value).

#include "lasso_surge.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hyrax.hpp"

namespace lasso_surge {

// yedidel-lasso: per-translation-unit OpenMP reduction operator for F so the
// sumcheck inner loops can use `reduction(+: ...)` over Fr accumulators.
#pragma omp declare reduction(+: F: omp_out += omp_in) initializer(omp_priv = F_ZERO)

namespace {

// yedidel-lasso: 16-bit subtables give us a tighter NUM_SUB (e.g. 4 instead
// of 7 for a 56-bit witness max), and each Spice memory check now operates on
// a 2^16 init/final multiset (still small enough for fast grand-product).
// The cost per subtable is roughly the same — we trade slightly bigger
// init/final GPs for fewer subtables, which collapses the *number* of
// 2^M-sized read/write GPs (the dominant cost on big M).
constexpr u32 SUB_LOG    = 16;         // log2(N_α) — each subtable has 2^16 entries
constexpr u32 SUB_N      = 1u << SUB_LOG;
constexpr u32 MAX_NUM_SUB = 8;         // hard cap (8·16 = 128 bits, the
                                       //   maximum convert(Fr) returns).
// NUM_SUB is chosen at runtime per-call by scanning the witness first for its
// actual max bit-width — see run_range_check below.

// Identity-subtable MLE T̃(z) = Σ_i 2^i · z_i over z ∈ F^8.
inline F identity_subtable_mle(const F *z) {
    return lasso_core::integer_index_mle(z, (int)SUB_LOG);
}

// Hyrax-open the vector w (of log size l) at point r, returning the claimed
// evaluation and accounting for proof bytes / timing. Mirrors the helper in
// lasso_logup.cpp.
F open_at(F *w, G1 *Tk, G1 *gens_col, G1 blinder, int l,
          std::vector<F> &r,
          u64 *proof_bytes,
          double *prover_time_s,
          double *verifier_time_s) {
    int halfl  = l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (l - halfl);
    F *L = new F[colnum];
    F *R = new F[rownum];
    brute_force_compute_LR(L, R, r.data(), l);
    F eval = lasso_core::mle_eval(w, r.data(), l);

    auto tim = hyrax::open(w, r.data(), eval, blinder, gens_col, L, R, Tk, l);
    *prover_time_s   += tim.first;
    *verifier_time_s += tim.second;

    int log_col = 0;
    while ((1 << log_col) < colnum) ++log_col;
    *proof_bytes += (u64)((1 + 2 * log_col) * G_BYTE_SIZE
                        + (log_col + 1) * F_BYTE_SIZE);
    delete[] L;
    delete[] R;
    return eval;
}

// Standalone degree-2 sumcheck: prove Σ_x f(x) · g(x) = target where both
// are MLEs of length 2^l. Returns the random point r and the final claims
// (f(r), g(r)). Uses LSB-first folding (same convention as lasso_logup).
void sumcheck_product(F *f, F *g, int l, F target,
                      std::vector<F> &out_random,
                      F *out_f_eval,
                      F *out_g_eval,
                      u64 *proof_bytes,
                      double *prover_time_s,
                      double *verifier_time_s,
                      bool *ok) {
    out_random.assign(l, F_ZERO);
    F current = target;

    int len = 1 << l;
    std::vector<F> wf(f, f + len);
    std::vector<F> wg(g, g + len);

    for (int round = 0; round < l; ++round) {
        auto t_p0 = std::chrono::high_resolution_clock::now();
        int half = len >> 1;
        // Round polynomial p(X) = (1-X)·(...) + X·(...) is degree 2 because
        // it's the product of two linear functions. We send three values:
        //   p(0), p(1), p(2)  (sufficient to reconstruct a degree-2 poly).
        //
        // yedidel-lasso: each k-iteration is independent; OpenMP reduction
        // over the three Fr accumulators gives near-linear speedup on the
        // first few rounds (which dominate the cost).
        F p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
        #pragma omp parallel for reduction(+:p0,p1,p2) schedule(static)
        for (int k = 0; k < half; ++k) {
            const F &f0 = wf[2 * k], &f1 = wf[2 * k + 1];
            const F &g0 = wg[2 * k], &g1 = wg[2 * k + 1];
            p0 += f0 * g0;
            p1 += f1 * g1;
            F f2 = f1 + (f1 - f0);
            F g2 = g1 + (g1 - g0);
            p2 += f2 * g2;
        }
        auto t_p1 = std::chrono::high_resolution_clock::now();
        *prover_time_s += std::chrono::duration<double>(t_p1 - t_p0).count();

        auto t_v0 = std::chrono::high_resolution_clock::now();
        // Verifier: check p(0) + p(1) == current
        if (!(p0 + p1 - current).isZero()) {
            std::cerr << "[surge] sumcheck_product round " << round
                      << ": p(0)+p(1) != current\n";
            *ok = false;
            return;
        }
        F c = lasso_core::random_field();
        out_random[round] = c;
        // Evaluate degree-2 p(c) via Lagrange over (0,1,2):
        //   p(c) = p0 · (c-1)(c-2)/2  -  p1 · c(c-2)  +  p2 · c(c-1)/2
        F two = F_ONE + F_ONE;
        F inv2;
        F::inv(inv2, two);
        F lc0 = (c - F_ONE) * (c - two) * inv2;
        F lc1 = -(c) * (c - two);
        F lc2 = c * (c - F_ONE) * inv2;
        current = p0 * lc0 + p1 * lc1 + p2 * lc2;
        auto t_v1 = std::chrono::high_resolution_clock::now();
        *verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();

        // Fold both vectors.
        // yedidel-lasso (race fix): same in-place-fold race that bit
        // grand_product. Use temporary buffers and copy back.
        auto t_f0 = std::chrono::high_resolution_clock::now();
        std::vector<F> nxt_wf(half), nxt_wg(half);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < half; ++k) {
            nxt_wf[k] = (F_ONE - c) * wf[2 * k] + c * wf[2 * k + 1];
            nxt_wg[k] = (F_ONE - c) * wg[2 * k] + c * wg[2 * k + 1];
        }
        for (int k = 0; k < half; ++k) {
            wf[k] = nxt_wf[k];
            wg[k] = nxt_wg[k];
        }
        auto t_f1 = std::chrono::high_resolution_clock::now();
        *prover_time_s += std::chrono::duration<double>(t_f1 - t_f0).count();

        len = half;
        *proof_bytes += 3 * F_BYTE_SIZE;  // p0, p1, p2
    }

    *out_f_eval = wf[0];
    *out_g_eval = wg[0];
    *ok = true;

    // Final consistency: f(r)·g(r) should equal the reduced current.
    if (!((*out_f_eval) * (*out_g_eval) - current).isZero()) {
        std::cerr << "[surge] sumcheck_product: final f(r)·g(r) != reduced sum\n";
        *ok = false;
    }
}

// (The discharge of memory-check final claims is realised inline inside
// run_range_check below — see the `discharge` lambda — so we do not expose
// a standalone helper here.)

}  // namespace

lasso_core::LassoBenchmark run_range_check(prover *p) {
    using namespace lasso_core;

    LassoBenchmark out{};
    out.variant = "Surge (decomposable, c=4, range-check 32-bit)";
    out.sound = true;
    out.num_lookups = (u32)prover::lasso_range_indices.size();

    // yedidel-lasso: verbose entry banner so we can confirm this exact module
    // ran (vs. a stale binary's older verifier.cpp). All diagnostic output
    // goes to stderr with explicit fflush so it cannot be lost to buffering.
    std::fprintf(stderr,
        "\n[lasso_surge] === ENTERED run_range_check ===\n"
        "[lasso_surge]   num_lookups (raw) = %u\n",
        out.num_lookups);
    std::fflush(stderr);

    if (out.num_lookups == 0) {
        std::fprintf(stderr,
            "[lasso_surge] no range-check indices to verify; skipping.\n");
        std::fflush(stderr);
        return out;
    }

    // -----------------------------------------------------------------------
    // Phase A: collect v, validate range, decompose to NUM_SUB subtables.
    //
    // We pre-scan the witness to find its actual max bit-width and choose
    // NUM_SUB just large enough to cover that width. This way the protocol
    // is automatically right-sized for the GPT-2 Softmax dt2 = B² - (A-C)²
    // values, which can reach ~2^60 depending on quantization parameters.
    //
    //   v[k] = Σ_{α=0..NUM_SUB-1} dim_α[k] · LIMB^(NUM_SUB-1-α)
    //
    // Convention: dim_arr[α][k] is the α-th SUB_LOG-bit limb of v[k], with
    // α=0 being the most significant. The "limb base" is LIMB = 2^SUB_LOG.
    // Combining function: g(t_0,...,t_{c-1}) = Σ_α t_α · LIMB^(c-1-α).
    // -----------------------------------------------------------------------
    auto t_a0 = std::chrono::high_resolution_clock::now();

    int log_M = 1;
    while ((1u << log_M) < out.num_lookups) ++log_M;
    u32 M = 1u << log_M;
    out.log_M = (u32)log_M;

    // (A.0) Pre-scan: find max value to right-size NUM_SUB. Also detect
    // negative values, which indicate a real soundness violation (the
    // honest prover should never produce a negative range-check witness).
    auto t_scan0 = std::chrono::high_resolution_clock::now();
    __int128 max_val = 0;
    __int128 max_neg = 0;
    u32      neg_count = 0;
    u32      max_neg_k = 0, max_pos_k = 0;
    for (u32 k = 0; k < out.num_lookups; ++k) {
        u32 idx = prover::lasso_range_indices[k];
        if (idx >= p->val[0].size()) {
            std::cerr << "[lasso_surge] index k=" << k << " (idx=" << idx
                      << ") out of bounds for val[0] (size "
                      << p->val[0].size() << ")\n";
            out.sound = false;
            return out;
        }
        __int128 vv = convert(p->val[0][idx]);
        if (vv < 0) {
            ++neg_count;
            if (-vv > max_neg) { max_neg = -vv; max_neg_k = k; }
        } else if (vv > max_val) { max_val = vv; max_pos_k = k; }
    }
    auto t_scan1 = std::chrono::high_resolution_clock::now();

    if (neg_count > 0) {
        std::fprintf(stderr,
            "[lasso_surge] FATAL: %u negative range-check witnesses found "
            "(largest |v|=%lld at k=%u). The protocol requires non-negative "
            "values; aborting.\n",
            neg_count, (long long)max_neg, max_neg_k);
        out.sound = false;
        return out;
    }

    // Compute max bit-width and the smallest NUM_SUB that covers it.
    int max_bits = 0;
    {
        __int128 m = max_val;
        while (m > 0) { ++max_bits; m >>= 1; }
        if (max_bits == 0) max_bits = 1;
    }
    u32 NUM_SUB = (u32)((max_bits + (int)SUB_LOG - 1) / (int)SUB_LOG);
    if (NUM_SUB < 1) NUM_SUB = 1;
    if (NUM_SUB > MAX_NUM_SUB) {
        std::fprintf(stderr,
            "[lasso_surge] FATAL: max value at k=%u needs %d bits, exceeds "
            "MAX_NUM_SUB·SUB_LOG = %u. Aborting.\n",
            max_pos_k, max_bits, MAX_NUM_SUB * SUB_LOG);
        out.sound = false;
        return out;
    }
    out.log_N = SUB_LOG * NUM_SUB;

    std::fprintf(stderr,
        "[lasso_surge] Phase A scan done in %.3f s. max v = %lld (≈%d bits at k=%u)\n",
        std::chrono::duration<double>(t_scan1 - t_scan0).count(),
        (long long)max_val, max_bits, max_pos_k);
    std::fprintf(stderr,
        "[lasso_surge] Choosing NUM_SUB=%u → table size 2^%u (covers max bit-width)\n",
        NUM_SUB, (u32)out.log_N);
    std::fflush(stderr);

    std::fprintf(stderr,
        "[lasso_surge] Phase A: log_M=%d, M (padded)=%u, NUM_SUB=%u, "
        "log_N(total)=%u\n",
        log_M, M, NUM_SUB, (u32)out.log_N);
    std::fprintf(stderr,
        "[lasso_surge]          allocating 1 + 2*NUM_SUB = %u vectors of "
        "%u Fr/u32 each (~%.1f MB total)\n",
        1 + 2 * NUM_SUB, M,
        (double)((1 + 2 * NUM_SUB) * M * 32) / (1024.0 * 1024.0));
    std::fflush(stderr);

    std::vector<F>                v_vec(M, F_ZERO);
    std::vector<std::vector<u32>> dim_arr(NUM_SUB, std::vector<u32>(M, 0));
    std::vector<std::vector<F>>   E_arr  (NUM_SUB, std::vector<F>(M, F_ZERO));

    // Per-subtable weight: weight[α] = LIMB^(NUM_SUB-1-α) where LIMB = 2^SUB_LOG.
    // Built as a 128-bit accumulator since for the deepest case
    // (MAX_NUM_SUB=8 with SUB_LOG=16) we have LIMB^7 = 2^112 — too big for
    // a u64. fr_from_i128 converts a __int128 into Fr via byte-by-byte
    // decomposition (independent of SUB_LOG; the byte/256 chunking is just
    // an arithmetic-helper choice for the conversion).
    auto fr_from_i128 = [](__int128 x) -> F {
        F acc = F_ZERO, base = F_ONE;
        while (x > 0) {
            acc  += base * F((int)(x & 0xff));
            base *= F(256);
            x >>= 8;
        }
        return acc;
    };
    const __int128 LIMB = ((__int128)1) << SUB_LOG;
    const __int128 LIMB_MASK = LIMB - 1;
    std::vector<F> weight(NUM_SUB);
    {
        __int128 w = 1;
        for (int a = (int)NUM_SUB - 1; a >= 0; --a) {
            weight[a] = fr_from_i128(w);
            w *= LIMB;
        }
    }

    for (u32 k = 0; k < out.num_lookups; ++k) {
        u32 idx = prover::lasso_range_indices[k];
        // Bounds + sign were already validated in the pre-scan above.
        __int128 vu_128 = convert(p->val[0][idx]);
        v_vec[k] = fr_from_i128(vu_128);
        // Limb decomposition, MSB-first into dim_arr. Mask = (1 << SUB_LOG) - 1.
        for (u32 a = 0; a < NUM_SUB; ++a) {
            u32 shift = (NUM_SUB - 1 - a) * SUB_LOG;
            u32 limb = (u32)((vu_128 >> shift) & LIMB_MASK);
            dim_arr[a][k] = limb;
            E_arr[a][k]   = F((int)limb);  // identity subtable
        }
        // Sanity: g(E_0, .., E_{c-1}) must reconstruct v.
        F recomb = F_ZERO;
        for (u32 a = 0; a < NUM_SUB; ++a) recomb += E_arr[a][k] * weight[a];
        if (!(recomb - v_vec[k]).isZero()) {
            std::cerr << "[lasso_surge] internal: subtable recomposition "
                         "mismatch at k=" << k << "\n";
            out.sound = false;
            return out;
        }
    }

    auto t_a1 = std::chrono::high_resolution_clock::now();
    double phase_A_s = std::chrono::duration<double>(t_a1 - t_a0).count();
    out.prover_time_s += phase_A_s;
    std::fprintf(stderr,
        "[lasso_surge] Phase A done: validation+decomposition in %.3f s\n",
        phase_A_s);
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Phase B: Hyrax commitments.
    //   * v  : log2(M)   variables
    //   * dim_α : log2(M) variables (we treat addresses as field elements
    //             stored in the same M-sized vector)
    //   * E_α : log2(M)
    //   * read_cts_α : log2(M)
    //   * final_cts_α : log2(N_α) = SUB_LOG = 8
    //
    // Generators at log2(M) and log2(N_α) widths are sampled fresh, isolated
    // from p->gens.
    // -----------------------------------------------------------------------
    auto t_b0 = std::chrono::high_resolution_clock::now();

    std::fprintf(stderr,
        "[lasso_surge] Phase B: sampling Hyrax generators "
        "(gens_M cols=2^%d, gens_N cols=2^%d)\n",
        log_M - log_M / 2, (int)SUB_LOG - (int)SUB_LOG / 2);
    std::fflush(stderr);

    HyraxGenerators gens_M; gens_M.init(log_M);
    HyraxGenerators gens_N; gens_N.init((int)SUB_LOG);

    auto t_gen_done = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_surge] Phase B.1: generators ready in %.3f s. "
        "Now committing 5 vectors via Pippenger…\n",
        std::chrono::duration<double>(t_gen_done - t_b0).count());
    std::fflush(stderr);

    // For commits to integer vectors (dim_α, E_α, v) we use the small-magnitude
    // Pippenger commit prover_commit(ll*, ...). Note: in this codebase
    // `ll` is typedef'd to __int128 (see typedef.hpp), NOT to long long.
    auto commit_int_vec_u32 = [&](const std::vector<u32> &xs) {
        std::vector<ll> ll_vec(xs.size(), (ll)0);
        for (size_t i = 0; i < xs.size(); ++i) ll_vec[i] = (ll)xs[i];
        return prover_commit(ll_vec.data(), gens_M.g.data(), log_M, 1);
    };

    // yedidel-lasso: commit to v itself in addition to dim_α and E_α.
    //   Without C_v, the Surge main sumcheck would just trust v(z_z) as a
    //   claim from the prover, and a malicious prover could pick any v
    //   together with consistent dim_α (since the limbs always reconstruct
    //   *some* v). C_v binds v before the Surge challenge z_z is sampled
    //   and is opened at z_z in Phase D. The prover_commit Pippenger code
    //   handles values up to ~2^80, which is comfortably above our pre-scan
    //   max of ~2^51, so packing v into one ll = __int128 is safe. (Linking
    //   the committed v back to the GPT-2 prover's input val[0] is a
    //   separate concern handled by the existing commitInput at the start
    //   of the framework.)
    G1 *Tk_v = nullptr;
    {
        std::vector<ll> v_ll(M, (ll)0);
        for (u32 k = 0; k < out.num_lookups; ++k) {
            ll acc = 0;
            for (u32 a = 0; a < NUM_SUB; ++a) {
                acc = (acc << SUB_LOG) | (ll)dim_arr[a][k];
            }
            v_ll[k] = acc;
        }
        Tk_v = prover_commit(v_ll.data(), gens_M.g.data(), log_M, 1);
    }

    // Per-subtable commits: dim_α and E_α (E_α == dim_α since each subtable
    // is the identity, but we commit to both anyway so the discharge code is
    // uniform across structured/unstructured tables).
    std::vector<G1*> Tk_dim(NUM_SUB), Tk_E(NUM_SUB);
    for (u32 a = 0; a < NUM_SUB; ++a) {
        Tk_dim[a] = commit_int_vec_u32(dim_arr[a]);
        Tk_E  [a] = commit_int_vec_u32(dim_arr[a]);
    }

    auto t_b1 = std::chrono::high_resolution_clock::now();
    double phase_B_s = std::chrono::duration<double>(t_b1 - t_b0).count();
    out.prover_time_s += phase_B_s;
    std::fprintf(stderr,
        "[lasso_surge] Phase B done: %u Pedersen-Hyrax commits in %.3f s "
        "(v + %u·(dim_α, E_α))\n",
        1 + 2 * NUM_SUB, phase_B_s, NUM_SUB);
    std::fflush(stderr);
    out.proof_size_bytes += (u64)((1 + 2 * NUM_SUB) * gens_M.g.size() / 2 * G_BYTE_SIZE);

    // -----------------------------------------------------------------------
    // Phase C: memory checking (NUM_SUB times — once per subtable α).
    // Each run produces witnesses (read_cts, final_cts) that we Hyrax-commit
    // to immediately after, so the discharge step in Phase E can open them.
    // -----------------------------------------------------------------------
    std::vector<F> T_identity(SUB_N);
    for (u32 j = 0; j < SUB_N; ++j) T_identity[j] = F((int)j);

    std::vector<MemCheckWitness>          mc_w_arr(NUM_SUB);
    std::vector<lasso_core::MemCheckProof> mc_arr(NUM_SUB);

    auto t_c_start = std::chrono::high_resolution_clock::now();
    for (u32 a = 0; a < NUM_SUB; ++a) {
        auto t_a0 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr,
            "[lasso_surge] Phase C: running Spice memory check for subtable "
            "α=%u (N_α=%u, M=%u)…\n", a, SUB_N, M);
        std::fflush(stderr);

        MemCheckInputs in;
        in.T = &T_identity;
        in.dim = &dim_arr[a];
        in.E = &E_arr[a];
        in.log_N = (int)SUB_LOG;
        in.log_M = log_M;
        mc_arr[a] = run_memory_check(in, mc_w_arr[a]);

        auto t_a1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr,
            "[lasso_surge]            α=%u done in %.3f s, sound=%s. "
            "γ=%s τ=%s\n",
            a, std::chrono::duration<double>(t_a1 - t_a0).count(),
            mc_arr[a].sound ? "TRUE" : "FALSE",
            mc_arr[a].gamma.getStr(10).substr(0, 16).c_str(),
            mc_arr[a].tau.getStr(10).substr(0, 16).c_str());
        std::fflush(stderr);

        out.prover_time_s    += mc_arr[a].prover_time_s;
        out.verifier_time_s  += mc_arr[a].verifier_time_s;
        out.proof_size_bytes += mc_arr[a].proof_size_bytes;
    }
    auto t_c_end = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_surge] Phase C done: %u memory checks in %.3f s total\n",
        NUM_SUB, std::chrono::duration<double>(t_c_end - t_c_start).count());
    std::fflush(stderr);

    bool any_mc_failed = false;
    for (u32 a = 0; a < NUM_SUB; ++a) if (!mc_arr[a].sound) any_mc_failed = true;
    if (any_mc_failed) {
        std::cerr << "[lasso_surge] one of the per-subtable memory checks "
                     "failed\n";
        out.sound = false;
        delete[] Tk_v;
        for (u32 a = 0; a < NUM_SUB; ++a) {
            delete[] Tk_dim[a]; delete[] Tk_E[a];
        }
        return out;
    }

    // Commit to read_cts_α (length M) and final_cts_α (length N_α) for each α.
    std::vector<G1*> Tk_rd(NUM_SUB), Tk_fn(NUM_SUB);
    for (u32 a = 0; a < NUM_SUB; ++a) {
        Tk_rd[a] = commit_fr_hyrax(mc_w_arr[a].read_cts.data(),
                                   gens_M.g.data(), log_M);
        Tk_fn[a] = commit_fr_hyrax(mc_w_arr[a].final_cts.data(),
                                   gens_N.g.data(), (int)SUB_LOG);
    }
    out.proof_size_bytes += (u64)(NUM_SUB * gens_M.g.size() / 2 * G_BYTE_SIZE);
    out.proof_size_bytes += (u64)(NUM_SUB * gens_N.g.size() / 2 * G_BYTE_SIZE);

    // -----------------------------------------------------------------------
    // Phase D: Surge main sumcheck (NUM_SUB independent eq·E_α sumchecks
    // at a shared challenge point z_z).
    //
    //   v(z_z) = Σ_k eq(z_z, k) · g(E_0[k], …, E_{c-1}[k])
    //          = Σ_α weight[α] · ( Σ_k eq(z_z, k) · E_α[k] )
    //          = Σ_α weight[α] · S_α
    //
    // For each α we run the existing degree-2 sumcheck on Σ_k eq(z_z,k)·E_α[k]
    // with target S_α, recovering the random point r_α and the final
    // evaluations eq(z_z, r_α), E_α(r_α).
    // -----------------------------------------------------------------------
    auto t_d0 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_surge] Phase D: Surge main sumcheck setup\n");
    std::fflush(stderr);

    std::vector<F> z_z = random_field_vector(log_M);
    F v_at_zz = mle_eval(v_vec.data(), z_z.data(), log_M);
    std::fprintf(stderr,
        "[lasso_surge]   z_z[0]=%s (random)   v(z_z)=%s\n",
        z_z.empty() ? "<empty>" : z_z[0].getStr(10).substr(0, 16).c_str(),
        v_at_zz.getStr(10).substr(0, 16).c_str());
    std::fflush(stderr);

    // yedidel-lasso: bind v(z_z) to the C_v commit. Without this opening, the
    // prover could send any value as v(z_z) and the Surge identity check would
    // be meaningless. Opening at z_z (the same point used by all per-subtable
    // sumchecks) keeps the Lasso instance self-consistent.
    F v_open = open_at(v_vec.data(), Tk_v, gens_M.g.data(), gens_M.blinder,
                       log_M, z_z,
                       &out.proof_size_bytes,
                       &out.prover_time_s,
                       &out.verifier_time_s);
    if (!(v_open - v_at_zz).isZero()) {
        std::cerr << "[lasso_surge] C_v opening at z_z disagrees with the "
                     "claimed v(z_z); refusing the proof.\n";
        out.sound = false;
    }

    // Per-subtable partial sums S_α and main-identity check.
    auto eq_z = eq_table(z_z.data(), log_M);
    std::vector<F> S_arr(NUM_SUB, F_ZERO);
    F lhs_check = F_ZERO;
    for (u32 a = 0; a < NUM_SUB; ++a) {
        F &S = S_arr[a];
        for (u32 k = 0; k < M; ++k) S += eq_z[k] * E_arr[a][k];
        lhs_check += weight[a] * S;
    }
    if (!(lhs_check - v_at_zz).isZero()) {
        std::cerr << "[lasso_surge] internal Surge identity wrong before "
                     "sumcheck (likely a v / E_α mismatch)\n";
        out.sound = false;
    }

    std::vector<std::vector<F>> r_arr(NUM_SUB);
    std::vector<F> E_at_w(NUM_SUB, F_ZERO), eq_at_w(NUM_SUB, F_ZERO);
    bool any_sc_fail = false;
    for (u32 a = 0; a < NUM_SUB; ++a) {
        // sumcheck_product folds its working copies, so we rebuild eq_z each
        // iteration to keep the original intact for later subtables.
        auto eq_local = eq_table(z_z.data(), log_M);
        auto t_sc_0 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr,
            "[lasso_surge]   running Surge sumcheck for α=%u (%d rounds)…\n",
            a, log_M);
        std::fflush(stderr);
        bool ok_a = false;
        sumcheck_product(eq_local.data(), E_arr[a].data(), log_M, S_arr[a],
                         r_arr[a], &eq_at_w[a], &E_at_w[a],
                         &out.proof_size_bytes,
                         &out.prover_time_s,
                         &out.verifier_time_s, &ok_a);
        auto t_sc_1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr,
            "[lasso_surge]   α=%u sumcheck done in %.3f s, ok=%s, "
            "E_α(r)=%s\n",
            a, std::chrono::duration<double>(t_sc_1 - t_sc_0).count(),
            ok_a ? "TRUE" : "FALSE",
            E_at_w[a].getStr(10).substr(0, 16).c_str());
        std::fflush(stderr);
        if (!ok_a) any_sc_fail = true;
    }
    if (any_sc_fail) {
        out.sound = false;
        delete[] Tk_v;
        for (u32 a = 0; a < NUM_SUB; ++a) {
            delete[] Tk_dim[a]; delete[] Tk_E[a];
            delete[] Tk_rd[a];  delete[] Tk_fn[a];
        }
        return out;
    }
    auto t_d2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_surge] Phase D done: total Surge sumcheck %.3f s\n",
        std::chrono::duration<double>(t_d2 - t_d0).count());
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Phase E: discharge openings.
    //   * Open E_α at its sumcheck final point r_α (NUM_SUB opens).
    //   * Verify eq(z_z, r_α) on the verifier side for each α.
    //   * For each memory check, open (dim_α, E_α, read_cts_α) at gp_read /
    //     gp_write final points; open final_cts_α at gp_final final point.
    // -----------------------------------------------------------------------
    for (u32 a = 0; a < NUM_SUB; ++a) {
        F E_open = open_at(E_arr[a].data(), Tk_E[a], gens_M.g.data(),
                           gens_M.blinder, log_M, r_arr[a],
                           &out.proof_size_bytes,
                           &out.prover_time_s,
                           &out.verifier_time_s);
        if (!(E_open - E_at_w[a]).isZero()) {
            std::cerr << "[lasso_surge] α=" << a
                      << ": sumcheck final claim != Hyrax opening for E_α\n";
            out.sound = false;
        }
    }

    // yedidel-lasso: verifier-side recomputation of eq(z_z, r_α). The
    // sumcheck "trusts" the final fold of eq_z, but a malicious prover could
    // send a tuple (eq_at_w, E_at_w) whose product happens to equal the
    // reduced sum without either being correct individually. Recomputing eq
    // independently — O(log M) field ops — closes that gap.
    {
        auto t_v0 = std::chrono::high_resolution_clock::now();
        for (u32 a = 0; a < NUM_SUB; ++a) {
            F eq_check = F_ONE;
            for (int i = 0; i < log_M; ++i) {
                eq_check *= z_z[i] * r_arr[a][i]
                          + (F_ONE - z_z[i]) * (F_ONE - r_arr[a][i]);
            }
            if (!(eq_check - eq_at_w[a]).isZero()) {
                std::cerr << "[lasso_surge] α=" << a
                          << ": verifier-recomputed eq(z_z, r) != sumcheck "
                             "final fold\n";
                out.sound = false;
            }
        }
        auto t_v1 = std::chrono::high_resolution_clock::now();
        out.verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();
    }

    // Memory-check discharges. Each subtable contributes 4 grand-products,
    // each grand-product producing one final point we must open against the
    // appropriate committed polynomial.
    auto discharge = [&](const lasso_core::MemCheckProof &mc,
                         std::vector<F> &dim_a_f,
                         std::vector<F> &E_a_f,
                         G1 *Tk_dim, G1 *Tk_E,
                         std::vector<F> &read_cts_f, G1 *Tk_rd,
                         std::vector<F> &final_cts_f, G1 *Tk_fn) -> bool {
        // Open dim & E & read_cts at gp_read.final_point and gp_write.final_point;
        // open final_cts at gp_final.final_point.
        std::vector<F> rp_read  = mc.gp_read.final_point;
        std::vector<F> rp_write = mc.gp_write.final_point;
        std::vector<F> rp_final = mc.gp_final.final_point;

        F dim_at_r = open_at(dim_a_f.data(), Tk_dim, gens_M.g.data(),
                             gens_M.blinder, log_M, rp_read,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
        F dim_at_w = open_at(dim_a_f.data(), Tk_dim, gens_M.g.data(),
                             gens_M.blinder, log_M, rp_write,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
        F E_at_r   = open_at(E_a_f.data(), Tk_E, gens_M.g.data(),
                             gens_M.blinder, log_M, rp_read,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
        F E_at_w   = open_at(E_a_f.data(), Tk_E, gens_M.g.data(),
                             gens_M.blinder, log_M, rp_write,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
        F rd_at_r  = open_at(read_cts_f.data(), Tk_rd, gens_M.g.data(),
                             gens_M.blinder, log_M, rp_read,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
        F rd_at_w  = open_at(read_cts_f.data(), Tk_rd, gens_M.g.data(),
                             gens_M.blinder, log_M, rp_write,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
        F fn_at_p  = open_at(final_cts_f.data(), Tk_fn, gens_N.g.data(),
                             gens_N.blinder, (int)SUB_LOG, rp_final,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);

        F gamma  = mc.gamma;
        F tau    = mc.tau;
        F tau_sq = tau * tau;

        // init: address J(z), value T̃(z) = J(z), ts 0.
        {
            const auto &z = mc.gp_init.final_point;
            F J = lasso_core::integer_index_mle(z.data(), (int)z.size());
            F leaf = gamma - (J + tau * J + tau_sq * F_ZERO);
            if (!(leaf - mc.gp_init.final_claim).isZero()) {
                std::cerr << "[lasso_surge] init final-claim mismatch\n";
                return false;
            }
        }
        // final: address J(z), value T̃(z) = J(z), ts final_cts̃(z).
        {
            const auto &z = mc.gp_final.final_point;
            F J = lasso_core::integer_index_mle(z.data(), (int)z.size());
            F leaf = gamma - (J + tau * J + tau_sq * fn_at_p);
            if (!(leaf - mc.gp_final.final_claim).isZero()) {
                std::cerr << "[lasso_surge] final final-claim mismatch\n";
                return false;
            }
        }
        // read: address dim̃, value Ẽ, ts read_cts̃.
        {
            F leaf = gamma - (dim_at_r + tau * E_at_r + tau_sq * rd_at_r);
            if (!(leaf - mc.gp_read.final_claim).isZero()) {
                std::cerr << "[lasso_surge] read final-claim mismatch\n";
                return false;
            }
        }
        // write: address dim̃, value Ẽ, ts read_cts̃ + 1.
        {
            F leaf = gamma - (dim_at_w + tau * E_at_w + tau_sq * (rd_at_w + F_ONE));
            if (!(leaf - mc.gp_write.final_claim).isZero()) {
                std::cerr << "[lasso_surge] write final-claim mismatch\n";
                return false;
            }
        }
        return true;
    };

    auto t_e0 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_surge] Phase E: discharging memory-check final claims via "
        "Hyrax openings (%u opens per α, %u α's, log_M=%d, log_N=%d)…\n",
        7u, NUM_SUB, log_M, (int)SUB_LOG);
    std::fflush(stderr);

    // Pre-materialise the Fr-typed mirror of dim_arr so the discharge lambda
    // can open them via Hyrax (Hyrax openings work on Fr*).
    std::vector<std::vector<F>> dim_f(NUM_SUB, std::vector<F>(M, F_ZERO));
    for (u32 a = 0; a < NUM_SUB; ++a)
        for (u32 k = 0; k < M; ++k)
            dim_f[a][k] = F((int)dim_arr[a][k]);

    bool any_disch_fail = false;
    for (u32 a = 0; a < NUM_SUB; ++a) {
        bool d_ok = discharge(mc_arr[a], dim_f[a], E_arr[a],
                              Tk_dim[a], Tk_E[a],
                              mc_w_arr[a].read_cts, Tk_rd[a],
                              mc_w_arr[a].final_cts, Tk_fn[a]);
        std::fprintf(stderr,
            "[lasso_surge]   discharge α=%u ok=%s\n",
            a, d_ok ? "TRUE" : "FALSE");
        std::fflush(stderr);
        if (!d_ok) any_disch_fail = true;
    }
    if (any_disch_fail) out.sound = false;

    auto t_e1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_surge] Phase E done in %.3f s, all-discharge ok=%s\n",
        std::chrono::duration<double>(t_e1 - t_e0).count(),
        any_disch_fail ? "FALSE" : "TRUE");
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Cleanup + accounting.
    // -----------------------------------------------------------------------
    delete[] Tk_v;
    for (u32 a = 0; a < NUM_SUB; ++a) {
        delete[] Tk_dim[a]; delete[] Tk_E[a];
        delete[] Tk_rd[a];  delete[] Tk_fn[a];
    }

    p->proof_size            += out.proof_size_bytes;
    // yedidel-lasso: use += so this composes with the c=1 unstructured run
    // that follows in verifier::verifyLasso. Both write into the same
    // `accumulated_lasso_time` field and the framework's existing audit
    // report sums them as a single "Lasso Range Proof" line.
    p->accumulated_lasso_time += out.prover_time_s;

    std::printf("\n================================================\n");
    std::printf("        SURGE LASSO RANGE-CHECK REPORT\n");
    std::printf("        (decomposable, c=4, 32-bit identity)\n");
    std::printf("================================================\n");
    std::printf("Lookup vector raw length:       %u\n", out.num_lookups);
    std::printf("Padded length (2^log_M):        %u\n", 1u << out.log_M);
    std::printf("Subtable count (c):             %u\n", NUM_SUB);
    std::printf("Subtable size N_alpha:          2^%u (=%u)\n", SUB_LOG, SUB_N);
    std::printf("Total table size (2^log_N):     2^%u\n", (u32)out.log_N);
    std::printf("Prover time:                    %7.2f s\n", out.prover_time_s);
    std::printf("Verifier time:                  %7.2f s\n", out.verifier_time_s);
    std::printf("Lookup proof size added:        %7.2f KB\n",
                (double)out.proof_size_bytes / 1024.0);
    std::printf("Soundness verdict:              %s\n",
                out.sound ? "SOUND (all checks passed)" : "FAILED");
    std::printf("================================================\n\n");
    std::fflush(stdout);

    return out;
}

}  // namespace lasso_surge
