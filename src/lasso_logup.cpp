// Sound LogUp range-check lookup module — implementation.
// See lasso_logup.hpp for the full design rationale and protocol summary.

#include "lasso_logup.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "hyrax.hpp"

namespace lasso_logup {

namespace {

constexpr u32 RANGE_LOG = 16;          // 16-bit range
constexpr u32 RANGE_N   = 1u << RANGE_LOG;  // 2^16 = 65536

// Multilinear extension of a vector w of length 2^l, evaluated at point r.
// Computes Σ_{x∈{0,1}^l} w[x] · eq(r, x) by brute force in O(l · 2^l).
F mle_eval(const Fr *w, const Fr *r, int l) {
    return brute_force_compute_eval(const_cast<Fr*>(w), const_cast<Fr*>(r), l);
}

// MLE of the integer-index polynomial J(y) = Σ_i 2^i · y_i, evaluated at y=r.
// This is the polynomial whose value at the {0,1}^k corner indexed by j ∈ [2^k]
// is exactly j as an integer in F.
F integer_index_mle(const Fr *r, int l) {
    F acc = F_ZERO;
    F two_pow = F_ONE;
    for (int i = 0; i < l; ++i) {
        acc += two_pow * r[i];
        two_pow += two_pow;  // two_pow doubles each round
    }
    return acc;
}

// Hyrax-style row-Pedersen commitment for a vector of arbitrary Fr scalars.
// Returns Tk[2^(l/2)] suitable for use with hyrax::open(...).
//
// Note: hyrax.cpp defines a Fr-overload of `perdersen_commit` (line 214) but
// it is *not* declared in hyrax.hpp — so we cannot call it from here. We
// invoke `G1::mulVec` directly (the same primitive that internal overload
// uses) to avoid touching hyrax.hpp.
G1 *commit_fr_hyrax(Fr *w, G1 *gens_col, int l) {
    int halfl  = l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (l - halfl);
    G1 *Tk = new G1[rownum];
    for (int i = 0; i < rownum; ++i) {
        G1::mulVec(Tk[i], gens_col, w + (i * colnum), colnum);
    }
    return Tk;
}

// Random Fr vector of given length.
std::vector<Fr> random_field_vector(int n) {
    std::vector<Fr> r(n);
    for (int i = 0; i < n; ++i) r[i].setByCSPRNG();
    return r;
}

// Single Fr challenge.
Fr random_field() {
    Fr r;
    r.setByCSPRNG();
    return r;
}

// Standalone sum-check of Σ_x f(x) = target, where f has 2^l entries.
// Implements degree-1 sumcheck with O(l) rounds. On success, returns f(r)
// where r = out_random in the same little-endian variable convention used by
// `lagrange()` / `brute_force_compute_eval()`: out_random[i] corresponds to
// r_i (the i-th MLE variable, which is bit i of the index k). The bookkeeping
// folds the LSB first so this matches the existing sum_check_product helper.
//
// On failure, prints an error and returns F_ZERO; the caller should check
// the .sound flag in the surrounding LookupBenchmark.
F sumcheck_sum_to_target(Fr *f, int l, F target,
                         std::vector<Fr> &out_random,
                         u64 *proof_bytes,
                         double *prover_time_s,
                         double *verifier_time_s,
                         bool *ok) {
    out_random.assign(l, F_ZERO);
    F current_sum = target;

    // Local fold buffer (LSB-first orientation: index k's LSB == round's variable).
    int len = 1 << l;
    std::vector<Fr> work(f, f + len);

    for (int round = 0; round < l; ++round) {
        auto p_start = std::chrono::high_resolution_clock::now();
        int half = len >> 1;
        // Round-polynomial g_round(X) = (1-X)·A + X·B. The LSB of every
        // pair is the variable being summed:
        //   A = Σ_{k} work[2k]   (variable = 0)
        //   B = Σ_{k} work[2k+1] (variable = 1)
        F A = F_ZERO, B = F_ZERO;
        for (int k = 0; k < half; ++k) {
            A += work[2 * k];
            B += work[2 * k + 1];
        }
        auto p_end = std::chrono::high_resolution_clock::now();
        *prover_time_s += std::chrono::duration<double>(p_end - p_start).count();

        // Verifier: check A + B == current_sum.
        auto v_start = std::chrono::high_resolution_clock::now();
        if (!(A + B - current_sum).isZero()) {
            std::cerr << "[lasso_logup] sumcheck_sum_to_target failed at round "
                      << round << " (A+B != current_sum)\n";
            *ok = false;
            return F_ZERO;
        }
        Fr challenge = random_field();
        out_random[round] = challenge;
        current_sum = (F_ONE - challenge) * A + challenge * B;
        auto v_end = std::chrono::high_resolution_clock::now();
        *verifier_time_s += std::chrono::duration<double>(v_end - v_start).count();

        // Prover folds: collapse the LSB.
        //   new_work[k] = (1-c)·work[2k] + c·work[2k+1]
        auto pf_start = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < half; ++k) {
            work[k] = (F_ONE - challenge) * work[2 * k]
                    + challenge        * work[2 * k + 1];
        }
        auto pf_end = std::chrono::high_resolution_clock::now();
        *prover_time_s += std::chrono::duration<double>(pf_end - pf_start).count();

        len = half;

        // Each round sends (A, B): 2 field elements.
        *proof_bytes += 2 * F_BYTE_SIZE;
    }

    *ok = true;
    return work[0];  // f(out_random) in lagrange-compatible convention.
}

// Pointwise opening + Schwartz-Zippel consistency check shared between the
// LHS (g(z)·(r+f(z)) = 1) and RHS (h(z)·(r+J(z)) = m(z)) cases.
struct OpenContext {
    Fr  *w;
    G1  *Tk;
    G1  *gens_col;
    G1   blinder;       // base generator
    int  l;
};

// Run hyrax::open at point z and accumulate timings/proof bytes.
// Returns the claimed evaluation w(z) (also recomputed locally for assertion).
F open_at(const OpenContext &ctx, std::vector<Fr> &z,
          u64 *proof_bytes,
          double *prover_time_s,
          double *verifier_time_s) {
    int halfl  = ctx.l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (ctx.l - halfl);
    Fr *L = new Fr[colnum];
    Fr *R = new Fr[rownum];
    brute_force_compute_LR(L, R, z.data(), ctx.l);
    F eval = mle_eval(ctx.w, z.data(), ctx.l);

    auto t0 = std::chrono::high_resolution_clock::now();
    // hyrax::open takes the blinding generator as a non-const G1&, so we
    // need a local mutable copy here (ctx is const&, which would otherwise
    // discard the qualifier).
    G1 blinder_local = ctx.blinder;
    auto tim = hyrax::open(ctx.w, z.data(), eval, blinder_local,
                           ctx.gens_col, L, R, ctx.Tk, ctx.l);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)t0; (void)t1;

    *prover_time_s   += tim.first;
    *verifier_time_s += tim.second;

    // Hyrax opening transmits one row-commitment (G), the bullet-reduce
    // transcript (~2 log(colnum) group elements + 2 field elements), and
    // the final evaluation. We use a conservative upper bound:
    //   1 group  +  2*log(colnum) groups  +  log(colnum)+1 field elements.
    int log_col = 0;
    while ((1 << log_col) < colnum) ++log_col;
    *proof_bytes += (u64)((1 + 2 * log_col) * G_BYTE_SIZE
                        + (log_col + 1) * F_BYTE_SIZE);

    delete[] L;
    delete[] R;
    return eval;
}

}  // namespace

LookupBenchmark run(prover *p) {
    LookupBenchmark out{};
    out.sound = true;
    out.num_lookups_raw = (u32)prover::lasso_range_indices.size();

    if (out.num_lookups_raw == 0) {
        std::cout << "[lasso_logup] no range-check indices to verify; skipping.\n";
        return out;
    }

    // -------------------------------------------------------------------
    // Phase 0: Build f_vec (length 2^log_M) and counts m (length 2^16).
    // Validates each f_i ∈ [0, 2^16) using the signed `convert(Fr)` helper;
    // values outside the range cause an explicit abort instead of being
    // silently masked.
    // -------------------------------------------------------------------
    auto t_phase0 = std::chrono::high_resolution_clock::now();

    int log_M = 1;
    while ((1u << log_M) < out.num_lookups_raw) ++log_M;
    if (log_M < 1) log_M = 1;
    out.log_M = (u32)log_M;
    u32 M = 1u << log_M;

    std::vector<Fr> f_vec(M, F_ZERO);
    std::vector<ll> f_vec_ll(M, 0);
    std::vector<Fr> m_vec(RANGE_N, F_ZERO);
    std::vector<ll> m_vec_ll(RANGE_N, 0);

    for (u32 i = 0; i < out.num_lookups_raw; ++i) {
        u32 idx = prover::lasso_range_indices[i];
        if (idx >= p->val[0].size()) {
            std::cerr << "[lasso_logup] index " << idx
                      << " out of bounds for val[0] (size "
                      << p->val[0].size() << ")\n";
            out.sound = false;
            return out;
        }
        __int128 v = convert(p->val[0][idx]);
        if (v < 0 || v >= (__int128)RANGE_N) {
            std::cerr << "[lasso_logup] range-check FAILED: value at val[0]["
                      << idx << "] = " << (long long)v
                      << " is outside [0, 2^16). Aborting (no soundness "
                         "shortcut via masking).\n";
            out.sound = false;
            return out;
        }
        u32 v_u = (u32)v;
        f_vec[i]     = F((int)v_u);
        f_vec_ll[i]  = (ll)v_u;
        m_vec_ll[v_u] += 1;
    }
    for (u32 j = 0; j < RANGE_N; ++j) {
        if (m_vec_ll[j] != 0) m_vec[j] = F((long long)m_vec_ll[j]);
    }

    auto t_phase0_end = std::chrono::high_resolution_clock::now();
    double phase0_s = std::chrono::duration<double>(t_phase0_end - t_phase0).count();
    out.prover_time_s += phase0_s;

    // -------------------------------------------------------------------
    // Phase 1: Pedersen-Hyrax commit to f_vec and m. Generators are sampled
    // fresh for this lookup so we don't depend on p->gens being large enough.
    // -------------------------------------------------------------------
    auto t_phase1 = std::chrono::high_resolution_clock::now();

    int half_M = log_M / 2;
    int gens_M_len = 1 << (log_M - half_M);   // column generators length
    int half_N = RANGE_LOG / 2;               // 8
    int gens_N_len = 1 << (RANGE_LOG - half_N); // 256

    std::vector<G1> gens_M(gens_M_len);
    G1 base_M = gen_gi(gens_M.data(), gens_M_len);
    std::vector<G1> gens_N(gens_N_len);
    G1 base_N = gen_gi(gens_N.data(), gens_N_len);

    // Use the integer Pippenger commit for the small-magnitude inputs.
    G1 *Tk_f = prover_commit(f_vec_ll.data(), gens_M.data(), log_M, 1);
    G1 *Tk_m = prover_commit(m_vec_ll.data(), gens_N.data(), RANGE_LOG, 1);

    auto t_phase1_end = std::chrono::high_resolution_clock::now();
    out.prover_time_s += std::chrono::duration<double>(t_phase1_end - t_phase1).count();

    // Two row-commitments transmitted (one Tk per polynomial). Use one G1
    // element per row as the conservative size estimate.
    out.proof_size_bytes += (u64)((1 << half_M) * G_BYTE_SIZE);
    out.proof_size_bytes += (u64)((1 << half_N) * G_BYTE_SIZE);

    // -------------------------------------------------------------------
    // Phase 2: Verifier samples r AFTER seeing the commits.
    // -------------------------------------------------------------------
    auto t_phase2 = std::chrono::high_resolution_clock::now();
    Fr r_chal = random_field();
    auto t_phase2_end = std::chrono::high_resolution_clock::now();
    out.verifier_time_s += std::chrono::duration<double>(t_phase2_end - t_phase2).count();
    out.proof_size_bytes += F_BYTE_SIZE;  // r transmitted prover-ward

    // -------------------------------------------------------------------
    // Phase 3: Compute reciprocals g[i] = 1/(r+f_i), h[j] = m_j/(r+j),
    // and Pedersen-Hyrax commit them. (Arbitrary Fr values, so naive
    // per-row commit is used.)
    // -------------------------------------------------------------------
    auto t_phase3 = std::chrono::high_resolution_clock::now();

    std::vector<Fr> g_vec(M);
    std::vector<Fr> h_vec(RANGE_N, F_ZERO);

    for (u32 i = 0; i < M; ++i) {
        Fr den = r_chal + f_vec[i];
        if (den.isZero()) {
            // r + f_i = 0 happens with probability 1/|F| over uniform r;
            // treat as protocol failure and fall back. The honest prover
            // would resample r in this rare case.
            std::cerr << "[lasso_logup] r + f_i = 0 collision at i=" << i
                      << " (negligible-probability event); aborting.\n";
            out.sound = false;
            delete[] Tk_f;
            delete[] Tk_m;
            return out;
        }
        Fr inv;
        Fr::inv(inv, den);
        g_vec[i] = inv;
    }

    F S_g = F_ZERO;
    for (u32 i = 0; i < M; ++i) S_g += g_vec[i];

    F S_h = F_ZERO;
    for (u32 j = 0; j < RANGE_N; ++j) {
        if (m_vec_ll[j] == 0) continue;  // h_j = 0
        Fr den = r_chal + F((int)j);
        if (den.isZero()) {
            std::cerr << "[lasso_logup] r + j = 0 collision at j=" << j
                      << " (negligible-probability event); aborting.\n";
            out.sound = false;
            delete[] Tk_f;
            delete[] Tk_m;
            return out;
        }
        Fr inv;
        Fr::inv(inv, den);
        h_vec[j] = m_vec[j] * inv;
        S_h += h_vec[j];
    }

    G1 *Tk_g = commit_fr_hyrax(g_vec.data(), gens_M.data(), log_M);
    G1 *Tk_h = commit_fr_hyrax(h_vec.data(), gens_N.data(), RANGE_LOG);

    auto t_phase3_end = std::chrono::high_resolution_clock::now();
    out.prover_time_s += std::chrono::duration<double>(t_phase3_end - t_phase3).count();

    // Two more row-commitments + two scalar sums S_g, S_h.
    out.proof_size_bytes += (u64)((1 << half_M) * G_BYTE_SIZE);
    out.proof_size_bytes += (u64)((1 << half_N) * G_BYTE_SIZE);
    out.proof_size_bytes += 2 * F_BYTE_SIZE;

    // -------------------------------------------------------------------
    // Phase 4: Verifier checks the LogUp identity S_g == S_h.
    // -------------------------------------------------------------------
    {
        auto v0 = std::chrono::high_resolution_clock::now();
        if (!(S_g - S_h).isZero()) {
            std::cerr << "[lasso_logup] LogUp identity FAILED: S_g != S_h\n";
            out.sound = false;
            delete[] Tk_f; delete[] Tk_m;
            delete[] Tk_g; delete[] Tk_h;
            return out;
        }
        auto v1 = std::chrono::high_resolution_clock::now();
        out.verifier_time_s += std::chrono::duration<double>(v1 - v0).count();
    }

    // -------------------------------------------------------------------
    // Phase 5: Two degree-1 sumchecks prove Σ_x g(x) = S_g and Σ_y h(y) = S_h.
    // The sumcheck framework returns the polynomials' evaluations at the
    // chosen random points r_A, r_B. Those evaluations must match the
    // openings of C_g, C_h at those points (verified in Phase 7).
    // -------------------------------------------------------------------
    std::vector<Fr> r_A, r_B;
    bool ok_A = true, ok_B = true;

    F g_at_rA = sumcheck_sum_to_target(g_vec.data(), log_M, S_g,
                                       r_A,
                                       &out.proof_size_bytes,
                                       &out.prover_time_s,
                                       &out.verifier_time_s,
                                       &ok_A);
    if (!ok_A) {
        out.sound = false;
        delete[] Tk_f; delete[] Tk_m; delete[] Tk_g; delete[] Tk_h;
        return out;
    }

    F h_at_rB = sumcheck_sum_to_target(h_vec.data(), (int)RANGE_LOG, S_h,
                                       r_B,
                                       &out.proof_size_bytes,
                                       &out.prover_time_s,
                                       &out.verifier_time_s,
                                       &ok_B);
    if (!ok_B) {
        out.sound = false;
        delete[] Tk_f; delete[] Tk_m; delete[] Tk_g; delete[] Tk_h;
        return out;
    }

    // -------------------------------------------------------------------
    // Phase 6: Sample Schwartz-Zippel points z_A, z_B AFTER all sumcheck
    // randomness has been issued. The pointwise consistency identities
    //   g(z_A) · (r + f(z_A)) == 1
    //   h(z_B) · (r + J(z_B)) == m(z_B)
    // are MLE-identities that hold at every {0,1}^k corner iff the
    // reciprocals are well-formed; checking at a single random point
    // catches any violation with soundness error ≤ 2k/|F|.
    // -------------------------------------------------------------------
    std::vector<Fr> z_A = random_field_vector(log_M);
    std::vector<Fr> z_B = random_field_vector(RANGE_LOG);

    OpenContext ctx_g{ g_vec.data(), Tk_g, gens_M.data(), base_M, log_M };
    OpenContext ctx_f{ f_vec.data(), Tk_f, gens_M.data(), base_M, log_M };
    OpenContext ctx_h{ h_vec.data(), Tk_h, gens_N.data(), base_N, (int)RANGE_LOG };
    OpenContext ctx_m{ m_vec.data(), Tk_m, gens_N.data(), base_N, (int)RANGE_LOG };

    // Phase 7a — LHS pointwise consistency: g(z_A)·(r + f(z_A)) == 1.
    F g_at_zA = open_at(ctx_g, z_A,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
    F f_at_zA = open_at(ctx_f, z_A,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
    {
        auto v0 = std::chrono::high_resolution_clock::now();
        F lhs = g_at_zA * (r_chal + f_at_zA);
        if (!(lhs - F_ONE).isZero()) {
            std::cerr << "[lasso_logup] LHS pointwise consistency FAILED: "
                         "g(z_A)·(r+f(z_A)) != 1\n";
            out.sound = false;
            delete[] Tk_f; delete[] Tk_m; delete[] Tk_g; delete[] Tk_h;
            return out;
        }
        auto v1 = std::chrono::high_resolution_clock::now();
        out.verifier_time_s += std::chrono::duration<double>(v1 - v0).count();
    }

    // Phase 7b — RHS pointwise consistency: h(z_B)·(r + J(z_B)) == m(z_B).
    F h_at_zB = open_at(ctx_h, z_B,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
    F m_at_zB = open_at(ctx_m, z_B,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
    {
        auto v0 = std::chrono::high_resolution_clock::now();
        F J_at_zB = integer_index_mle(z_B.data(), (int)RANGE_LOG);
        F lhs = h_at_zB * (r_chal + J_at_zB);
        if (!(lhs - m_at_zB).isZero()) {
            std::cerr << "[lasso_logup] RHS pointwise consistency FAILED: "
                         "h(z_B)·(r+J(z_B)) != m(z_B)\n";
            out.sound = false;
            delete[] Tk_f; delete[] Tk_m; delete[] Tk_g; delete[] Tk_h;
            return out;
        }
        auto v1 = std::chrono::high_resolution_clock::now();
        out.verifier_time_s += std::chrono::duration<double>(v1 - v0).count();
    }

    // -------------------------------------------------------------------
    // Phase 7c: link the sumcheck-final evaluations g(r_A) and h(r_B) to
    // the openings. Because we already opened g at z_A and h at z_B above,
    // we now also open them at r_A / r_B respectively to discharge the
    // sumcheck final claims. (A production implementation would batch these
    // into a single opening per polynomial via random linear combination;
    // we keep them separate here for clarity.)
    // -------------------------------------------------------------------
    F g_at_rA_open = open_at(ctx_g, r_A,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
    F h_at_rB_open = open_at(ctx_h, r_B,
                             &out.proof_size_bytes,
                             &out.prover_time_s,
                             &out.verifier_time_s);
    {
        auto v0 = std::chrono::high_resolution_clock::now();
        if (!(g_at_rA_open - g_at_rA).isZero()) {
            std::cerr << "[lasso_logup] sumcheck-A final claim does not match "
                         "C_g opening at r_A\n";
            out.sound = false;
            delete[] Tk_f; delete[] Tk_m; delete[] Tk_g; delete[] Tk_h;
            return out;
        }
        if (!(h_at_rB_open - h_at_rB).isZero()) {
            std::cerr << "[lasso_logup] sumcheck-B final claim does not match "
                         "C_h opening at r_B\n";
            out.sound = false;
            delete[] Tk_f; delete[] Tk_m; delete[] Tk_g; delete[] Tk_h;
            return out;
        }
        auto v1 = std::chrono::high_resolution_clock::now();
        out.verifier_time_s += std::chrono::duration<double>(v1 - v0).count();
    }

    // Cleanup.
    delete[] Tk_f;
    delete[] Tk_m;
    delete[] Tk_g;
    delete[] Tk_h;

    // Update prover-side accounting that the rest of the framework uses.
    p->proof_size           += out.proof_size_bytes;
    p->accumulated_lasso_time = out.prover_time_s;

    std::printf("\n================================================\n");
    std::printf("        SOUND LogUp RANGE-CHECK REPORT\n");
    std::printf("================================================\n");
    std::printf("Lookup vector raw length (M_raw):  %u\n",  out.num_lookups_raw);
    std::printf("Padded length (2^log_M):           %u\n",  1u << out.log_M);
    std::printf("Table N = 2^16:                    %u\n",  RANGE_N);
    std::printf("Prover time:                       %7.2f s\n", out.prover_time_s);
    std::printf("Verifier time:                     %7.2f s\n", out.verifier_time_s);
    std::printf("Lookup proof size added:           %7.2f KB\n",
                (double)out.proof_size_bytes / 1024.0);
    std::printf("Soundness verdict:                 %s\n",
                out.sound ? "SOUND (all checks passed)" : "FAILED");
    std::printf("================================================\n\n");
    std::fflush(stdout);

    return out;
}

}  // namespace lasso_logup
