// yedidel-lasso: c=1 (unstructured) Lasso — implementation.
// See lasso_unstructured.hpp for the protocol summary.

#include "lasso_unstructured.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "hyrax.hpp"
#include "lasso_memcheck.hpp"

namespace lasso_unstructured {

// yedidel-lasso: per-translation-unit OpenMP reduction operator for F.
#pragma omp declare reduction(+: F: omp_out += omp_in) initializer(omp_priv = F_ZERO)

// yedidel-lasso (adv-test): mirror of the harness in lasso_surge.cpp.
// Recognised LASSO_ADV_TEST values that this module handles:
//     WRONG-T-EXP — tamper with one entry of the exp-table polynomial
//                   AFTER the Hyrax commit was taken on its honest value.
//                   The init / final multiset hashes captured the original
//                   T value, but the discharge re-opens T at a random
//                   point against the tampered local copy, so the leaf
//                   reconstruction must mismatch.
namespace {
inline std::string adv_test_mode() {
    const char *e = std::getenv("LASSO_ADV_TEST");
    return e ? std::string(e) : "";
}
inline bool adv_is(const char *name) { return adv_test_mode() == name; }
}  // namespace

namespace {

// Hyrax open helper, identical in shape to the one in lasso_surge.cpp;
// duplicated here to keep the modules self-contained without a new header.
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

// Degree-2 sumcheck for Σ_x f(x) · g(x) = target. Identical in spirit to the
// helper in lasso_surge.cpp; kept local to avoid a cross-module header.
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
        // yedidel-lasso: parallelised — see lasso_surge.cpp for the same
        // pattern.
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
        if (!(p0 + p1 - current).isZero()) {
            std::cerr << "[unstructured] sumcheck round " << round
                      << ": p(0)+p(1) != current\n";
            *ok = false;
            return;
        }
        F c = lasso_core::random_field();
        out_random[round] = c;
        F two = F_ONE + F_ONE;
        F inv2; F::inv(inv2, two);
        F lc0 = (c - F_ONE) * (c - two) * inv2;
        F lc1 = -(c) * (c - two);
        F lc2 = c * (c - F_ONE) * inv2;
        current = p0 * lc0 + p1 * lc1 + p2 * lc2;
        auto t_v1 = std::chrono::high_resolution_clock::now();
        *verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();

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
        *proof_bytes += 3 * F_BYTE_SIZE;
    }

    *out_f_eval = wf[0];
    *out_g_eval = wg[0];
    *ok = true;

    if (!((*out_f_eval) * (*out_g_eval) - current).isZero()) {
        std::cerr << "[unstructured] sumcheck final f(r)·g(r) != reduced\n";
        *ok = false;
    }
}

}  // namespace

lasso_core::LassoBenchmark run_exp_lookup(prover *p,
                                          const std::vector<LookupPair> &pairs,
                                          const std::vector<F> &T_padded,
                                          int log_N) {
    using namespace lasso_core;

    LassoBenchmark out{};
    out.variant = "Unstructured Lasso (c=1, exp table)";
    out.sound = true;
    out.num_lookups = (u32)pairs.size();

    std::fprintf(stderr,
        "\n[lasso_unstructured] === ENTERED run_exp_lookup ===\n"
        "[lasso_unstructured]   pairs.size()    = %u\n"
        "[lasso_unstructured]   T_padded.size() = %zu\n"
        "[lasso_unstructured]   log_N (caller)  = %d\n",
        out.num_lookups, T_padded.size(), log_N);
    std::fflush(stderr);

    if (out.num_lookups == 0) {
        std::fprintf(stderr,
            "[lasso_unstructured] no exp-table pairs to verify; skipping.\n");
        std::fflush(stderr);
        return out;
    }

    u32 N = (u32)T_padded.size();
    if (N != (1u << log_N)) {
        std::fprintf(stderr,
            "[lasso_unstructured] T_padded size != 2^log_N (size=%u, "
            "expected=%u). The exp table was either not initialised or "
            "padded incorrectly. Skipping c=1 Lasso.\n",
            N, 1u << log_N);
        std::fflush(stderr);
        out.sound = false;
        return out;
    }

    int log_M = 1;
    while ((1u << log_M) < out.num_lookups) ++log_M;
    u32 M = 1u << log_M;
    out.log_M = (u32)log_M;
    out.log_N = (u32)log_N;
    std::fprintf(stderr,
        "[lasso_unstructured] log_M=%d (M=%u), log_N=%d (N=%u)\n",
        log_M, M, log_N, N);
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Phase A: build dim, E from the (t_idx, E_idx) pairs in p->val[0].
    //          Validate dim[k] ∈ [0, N) and E[k] == T[dim[k]] (the latter is
    //          enforced cryptographically below; we sanity-check here so the
    //          honest prover catches issues early instead of failing memcheck).
    // -----------------------------------------------------------------------
    auto t_a0 = std::chrono::high_resolution_clock::now();

    // yedidel-lasso: padding entries (k >= num_lookups) must look like
    // *consistent* reads of address 0 — that is (dim=0, E=T[0]) — so the
    // Spice memory-checking multiset identity stays balanced. The previous
    // version padded with (0, F_ZERO), which only worked when T[0] == 0
    // (true for the identity subtables in Surge but false for the exp table
    // in unstructured Lasso, where T[0] = round(exp(0)/Se) = 2^20). The
    // mismatch caused H(init)·H(write) ≠ H(read)·H(final) at runtime.
    std::vector<u32> dim(M, 0);
    std::vector<F>   E_f(M, T_padded[0]);  // pad with T[0] for memcheck consistency

    for (u32 k = 0; k < out.num_lookups; ++k) {
        const auto &p_pair = pairs[k];
        if (p_pair.t_idx >= p->val[0].size() ||
            p_pair.E_idx >= p->val[0].size()) {
            std::cerr << "[lasso_unstructured] pair index out of bounds\n";
            out.sound = false;
            return out;
        }
        __int128 t_val = convert(p->val[0][p_pair.t_idx]);
        if (t_val < 0 || t_val >= (__int128)N) {
            std::cerr << "[lasso_unstructured] t_idx value " << (long long)t_val
                      << " outside [0, " << N << ")\n";
            out.sound = false;
            return out;
        }
        u32 t_u = (u32)t_val;
        F   E_v = p->val[0][p_pair.E_idx];
        // Sanity (honest prover): E should equal table[t].
        if (!(E_v - T_padded[t_u]).isZero()) {
            std::cerr << "[lasso_unstructured] honest-prover sanity FAILED at "
                         "pair " << k << " (E != T[t])\n";
            out.sound = false;
            return out;
        }
        dim[k] = t_u;
        E_f[k] = E_v;
    }

    auto t_a1 = std::chrono::high_resolution_clock::now();
    double phase_A_s = std::chrono::duration<double>(t_a1 - t_a0).count();
    out.prover_time_s += phase_A_s;
    std::fprintf(stderr,
        "[lasso_unstructured] Phase A done: %u pair validations in %.3f s\n",
        out.num_lookups, phase_A_s);
    std::fflush(stderr);

    // -----------------------------------------------------------------------
    // Phase B: Hyrax commitments to dim, E, and T.
    // -----------------------------------------------------------------------
    auto t_b0 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase B: sampling generators (M=%d, N=%d) and "
        "committing dim, E, T\n", log_M, log_N);
    std::fflush(stderr);

    HyraxGenerators gens_M; gens_M.init(log_M);
    HyraxGenerators gens_N; gens_N.init(log_N);

    // yedidel-lasso: `ll` here is __int128 per typedef.hpp, not long long.
    std::vector<ll> dim_ll(M, (ll)0);
    for (u32 k = 0; k < M; ++k) dim_ll[k] = (ll)dim[k];
    G1 *Tk_dim = prover_commit(dim_ll.data(), gens_M.g.data(), log_M, 1);

    std::vector<F> E_local(E_f);  // commit_fr_hyrax mutates? No — but be safe.
    G1 *Tk_E = commit_fr_hyrax(E_local.data(), gens_M.g.data(), log_M);

    std::vector<F> T_local(T_padded);

    // yedidel-lasso (adv-test:WRONG-T-EXP): tamper with one entry of T_local
    // BEFORE the Hyrax commit. Spice memory checking still sees the
    // honest T_padded (its const& parameter), so the init / final
    // multisets capture the ORIGINAL T values; the commit on T_local
    // captures the tampered values. In Phase E discharge, the open of
    // Tk_T returns the tampered MLE, but mc.gp_init.final_claim comes
    // from the honest-T multiset, so the leaf reconstruction must
    // mismatch -- graceful rejection without firing a Hyrax binding
    // assertion.
    if (adv_is("WRONG-T-EXP") && N > 1) {
        T_local[1] = T_local[1] + F_ONE;
        std::fprintf(stderr,
            "[adv-test:WRONG-T-EXP] Tampered T_local[1] += 1 before Hyrax commit\n");
        std::fflush(stderr);
    }

    G1 *Tk_T = commit_fr_hyrax(T_local.data(), gens_N.g.data(), log_N);

    auto t_b1 = std::chrono::high_resolution_clock::now();
    double phase_B_s = std::chrono::duration<double>(t_b1 - t_b0).count();
    out.prover_time_s += phase_B_s;
    std::fprintf(stderr,
        "[lasso_unstructured] Phase B done: 3 commits (dim, E, T) in %.3f s\n",
        phase_B_s);
    std::fflush(stderr);
    out.proof_size_bytes += (u64)(3 * gens_M.g.size() / 2 * G_BYTE_SIZE);
    out.proof_size_bytes += (u64)(gens_N.g.size() / 2 * G_BYTE_SIZE);

    // -----------------------------------------------------------------------
    // Phase C: Spice memory checking on (T, dim, E).
    // -----------------------------------------------------------------------
    MemCheckInputs in;
    in.T = &T_padded;
    in.dim = &dim;
    in.E = &E_f;
    in.log_N = log_N;
    in.log_M = log_M;

    auto t_c0 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase C: running Spice memory check "
        "(M=%u, N=%u)…\n", M, N);
    std::fflush(stderr);
    MemCheckWitness mc_w;
    auto mc = run_memory_check(in, mc_w);
    auto t_c1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase C done in %.3f s, sound=%s. "
        "γ=%s τ=%s\n",
        std::chrono::duration<double>(t_c1 - t_c0).count(),
        mc.sound ? "TRUE" : "FALSE",
        mc.gamma.getStr(10).substr(0, 16).c_str(),
        mc.tau.getStr(10).substr(0, 16).c_str());
    std::fflush(stderr);

    out.prover_time_s    += mc.prover_time_s;
    out.verifier_time_s  += mc.verifier_time_s;
    out.proof_size_bytes += mc.proof_size_bytes;

    if (!mc.sound) {
        std::cerr << "[lasso_unstructured] memory check failed\n";
        out.sound = false;
        delete[] Tk_dim; delete[] Tk_E; delete[] Tk_T;
        return out;
    }

    G1 *Tk_rd = commit_fr_hyrax(mc_w.read_cts.data(), gens_M.g.data(), log_M);
    G1 *Tk_fn = commit_fr_hyrax(mc_w.final_cts.data(), gens_N.g.data(), log_N);
    out.proof_size_bytes += (u64)(gens_M.g.size() / 2 * G_BYTE_SIZE);
    out.proof_size_bytes += (u64)(gens_N.g.size() / 2 * G_BYTE_SIZE);

    // -----------------------------------------------------------------------
    // Phase D: Surge main sumcheck for c=1.
    //   E(z_z) = Σ_k eq(z_z, k) · E[k]
    // After the sumcheck, open E at the final point and check the identity.
    // -----------------------------------------------------------------------
    auto t_d0 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase D: c=1 main sumcheck (%d rounds)\n",
        log_M);
    std::fflush(stderr);

    std::vector<F> z_z = random_field_vector(log_M);
    F E_at_zz = mle_eval(E_f.data(), z_z.data(), log_M);
    std::fprintf(stderr,
        "[lasso_unstructured]   z_z[0]=%s   E(z_z)=%s\n",
        z_z.empty() ? "<empty>" : z_z[0].getStr(10).substr(0, 16).c_str(),
        E_at_zz.getStr(10).substr(0, 16).c_str());
    std::fflush(stderr);

    auto eq_z = eq_table(z_z.data(), log_M);
    F S_check = F_ZERO;
    for (u32 k = 0; k < M; ++k) S_check += eq_z[k] * E_f[k];
    if (!(S_check - E_at_zz).isZero()) {
        std::cerr << "[lasso_unstructured] internal: E(z_z) != Σ eq·E (likely "
                     "a bookkeeping bug)\n";
        out.sound = false;
    }

    std::vector<F> r_E;
    F E_final = F_ZERO, eq_final = F_ZERO;
    bool ok = false;
    sumcheck_product(eq_z.data(), E_f.data(), log_M, S_check,
                     r_E, &eq_final, &E_final,
                     &out.proof_size_bytes,
                     &out.prover_time_s,
                     &out.verifier_time_s, &ok);
    auto t_d1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase D done in %.3f s, ok=%s, "
        "E(r)=%s\n",
        std::chrono::duration<double>(t_d1 - t_d0).count(),
        ok ? "TRUE" : "FALSE",
        E_final.getStr(10).substr(0, 16).c_str());
    std::fflush(stderr);
    if (!ok) {
        out.sound = false;
        delete[] Tk_dim; delete[] Tk_E; delete[] Tk_T;
        delete[] Tk_rd; delete[] Tk_fn;
        return out;
    }

    F E_open = open_at(E_f.data(), Tk_E, gens_M.g.data(), gens_M.blinder,
                       log_M, r_E,
                       &out.proof_size_bytes,
                       &out.prover_time_s,
                       &out.verifier_time_s);
    if (!(E_open - E_final).isZero()) {
        std::cerr << "[lasso_unstructured] sumcheck final claim != Hyrax "
                     "opening for E\n";
        out.sound = false;
    }

    // yedidel-lasso: verifier-side recomputation of eq(z_z, r_E). Same
    // soundness gap closure as in lasso_surge: a malicious prover could send
    // (eq_final, E_final) whose product matches the reduced sum without
    // either being correct individually. O(log M) field ops to recompute.
    {
        auto t_v0 = std::chrono::high_resolution_clock::now();
        F eq_check = F_ONE;
        for (int i = 0; i < log_M; ++i) {
            eq_check *= z_z[i] * r_E[i] + (F_ONE - z_z[i]) * (F_ONE - r_E[i]);
        }
        if (!(eq_check - eq_final).isZero()) {
            std::cerr << "[lasso_unstructured] verifier-recomputed eq(z_z, r) "
                         "does not match sumcheck final fold\n";
            out.sound = false;
        }
        auto t_v1 = std::chrono::high_resolution_clock::now();
        out.verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();
    }

    // -----------------------------------------------------------------------
    // Phase E: discharge memory-check final claims.
    //   For each multiset (init / read / write / final) the leaf at the
    //   GKR final point has the form
    //       gamma - (address + tau·value + tau²·timestamp)
    //   We need to recompute that leaf using committed openings.
    //
    //   Address column:
    //     init / final : J(point)        (analytic integer-index MLE)
    //     read / write : dim̃(point)      (Hyrax open of C_dim)
    //   Value column:
    //     init / final : T̃(point)        (Hyrax open of C_T)
    //     read / write : Ẽ(point)        (Hyrax open of C_E)
    //   Timestamp column:
    //     init         : 0
    //     final        : final_cts̃(point)
    //     read         : read_cts̃(point)
    //     write        : read_cts̃(point) + 1
    // -----------------------------------------------------------------------
    F gamma = mc.gamma;
    F tau   = mc.tau;
    F tau_sq = tau * tau;

    auto check_init = [&]() {
        std::vector<F> z = mc.gp_init.final_point;
        F J  = lasso_core::integer_index_mle(z.data(), (int)z.size());
        F Tv = open_at(T_local.data(), Tk_T, gens_N.g.data(),
                       gens_N.blinder, log_N, z,
                       &out.proof_size_bytes,
                       &out.prover_time_s,
                       &out.verifier_time_s);
        F leaf = gamma - (J + tau * Tv + tau_sq * F_ZERO);
        if (!(leaf - mc.gp_init.final_claim).isZero()) {
            std::cerr << "[lasso_unstructured] init multiset mismatch\n";
            return false;
        }
        return true;
    };

    auto check_final_set = [&]() {
        std::vector<F> z = mc.gp_final.final_point;
        F J  = lasso_core::integer_index_mle(z.data(), (int)z.size());
        F Tv = open_at(T_local.data(), Tk_T, gens_N.g.data(),
                       gens_N.blinder, log_N, z,
                       &out.proof_size_bytes,
                       &out.prover_time_s,
                       &out.verifier_time_s);
        F fnv = open_at(mc_w.final_cts.data(), Tk_fn, gens_N.g.data(),
                        gens_N.blinder, log_N, z,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
        F leaf = gamma - (J + tau * Tv + tau_sq * fnv);
        if (!(leaf - mc.gp_final.final_claim).isZero()) {
            std::cerr << "[lasso_unstructured] final multiset mismatch\n";
            return false;
        }
        return true;
    };

    auto check_read = [&]() {
        std::vector<F> z = mc.gp_read.final_point;
        std::vector<F> dim_f(M);
        for (u32 k = 0; k < M; ++k) dim_f[k] = F((int)dim[k]);
        F dimv = open_at(dim_f.data(), Tk_dim, gens_M.g.data(),
                         gens_M.blinder, log_M, z,
                         &out.proof_size_bytes,
                         &out.prover_time_s,
                         &out.verifier_time_s);
        F Ev = open_at(E_f.data(), Tk_E, gens_M.g.data(),
                       gens_M.blinder, log_M, z,
                       &out.proof_size_bytes,
                       &out.prover_time_s,
                       &out.verifier_time_s);
        F rdv = open_at(mc_w.read_cts.data(), Tk_rd, gens_M.g.data(),
                        gens_M.blinder, log_M, z,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
        F leaf = gamma - (dimv + tau * Ev + tau_sq * rdv);
        if (!(leaf - mc.gp_read.final_claim).isZero()) {
            std::cerr << "[lasso_unstructured] read multiset mismatch\n";
            return false;
        }
        return true;
    };

    auto check_write = [&]() {
        std::vector<F> z = mc.gp_write.final_point;
        std::vector<F> dim_f(M);
        for (u32 k = 0; k < M; ++k) dim_f[k] = F((int)dim[k]);
        F dimv = open_at(dim_f.data(), Tk_dim, gens_M.g.data(),
                         gens_M.blinder, log_M, z,
                         &out.proof_size_bytes,
                         &out.prover_time_s,
                         &out.verifier_time_s);
        F Ev = open_at(E_f.data(), Tk_E, gens_M.g.data(),
                       gens_M.blinder, log_M, z,
                       &out.proof_size_bytes,
                       &out.prover_time_s,
                       &out.verifier_time_s);
        F rdv = open_at(mc_w.read_cts.data(), Tk_rd, gens_M.g.data(),
                        gens_M.blinder, log_M, z,
                        &out.proof_size_bytes,
                        &out.prover_time_s,
                        &out.verifier_time_s);
        F leaf = gamma - (dimv + tau * Ev + tau_sq * (rdv + F_ONE));
        if (!(leaf - mc.gp_write.final_claim).isZero()) {
            std::cerr << "[lasso_unstructured] write multiset mismatch\n";
            return false;
        }
        return true;
    };

    auto t_e0 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase E: discharging 4 memory-check final claims "
        "(opens of T, dim, E, read_cts, final_cts at GKR final points)…\n");
    std::fflush(stderr);
    bool ci = check_init();
    bool cf = check_final_set();
    bool cr = check_read();
    bool cw = check_write();
    auto t_e1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "[lasso_unstructured] Phase E done in %.3f s, "
        "init=%s final=%s read=%s write=%s\n",
        std::chrono::duration<double>(t_e1 - t_e0).count(),
        ci ? "OK" : "FAIL", cf ? "OK" : "FAIL",
        cr ? "OK" : "FAIL", cw ? "OK" : "FAIL");
    std::fflush(stderr);
    if (!(ci && cf && cr && cw)) {
        out.sound = false;
    }

    // -----------------------------------------------------------------------
    // Cleanup + accounting.
    // -----------------------------------------------------------------------
    delete[] Tk_dim; delete[] Tk_E; delete[] Tk_T;
    delete[] Tk_rd;  delete[] Tk_fn;

    p->proof_size           += out.proof_size_bytes;
    p->accumulated_lasso_time += out.prover_time_s;

    std::printf("\n================================================\n");
    std::printf("    UNSTRUCTURED LASSO REPORT (c=1, exp table)\n");
    std::printf("================================================\n");
    std::printf("Lookup pair count:              %u\n", out.num_lookups);
    std::printf("Padded length (2^log_M):        %u\n", 1u << out.log_M);
    std::printf("Table size (2^log_N):           %u\n", 1u << out.log_N);
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

}  // namespace lasso_unstructured
