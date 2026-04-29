// lasso-fork: Grand-Product GKR — implementation (corrected).
// See lasso_grand_product.hpp for the protocol summary and rationale.
//
// IMPORTANT (lasso-fork, post-mortem):
//   The first version of this file performed a naive layer-by-layer reduction
//   that only verified  a · b == c  pointwise on {0,1}, which is sound for
//   the boolean-cube corners but FALSE at random field points used during the
//   GKR descent. The product of two MLEs is *not* the MLE of the pointwise
//   product (cross-terms of the form γ·(1-γ)·v[i]·v[j] appear). This rewrite
//   replaces the naive check with the standard Thaler13 construction:
//
//     V_k(z_k) = sum_{j ∈ {0,1}^k} eq(z_k, j) · V_{k+1}(0, j) · V_{k+1}(1, j)
//
//   which is a degree-3 sumcheck of k rounds. After the sumcheck, two
//   evaluations V_{k+1}(0, w) and V_{k+1}(1, w) are reduced to one evaluation
//   V_{k+1}(γ, w) using a verifier-chosen γ ∈ F. Total transcript across L
//   layers is O(L²) field elements, which is fine for our use cases
//   (L ≤ 23 → ~265 sumcheck rounds total per grand product).

#include "lasso_grand_product.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lasso_core {

// lasso-fork: per-translation-unit OpenMP reduction operator for F (mcl::Fr).
// Lets us write `#pragma omp parallel for reduction(+:acc)` over field-element
// accumulators in the sumcheck inner loops below.
#pragma omp declare reduction(+: F: omp_out += omp_in) initializer(omp_priv = F_ZERO)

namespace {

// Build the binary product tree bottom-up.
//   layers[L]   = leaves (copy of v),   size 2^L
//   layers[k]   = pairwise products,    size 2^k
//   layers[0]   = root (single value),  size 1
//
// lasso-fork: each level's pairwise products are independent so OpenMP
// gives near-linear speedup for the lower levels (which dominate the cost).
std::vector<std::vector<F>> build_product_tree(const std::vector<F> &v) {
    int L = 0;
    while ((1u << L) < v.size()) ++L;
    assert((1u << L) == v.size() &&
           "grand_product: input length must be a power of two");

    std::vector<std::vector<F>> layers(L + 1);
    layers[L] = v;
    for (int k = L - 1; k >= 0; --k) {
        int sz = 1 << k;
        layers[k].resize(sz);
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < sz; ++j) {
            layers[k][j] = layers[k + 1][2 * j] * layers[k + 1][2 * j + 1];
        }
    }
    return layers;
}

// Multilinear-equality table: returns out[k] = eq(z, k) for k ∈ {0,1}^|z|,
// matching the LSB-first bit convention used by `lagrange()` in hyrax.cpp.
std::vector<F> make_eq_table(const std::vector<F> &z) {
    int l = (int)z.size();
    std::vector<F> out(1ULL << l, F_ZERO);
    out[0] = F_ONE;
    for (int i = 0; i < l; ++i) {
        int half = 1 << i;
        for (int k = half - 1; k >= 0; --k) {
            F v = out[k];
            out[k]        = v * (F_ONE - z[i]);
            out[k + half] = v * z[i];
        }
    }
    return out;
}

// Verifier-side direct evaluation of eq(z, w) = Π_i (z_i w_i + (1-z_i)(1-w_i))
// in O(|z|) field operations.
F eq_direct(const std::vector<F> &z, const std::vector<F> &w) {
    assert(z.size() == w.size());
    F acc = F_ONE;
    for (size_t i = 0; i < z.size(); ++i) {
        acc *= z[i] * w[i] + (F_ONE - z[i]) * (F_ONE - w[i]);
    }
    return acc;
}

// Lagrange interpolation of a degree-3 polynomial g, given its values at
// X ∈ {0, 1, 2, 3}, evaluated at point t. Returns g(t).
F deg3_eval(const F &g0, const F &g1, const F &g2, const F &g3, const F &t) {
    // Lagrange basis L_i(t) for nodes {0,1,2,3}:
    //   L_0 = (t-1)(t-2)(t-3) / (-6)
    //   L_1 = t(t-2)(t-3)     /   2
    //   L_2 = t(t-1)(t-3)     /  (-2)
    //   L_3 = t(t-1)(t-2)     /   6
    F t1 = t - F_ONE;
    F t2 = t - (F_ONE + F_ONE);
    F t3 = t - (F_ONE + F_ONE + F_ONE);

    F six = F_ONE + F_ONE + F_ONE + F_ONE + F_ONE + F_ONE;  // 6
    F two = F_ONE + F_ONE;
    F inv6, inv2;
    F::inv(inv6, six);
    F::inv(inv2, two);
    F neg_inv6 = F_ZERO - inv6;
    F neg_inv2 = F_ZERO - inv2;

    F l0 =  (t1 * t2 * t3) * neg_inv6;
    F l1 =  (t  * t2 * t3) * inv2;
    F l2 =  (t  * t1 * t3) * neg_inv2;
    F l3 =  (t  * t1 * t2) * inv6;
    return g0 * l0 + g1 * l1 + g2 * l2 + g3 * l3;
}

}  // namespace

GrandProductProof prove_grand_product(const std::vector<F> &v) {
    GrandProductProof out{};
    out.sound = true;

    if (v.empty()) {
        out.claimed_product = F_ONE;
        return out;
    }

    auto t_build0 = std::chrono::high_resolution_clock::now();
    auto layers = build_product_tree(v);
    int L = (int)layers.size() - 1;
    auto t_build1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t_build1 - t_build0).count();
    out.prover_time_s += build_s;

    out.claimed_product = layers[0][0];

    std::fprintf(stderr,
        "    [grand_product] L=%d (leaves=%zu), product-tree built in %.3fs\n",
        L, v.size(), build_s);
    std::fflush(stderr);

    // GKR descent: maintain (z, c) such that V_k(z) = c.
    // Start at root: z is empty (length 0), c = layers[0][0] = total product.
    std::vector<F> z;          // length k after iteration k
    F              c = out.claimed_product;

    auto t_loop0 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < L; ++k) {
        // ---------------------------------------------------------------
        // Build the three vectors over j ∈ {0,1}^k that the round-k
        // sumcheck operates on:
        //   eq_tab[j]    = eq(z, j)         (verifier-known once z is fixed)
        //   even_tab[j]  = layers[k+1][2j]  = V_{k+1}(0, j)
        //   odd_tab[j]   = layers[k+1][2j+1] = V_{k+1}(1, j)
        // Identity to be proved:
        //   c == sum_{j ∈ {0,1}^k} eq_tab[j] · even_tab[j] · odd_tab[j]
        // ---------------------------------------------------------------
        size_t span = (size_t)1 << k;
        const auto &nxt = layers[k + 1];
        std::vector<F> eq_tab  = make_eq_table(z);
        std::vector<F> even_tab(span), odd_tab(span);
        for (size_t j = 0; j < span; ++j) {
            even_tab[j] = nxt[2 * j];
            odd_tab[j]  = nxt[2 * j + 1];
        }

        // ---------------------------------------------------------------
        // k rounds of degree-3 sumcheck (LSB-first folding to match the
        // existing convention in lasso_logup / lasso_surge).
        // Each round sends 4 evaluations g_round(0..3) of the round
        // polynomial. After all rounds, w[0..k-1] is the random point and
        // we have the final claims even(w) and odd(w) in even_tab[0] /
        // odd_tab[0] (and eq(z, w) in eq_tab[0]).
        // ---------------------------------------------------------------
        std::vector<F> w(k);
        F current_target = c;

        for (int round = 0; round < k; ++round) {
            auto t_p0 = std::chrono::high_resolution_clock::now();
            int half = (int)(span >> 1);

            // Evaluations of the round polynomial at X ∈ {0, 1, 2, 3}.
            // For each pair (eq[2j], eq[2j+1]) view it as a linear function
            // of X; same for even and odd. Their product is degree 3 in X.
            //
            // lasso-fork: parallelised with OpenMP. Each iteration is
            // independent, accumulating into four reductions g0..g3. The
            // declared `+: F` reduction at the top of this file makes mcl::Fr
            // safe for `reduction(+:...)`.
            F g0 = F_ZERO, g1 = F_ZERO, g2 = F_ZERO, g3 = F_ZERO;
            F two = F_ONE + F_ONE;
            F three = two + F_ONE;
            #pragma omp parallel for reduction(+:g0,g1,g2,g3) schedule(static)
            for (int j = 0; j < half; ++j) {
                const F &eq0 = eq_tab[2 * j],   &eq1 = eq_tab[2 * j + 1];
                const F &ev0 = even_tab[2 * j], &ev1 = even_tab[2 * j + 1];
                const F &od0 = odd_tab[2 * j],  &od1 = odd_tab[2 * j + 1];

                F deq = eq1 - eq0, dev = ev1 - ev0, dod = od1 - od0;
                g0 += eq0 * ev0 * od0;
                g1 += eq1 * ev1 * od1;
                F eq_2 = eq0 + two * deq;
                F ev_2 = ev0 + two * dev;
                F od_2 = od0 + two * dod;
                g2 += eq_2 * ev_2 * od_2;
                F eq_3 = eq0 + three * deq;
                F ev_3 = ev0 + three * dev;
                F od_3 = od0 + three * dod;
                g3 += eq_3 * ev_3 * od_3;
            }
            auto t_p1 = std::chrono::high_resolution_clock::now();
            out.prover_time_s += std::chrono::duration<double>(t_p1 - t_p0).count();

            // Verifier: check g(0) + g(1) == current_target, sample t,
            // update target := g(t), append t to w.
            auto t_v0 = std::chrono::high_resolution_clock::now();
            if (!(g0 + g1 - current_target).isZero()) {
                std::cerr << "[grand_product] layer " << k << " round " << round
                          << ": g(0)+g(1) != current_target\n";
                out.sound = false;
                return out;
            }
            F t = random_field();
            w[round] = t;
            current_target = deg3_eval(g0, g1, g2, g3, t);
            auto t_v1 = std::chrono::high_resolution_clock::now();
            out.verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();

            // Prover folds eq, even, odd by t (LSB collapse).
            //
            // lasso-fork (race fix): we MUST NOT do the fold in place when
            // running under OpenMP. The sequential version was safe because
            // thread-of-iter-j2 writes to index j2 only after iter-j1 < j2
            // has already read 2*j1 and 2*j1+1 — but in parallel, thread j2
            // can write to eq_tab[j2] while thread j1 is still reading
            // eq_tab[2*j1] / eq_tab[2*j1+1]. Concretely: at half=4 (first
            // appears at layer k=3, round 0), thread iter j2=2 writes
            // eq_tab[2] while thread iter j1=1 reads it ⇒ race ⇒ "layer 3
            // round 1: g(0)+g(1) != current_target".
            // Fix: write into a fresh buffer, then move it back.
            auto t_f0 = std::chrono::high_resolution_clock::now();
            std::vector<F> nxt_eq(half), nxt_even(half), nxt_odd(half);
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < half; ++j) {
                nxt_eq[j]   = (F_ONE - t) * eq_tab[2 * j]   + t * eq_tab[2 * j + 1];
                nxt_even[j] = (F_ONE - t) * even_tab[2 * j] + t * even_tab[2 * j + 1];
                nxt_odd[j]  = (F_ONE - t) * odd_tab[2 * j]  + t * odd_tab[2 * j + 1];
            }
            // Replace contents (move keeps the storage of the next iteration
            // identical to a sequential fold).
            for (int j = 0; j < half; ++j) {
                eq_tab[j]   = nxt_eq[j];
                even_tab[j] = nxt_even[j];
                odd_tab[j]  = nxt_odd[j];
            }
            span = half;
            auto t_f1 = std::chrono::high_resolution_clock::now();
            out.prover_time_s += std::chrono::duration<double>(t_f1 - t_f0).count();

            out.proof_size_bytes += 4 * F_BYTE_SIZE;
        }

        // After k rounds: span == 1, eq_tab[0] = eq(z, w), even_tab[0] = V_{k+1}(0, w),
        // odd_tab[0] = V_{k+1}(1, w).
        F eq_at_w   = eq_tab[0];
        F even_at_w = even_tab[0];
        F odd_at_w  = odd_tab[0];

        // Verifier-side independent recomputation of eq(z, w) — the
        // sumcheck folded eq alongside the others, but the verifier must
        // not blindly trust that fold; otherwise a malicious prover could
        // ship a fabricated triple whose product matches current_target.
        {
            auto t_v0 = std::chrono::high_resolution_clock::now();
            F eq_check = eq_direct(z, w);
            if (!(eq_check - eq_at_w).isZero()) {
                std::cerr << "[grand_product] layer " << k
                          << ": verifier eq(z,w) != prover folded eq\n";
                out.sound = false;
                return out;
            }
            // Final identity at point w.
            if (!(eq_at_w * even_at_w * odd_at_w - current_target).isZero()) {
                std::cerr << "[grand_product] layer " << k
                          << ": eq·even·odd != reduced target after sumcheck\n";
                out.sound = false;
                return out;
            }
            auto t_v1 = std::chrono::high_resolution_clock::now();
            out.verifier_time_s += std::chrono::duration<double>(t_v1 - t_v0).count();
        }

        // ---------------------------------------------------------------
        // Reduce the two claims about V_{k+1} (at points (0,w) and (1,w))
        // to a single claim at (γ, w) for verifier-chosen γ ∈ F.
        //   V_{k+1}((γ, w)) = (1-γ) · V_{k+1}(0, w) + γ · V_{k+1}(1, w)
        // The new sumcheck-target z for the next layer is (γ, w_0, ..., w_{k-1}),
        // matching our LSB-first convention (γ goes to the new variable).
        // ---------------------------------------------------------------
        F gamma = random_field();
        F new_c = (F_ONE - gamma) * even_at_w + gamma * odd_at_w;

        std::vector<F> new_z;
        new_z.reserve(w.size() + 1);
        new_z.push_back(gamma);
        for (const F &t : w) new_z.push_back(t);

        // Transcript: even(w), odd(w) (2 field elements). γ is a verifier
        // challenge so it doesn't count toward proof size, but the per-layer
        // 2 claims do.
        out.proof_size_bytes += 2 * F_BYTE_SIZE;

        z = std::move(new_z);
        c = new_c;
    }

    auto t_loop1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr,
        "    [grand_product] descent done in %.3fs (%d sumcheck rounds total)\n",
        std::chrono::duration<double>(t_loop1 - t_loop0).count(),
        L * (L - 1) / 2);
    std::fflush(stderr);

    out.final_point = std::move(z);
    out.final_claim = c;
    return out;
}

bool verify_grand_product(const GrandProductProof &proof,
                          F final_claim_check) {
    // The verifier-side checks live inside prove_grand_product because both
    // parties run in-process. This wrapper performs the only check the
    // surrounding protocol can independently make: the final claim from the
    // GKR descent must match the value the caller opened from commit(v).
    if (!proof.sound) return false;
    if (!(proof.final_claim - final_claim_check).isZero()) {
        std::cerr << "[grand_product] verify_grand_product: final claim "
                     "mismatch against committed v's opening\n";
        return false;
    }
    return true;
}

}  // namespace lasso_core
