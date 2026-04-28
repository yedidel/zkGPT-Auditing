// yedidel-lasso: Shared types and helpers for the Lasso lookup family
// (eprint 2023/1216 by Setty-Thaler-Wahby).
//
// The zkGPT codebase uses two lookup tables:
//
//   (A) An identity range table T[j] = j over j ∈ [0, 2^16). Used for the
//       quantization auxiliaries d1, d2, d3, s, dt2 in LayerNorm / GELU /
//       Round / Softmax. This table is "structured + decomposable":
//       T splits as T_1 ⊗ T_2 with 8-bit subtables and combining function
//       g(t_1, t_2) = t_1·256 + t_2. Both subtables are identities whose MLE
//       evaluates in O(log √N) without commitment. → Surge protocol, c=2.
//
//   (B) An exponentiation table built by `neuralNetwork::compute_e_table()`
//       (size 655360 ≈ 2^20), holding round(exp(-St·i)/Se) for the Softmax
//       (t, E) pair lookup. The MLE has no closed-form, so by Bing-Jyue's
//       guidance we set c=1 and treat the table itself as a committed
//       polynomial. → Unstructured Lasso, c=1.
//
// Both variants share three pieces of cryptographic machinery:
//   * Spice-style offline memory checking      (lasso_memcheck.*)
//   * Grand-product GKR for multiset hashing   (lasso_grand_product.*)
//   * Hyrax row-Pedersen commits + openings    (existing hyrax.*)
//
// Variant-specific pieces:
//   * Surge main sumcheck for decomposable tables       (lasso_surge.*)
//   * Pedersen-T commit + main sumcheck for unstructured (lasso_unstructured.*)
//
// Naming convention: every new symbol introduced for Lasso lives in
// namespace `lasso_core` (shared) or `lasso_surge` / `lasso_unstructured`
// (variant drivers). The "core" namespace contains data structures, helpers,
// and primitives that both variants need.

#ifndef ZKCNN_LASSO_TYPES_HPP
#define ZKCNN_LASSO_TYPES_HPP

#include <cstdint>
#include <vector>

#include "global_var.hpp"
#include "hyrax.hpp"

namespace lasso_core {

// ---------------------------------------------------------------------------
//  Multilinear-extension utilities used across the Lasso modules.
// ---------------------------------------------------------------------------

// Brute-force MLE evaluation of a length-2^l vector w at point r ∈ F^l, using
// the same little-endian variable convention as `lagrange()` in hyrax.cpp:
// bit i of the index k corresponds to r[i].
// O(l · 2^l) — fine for tables up to ~2^20.
inline F mle_eval(const F *w, const F *r, int l) {
    return brute_force_compute_eval(const_cast<F*>(w), const_cast<F*>(r), l);
}

// MLE of the integer-index polynomial J(y) = Σ_i 2^i · y_i, evaluated at y=r.
// On the {0,1}^k corner indexed by integer j, J equals j exactly.
inline F integer_index_mle(const F *r, int l) {
    F acc = F_ZERO;
    F two_pow = F_ONE;
    for (int i = 0; i < l; ++i) {
        acc += two_pow * r[i];
        two_pow += two_pow;
    }
    return acc;
}

// MLE of the equality polynomial eq(z, x) = Π_i (z_i x_i + (1-z_i)(1-x_i)),
// fully expanded over x ∈ {0,1}^l. Returns a fresh vector of length 2^l
// where entry k holds eq(z, k).
//
// Convention: bit i of the index k corresponds to variable z_i, matching
// the existing `lagrange()` helper in hyrax.cpp (`if (k & (1<<i)) ret *= r[i]`).
// We extend by adding the i-th variable as bit i of the new index, growing
// the active region from [0, 2^i) to [0, 2^(i+1)) by attaching the bit-i = 1
// successor at offset 2^i above each existing entry.
inline std::vector<F> eq_table(const F *z, int l) {
    std::vector<F> out(1ULL << l, F_ZERO);
    out[0] = F_ONE;
    for (int i = 0; i < l; ++i) {
        int half = 1 << i;
        // Process existing indices from high to low so writes to (k + half)
        // never overwrite an entry we still need to read.
        for (int k = half - 1; k >= 0; --k) {
            F v = out[k];
            out[k]        = v * (F_ONE - z[i]);
            out[k + half] = v * z[i];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Random sampling helpers. These approximate Fiat-Shamir with the same
//  CSPRNG that the rest of the codebase uses; full FS-from-transcript can be
//  layered on later without affecting the protocol structure.
// ---------------------------------------------------------------------------

inline F random_field() {
    F r;
    r.setByCSPRNG();
    return r;
}

inline std::vector<F> random_field_vector(int n) {
    std::vector<F> r(n);
    for (int i = 0; i < n; ++i) r[i].setByCSPRNG();
    return r;
}

// ---------------------------------------------------------------------------
//  Hyrax row-commitment helper for arbitrary Fr values.
//
//  The existing prover_commit(ll*, ...) in hyrax.cpp only handles small-
//  magnitude __int128 integer inputs (Pippenger over bounded buckets). For
//  Lasso we frequently commit to vectors of arbitrary field elements (read
//  counts in F, reciprocals, table values, etc.).
//
//  hyrax.cpp also defines `perdersen_commit(G1*, Fr*, int)` as a thin wrapper
//  around `G1::mulVec`, but that overload is NOT declared in hyrax.hpp — it
//  is private to that translation unit. We therefore call mcl's `mulVec`
//  directly from this header to avoid touching hyrax.hpp.
// ---------------------------------------------------------------------------
inline G1 *commit_fr_hyrax(F *w, G1 *gens_col, int l) {
    int halfl  = l / 2;
    int rownum = 1 << halfl;
    int colnum = 1 << (l - halfl);
    G1 *Tk = new G1[rownum];
    for (int i = 0; i < rownum; ++i) {
        G1::mulVec(Tk[i], gens_col, w + (i * colnum), colnum);
    }
    return Tk;
}

// ---------------------------------------------------------------------------
//  A bundled Hyrax commitment handle: keeps the row-table Tk plus the
//  generators that produced it, so an opening can be performed later.
// ---------------------------------------------------------------------------
struct HyraxHandle {
    G1     *Tk        = nullptr;  // owned, must be `delete[]`d
    G1     *gens_col  = nullptr;  // borrowed (lives in HyraxGenerators)
    G1      blinder;              // the base generator returned by gen_gi
    int     l         = 0;        // log size of the committed vector
    int     halfl() const { return l / 2; }
    int     rownum()  const { return 1 << halfl(); }
    int     colnum()  const { return 1 << (l - halfl()); }
};

// ---------------------------------------------------------------------------
//  A collection of fresh Hyrax generators sized for a particular log-length.
//  We sample these per-protocol-instance (inside lasso_core) instead of
//  reusing p->gens, which is sized for the input commitment and may not have
//  enough columns for our intermediate vectors.
// ---------------------------------------------------------------------------
struct HyraxGenerators {
    std::vector<G1> g;       // column generators of length 2^(l - l/2)
    G1              blinder; // base generator
    int             l = 0;

    void init(int total_log_size) {
        l = total_log_size;
        int halfl  = l / 2;
        int colnum = 1 << (l - halfl);
        g.resize(colnum);
        blinder = gen_gi(g.data(), colnum);
    }
};

// ---------------------------------------------------------------------------
//  Per-call benchmark accounting, mirroring lasso_logup::LookupBenchmark.
// ---------------------------------------------------------------------------
struct LassoBenchmark {
    bool   sound            = true;
    u32    num_lookups      = 0;
    u32    log_M            = 0;
    u32    log_N            = 0;
    double prover_time_s    = 0.0;
    double verifier_time_s  = 0.0;
    u64    proof_size_bytes = 0;
    const char *variant     = "";  // "Surge (decomposable)" or "Unstructured (c=1)"
};

}  // namespace lasso_core

#endif  // ZKCNN_LASSO_TYPES_HPP
