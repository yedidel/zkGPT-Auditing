# CHANGELOG

This document describes the modifications applied to the zkGPT codebase
relative to the original publicly available implementation accompanying
Qu et al., "zkGPT: An Efficient Non-interactive Zero-knowledge Proof
Framework for LLM Inference" (USENIX Security 2025). The original
artifact is publicly distributed by the original authors and is *not*
redistributed in this fork; only our delta is included here.

The motivation, evaluation, and adversarial validation of the changes
described below are reported in the accompanying paper submitted to
the NeurIPS 2026 Datasets & Benchmarks Track.

---

## 1. Scope

### 1.0 Which upstream release this fork is based on

All statements below refer to the Zenodo deposit
[`10.5281/zenodo.14727819`](https://doi.org/10.5281/zenodo.14727819)
(`zktransformer.zip`, md5 `e2d0a05dd4e4c5b1a3dfae42436ff698`, deposited
23 January 2025), which is the artifact the zkGPT paper designates.
`src/main_demo_llm.cpp` in this fork is byte-identical to the copy in that
deposit.

A separate GitHub repository, `security-Anonymous/zkGPT` (single commit,
27 August 2025), contains a different release of the same work. It adds a
standalone `range_prover` component implementing a LogUp lookup argument
with Hyrax commitments (`src/range_prover.{hpp,cpp}`, `src/hyrax_rp.hpp`,
`verifier::range_prove()`). None of those files, and no occurrence of
`logup`, `range_prover`, `memcheck`, `spice`, or `grand-product`, appear
anywhere in the Zenodo deposit.

The findings recorded here are scoped to the deposit. They are not claims
about the GitHub release, which we have not measured. Two observations
about that release are worth recording for anyone comparing the two:

1. Its range prover is invoked from `main`, not from the verifier.
   `range_prover::prove()` returns a `double` holding elapsed prover time,
   and `verifier::range_prove(double)` stores that value. The only
   subsequent use of it is `prover_time + range_prover_time` in a printed
   total, so no proof object crosses the prover-verifier boundary.
2. `grand-product` and memory-checking subprotocols are absent from that
   release as well.

### 1.1 The gap in the audited deposit

The deposited zkGPT implementation enforces nonlinear-layer constraints
(LayerNorm, GeLU, Softmax) by performing a wiring check whose witness
range is masked with `& 0xFFFF` rather than via a cryptographic lookup
argument. The 7.2M nonlinear constraints in a 12-block GPT-2 forward
pass are therefore *not* bound to any committed table, and a malicious
prover can substitute out-of-range witnesses without being caught.

This fork replaces the masked wiring check with a faithful Lasso
pipeline that integrates natively with the existing GKR backend and
uses the Hyrax row-Pedersen commitment already present in the codebase.

Two Lasso variants are implemented:

- **Variant A -- Surge** (decomposable tables). Used for LayerNorm
  and GeLU range checks. Subtable parameters are chosen dynamically
  (`c in {1..8}`, `SUB_LOG = 16`) based on a witness pre-scan so the
  composed table covers up to 64-bit values.
- **Variant B -- Unstructured** (`c = 1`). Used for the Softmax
  exponentiation table (655,360 entries, padded to 2^20). The full
  table is committed once with Hyrax and re-used per call.

Both variants run Spice offline memory checking and verify the four
multiset-hash equalities (init/read/write/final) via a Thaler13
sumcheck-based grand-product GKR.

---

## 2. New files

All new files live under `src/` and are prefixed `lasso_`:

| File | Purpose |
|------|---------|
| `src/lasso_types.hpp`            | Shared types, multilinear-extension helpers, `eq` table, `HyraxGenerators`, lightweight benchmark struct. |
| `src/lasso_grand_product.hpp/cpp`| Thaler13 grand-product GKR (degree-3 sumcheck per layer, k rounds at layer k). |
| `src/lasso_memcheck.hpp/cpp`     | Spice multiset-hash construction and four grand-product invocations per lookup. |
| `src/lasso_surge.hpp/cpp`        | Variant A driver. Dynamic `NUM_SUB` selection from a pre-scan, per-subtable Hyrax openings. |
| `src/lasso_unstructured.hpp/cpp` | Variant B driver for the exp table. Single Hyrax commit shared across calls. |
| `src/lasso_logup.hpp/cpp`        | Earlier LogUp-style prototype, retained for reference but unused at runtime. |

`src/CMakeLists.txt` was switched from `aux_source_directory` to an
explicit list including the files above, so the new translation units
participate in the build deterministically across CMake versions.

---

## 3. Modifications to existing files

| File | Change |
|------|--------|
| `src/CMakeLists.txt` | Explicit listing of all `lasso_*.cpp` translation units. |
| `src/prover.hpp`     | `ExpLookupPair` struct + statics for collecting (table, evaluation) pairs of the exp table during the forward pass. |
| `src/prover.cpp`     | Initializers and accessors for the new statics; no change to existing prover-side wiring beyond hooking the Lasso entry points after the matching GKR layer completes. |
| `src/neuralNetwork.cpp` | Builds `exp_table_padded` (2^20 entries) and accumulates `(t, E)` pairs that are forwarded to the unstructured Lasso variant. |
| `src/verifier.cpp`   | Removes the previous masked-range wiring block and calls into the new Surge / Unstructured verifiers. A `BUILD-MARKER` log line was added so reviewers can confirm the corrected binary is the one being measured. |

No public API of the original codebase was renamed. Removed code paths
were paths that were unsound (the `& 0xFFFF` mask block).

---

## 4. Soundness implications

The corrected pipeline binds every nonlinear-layer witness to a
committed table:

- Schwartz--Zippel error per multiset-hash equality is bounded by
  `3 * max(N_alpha, M) / |F_r|`, which for our largest instance
  (M = 2^23) gives `< 2^-225`.
- Each round of the degree-3 grand-product sumchecks contributes at
  most `4 / |F_r|`.
- Total knowledge-soundness error for a full GPT-2 forward proof
  remains below `2^-200`, well under the conventional `2^-100`
  cryptographic threshold.

A more detailed accounting is given in Section 3.1 of the paper
("Soundness parameters").

---

## 5. Adversarial validation harness

`src/lasso_surge.cpp` and `src/lasso_unstructured.cpp` ship with
controlled tampering injection points gated on the `LASSO_ADV_TEST`
environment variable. Each scenario tampers a single witness value at
a controlled stage of the pipeline; the prover is then run to
completion against the *unmodified* verifier so the cryptographic
check that catches the tampering can be observed.

Available scenarios (Variant A -- Surge):

| `LASSO_ADV_TEST` value | Witness tampered | Expected guard |
|------------------------|-------------------|----------------|
| `NEG`             | Negative value injected before the pre-scan.            | Phase A pre-scan |
| `WRONG-E`         | Single subtable evaluation `E_alpha`.                   | Spice multiset identity |
| `WRONG-DIM`       | Subtable dimension witness `dim_alpha`.                 | Spice multiset / discharge |
| `WRONG-CTS`       | A read-count entry, between memcheck and Hyrax commit.  | Phase E discharge (GKR final-claim) |
| `WRONG-FN`        | A final-count entry, between memcheck and Hyrax commit. | Phase E discharge (GKR final-claim) |
| `WRONG-CTS-POST`  | A read-count entry, **after** its Hyrax commit.         | Hyrax binding/opening |
| `WRONG-FN-POST`   | A final-count entry, **after** its Hyrax commit.        | Hyrax binding/opening |

Variant B -- Unstructured:

| `LASSO_ADV_TEST` value | Witness tampered | Expected guard |
|------------------------|-------------------|----------------|
| `WRONG-T-EXP` | A single entry of the committed exp table. | Unstructured discharge (init multiset) |

### Property-based randomization (LASSO_ADV_SEED)

Every scenario picks the tampered index from a pseudo-random draw
seeded by the `LASSO_ADV_SEED` env var (a non-zero unsigned integer).
When the seed is unset (or set to `0`), the index defaults to `1` for
backward compatibility with the original harness.

Setting different seeds replays the same tamper class at different
witness positions. Setting a fixed seed makes the run reproducible.
Sweeping `N` seeds per scenario gives an empirical coverage statement
("expected guard fired in `N/N` cells") rather than a single point check.

Run a single scenario at a fixed seed:

```bash
LASSO_ADV_TEST=WRONG-E LASSO_ADV_SEED=42 ./cmake-build-release/src/demo_llm_run
LASSO_ADV_TEST=NEG                       ./cmake-build-release/src/demo_llm_run   # seed defaults to 0 -> index 1
```

The notebook (`zkGPT_repro.ipynb`, section 12.5) ships a sweep cell
that runs every scenario at `LASSO_ADV_SEED in {1..N}` and emits a
coverage matrix to `paper/discussion/adv_coverage.json`.

Results for the full sweep (eight tampering scenarios plus the honest
baseline) are reported in Appendix B of the paper.

---

## 6. Build and run

The build steps are unchanged from the upstream README:

```bash
mkdir -p build && cd build
cmake ..
make -j
```

The corrected binary prints `[BUILD-MARKER ...]` lines so the running
artifact can be distinguished from any previously built upstream
binary that may exist on the same machine.

System and dependency versions used for the measurements in the paper
are documented in Section 3.1 ("Experimental setup"). In short:
Lambda Cloud `gpu_1x_a100_sxm4`, NVIDIA A100-SXM4-40GB, Ubuntu 22.04,
GCC 11.4, CMake 3.22, mcl-bn254 commit `e4c8bbe4` (2025-10-16).

---

## 7. Known limitations and non-goals

- The proof system is unchanged: BN254 / Hyrax / GKR. We did not
  port the protocol to a different curve or commitment scheme.
- The corrected metrics reflect a single-machine deterministic run.
  Variance across runs is well within the rounding shown in the paper
  tables and is therefore not reported separately.
- The legacy LogUp prototype (`src/lasso_logup.*`) is retained only
  to preserve historical context and is not invoked at runtime.

---

## 8. File-level summary

```
NEW   src/lasso_types.hpp
NEW   src/lasso_grand_product.hpp
NEW   src/lasso_grand_product.cpp
NEW   src/lasso_memcheck.hpp
NEW   src/lasso_memcheck.cpp
NEW   src/lasso_surge.hpp
NEW   src/lasso_surge.cpp
NEW   src/lasso_unstructured.hpp
NEW   src/lasso_unstructured.cpp
NEW   src/lasso_logup.hpp        # retained, unused
NEW   src/lasso_logup.cpp        # retained, unused

MOD   src/CMakeLists.txt         # explicit source listing
MOD   src/prover.hpp             # ExpLookupPair + statics
MOD   src/prover.cpp             # initializers / accessors
MOD   src/neuralNetwork.cpp      # exp_table_padded, (t,E) collection
MOD   src/verifier.cpp           # Lasso wiring, BUILD-MARKER, removal of masked range check
```
