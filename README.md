# zkGPT (anonymous artifact -- faithful Lasso variants)

This repository is an anonymous research artifact accompanying our
NeurIPS 2026 Datasets & Benchmarks Track submission. It is a fork of
the publicly available zkGPT framework of Qu et al. (USENIX Security
2025; cited in the accompanying paper) that replaces the framework's
masked range-check wiring with a faithful Lasso lookup pipeline. The
original unmodified codebase is *not* included here; only our delta
plus the unmodified files that our delta depends on.

The full set of changes relative to upstream is documented in
[`CHANGELOG.md`](CHANGELOG.md). The motivation, evaluation, and
adversarial validation are reported in the accompanying paper.

The current artifact targets GPT-2 (12 transformer blocks, 12 attention
heads, hidden dimension 768, sequence length 64).

---

## Summary of changes

See [`CHANGELOG.md`](CHANGELOG.md) for the authoritative list. In
short:

- Replaces the masked wiring check (`& 0xFFFF`) with two Lasso
  variants: **Surge** (decomposable, used for LayerNorm/GeLU) and
  **Unstructured** (`c=1`, used for the Softmax exponentiation table).
- Adds Spice offline memory checking and a Thaler13 sumcheck-based
  grand-product GKR.
- Adds an adversarial test harness (`LASSO_ADV_TEST=...`) covering six
  tampering scenarios. See Section 5 of `CHANGELOG.md` and Appendix B
  of the paper.
- Updates `src/CMakeLists.txt` to enumerate sources explicitly so the
  new modules participate in every build.

---

## Requirements

### Software
- C++14 (tested with GCC 11.4)
- CMake >= 3.10 (tested with CMake 3.22.1)
- GMP
- OpenMP
- The `mcl-bn254` submodule under `3rd/` (commit `e4c8bbe4`,
  2025-10-16, as used in the paper measurements)

### Hardware (for full GPT-2 reproduction)
- Linux (Ubuntu 22.04 used for the paper measurements)
- A multi-core CPU; the paper measurements used 30 vCPUs of an
  AMD EPYC 7J13.
- At least ~200 GB RAM for the largest configurations.
- An NVIDIA GPU is required only for the (separate) zkLLM artifacts;
  zkGPT itself runs on the CPU side of the BN254 backend.

The exact configuration used to produce the numbers in the paper is
documented in Section 3.1 ("Experimental setup").

---

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j
```

The corrected binary prints `[BUILD-MARKER ...]` lines at startup so
you can confirm at runtime that the corrected code path -- not a
stale upstream build that may exist on the same machine -- is the one
producing the measurements.

---

## Reproducing the headline numbers

The measurements reported in Table 2 of the paper (GPT-2, 12 blocks)
are produced by the default run of the main binary built above. To
inspect the per-component breakdown shown in Appendix A
("Per-component cost") run with verbose logging enabled.

To reproduce a tampering scenario from Appendix B (Table 3), set the
`LASSO_ADV_TEST` environment variable before launching the binary:

```bash
LASSO_ADV_TEST=NEG     ./build/main_zkgpt   # rejected at pre-scan
LASSO_ADV_TEST=WRONG-E ./build/main_zkgpt   # rejected at multiset-hash check
```

The full list of supported scenarios and the guard that catches each
one is given in Section 5 of `CHANGELOG.md`.

---

## Rule-based no-LLM baseline

The script [`tools/rule_based_baseline.py`](tools/rule_based_baseline.py) is
the deterministic, no-LLM baseline reported in Section 5.2 of the paper. It
scans a target codebase for the canonical patterns associated with the
soundness guards that `zkml-audit-benchmark` artifacts target (presence of
Fiat-Shamir transcript binding before each challenge draw, commit-before-
challenge ordering for lookup arguments, lookup-argument invocation, Spice/
memcheck companion calls, per-stage output commitment chain). It runs in
under a second per codebase, has no LLM dependency, and is intended as a
floor any future LLM-based auditor should clear.

Why include it: reviewers asked for at least one ablation isolating the LLM's
contribution. The single-agent vs multi-agent comparison in Table 4 of the
paper is the architecture ablation; this script is the no-LLM ablation,
giving a deterministic reference point that shows the gap LLM-based reasoning
must justify.

How to run:

```bash
python tools/rule_based_baseline.py --codebase /path/to/codebase \
                                    --report /path/to/output.json
```

The JSON report includes the per-pattern checklist for each scanned codebase
and a 0--1 soundness-checklist score. The script depends only on the Python
standard library (no extra installs).

---

## License

Released under the same terms as the upstream zkGPT codebase. See
[`LICENSE.md`](LICENSE.md).
