# zkGPT (faithful Lasso variants)

This repository is a research artifact accompanying our
NeurIPS 2026 Datasets & Benchmarks Track submission. It is a fork of
the zkGPT framework of Qu et al. (USENIX Security 2025; cited in the
accompanying paper) that replaces the framework's masked range-check
wiring with a faithful Lasso lookup pipeline. The original unmodified
codebase is *not* redistributed here; only our delta plus the
unmodified files that our delta depends on.

The full set of changes relative to upstream is documented in
[`CHANGELOG.md`](CHANGELOG.md). The motivation, evaluation, and
adversarial validation are reported in the accompanying paper.

The current artifact targets GPT-2 (12 transformer blocks, 12 attention
heads, hidden dimension 768, sequence length 64).

---

## Upstream baseline

Every measurement and every soundness finding in this repository refers to
the Zenodo deposit that the zkGPT paper designates as its artifact:

| | |
|---|---|
| DOI | [`10.5281/zenodo.14727819`](https://doi.org/10.5281/zenodo.14727819) |
| File | `zktransformer.zip` |
| Size | 668,493 bytes |
| md5 | `e2d0a05dd4e4c5b1a3dfae42436ff698` |
| Deposited | 23 January 2025 |

[`zkGPT_repro.ipynb`](zkGPT_repro.ipynb) downloads this deposit, verifies the
md5 against the value published by Zenodo, and builds from it. No account or
manual download is required.

### Two upstream distributions exist, and they differ

A GitHub repository, `security-Anonymous/zkGPT`, also exists. It carries a
single commit dated 27 August 2025, seven months after the Zenodo deposit, and
it is **not** the same code. It adds a standalone range-proof component that is
absent from the deposit:

| | Zenodo deposit (audited) | GitHub repository |
|---|---|---|
| `src/range_prover.{hpp,cpp}` | absent | present |
| `src/hyrax_rp.hpp` | absent | present |
| `verifier::range_prove()` | absent | present |
| LogUp lookup argument | absent | present |
| grand-product / memory checking | absent | absent |
| range enforcement in `verifier.cpp` | `FILTER = 65535` mask | mask plus separate range prover |

`src/main_demo_llm.cpp` in this fork is byte-identical to the copy in the
Zenodo deposit, which fixes the baseline unambiguously.

We audited the deposit because it is the artifact the paper cites. Readers
comparing our results against the GitHub tree should expect them not to line
up: the two releases are not interchangeable, and the notebook asserts that the
build tree is the deposit rather than the GitHub variant before proceeding.

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

[`REPRODUCTION.md`](REPRODUCTION.md) is the full manifest: it pins the
upstream baseline by DOI and checksum, gives the exact build and run
commands, and lists the expected output of every run this repository
contributes to the paper. The short version follows.

Build and run:

```bash
./llm.sh                                    # build and run
./cmake-build-release/src/demo_llm_run      # run an existing build
```

The default run reproduces the Table 2 measurements for GPT-2 (12 blocks,
12 heads, hidden dimension 768, sequence length 64). No arguments are
needed. The per-component breakdown in Appendix A comes from the same run
with verbose logging enabled.

To reproduce a tampering scenario from Appendix C (Table 3), set
`LASSO_ADV_TEST` before launching the binary:

```bash
LASSO_ADV_TEST=NEG     ./cmake-build-release/src/demo_llm_run  # rejected at pre-scan
LASSO_ADV_TEST=WRONG-E ./cmake-build-release/src/demo_llm_run  # rejected at Spice multiset identity
```

All nine scenarios and the guard that catches each are tabulated in
[`REPRODUCTION.md`](REPRODUCTION.md) section 4, and described in section 5
of [`CHANGELOG.md`](CHANGELOG.md).

---

## License

Released under the same terms as the upstream zkGPT codebase. See
[`LICENSE.md`](LICENSE.md).
