# Reproduction manifest

This file pairs the frozen upstream snapshot with the corrected fork, and gives
the exact commands, parameters, and expected outputs for every number this
artifact contributes to the paper. It is written so that a reader can see what
each run produces without executing anything, and execute it if they wish.

Everything below was checked against the source tree in this repository. Where a
value depends on hardware, the machine is stated.

---

## 1. The two trees being compared

| | Upstream baseline | Corrected fork |
|---|---|---|
| Source | Zenodo [`10.5281/zenodo.14727819`](https://doi.org/10.5281/zenodo.14727819) | this repository |
| File | `zktransformer.zip`, 668,493 bytes | — |
| md5 | `e2d0a05dd4e4c5b1a3dfae42436ff698` | — |
| Deposited | 23 January 2025 | — |
| Range enforcement | `FILTER = 65535` mask in `verifier.cpp` | Surge + Unstructured Lasso |
| Lookup commitment | none | Hyrax, opened by the verifier |
| Memory checking | none | Spice |

`src/main_demo_llm.cpp` in this repository is byte-identical to the copy in the
deposit, which fixes the baseline unambiguously. A second upstream distribution
exists on GitHub and is **not** the same code; see the comparison table in
[`README.md`](README.md) before comparing any number against it.

[`zkGPT_repro.ipynb`](zkGPT_repro.ipynb) performs the download, verifies the md5
against the value published by Zenodo, and asserts that the resulting tree is
the deposit rather than the GitHub variant before building.

---

## 2. Build

```bash
./llm.sh
```

`llm.sh` runs `build.sh` (which configures CMake in `cmake-build-release` with
`-DCMAKE_BUILD_TYPE=Release`), builds the `demo_llm_run` target, and launches
it. To build without running:

```bash
./build.sh
cmake --build ./cmake-build-release --target demo_llm_run -- -j6
```

The binary is produced at:

```
./cmake-build-release/src/demo_llm_run
```

Dependency: `mcl` must be present at `3rd/mcl`. The Zenodo deposit does not
vendor it; the notebook fetches it, or clone it manually into `3rd/`.

---

## 3. Headline cost measurements (paper Table 2)

```bash
./cmake-build-release/src/demo_llm_run
```

Default configuration is GPT-2: 12 transformer blocks, 12 attention heads,
hidden dimension 768, sequence length 64. No arguments or environment variables
are required.

Machine for the reported numbers: single Lambda Cloud A100-SXM4-40GB instance,
host AMD EPYC CPU, BN254 curve via `mcl-bn254`, 32 OpenMP threads. zkGPT proving
runs on the CPU, so the GPU is not used by this binary.

| Metric | Upstream baseline | Corrected fork | Ratio |
|---|---|---|---|
| Total prover time | 20.7 s | 101.3 s | 4.9x |
| Total verifier time | 0.3 s | 308.6 s | 1029x |
| Total proof size | 88.3 kB | 1284.2 kB | 14.5x |

Component split inside the corrected total, reported in Appendix A: Surge
prover 49.1 s, Unstructured prover 2.7 s. Verifier time is dominated by
approximately 43 Hyrax openings, which is intrinsic to Lasso verification
rather than to this implementation, since Hyrax verification scales with
witness size.

**Known documentation fault, now corrected.** Earlier revisions of this
repository referred to the binary as `./build/main_zkgpt`. That path never
existed; the target is `demo_llm_run` and it lands under `cmake-build-release`.

**Known fault in the notebook, now corrected.** The notebook previously
retained the upstream 20.7 s figure as its headline output rather than the
corrected measurement, and cloned a repository that has since been removed. It
now fetches the Zenodo deposit and verifies its checksum.

---

## 4. Adversarial validation (paper Appendix C, Table 3)

The corrected pipeline passing on honest input does not by itself show that the
restored checks are evaluated rather than merely wired in. The harness below
tampers with one witness value at a controlled stage and records which
cryptographic guard rejects the run.

```bash
LASSO_ADV_TEST=<SCENARIO> ./cmake-build-release/src/demo_llm_run
```

`LASSO_ADV_SEED` optionally fixes the index of the tampered element; omit it for
a seed drawn from the run.

| `LASSO_ADV_TEST` | Tamper | Expected outcome |
|---|---|---|
| `HONEST` | none (baseline) | passes, both variants sound |
| `NEG` | inject `-1` into `val[0]` before pre-scan | rejected at Phase A pre-scan (range validation) |
| `WRONG-E` | flip one `E_alpha[k]` after Phase A | rejected by Spice multiset identity |
| `WRONG-DIM` | flip one `dim_alpha[k]` after Phase A | rejected by Spice multiset / discharge |
| `WRONG-CTS` | flip one `read_cts[k]` between memcheck and commit | rejected at Phase E discharge (GKR final claim) |
| `WRONG-FN` | flip one `final_cts[j]` between memcheck and commit | rejected at Phase E discharge (GKR final claim) |
| `WRONG-CTS-POST` | flip one `read_cts[k]` **after** its Hyrax commit | rejected by Hyrax opening (binding) |
| `WRONG-FN-POST` | flip one `final_cts[j]` **after** its Hyrax commit | rejected by Hyrax opening (binding) |
| `WRONG-T-EXP` | flip one `T[j]` before Hyrax commit (Unstructured) | rejected at Unstructured discharge (init multiset) |

The eight tamper scenarios exercise four structurally distinct guard families:
input range validation, Spice multiset identity, GKR final-claim discharge, and
Hyrax binding. A scenario that passed would indicate a check that is present in
the code but never evaluated, which is the failure mode this harness exists to
rule out.

This harness covers zkGPT only. The zkLLM fork has no equivalent runtime
harness, so its findings rest on the protocol-to-implementation mapping rather
than on execution. That asymmetry is a real limitation and is stated as such.

---

## 5. What this manifest does not cover

* The zkLLM measurements (paper Table 1) live in the separate zkLLM fork and
  are reproduced by the executed notebook shipped there.
* The benchmark, the artifact corpus, and the auditor evaluation are separate
  artifacts and are not built from this repository.
* Numbers here are upper bounds from a non-aggregated instantiation. A
  batched or aggregated commitment scheme would reduce the constant factors,
  though not the presence or absence of the enforcement steps themselves.
