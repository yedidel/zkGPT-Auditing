# zkGPT

## Introduction

This is an enhanced implementation of **zkGPT**, a SNARK framework for LLM inference. 
This project is an extension of the original work which can be found at: [https://zenodo.org/records/14727819](https://zenodo.org/records/14727819).

 **Current implementation supports GPT-2.** The research paper describing zkGPT in detail is included in this repository as a PDF file: `zkGPT-fullversion.pdf`.

---

## Recent Cryptographic Audit & Fixes (This Extension)

While the original implementation provided a foundation for proving LLM layers, our audit revealed critical missing components required for a complete and sound zero-knowledge proof. This version adds the following enhancements:

### 1. Full Soundness Implementation
The original code lacked the **Evaluation Opening Proof** for the Lasso lookup protocol. Without this, the prover's claims about range constraints could not be cryptographically verified against the committed data. 
* **Added:** We integrated a full **Hyrax-based polynomial opening** at the challenge point $r$ (as described in Protocol 1 of the zkGPT paper).
* **Added:** Missing auxiliary commitments to Lasso polynomials (counts and lookups) are now computed and verified.

### 2. Corrected Proof Metrics
Previously, the reported proof size only accounted for Sumcheck field elements, leading to an artificially low figure (~1.8 KB). 
* **Update:** We updated the metrics to include the actual cryptographic overhead (Commitments + Opening Proofs). The corrected proof size is now **~95.26 KB**, which aligns with the academic benchmarks for GPT-2.

### 3. Performance Analysis (Baseline vs. Optimized)
We introduced a detailed performance breakdown to distinguish between the theoretical "True Cost" of the protocol and our engineering optimizations:
* **Baseline Audit:** We measured the Lasso commitment at **~72.8s** in a single-threaded baseline, proving it to be the primary bottleneck in a naive implementation.
* **Optimized Proof:** By utilizing our 32-thread Pippenger and Circuit Squeeze optimizations, this overhead is reduced to **~0.73s**, making the non-linear layer verification practical.

---

## Requirement
### Software Requirement
- C++14
- cmake >= 3.10
- GMP library

### Recommended Server Configuration
To ensure smooth execution, we recommend using a server with the following specifications:
- Operating System (OS): Linux (e.g., Ubuntu 18.04 or later)
- Processor (CPU): Multi-core CPU, preferably with 16 cores or more
- Memory (RAM): at least 200GB


## Experiment Script
### Clone the repo
To run the code, make sure you clone with
``` bash
git clone --recurse-submodules git@github.com:security-Anonymous/zkTransformer.git