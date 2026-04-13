# zkGPT — Repository Summary

## Overview

**zkGPT** is a C++ implementation of a **SNARK (Succinct Non-interactive Argument of Knowledge) for LLM inference**, specifically targeting **GPT-2** (12 transformer blocks). The system allows a prover to demonstrate that a given LLM inference was performed correctly, and a verifier can check this proof efficiently — without re-executing the model.

The project builds upon prior work from **zkCNN** (zero-knowledge proofs for CNNs) and **Lasso** (lookup arguments), adapting and extending their techniques to handle the unique operations found in transformer architectures (multi-head attention, softmax, GELU, layer normalization). The accompanying paper is `zkGPT-fullversion.pdf`.

**License:** MIT (Copyright 2022, TAMUCrypto)

---

## High-Level Architecture

The proof system follows the **GKR (Goldwasser-Kalai-Rothblum) interactive proof** paradigm combined with **sumcheck protocols**. The entire GPT-2 computation is encoded as an **arithmetic layered circuit**, and the prover demonstrates correct evaluation of this circuit layer by layer.

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Model       │────▶│  Arithmetic      │────▶│  GKR-based   │
│  Definition  │     │  Circuit Builder │     │  Prover      │
│  (GPT-2)     │     │  (neuralNetwork) │     │  (prover)    │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                                              Proof transcript
                                                     │
                                              ┌──────▼───────┐
                                              │  Verifier     │
                                              │  (verifier)   │
                                              └──────────────┘
```

### Proof Pipeline

1. **Circuit Initiation** — The GPT-2 model is encoded as a layered arithmetic circuit with all non-linear operations (LayerNorm, GELU, Softmax) decomposed into verifiable sub-circuits.
2. **Input Commitment** — Model weights are committed using a Hyrax-style Pedersen vector commitment (over BN254 elliptic curve).
3. **GKR Protocol** — The verifier checks correct circuit evaluation layer-by-layer via interactive sumcheck protocols (made non-interactive via Fiat-Shamir in practice).
4. **Lasso (Input Consistency)** — A sumcheck-based argument ensures the claimed input layer values are consistent with the committed weights.
5. **Commitment Opening** — The Hyrax polynomial commitment is opened via a Bulletproofs-style inner-product argument.

---

## Cryptographic Primitives Used

| Primitive | Purpose | Implementation Location |
|---|---|---|
| **BN254 pairing curve** | Finite field & group operations | mcl library (`3rd/mcl`) |
| **Pedersen vector commitment (Hyrax)** | Commit to model weights & inputs | `hyrax.cpp` / `hyrax.hpp` |
| **GKR interactive proof** | Prove correct layered circuit evaluation | `prover.cpp` / `verifier.cpp` |
| **Sumcheck protocol** | Core sub-protocol within GKR | `prover.cpp` (`sumcheckUpdate*`) |
| **Lasso lookup argument** | Input consistency (replaces Virgo) | `verifier.cpp` (`verifyLasso`) |
| **Bulletproofs inner-product argument** | Commitment opening | `hyrax.cpp` (`bullet_reduce`) |
| **Pippenger's algorithm** | Fast multi-scalar multiplication for commitments | `hyrax.cpp` (`perdersen_commit`) |

---

## Source File Breakdown

### Entry Point

#### `main_demo_llm.cpp`
The main executable (~30 lines). Initializes the BN254 pairing, creates a GPT-2 model with 12 transformer blocks, builds the arithmetic circuit, and runs the full prove-then-verify pipeline with 32 threads.

```cpp
initPairing(mcl::BN254);
prover p;
LLM nn(12);        // 12-layer GPT-2
nn.create(p, 1);   // Build circuit + merge layers
verifier v(&p, p.C);
v.prove(32);        // Run prover + verifier
```

---

### Model Definition

#### `models.hpp` / `models.cpp`
Defines the `LLM` class (extends `neuralNetwork`). Sets up the GPT-2 architecture:
- **12 transformer blocks**, each containing 4 fully-connected layers:
  - FC1: 768 → 2304 (QKV projection)
  - FC2: 768 → 768 (attention output projection)
  - FC3: 768 → 2304 (FFN first layer, with GELU)
  - FC4: 2304 → 768 (FFN second layer)
- Sequence length is hardcoded to **30 tokens**.
- The constructor initializes the `full_conn` vector with `fconKernel` structs describing each FC layer's dimensions.

---

### Neural Network / Circuit Builder

#### `neuralNetwork.hpp` / `neuralNetwork.cpp` (~1567 lines)
The largest and most complex file. Responsible for encoding the GPT-2 computation as a layered arithmetic circuit by creating gates (unary and binary) for each operation.

##### Quantization Scheme
All floating-point values are quantized to integers with tracked scale factors `(c, e)` representing `value ≈ c × 2^e`. The `search()` helper finds the best `(c, e)` pair for a given floating-point scale by exhaustive search over `e ∈ [-10, 10]` and `c ∈ [1, 800]`.

##### Layer Types (defined in `circuit.h`)

| Layer Type | Enum | Purpose |
|---|---|---|
| `INPUT` | Input data + model weights layer | Holds all model weights, input embeddings, and auxiliary values |
| `FCONN` | Fully-connected (matrix multiplication) | Proven via Thaler'13 / sumcheck product protocol |
| `RELU` / `Sqr` | Rounding layer | Proves quantized rounding correctness |
| `LAYER_NORM_1/2/3` | Layer normalization (3-phase checker) | Decomposes LN into arithmetic constraints |
| `GELU_1/2/3` | GELU activation (3-phase checker) | Piecewise polynomial approximation |
| `MHA_QK` | Multi-head attention Q·K^T product | 12-head, 64-dim causal attention |
| `SOFTMAX_1/2/3` | Softmax (3-phase checker) | Uses exp lookup table + division rounding |

##### Key Methods

**`create(prover &pr, bool merge)`** — Master orchestrator:
1. Calls `compute_e_table()` to precompute the exponential lookup table for softmax.
2. Calls `initParam()` to set up FC layer dimensions, padding to powers of 2, and compute total input layer size.
3. Initializes the input layer and reads FC weights (currently randomized via `rand()%1024`).
4. Generates random curve points for Hyrax commitment generators.
5. Commits model weights via `pr.commitInput()` with 32 threads.
6. For each of the 48 FC layers (4 per transformer block × 12 blocks):
   - Before FC layers 0,2 (mod 4): builds LayerNorm checker (3 phases).
   - Builds the FC layer.
   - Builds the rounding layer.
   - After FC layer 0 (mod 4): builds MHA Q·K^T → Softmax (3 phases).
   - After FC layer 2 (mod 4): builds GELU checker (3 phases).
7. If `merge=true`, calls `merge_layer()` to coalesce same-type layers.
8. Calls `C.initSubset()` to set up Lasso subset mappings.

**`ln_checker_layer1/2/3()`** — Three-phase LayerNorm verification:
- **Phase 1**: For each position `i` in the sequence:
  - Computes `a[co] = x[co]·N - Σx` (centered input).
  - Computes variance `B = 1 + Σa²` and approximate `σ ≈ √B` via integer square root.
  - Verifies `σ² + σ + 1 ≥ B` and `B ≥ σ² - σ` (integer square root bounds).
  - Computes normalized output `y ≈ (w·a/σ + b)` via quantized rounding.
  - Stores rounding residual `δ₁ = (σ²+σ+1-B)(B-σ²+σ)` as a non-negative product check.
  - Also stores `δ₂ = term1 × term2` for each output element's rounding correctness.
- **Phase 2**: Reconstructs the two rounding-bound terms for each output:
  - `term1 = (2y+1)·2^(m-1)·σ + 1 - c₁·w·a·2^(e₁+m) - c₂·b·σ·2^(e₂+m)`
  - `term2 = c₁·w·a·2^(e₁+m) + c₂·b·σ·2^(e₂+m) - (2y-1)·2^(m-1)·σ + 1`
  - Also checks that `Σa²` is consistent (via cross-layer product constraint).
- **Phase 3**: Verifies `δ₂ = term1 × term2` (product of the two terms from Phase 2).

**`gelu_checker_layer1/2/3()`** — Three-phase GELU activation verification:
- Approximates GELU as a piecewise function:
  - For large `|x|` (when `cd·2^ed ≤ cx·2^ex·|x|`): `GELU(x) ≈ x`
  - For small `|x|`: a cubic approximation `GELU(x) ≈ x + |x| - t·(ca·|x|³·2^(ea+2ex) - cb·|x|²·2^(eb+ex) + cc·|x|·2^ec)`
- **Phase 1**: Computes auxiliary values: `|x|`, threshold indicator `t ∈ {0,1}`, `|x|·t`, `x²`, and verifies `|x|² = x²` (absolute value check) and `t² = t` (binary check). Also checks `δ₁ = |x| + 1 ≥ 0` and `δ₂ = t + (1-2t)(C₇·|x| - C₆)` (threshold consistency).
- **Phase 2**: Constructs the two rounding-bound terms using the cubic polynomial approximation coefficients.
- **Phase 3**: Verifies `δ₃ = term1 × term2` (the product check for rounding correctness).

**`multi_head_matrix_QK()`** — Builds the circuit for the multi-head attention Q·K^T computation:
- 12 heads, each with 64-dimensional head size.
- Computes only the **lower-triangular** (causal) part: for each `(i, j)` where `j ≤ i`, sums `Q[i, head·64+k] · K[j, 12·64+head·64+k]` over `k = 0..63`.
- Output size: `12 × len × (len+1)/2` elements.

**`softmax_layer_1/2/3()`** — Three-phase Softmax verification:
- **Phase 1**:
  - For each head and each row `i`, finds `pmax` (maximum attention score).
  - Computes `t[j] ≈ (pmax - p[j]) · scale` (quantized distance from max).
  - Looks up `E[j] = exp_table[t[j]]` from the precomputed table.
  - Computes `ΣE[j]` (sum of exponentials) and `V·E` products (attention-weighted values).
  - Verifies rounding of `t` via `δ₂ = term1 × term2` (same pattern as other rounding checks).
  - Checks `ΣE = Σ E[j]` (sum consistency).
- **Phase 2**: Computes the final softmax output `S[i,j] ≈ E[i,j]·V / ΣE`:
  - Builds term1/term2 for division rounding correctness.
  - Also verifies the rounding `δ₂` from Phase 1 was correct (product cross-check).
- **Phase 3**: Verifies `δ₃ = term1 × term2` for the division rounding.

**`roundLayer()`** — Quantized rounding layer:
- Proves that output `q ≈ round(p × c · 2^m)` by verifying:
  - `(c·p·2^(m+M+1) + 2^M - q·2^(M+1)) × (q·2^(M+1) + 2^M - c·p·2^(m+M+1)) ≥ 0`
  - Encoded as: the circuit gate output is `- δ + (computed product)` and asserts this is zero, where `δ` is precomputed and stored as a positive auxiliary value.

**`fullyConnLayer()`** — Sets up the fully-connected matrix multiplication layer:
- Brute-force computes `C = X × W^T` on the CPU (noted as slow; GPU optimization planned).
- The proving of correctness is handled by the **Thaler'13** protocol in `verifier.cpp`.

**`merge_layer()`** — Optimization pass:
- Collapses all same-type checker layers across the 12 transformer blocks into single merged layers.
- E.g., all 12 `LAYER_NORM_1` layers → 1 merged `LAYER_NORM_1` layer with adjusted gate offsets.
- This reduces circuit depth and improves prover efficiency.
- Uses `uni_interval` / `bin_interval` to track which gates belong to which original block (enabling parallel processing).

**`compute_e_table()`** — Precomputes a lookup table of **655,360 entries** for `exp(-x)`:
- `St = 2^(-9)`, `Se = 2^(-20)`, `table[i] = round(exp(-St·i) / Se)`.
- Guard: `max(t, 1)` to avoid division by zero in softmax when `ΣE = 0`.

---

### Circuit Representation

#### `circuit.h` / `circuit.cpp`
Defines the layered arithmetic circuit data structures:

- **`uniGate`** — Unary gate: `output[g] += scalar × input[u]` (from layer `lu`).
  - `g`: output gate index, `u`: input gate index, `lu`: input layer (0 = input layer, non-zero = previous layer), `sc`: scalar multiplier (`__int128`).
  
- **`binGate`** — Binary (multiplication) gate: `output[g] += scalar × input[u] × input[v]`.
  - `g`: output gate index, `u, v`: input gate indices, `l`: layer flag (0 = both from input, 1 = both from prev layer, 2 = u from prev layer and v from input), `sc`: scalar multiplier.
  - `getLayerIdU()` / `getLayerIdV()`: decode which layer each input comes from.

- **`layer`** — A single circuit layer:
  - `ty`: layer type enum, `size`: number of gates.
  - `uni_gates` / `bin_gates`: vectors of gates.
  - `size_u[2]` / `size_v[2]`: sizes for input subsets ([0] = from input layer, [1] = from previous layer).
  - `bit_length_u[2]` / `bit_length_v[2]`: log₂ sizes (for power-of-2 alignment).
  - `need_phase2`: whether the layer has binary gates requiring a second sumcheck phase.
  - `zero_start_id`: in layers with a "zero-check" section, gates from this index onward must evaluate to zero.
  - `uni_interval` / `bin_interval`: pairs tracking gate ranges for merged layers (enabling parallel processing).
  - `ori_id_u` / `ori_id_v`: mapping from compact subset indices back to original input layer indices.

- **`layeredCircuit`** — The complete circuit: vector of layers.
  - `initSubset()`: Critical post-processing that:
    1. Scans all gates in each layer to find which input-layer indices are referenced.
    2. Builds a compact subset (deduplication) and remaps gate indices.
    3. Computes `ori_id_u` / `ori_id_v` mappings from subset back to original indices.
    4. Sets `bit_length_u/v` for power-of-2 alignment.
    5. This is essential for the Lasso input-consistency argument — it tells the prover/verifier exactly which input-layer elements each circuit layer touches.

---

### Prover

#### `prover.hpp` / `prover.cpp` (~666 lines)
Implements the prover's side of the GKR protocol:

- **`init()`** — Allocates large working arrays (up to `2^28` entries each) for sumcheck bookkeeping: `V_mult`, `mult_array`, `lasso_mult_v`, and their temporary counterparts.

- **`sumcheckInitAll(r_0)`** — Stores the verifier's random evaluation point for the output layer.

- **`sumcheckInit(α, β)`** — Prepares for a new layer's sumcheck with random combination coefficients. The GKR protocol reduces verification of layer `i` to claims about layers `i-1` and `0` (input). When both claims exist, they're linearly combined with `α, β`.

- **`sumcheckInitPhase1(relu_rou)`** — Phase 1 initialization:
  1. Builds `beta_g` table: the multilinear extension of the "equality" polynomial over the output positions.
  2. For each unary gate: accumulates `mult_array[idx][u] += beta_g[g] · sc`.
  3. For each binary gate: accumulates `mult_array[idx][u] += val_v · beta_g[g] · sc` (folding in the second operand's value).
  4. Sets up `V_mult[b][u]` with the actual circuit values at each input position.
  5. Uses **32-thread parallelism** via `ThreadSafeQueue` for large merged layers (processes gate intervals in parallel).

- **`sumcheckUpdate1(r)` / `sumcheckUpdate2(r)`** — Iterative sumcheck rounds: reduces the polynomial one variable at a time. Returns a quadratic polynomial `p(x) = ax² + bx + c` for the verifier to check `p(0) + p(1) = previousSum`. Uses `sumcheckUpdateEach()` internally which interpolates `mult_array` and `V_mult` entries to form the quadratic.

- **`sumcheckInitPhase2()`** — Phase 2: handles the second input wire of binary gates. Re-initializes `beta_u` using the now-fixed `r_u` values, and accumulates:
  - `add_term`: contributions from unary gates (now fully determined).
  - `mult_array[idx][v]`: contributions from binary gates, folding in the first operand's fixed value.

- **`sumcheckLassoInit(sig_u, sig_v, r_uu, r_vv)`** — Initializes the Lasso input-consistency check:
  1. For each circuit layer `i`, computes `beta_g` over the layer's subset indices using the random challenges `σ_u[i]` / `σ_v[i]`.
  2. Accumulates `lasso_mult_v[original_idx]` across all layers — this forms the "multiplicity" vector telling how many times each input element is accessed (weighted by random challenges).

- **`sumcheckFinalize1/2(r, claim_0, claim_1)`** — Finalizes each sumcheck phase, evaluating the interpolated polynomials at the final random point to produce claimed evaluations `V_u0, V_u1` (or `V_v0, V_v1` for phase 2).

- **`commitInput(gens, thr)`** — Commits the entire input layer (weights + auxiliary values):
  1. Converts all `Fr` values to `__int128` for efficient Pippenger commitment.
  2. Calls `prover_commit(vi, gens, l, thr)` which splits the vector into a matrix and commits each row using multi-threaded Pippenger MSM.
  3. Stores the commitment data in `cc` struct for later opening.

- **`Vres(r, size, r_size, layer_id)`** — Evaluates the multilinear extension of a layer's values at a random point `r`. Used to compute the initial claim for the output layer: iteratively reduces by one variable at a time via the standard MLE evaluation algorithm.

---

### Verifier

#### `verifier.hpp` / `verifier.cpp` (~908 lines)
Implements the verifier's side. The verifier drives the protocol by generating random challenges and checking prover responses.

- **`prove(commit_thread)`** — Top-level orchestrator:
  1. Calls `verifyGKR()` — layer-by-layer GKR verification.
  2. Calls `verifyLasso()` — input consistency check.
  3. Calls `openCommit()` — opens the Hyrax commitment.
  4. Prints timing results: matrix multiplication time, total prover time, verifier time, and proof size.

- **`verifyGKR()`** — Layer-by-layer GKR verification:
  - Generates random evaluation point `r` for the output layer.
  - Computes the initial claim via `p->Vres()`.
  - For each layer from output to input:
    - **FCONN layers** (matrix multiplication): Uses the **Thaler'13** protocol:
      1. Sets up evaluation points `r_u`, `r_v` incorporating the random challenges.
      2. Calls `init_book_keeping()` to reduce the weight matrix by fixing `n` variables.
      3. Calls `init_book_keeping_fast()` for the integer-valued weight matrix using a Pippenger-like trick: precomputes `fm[R][val]` tables so that bookkeeping for integer weights becomes a table lookup + accumulation.
      4. Runs `sum_check_product()` — a recursive product sumcheck that verifies `Σ_m f(m)·g(m) = claimed_value` by reducing one variable at a time.
    - **Other layers** (checker layers): Standard sumcheck with two phases:
      1. Phase 1: Fix `u` variables — for each round, prover sends quadratic `p(x)`, verifier checks `p(0)+p(1) = previousSum`, samples random `r`, sets `previousSum = p(r)`.
      2. Phase 2 (if `need_phase2`): Fix `v` variables — same protocol.
      3. After both phases, calls `predicatePhase1/2()` to compute the wiring predicates, then `getFinalValue()` to verify consistency.

- **`predicatePhase1(layer_id)` / `predicatePhase2(layer_id)`** — Compute the "wiring polynomial" values at the fixed random points. For phase 1: `uni_value[idx] = Σ beta_g[g] · beta_u[u] · sc`. For phase 2: `bin_value[l] = Σ beta_g[g] · beta_u[u] · beta_v[v] · sc`. Uses 32-thread parallelism for merged layers.

- **`getFinalValue(claim_u0, claim_u1, claim_v0, claim_v1)`** — Computes the expected value:
  ```
  test = bin_value[0]·(claim_u0·claim_v0) + bin_value[1]·(claim_u1·claim_v1)
       + bin_value[2]·(claim_u1·claim_v0) + uni_value[0]·claim_u0 + uni_value[1]·claim_u1
  ```
  This must equal `previousSum` for verification to pass.

- **`verifyLasso()`** — Input consistency check:
  1. Generates random challenges `σ_u, σ_v` for each layer and random evaluation point `r_u[0]` for the input layer.
  2. Computes `previousSum = Σ (σ_u[i] · claim_u0[i] + σ_v[i] · claim_v0[i])` — the target value.
  3. Calls `p->sumcheckLassoInit()` to build the `lasso_mult_v` vector.
  4. Runs a product sumcheck: verifies `Σ lasso_mult_v[i] · val[0][i] = previousSum`.
     - For the first ~9 rounds, uses **32-thread parallelism** with worker functions.
     - For remaining rounds, uses sequential scalar reduction.
  5. Computes `eval_in = pa2[n][0]` (the evaluation of input layer at `r_u[0]`).
  6. Independently computes `gr = Σ beta_g[ori_id] · beta_u/v[subset_id]` across all layers.
  7. Final check: `eval_in × gr == previousSum`.

- **`openCommit()`** — Opens the Hyrax commitment:
  1. Computes Lagrange basis vectors `L, R` via `brute_force_compute_LR()`.
  2. Calls `hyrax::open()` which:
     - Computes `R^T · w` (matrix-vector product, multi-threaded).
     - Computes `T' = Σ R_k · T_k` (commitment folding).
     - Runs the Bulletproofs `prove_dot_product()` inner-product argument.

- **`init_book_keeping(m, n, vec, offset, ra)`** — Standard bookkeeping: fixes `n` variables of a multilinear polynomial, reducing dimension from `m+n` to `m`. Iteratively folds: `pa[j][k] = pa[j-1][k + 2^(m+n-j)] · ra[m+n-j] + pa[j-1][k] · (1 - ra[m+n-j])`.

- **`init_book_keeping_fast(m, n, vec, ra)`** — Optimized bookkeeping for integer-valued weight matrices:
  1. Precomputes equality polynomial tables `eq1`, `eq2` (splitting `n` variables into two halves).
  2. Builds a fast multiplication table `fm[R][val] = val · eq2[R]` (up to 65536 entries per row).
  3. For each output position `kp`, accumulates `ret[kp] = Σ_L eq1[L] · Σ_R fm[R][vec[kp + (L·2^(n/2)+R)·2^m]]`.
  4. Uses 32 threads for both the table construction and the accumulation.

---

### Polynomial Commitment (Hyrax)

#### `hyrax.hpp` / `hyrax.cpp` (~458 lines)
Implements the **Hyrax polynomial commitment scheme** — a matrix-structured Pedersen commitment with Bulletproofs opening.

- **`perdersen_commit(G1* g, ll* f, int n, G1* W)`** — Optimized Pedersen commitment using **Pippenger's algorithm**:
  - Decomposes each `__int128` scalar `f[i]` into 16-bit blocks (5 blocks → supports up to 2^80).
  - Accumulates group elements into bucket arrays `W[value + block·65536]`.
  - For each bucket bit position, sums the accumulated elements.
  - Final combination via doublings: `ret += gg[j] · 2^j` for each bit position `j`.
  - This is significantly faster than naive MSM for integer scalars.

- **`perdersen_commit(G1* g, int* f, int n, G1* W)`** — Simplified Pippenger for small (32-bit) integer scalars. Uses a single block of 65536 buckets.

- **`perdersen_commit(G1* g, Fr* f, int n)`** — Standard MSM using mcl's optimized `G1::mulVec`.

- **`prover_commit(w, g, l, thread_n)`** — Commits a vector:
  1. Splits the `2^l`-length vector into a `rownum × colnum` matrix.
  2. Commits each row independently → produces `T_k` values.
  3. Uses `ThreadSafeQueue` for multi-threaded row commitment.

- **`compute_RT(w, R, l, g, ret)`** — Computes `R^T · w` (the half-evaluation matrix-vector product). Multi-threaded: splits columns across `hardware_concurrency()` threads.

- **`compute_Tprime(l, R, Tk)`** — Computes `T' = Σ R_k · T_k` for commitment folding.

- **`bullet_reduce(gamma, a, g, n, G, x, y)`** — Recursive **Bulletproofs inner-product argument**:
  1. Base case: `n=1` → return the single element.
  2. Prover computes cross-products `x₁·a₂`, `x₂·a₁`.
  3. Prover sends folded commitments `γ₋₁`, `γ₁`.
  4. Verifier samples random `c`.
  5. Both compute folded vectors: `a' = a[0..n/2]·c⁻¹ + a[n/2..n]·c`, similarly for `g'`.
  6. For `n ≥ 2048`, generator folding `g'` is parallelized with 16 threads.
  7. Recurse with `n/2`.

- **`prove_dot_product(comm_x, comm_y, a, g, G, x, y, n)`** — Full inner-product argument wrapper. Runs `bullet_reduce()` and asserts `p.y == p.x · p.a` and `p.gamma == p.g · p.x + G · p.y` at the base case.

- **`hyrax::open(w, r, eval, G, g, L, R, tk, l)`** — Full commitment opening:
  1. Computes `R^T · w` via `compute_RT()`.
  2. Computes `T' = Σ R_k · T_k` via `compute_Tprime()`.
  3. Runs `prove_dot_product(T', G·eval, L, g, G, RT, eval, colnum)`.
  4. Returns `(prover_time, verifier_time)` pair.

---

### Supporting Modules

#### `polynomial.h` / `polynomial.cpp`
Univariate polynomial classes used during sumcheck:
- `linear_poly` (degree 1: `ax + b`), `quadratic_poly` (degree 2: `ax² + bx + c`), `cubic_poly` (degree 3), `quadruple_poly` (degree 4), `quintuple_poly` (degree 5).
- Each supports addition, scalar multiplication, and evaluation.
- `linear_poly * linear_poly → quadratic_poly` and `quadratic_poly * linear_poly → cubic_poly` allow symbolic polynomial manipulation during sumcheck rounds.

#### `utils.hpp` / `utils.cpp` (~262 lines)
Utility functions:
- **`initBetaTable(beta_g, gLength, r_0, r_1, α, β)`** — Computes the multilinear extension "equality" polynomial `β(r, x) = α·eq(r₀, x) + β·eq(r₁, x)` for all Boolean inputs. Uses a split-half optimization: precomputes `beta_f` and `beta_s` for the first/second halves, then combines via `beta_g[i] = beta_f[i & mask] · beta_s[i >> half]`. Multi-threaded (32 threads, splitting into 1024 work chunks) when `gLength ≥ 15`.
- **`initBetaTable(beta_g, gLength, r, init)`** — Single-point version (used for Lasso and phase 2).
- **`initLayer(circuit, size, ty)`** — Sets `circuit.size = size`, `circuit.bit_length = ceilPow2BitLength(size)`, `circuit.ty = ty`.
- **`ceilPow2BitLength(n)`** — Returns ⌈log₂(n)⌉. Used heavily for power-of-2 padding.
- **`matIdx(x, y, n)`** — 2D → 1D index: `x·n + y`.
- **`cubIdx(x, y, z, n, m)`** — 3D → 1D index.
- **`tesIdx(w, x, y, z, n, m, l)`** — 4D → 1D index.

#### `timer.hpp` / `timer.cpp`
Simple high-resolution timer for benchmarking prover/verifier times. Uses `std::chrono::high_resolution_clock`.

#### `typedef.hpp`
Basic integer type aliases:
- `u64`, `u32`, `u8` — unsigned integers.
- `i64`, `i32`, `i8` — signed integers.
- `ll = __int128` — 128-bit signed integer (used extensively for large intermediate products in quantized arithmetic).

#### `global_var.hpp`
Global type definitions and constants:
- `F = Fr` — BN254 scalar field element (~254-bit prime field).
- `G = G1` — BN254 curve point.
- `F_ONE = Fr::one()`, `F_ZERO = Fr(0)`.
- `F_BYTE_SIZE = 16`, `G_BYTE_SIZE = 32` — used for proof size tracking.
- Output table format constants for benchmarking display.

#### `hyrax.hpp` — `ThreadSafeQueue<T>`
A lock-based concurrent work queue used throughout:
- `Push(T)`, `TryPop(T&)`, `WaitPop(T&)`, `Empty()`, `Size()`, `Clear()`.
- Used for dispatching work to thread pools across prover and verifier.

---

## Non-linear Operation Verification Strategy

The key challenge in zkGPT is verifying non-linear operations (LayerNorm, GELU, Softmax) that involve divisions, square roots, and exponentials — none of which exist natively in arithmetic circuits over finite fields.

### Quantization + Rounding Proofs
Every non-linear operation is decomposed as:
1. **Quantize** all values to integers with explicit scales `(c, e)`.
2. **Approximate** the non-linear function with integer arithmetic.
3. **Prove rounding correctness** by showing that the rounded value `q` satisfies:
   - `(2q + 1) · denominator ≥ numerator` (upper bound)
   - `numerator ≥ (2q - 1) · denominator` (lower bound)
   - Which is equivalent to proving that `δ = upper_term × lower_term ≥ 0` (a non-negative product).
   - The δ value is precomputed and stored as an auxiliary input; the circuit checks that the product matches.

### Three-Phase Decomposition
Each non-linear operation is split into 3 circuit layers to keep the gate degree manageable (at most degree-2 gates):
- **Phase 1**: Compute intermediate values and basic consistency checks (absolute values, thresholds, sums).
- **Phase 2**: Compute the two rounding-bound terms (involving products of intermediate values with constants).
- **Phase 3**: Verify that `term1 × term2 = δ` (the precomputed auxiliary — a single binary gate).

### Lookup Tables for Exponentiation
Softmax uses a precomputed table of **655,360 entries** for `exp(-x)` at fixed precision (`St = 2^(-9)`, `Se = 2^(-20)`). The prover stores `(t, E)` pairs in the input layer and the circuit verifies:
1. The rounding `t ≈ (pmax - p[j]) · scale` is correct (via rounding proof).
2. The lookup `E = table[t]` is consistent (via Lasso lookup argument).

---

## Matrix Multiplication Protocol (Thaler'13)

Fully-connected layers are the most expensive part. They are handled via a specialized protocol:

1. The verifier selects random evaluation points `r_u` (for the output matrix rows/shared dimension) and `r_v` (for the shared dimension/columns).
2. The matrix product `C = A × B` is verified by checking that `C̃(r) = Σ_m Ã(r_u, m) · B̃(m, r_v)` where tilde denotes multilinear extensions.
3. This is reduced to a **product sumcheck** `sum_check_product()` over the shared dimension `m`.
4. At each round, the product sumcheck sends a degree-2 polynomial `g(x) = ax² + bx + c` and verifies `g(0) + g(1) = previousSum`.
5. `init_book_keeping_fast()` uses a Pippenger-like trick: since the weight matrix has integer values (all ≤ 1024), it precomputes `fm[R][val] = val · eq2[R]` tables, turning the inner sum into a table lookup. This is significantly faster than the standard approach for integer-valued matrices.

---

## Performance & Threading

The implementation is heavily multi-threaded (32 threads by default for most operations):

| Operation | Thread Count | Parallelization Strategy |
|---|---|---|
| Beta table initialization | 32 | Split into 1024 chunks via `ThreadSafeQueue` |
| Sumcheck Phase 1/2 bookkeeping | 32 | Parallel gate processing via intervals |
| Commitment (Pippenger MSM) | configurable (32 for main commit) | One thread per matrix row |
| Bulletproofs generator folding | 16 | Column-based splitting |
| Lasso sumcheck | 32 | Worker threads per chunk |
| Matrix bookkeeping (`init_book_keeping_fast`) | 32 | Row-parallel table construction + accumulation |
| RT computation | `hardware_concurrency()` | Column-based splitting |
| Predicate computation | 32 | Interval-based gate processing |

**Recommended hardware**: 16+ core CPU, 200+ GB RAM (the circuit for GPT-2 with 12 transformer blocks is very large — working arrays are `2^28` entries).

---

## Data Flow Through One Transformer Block

```
Input (30×768)
    │
    ├─── LayerNorm ──────── (ln_checker_1/2/3 — 3 circuit layers)
    │
    ├─── FC1 (768→2304) ─── (fullyConnLayer — 1 FCONN layer)
    │
    ├─── Round ──────────── (roundLayer — 1 RELU layer)
    │
    ├─── Split Q/K/V (each 768)
    │
    ├─── MHA Q·K^T ──────── (multi_head_matrix_QK — 1 MHA_QK layer)
    │    (12 heads × 64-dim, causal)
    │
    ├─── Softmax ─────────── (softmax_1/2/3 — 3 circuit layers)
    │    (with exp lookup table)
    │
    ├─── FC2 (768→768) ───── (fullyConnLayer — 1 FCONN layer)
    ├─── Round ────────────── (roundLayer — 1 RELU layer)
    │
    ├─── LayerNorm ──────── (ln_checker_1/2/3 — 3 circuit layers)
    │
    ├─── FC3 (768→2304) ──── (fullyConnLayer — 1 FCONN layer)
    ├─── Round ────────────── (roundLayer — 1 RELU layer)
    │
    ├─── GELU ──────────── (gelu_checker_1/2/3 — 3 circuit layers)
    │    (piecewise cubic approximation)
    │
    ├─── FC4 (2304→768) ──── (fullyConnLayer — 1 FCONN layer)
    └─── Round ────────────── (roundLayer — 1 RELU layer)
            │
            ▼
      Output (30×768) → next block
```

Each transformer block produces ~21 circuit layers. After all 12 blocks, layer merging collapses same-type checker layers, reducing the total circuit depth significantly.

---

## Build System

- **CMake** (minimum 3.10), C++14 standard.
- **Compiler flags**: `-mcmodel=large -O3 -lpthread -pthread` (large memory model needed for the large working arrays).
- Depends on **GMP** (GNU Multiple Precision) library and the **mcl** library (included as a Git submodule under `3rd/mcl/` for BN254 curve operations).
- The `src/CMakeLists.txt` builds a static library `gpt_lib` from all source files (excluding `main*`), then links `demo_llm_run` against `gpt_lib`, `mcl`, and `mclbn256`.
- Build: `./build.sh` → creates `cmake-build-release/`.
- Run: `./llm.sh` → builds and executes `demo_llm_run`.

---

## Key Limitations & TODOs (noted in code)

1. **Circuit initiation is slow** — brute-force CPU matrix multiplication in `fullyConnLayer()` computes each FC layer sequentially on a single thread. GPU acceleration is planned.
2. **Weight values are randomized** — `readFconWeight()` uses `rand()%1024` instead of loading from a real GPT-2 checkpoint. This is sufficient for benchmarking the proof system but doesn't represent real inference.
3. **Bias addition is disabled** — FC layers don't add bias for simplicity (commented out).
4. **LayerNorm channel dimensions** — Several `TODO` comments indicate `channel_out` vs. `real_cn_in` may need adjustment for accurate LayerNorm computation.
5. **GELU rounding for Python compatibility** — A `TODO` notes potential discrepancies between C++ and Python GELU rounding behavior.
6. **Softmax exp table guard** — `max(t, 1)` avoids `ΣE = 0` division by zero in edge cases but introduces a small approximation error.
7. **Input data** — The `calcInputLayer()` reads from a CSV file (path hardcoded in constructor, though the actual data format is `len × 768` floating-point values).
8. **Memory usage** — Working arrays of `2^28` entries (~1 GB each) and multiple commitment/bookkeeping arrays require 200+ GB RAM.
