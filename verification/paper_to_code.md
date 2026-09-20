# Paper → Code Verification

Evidence from code inspection + `verification/tests` + Stage-1 artifacts. Status vocabulary matches the audit prompt.

---

## Component: Tokenizer

### Mathematical definition

Character identity map \(c \mapsto \mathrm{id}(c)\) over corpus alphabet. Not BPE.

### Expected behavior

Deterministic encode/decode for in-vocab characters; OOV → space/0.

### Implementation

- file: `DeepSeekFlashV4-Mini/tokenizer/tokenizer.py`
- class: `CharTokenizer`
- function: `encode` / `decode`

### Verification

- shape/round-trip: `verification/tests/test_tokenizer.py`
- numerical: N/A
- edge: empty, Unicode, unknown
- integration: inference pipeline test

### Status

⚠️ PASS WITH SIMPLIFICATION — labeled `[SIMPLIFIED]` in Stage-1; not production ~129k BPE.

---

## Component: RMSNorm

### Mathematical definition

\[
\mathrm{RMS}(x)=\sqrt{\mathrm{mean}(x^2)+\varepsilon},\quad y=(x/\mathrm{RMS}(x))\odot\gamma
\]

### Expected behavior

Normalize last dim; preserve dtype; learnable gain only.

### Implementation

- file: `DeepSeekFlashV4-Mini/model/normalization.py`
- class: `RMSNorm`

### Verification

- Stage-1 train uses it; gradient test in `test_training.py`

### Status

✅ PASS — `[FAITHFUL]` miniature.

---

## Component: RoPE

### Mathematical definition

For pair \((2i,2i+1)\) at position \(t\):

\[
\theta_i = \mathrm{base}^{-2i/d},\quad
\begin{pmatrix}x'\\y'\end{pmatrix}
=
\begin{pmatrix}\cos(t\theta_i)&-\sin(t\theta_i)\\ \sin(t\theta_i)&\cos(t\theta_i)\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix}
\]

Complex form: \(x_c \cdot e^{i t \theta}\).

### Expected behavior

Position-0 identity; L2 norm preserved per head; matches rotate-half reference.

### Implementation

- file: `DeepSeekFlashV4-Mini/model/rope.py`
- functions: `rope_freqs`, `apply_rope`, `rope_reference_rotate_half`

### Verification

- shape, pos-0, norm, gradient, vs `verification/reference.reference_rope`
- max abs error < 1e-5 vs reference (`test_rope.py`)
- Stage-1 + RoPE A0 vs A1 measured delta (`results/v4/rope_formulation/`)

### Status

✅ PASS — mathematically verified; A1 rotate-half measured **REGRESSION** vs A0 complex (keep A0).

---

## Component: Attention (causal / SWA)

### Mathematical definition

\[
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
\]

Causal \(M_{ij}=-\infty\) for \(j>i\); SWA also requires \(i-j < n_{\mathrm{win}}\).

### Expected behavior

Prefix outputs invariant to future tokens; window blocks distant past.

### Implementation

- file: `DeepSeekFlashV4-Mini/model/attention.py`
- classes: `CausalSelfAttention`, `SlidingWindowAttention`

### Verification

- core scores vs `reference_causal_attention`
- causal leakage + SWA window tests **PASS**
- Stage gate (`docs/progress.md`): Stage 2 **not unlocked**; Stage-1 train uses **separate** `TemporaryCausalBlock`

### Status

⚠️ PARTIALLY IMPLEMENTED — code works numerically, but not the Stage-1 training path and no formal attention B0 benchmark.

---

## Component: Compression / CSA / HCA / DSA

### Mathematical definition (paper intent)

Compress KV along sequence (overlap / non-overlap pools), index Top-K sparse attention, combine with SWA.

### Expected behavior

Fewer KV positions than \(T\); sparse gather; causal constraints on compressed indices.

### Implementation

- public: `compression.py`, `dsa.py` → **DEFERRED stubs**
- provisional: `mla.py` (`compress_overlap`, `compress_nonoverlap`, `CSA`, `HCA`)

### Verification

- compression ratio >1.5 for overlap helper
- CSA finite outputs; **not** equal to dense attention (expected)
- CSA applies RoPE to **Q only** (compressed K not rotated) — paper-faithfulness gap
- public DSA/compression: ❌ NOT IMPLEMENTED

### Status

- helpers/CSA/HCA: ⚠️ PASS WITH SIMPLIFICATION
- public DSA/compression API: ❌ NOT IMPLEMENTED

---

## Component: MLA

### Mathematical definition (DeepSeek-V2 style, educational)

Down-project to latent \(c_{KV}\), up-project to multi-head K/V.

### Implementation

`MLA` in `V4/model/mla.py` and `V4.1/model/mla.py`

### Verification

Shape + causal mask; not compared to production MLA ranks.

### Status

⚠️ SCALE-ADAPTED

---

## Component: MoE / Router / Shared experts

### Mathematical definition (lab)

Scores: \(\sqrt{\mathrm{softplus}(W x)}\); select top-\(k\) (+ bias for balance); renormalize; scale by `routed_scaling`. Shared experts always on; routed add weighted expert FFNs (SwiGLU).

### Implementation

`router.py`, `experts.py`, `shared_experts.py`, `moe.py`

### Verification

- router vs `reference_router_topk` (indices + weights)
- shared always active; active params ≪ total
- hash router deterministic
- `n_activated > n_routed` rejected by reference; model `topk` would error at runtime

### Status

⚠️ SCALE-ADAPTED — trainable; not production DeepSeekMoE (capacity, EP, aux loss schedule).

---

## Component: KV Cache

### Expected behavior

Prefill store KV; decode append; logits match full recompute.

### Implementation

Accounting notes only (`optimization/kv_cache.py`). `forward` / `generate` recompute full context — **no** `past_key_values`.

### Verification

`test_kv_cache.py` asserts missing cache API.

### Status

❌ NOT IMPLEMENTED (runtime). ⚠️ THEORETICAL ONLY (byte accounting).

---

## Component: Quantization / FP4 KV

### Expected behavior

True FP4: scale → pack 4-bit → unpack → dequant; memory ≈ 4× smaller than FP16.

### Implementation

- `fake_quantize` / `mxfp4_sim` round in float32
- CSA2 `_quantize_fp4_sim` same idea
- public `fp4_kv_cache.py` DEFERRED

### Verification

Fake-quant matches reference; dtype remains float32; **no packed storage**.

### Status

⚠️ SIMULATED — do not claim practical FP4 memory savings.

---

## Component: CED (V4.1)

### Expected behavior

Causal encoder → compressed/global state → decoder; prefill vs decode activation split.

### Implementation

- public `ced.py` / `causal_encoder.py` DEFERRED
- provisional `ToyDSV41` encoder/decoder stacks in `_transformer.py`

### Verification

Forward/backward finite; with bounded replay, **prefix logits are zeros** (documented hazard for LM loss).

### Status

⚠️ PASS WITH SIMPLIFICATION (provisional). Public API ❌ NOT IMPLEMENTED.

---

## Component: CSA2

### Expected behavior

Full / Reindex / Reuse modes + hierarchical candidate pool + cross-layer shared KV.

### Implementation

`_csa2.py` provisional; public `csa2.py` DEFERRED.

### Verification

Runs inside ToyDSV41; no dense-vs-CSA2 numerical fidelity suite beyond finiteness.

### Status

⚠️ PASS WITH SIMPLIFICATION / ⚠️ SCALE-ADAPTED

---

## Component: SWA Bounded Replay

### Expected behavior

Persist global KV; rebuild SWA from last \(n_{\mathrm{win}}\) (paper: not exact \(L\times n_{\mathrm{win}}\)).

### Implementation

`swa_bounded_replay` truncates hidden states; decoder runs on tail; logits padded with zeros for prefix.

### Verification

`test_v41_replay_zeros_prefix_logits` confirms zero prefix logits when replayed.

### Status

⚠️ PARTIALLY IMPLEMENTED — truncation demo, not exact SWA state rebuild; public module deferred.
