# DeepSeek Flash Lab Verification Report

**Date:** 2026-09-20  
**Command:** `python verification/run_full_verification.py` → **ok=true** (57 pytest tests)  
**Hardware:** CPU only — Python 3.12.11, PyTorch 2.13.0+cu130, **CUDA unavailable**  
**Rule followed:** Audit first; no silent “feature completion” of deferred Stage-2+ unlocks.

---

## Executive Summary

The lab is **not** a fully verified V4 / V4.1 Flash clone. What **is** verified is Stage-1 primitives (tokenizer, embedding, RMSNorm, RoPE) plus a **provisional** V4 Mini stack (attention/CSA/HCA/MoE) that can train and overfit on CPU. Public APIs for DSA, compression, CED, CSA2, FP4 KV, and SWA replay are **deferred stubs**. Soft-pass deferred tests are a **test-quality failure**. No incremental KV cache exists. FP4/FP8 are **simulated** float rounding. Optimization evidence with BEFORE→AFTER→Δ exists only for Stage-1 B0 and RoPE formulation (A1 slower).

**Is this implementation actually working correctly?**  
→ **Partially.** Stage-1 math is correct. Later paper mechanisms are either provisional miniatures, simulations, or not implemented behind the public deferred modules. Do not treat the repo as paper-verified end-to-end.

---

## Repository Audit

- Two Mini packages + `bench/`, `configs/`, `datasets/`, `docs/`, `results/`, legacy `oldDATA/`.
- Stage gate (`docs/progress.md`): Stage 1 ✅; Stages 2–4 **locked**.
- Collision hazard: both packages use `model` / `optimization` top-level names (`verification/pkg_import.py` isolates them).

## Paper-to-Code Audit

See `verification/paper_to_code.md` and `verification/PAPER_CLAIM_AUDIT.md`.

## Mathematical Verification

| Item | Result |
|------|--------|
| Config `d_model == n_heads * d_head` | PASS (V4 + V4.1 defaults) |
| V4.1 override `n_decoder_layers` without modes | Assert fails (documented) |
| RoPE vs reference | max_abs_error < 1e-5 |
| Causal attention core vs reference | max_abs_error < 1e-5 |
| Router vs reference | indices equal; weights < 1e-5 |

## Tensor Shape Verification

V4 Mini, CSA/HCA/MLA, V4.1 ToyDSV41 forward shapes: **PASS** (`test_shapes.py`).

## Numerical Verification

Finite forward/backward on V4 Mini: **PASS**. CSA ≠ dense (expected gap). Fake-quant matches reference.

## Attention Verification

Causal leakage + SWA window: **PASS**. Not wired into Stage-1 official train loop.

## Compression Verification

Helpers compress sequence length (ratio >1.5). Public `compression.py`: **NOT IMPLEMENTED**.

## DSA Verification

Public `dsa.py`: **NOT IMPLEMENTED**. CSA/HCA provisional only. No measured sparse speedup.

## MLA/KV Verification

Tiny MLA exists. **No** runtime KV cache; decode recomputes. Accounting-only notes: **THEORETICAL ONLY**.

| Metric | Full Recompute | KV Cache | Difference |
|--------|---------------:|---------:|-----------:|
| Memory | full activations each step | N/A | N/A |
| Decode latency | measured recompute only | N/A | N/A |
| Tokens/sec | microbench CPU ~prefill | N/A | N/A |
| Output error | N/A | N/A | cannot compare |

## MoE Verification

Shared+routed path finite; active params ≪ total; router reference match: **PASS** at toy scale.

## CED Verification

Provisional ToyDSV41 CED: runs. Public CED: deferred. Bounded replay zeros prefix logits: **correctness hazard** for full-sequence LM loss.

## CSA2 Verification

Provisional `_csa2.py` modes exist. Public API deferred. No FLOPs/latency proof of speedup.

## FP4 KV Verification

Simulated; remains float32. Public module deferred. **No** proven memory save from packing.

## SWA Replay Verification

Tail truncate implemented; exact multi-layer SWA state rebuild **not** implemented. Public module deferred.

## Training Verification

- Stage-1 official: loss ~1.72 train / ~1.68 val on Level-0 (CPU) — **PASS** for Stage-1 gate.
- Provisional V4 Mini overfit 40 steps: loss drops ≥15% — **PASS**.
- Checkpoint round-trip logits identical — **PASS**.

## Gradient Verification

See `gradient_audit.md`. Finite grads verified; full finite-difference suite not completed.

## Inference Verification

Tokenize → logits → generate → decode: **PASS** on V4 Mini. Deterministic near-zero temperature with fixed seed: **PASS**.

## Quantization Verification

| Metric | FP32/BF16 | Quantized | Δ |
|--------|----------:|----------:|--:|
| Model Memory | baseline float | same float tensors | ~0 (sim) |
| Peak Memory | — | — | not reduced by packing |
| Prefill/Decode | — | — | not measured as win |
| Max Error | 0 | >0 for 4-bit sim | expected lossy |

## Memory Verification

Independent param count matches `num_parameters()`. Stage-1 estimate ~1.2 MB model+opt+act (tiny). CPU peak ~789 MB (process). GPU: **unavailable**.

## CPU Verification

Primary path. Micro prefill T=64 ≈ **7.5 ms** (this host). Stage-1 ~51k tokens/s (train loop metric from verify).

## 2 GB GPU Verification

**Cannot verify** — `cuda_available=false` on this machine. Configs declare `memory_budget_mb: 2048` but no on-device fit test was run.

## Benchmark Verification

| Artifact | Status |
|----------|--------|
| `results/v4/stage01/baseline/` | Present |
| `results/v4/rope_formulation/` A0 vs A1 | Present — A1 **REGRESSION** on `forward_ms` |
| Long-seq 128–2048 gen grids | Not run (no unlocked Stage-2+ protocol runs) |
| DSA/CED/FP4 BEFORE/AFTER | Missing |

## Reproducibility Verification

Eval forward equal under same seed: **PASS**. Different seeds differ: **PASS**.

## Regression Verification

Harness exists (`bench/regression.py`). Only RoPE candidate compared; correctly marked worse.

## Known Simplifications

Char vocab; tiny d_model/layers/experts; mean-pool compression; fake FP4/FP8; educational MLA; toy Engram/ViT/DSpark.

## Known Bugs / Hazards

1. **Deferred soft-pass tests** (`assert True`) — misleading green.
2. **Public vs provisional split** — stubs say “not ready” while `_*.py` / `mla.py` contain real code; easy to mis-claim completeness.
3. **No incremental KV cache** — generate is O(T²) recompute.
4. **V4.1 replay zero-pads prefix logits** — unsafe for full-seq CE with replay on.
5. **CSA/HCA RoPE on Q only** — compressed keys unrotated.
6. **Package name collision** between Minis.
7. **Stage-1 train ≠ `attention.py`** — temporary inline block.

## Missing Components

Public: DSA, compression, CED, causal_encoder, CSA2, FP4 KV, SWA replay, multimodal fusion, vision_encoder, csa2_cache. Runtime KV cache for both Minis.

## Optimization Results

| Optimization | Memory Δ | Prefill Δ | Decode Δ | Throughput Δ | KV Δ | Quality Δ | Verdict |
|--------------|---------:|----------:|---------:|-------------:|-----:|----------:|---------|
| Stage-1 B0 | baseline | baseline | — | baseline | — | baseline | reference |
| RoPE A1 rotate-half | ~0 | slower (`forward_ms` Improve% ≪ 0) | — | worse | — | ≈0 abs err | **REGRESSION** |
| DSA/CSA2/FP4/CED | — | — | — | — | — | — | **no measured Δ** |

## Final Status

| Area | Status |
|------|--------|
| Stage-1 primitives | ✅ PASS |
| Provisional V4 Mini stack | ⚠️ PARTIALLY IMPLEMENTED / SCALE-ADAPTED |
| Public Stage-2+ paper APIs | ❌ NOT IMPLEMENTED |
| Runtime KV / real FP4 | ❌ NOT IMPLEMENTED / ⚠️ SIMULATED |
| Optimization claims beyond RoPE | ⚠️ THEORETICAL ONLY or missing |
| Soft-pass deferred tests | ❌ FAIL |

## Final Scoreboard

| Component | Implementation | Math | Tests | Training | Inference | Benchmark | Status |
| --------- | -------------- | ---- | ----- | -------- | --------- | --------- | ------ |
| Tokenizer | Yes | N/A | Yes | Yes | Yes | No | ⚠️ PASS WITH SIMPLIFICATION |
| RoPE | Yes | Yes | Yes | Yes | Yes | Yes (Δ) | ✅ PASS |
| Attention | Provisional | Yes (core) | Yes | Partial* | Yes | No | ⚠️ PARTIALLY IMPLEMENTED |
| Compression | Stub / helpers | Partial | Partial | No | No | No | ⚠️ PARTIALLY / ❌ public |
| DSA | Stub | — | Soft-pass | No | No | No | ❌ NOT IMPLEMENTED |
| MLA | Tiny | Partial | Shape | Via model | Via model | No | ⚠️ SCALE-ADAPTED |
| MoE | Yes | Partial | Yes | Overfit | Yes | No | ⚠️ SCALE-ADAPTED |
| KV Cache | Accounting only | — | Documents absence | No | Recompute | Theoretical | ❌ NOT IMPLEMENTED |
| Quantization | Fake | Partial | Yes | No | No | No | ⚠️ SIMULATED |
| CED | Provisional / stub | Partial | Partial | Smoke | Smoke | No | ⚠️ PARTIALLY / ❌ public |
| CSA2 | Provisional / stub | Partial | Partial | Smoke | Smoke | No | ⚠️ PARTIALLY / ❌ public |
| FP4 KV | Sim / stub | Partial | Partial | No | No | No | ⚠️ SIMULATED / ❌ public |
| SWA Replay | Trim / stub | Partial | Documents zeros | Hazardous | Partial | Theoretical | ⚠️ PARTIALLY IMPLEMENTED |

\*Stage-1 training uses `TemporaryCausalBlock`, not `model.attention.py`.

---

## Answers to the mandatory questions

1. **What is correct?** Stage-1 tokenizer/embedding/RMSNorm/RoPE; causal/SWA core math; router top-k vs reference; provisional V4 Mini can forward/backward/overfit; checkpoint I/O.
2. **What is mathematically verified?** RoPE, RMSNorm form, causal attention scores, router scoring, config head arithmetic.
3. **What is experimentally verified?** Stage-1 train/val metrics; RoPE A0 vs A1 delta (regression); V4 Mini overfit; CPU micro latency; causal leakage tests.
4. **What is only simplified?** Char tokenizer, toy MoE sizes, mean-pool compression, educational MLA/CSA/HCA/CSA2/CED.
5. **What is broken?** Soft-pass deferred tests; claiming Stage-2+ complete; replay+full CE; missing KV cache correctness.
6. **What is missing?** Public DSA/compression/CED/CSA2/FP4/SWA modules; incremental KV; packed FP4; measured opt Δ for Stage-2+.
7. **What cannot be verified on current hardware?** Any 2 GB GPU / CUDA path; long-context 1k–2k GPU grids.
8. **What benchmarks prove an optimization works?** **None positive.** Only RoPE A1 proves a candidate was *worse*. Stage-1 B0 is a baseline, not an optimization win.
9. **Memory savings?** Not proven for FP4/DSA/CSA2. Theoretical accounting only.
10. **Latency savings?** Not proven for paper mechanisms. RoPE A1 increased latency.
11. **Throughput improvements?** Not proven beyond Stage-1 baseline recording.
12. **Quality tradeoffs?** Fake quant / FP4 sim introduce error; CSA approximates dense; replay zeros prefix logits.
13. **What should be fixed first?** (priority order)
    1. Replace soft-pass deferred tests with hard skip/`pytest.importorskip` or explicit `xfail`.
    2. Unlock Stage-2 properly: wire `attention.py` into official train + B0, then causal suite as gate.
    3. Implement real incremental KV cache + cache vs recompute greedy parity.
    4. Resolve public-stub vs provisional-code dual paths (one source of truth).
    5. Only then CED/CSA2/FP4 with BEFORE/AFTER benches under `docs/optimization_protocol.md`.

---

## How to re-run

```bash
python verification/run_fast_verification.py
python verification/run_numerical_verification.py
python verification/run_benchmark_verification.py
python verification/run_full_verification.py
```
