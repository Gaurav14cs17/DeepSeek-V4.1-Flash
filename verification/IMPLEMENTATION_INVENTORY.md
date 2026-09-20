# Implementation Inventory

Generated from repository inspection + `python verification/run_full_verification.py` (2026-09-20).

**Legend**

| Status | Meaning |
|--------|---------|
| ✅ PASS | Evidence from math + tests |
| ⚠️ PASS WITH SIMPLIFICATION | Works but not paper-faithful |
| ⚠️ SCALE-ADAPTED | Tiny educational scale |
| ⚠️ THEORETICAL ONLY | Accounting / complexity claims without measured speedup |
| ⚠️ PARTIALLY IMPLEMENTED | Provisional code exists; public API deferred or incomplete |
| ❌ FAIL | Broken / misleading / soft-pass tests |
| ❌ NOT IMPLEMENTED | Deferred stub (`DEFERRED=True` / `NotImplementedError`) |

| Component | File(s) | Implemented | Tested | Benchmarked | Status |
| --------- | ------- | ----------: | -----: | ----------: | ------ |
| Tokenizer (char) | `DeepSeekFlashV4-Mini/tokenizer/` | Yes | Yes (`verification/tests/test_tokenizer.py`) | No | ⚠️ PASS WITH SIMPLIFICATION |
| Embedding | `V4/model/embedding.py` | Yes | Yes (Stage-1 + verification) | Stage-1 B0 | ✅ PASS |
| RMSNorm | `V4/model/normalization.py` | Yes | Yes | Stage-1 B0 | ✅ PASS |
| RoPE | `V4/model/rope.py`, `V4.1/model/rope.py` | Yes | Yes (ref + Stage-1) | RoPE A0/A1 Δ | ✅ PASS |
| Causal / SWA Attention | `V4/model/attention.py` | Yes (provisional) | Yes (shape + causal) | No formal B0 | ⚠️ PARTIALLY IMPLEMENTED |
| Compression (public) | `V4/model/compression.py` | No (stub) | Soft-pass only | No | ❌ NOT IMPLEMENTED |
| Compression (helpers) | `V4/model/mla.py` `compress_*` | Yes | Yes (ratio test) | No | ⚠️ SCALE-ADAPTED |
| DSA (public) | `V4/model/dsa.py` | No (stub) | Soft-pass only | No | ❌ NOT IMPLEMENTED |
| CSA / HCA | `V4/model/mla.py` | Yes (provisional) | Shape + finite | No | ⚠️ PASS WITH SIMPLIFICATION |
| MLA | `V4/model/mla.py`, `V4.1/model/mla.py` | Yes (tiny LoRA-KV) | Shape | No | ⚠️ SCALE-ADAPTED |
| MoE / Router / Experts | `V4/model/{moe,router,experts,shared_experts}.py` | Yes | Yes (router ref + MoE) | No | ⚠️ SCALE-ADAPTED |
| mHC / MTP | `V4/model/{mhc,mtp}.py` | Yes | Via full model train | No | ⚠️ SCALE-ADAPTED |
| Full V4 Mini model | `V4/model/model.py` + `decoder.py` | Yes | Overfit + checkpoint | Micro CPU only | ⚠️ PARTIALLY IMPLEMENTED |
| Stage-1 train model | `V4/training/stage01_train.py` `TemporaryCausalBlock` | Yes | Stage-1 gate | B0 JSON | ✅ PASS (experimental attn) |
| KV Cache (runtime) | — | No incremental cache | Documented absence | No | ❌ NOT IMPLEMENTED |
| KV accounting | `V4/optimization/kv_cache.py` | Yes (notes only) | Smoke | Theoretical | ⚠️ THEORETICAL ONLY |
| Quantization | `*/optimization/quantization.py` | Fake-quant only | Yes vs ref | No | ⚠️ SIMULATED / THEORETICAL |
| CED (public) | `V4.1/model/ced.py` | No (stub) | Soft-pass | No | ❌ NOT IMPLEMENTED |
| Causal encoder (public) | `V4.1/model/causal_encoder.py` | No (stub) | Soft-pass | No | ❌ NOT IMPLEMENTED |
| CED (provisional) | `V4.1/model/_transformer.py` ToyDSV41 | Yes | Shape + train smoke | No | ⚠️ PASS WITH SIMPLIFICATION |
| CSA2 (public) | `V4.1/model/csa2.py` | No (stub) | Soft-pass | No | ❌ NOT IMPLEMENTED |
| CSA2 (provisional) | `V4.1/model/_csa2.py` | Yes | Shape via model | No | ⚠️ PASS WITH SIMPLIFICATION |
| FP4 KV (public) | `V4.1/optimization/fp4_kv_cache.py` | No (stub) | Soft-pass | No | ❌ NOT IMPLEMENTED |
| FP4 (provisional sim) | `_csa2._quantize_fp4_sim` | Fake round in float | Yes (not packed) | No | ⚠️ SIMULATED |
| SWA Replay (public) | `V4.1/optimization/swa_replay.py` | No (stub) | Soft-pass | No | ❌ NOT IMPLEMENTED |
| SWA Bounded Replay | `V4.1/model/kv_stats.py` | Trim last `n_win` | Yes (zeros prefix logits) | Theoretical | ⚠️ PARTIALLY IMPLEMENTED |
| Vision / multimodal | `V4.1/model/{vision_encoder,multimodal_fusion}.py` | Public stubs; `_vision.py` exists | Soft-pass / unused by default | No | ❌ NOT IMPLEMENTED (public) |
| Bench harness | `bench/` | Yes | Import + Stage-1 artifacts | Stage-1 + RoPE Δ | ✅ PASS (harness) |
| Deferred soft-pass tests | `*/tests/test_{dsa,ced,...}.py` | N/A | Misleading `assert True` | N/A | ❌ FAIL (test quality) |

## Package collision note

Both Minis export top-level names `model` and `optimization`. Verification uses `verification/pkg_import.py` to isolate imports. Mixing both packages on `sys.path` without purging causes silent wrong-module imports.
