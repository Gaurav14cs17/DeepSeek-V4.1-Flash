# Progress checklist

Principle: **PAPER → MATH → IMPLEMENTATION → TRAINING → MEASUREMENT → ANALYSIS → OPTIMIZATION**

Do not mark a component **done** until: paper note + code + unit test + tiny train/measure JSON exist.

**Optimization rule:** every change needs BEFORE baseline → AFTER → Δ table → report. See `docs/optimization_protocol.md`.

---

## Stage 0 — Lab scaffold

- [x] Repository structure (matches research-lab tree)
- [x] `requirements.txt`
- [x] `README.md`
- [x] `configs/v4_tiny.yaml` / `v4_small.yaml` / `v4_1_tiny.yaml` / `v4_1_small.yaml`
- [x] `datasets/` ladder + capped `download.py`
- [x] `docs/progress.md`
- [x] V4 / V4.1 `paper_notes/architecture.md`

## Optimization harness (always required)

- [x] `bench/` (protocol, memory, latency, delta, compare, regression, report, history)
- [x] `docs/optimization_protocol.md`
- [x] `comparison/optimization_history.csv`
- [x] README V4 / V4.1 optimization dashboards

## Stage 1 — V4-Mini primitives ✅ GATE PASSED

- [x] tokenizer (`[SIMPLIFIED]` char vocab)
- [x] embedding (`[FAITHFUL]`)
- [x] normalization / RMSNorm (`[FAITHFUL]`)
- [x] RoPE (`[FAITHFUL]`)
- [x] `tests/test_rope.py`
- [x] `experiments/01_transformer.ipynb`
- [x] tiny train → `results/stage01/metrics.json`
- [x] formal B0 → `results/v4/stage01/baseline/`
- [x] RoPE A0 vs A1 delta demo → `results/v4/rope_formulation/` (shows regressions honestly)

## Stage 2 — V4-Mini attention stack (NOT UNLOCKED)

- [ ] vanilla attention — paper note + B0 + research loop
- [ ] Transformer decoder
- [ ] language-model loss + baseline training
- [ ] compression mechanism
- [ ] DeepSeek attention (CSA / HCA / DSA as documented)
- [ ] MLA (comparison mini)

> Files may exist as provisional or deferred stubs. **Do not treat as complete** until this checklist is checked **and** Δ reports exist.

## Stage 3 — V4-Mini MoE + systems

- [ ] MoE / router / experts / shared experts
- [ ] expert load statistics
- [ ] KV cache
- [ ] inference
- [ ] quantization
- [ ] profiling
- [ ] full V4-Mini integration

## Stage 4 — V4.1-Mini (only after V4-Mini Stages 1–3)

- [ ] copy reusable primitives + document diffs
- [ ] causal encoder / decoder / CED
- [ ] CSA2 + cross-layer KV reuse
- [ ] FP4 KV + SWA Bounded Replay
- [ ] multimodal path
- [ ] comparison suite → `comparison/results.json`

---

## How to verify Stage 1 + bench harness

```bash
python scripts/verify_stage01.py
python scripts/run_stage01_baseline.py
python scripts/run_rope_delta_demo.py
```
