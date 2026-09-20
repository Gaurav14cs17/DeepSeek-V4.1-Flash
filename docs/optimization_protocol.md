# Optimization → Benchmark → Delta protocol

**This is one of the most important requirements of the entire project.**

Never implement an optimization without measuring its effect against a clearly defined baseline. Every optimization must produce a BEFORE vs AFTER comparison.

The project is not complete when an optimized implementation works. It is complete only when we know:

1. How much memory changed.
2. How much latency changed.
3. How much throughput changed.
4. How much inference time changed.
5. How much training time changed where applicable.
6. How much KV-cache memory changed.
7. How much CPU RAM changed.
8. How parameter count changed.
9. Whether numerical accuracy changed.
10. Whether loss/perplexity changed.
11. Why the change happened.
12. What trade-offs were introduced.

---

## MOST IMPORTANT RULE

```text
BEFORE → MEASURE → IMPLEMENT → AFTER → COMPARE → %Δ → EXPLAIN → SAVE
```

Never skip the BEFORE measurement. Never report an optimization without quantitative evidence.

---

## Core experiment loop

Every major component MUST follow:

```text
PAPER
  ↓
REFERENCE IMPLEMENTATION
  ↓
BASELINE BENCHMARK
  ↓
OPTIMIZATION
  ↓
OPTIMIZED IMPLEMENTATION
  ↓
SAME BENCHMARK
  ↓
BEFORE vs AFTER
  ↓
DELTA
  ↓
ANALYSIS
  ↓
CONCLUSION
```

Never compare two implementations using different:

* hardware
* batch size
* sequence length
* dataset
* number of generated tokens
* precision
* warm-up procedure
* number of iterations

unless the experiment explicitly studies that variable.

Unfair compares are **rejected** by `bench.compare` (`assert_same_protocol` + hardware check).

---

## Baseline requirement

Every optimization must first establish a baseline.

Example:

```text
Experiment: Vanilla Attention
Baseline:   attention_reference.py
B0 = benchmark(attention_reference)

# only then
attention_optimized.py
B1 = benchmark(attention_optimized)
Delta = B1 - B0
```

**Improve %** (positive = better for that metric’s direction):

* Lower-is-better: `(baseline - optimized) / baseline * 100`
* Higher-is-better: `(optimized - baseline) / baseline * 100`

Clearly label whether an increase or decrease is beneficial (`Better?` = YES / NO / SAME / CHECK).

---

## Shared tools (`bench/`)

| Module | Role |
|--------|------|
| `protocol.py` | Hardware/software metadata, result bundles |
| `memory.py` | Memory breakdown reports |
| `latency.py` | Warmup + timed iters + component breakdown |
| `inference.py` | Prompt/gen grid, TTFT / prefill / decode |
| `training_metrics.py` | Forward / backward / optimizer / step |
| `kv.py` | KV context sweep + ASCII chart |
| `accuracy.py` | Numerical + quality trade-off tables |
| `ladder.py` | Level 0–5 cumulative tables |
| `delta.py` | Absolute/relative Δ + Better? column |
| `compare.py` | Fair protocol check + print tables |
| `regression.py` | WARN if memory/latency/throughput worsen >10% |
| `report.py` | Write `optimization_report.md` |
| `history.py` | Append `comparison/optimization_history.csv` |

---

## Result layout

```text
results/
├── v4/
│   ├── baseline/          # or stage01/baseline/
│   │   ├── config.json
│   │   ├── metrics.json
│   │   └── memory.json
│   ├── rope_formulation/
│   │   ├── A0_complex/
│   │   ├── A1_rotate_half/
│   │   └── optimization_report.md
│   └── comparison.csv     # optional per-experiment
└── v4_1/
    └── ...
```

Every bundle records: CPU/GPU/VRAM, Python/PyTorch/CUDA, model, config, precision, batch, seq, gen length, warmup, iters, seed, git commit, timestamp.

---

## Mandatory metrics (when applicable)

### Memory

peak / allocated / reserved GPU · CPU RAM · param / activation / optimizer · KV-cache · checkpoint size

### Latency

init · TTFT · prefill · decode · avg token · p50 · p95

### Throughput

training / inference / prefill / decode tokens/sec · samples/sec

### Training

step / epoch / total time · loss · val loss · perplexity · convergence

### Model

total / trainable / active params · FLOPs est · param memory

### KV cache

bytes/token · total size · growth rate · compression ratio

### MoE

expert utilization · tokens/expert · routing entropy · imbalance · dropped tokens · active experts/token

### Numerical correctness

max abs error · mean abs error · relative error · cosine similarity

---

## Standard benchmark table

Every optimization experiment MUST produce:

| Metric | Baseline | Optimized | Absolute Δ | Improve % | Better? |
| --- | ---: | ---: | ---: | ---: | --- |

Do not report “Optimized version is faster.” Report exact measurements.

---

## Memory / latency breakdowns

Use `bench.memory.format_memory_report` and `memory_delta` — explain **where** memory changed (params vs activations vs KV vs temps).

Use `bench.latency.time_components` + `format_component_table` for:

tokenizer · embedding · attention · MoE/router · experts/FFN · LM head · sampling · prefill total · decode total

---

## Inference benchmark protocol

Prompt lengths: **128, 256, 512, 1024, 2048**  
Generated tokens: **32, 128, 512** (subset if hardware-limited; record subset in `config.extra`)

Per config measure: TTFT, prefill, decode, total generation time, tokens/sec, peak GPU, CPU RAM, KV size.

Helpers: `bench.inference.allowed_configs`, `measure_inference_point`, `format_inference_grid`, `inference_delta_summary`.

---

## Training benchmark protocol

Measure: forward · backward · optimizer · step · tokens/sec · GPU/CPU memory · loss · val loss.

Helpers: `bench.training_metrics.measure_training_step`, `format_training_table`, `training_delta_summary`.

---

## KV cache experiment

Context lengths: **128, 256, 512, 1024, 2048, 4096**

Measure: KV bytes/token · total KV · prefill · decode · peak memory · tokens/sec

Helpers: `bench.kv.format_kv_sweep_table`, `format_kv_ascii_chart`.

Especially important for V4 compression/DSA/KV and V4.1 CSA2 / FP4 KV / SWA Bounded Replay / cross-layer reuse.

---

## Optimization ladder

Do not jump from baseline to final optimized. After **every** level, benchmark again:

| Level | Meaning |
|------:|---------|
| 0 | Reference |
| 1 | Memory |
| 2 | Algorithmic |
| 3 | Cache |
| 4 | Quantization |
| 5 | Kernel/runtime |

Helper: `bench.ladder.format_ladder_table` (cumulative Δ vs A0).

---

## No false optimization

An optimization is NOT successful because code is more complicated, theoretical FLOPs fell, or tensor/cache size shrank. It must be **measured**.

Example: KV −50% but decode +12% → report both. Do **not** call it simply “faster.”

---

## Accuracy / quality trade-off

If quality can change, compare loss / val loss / perplexity (and generation quality where measurable) alongside memory/latency. Slight quality regressions → `Better? = CHECK`.

Helper: `bench.accuracy.quality_tradeoff_table`, `numerical_diff`.

---

## Automatic regression check

```bash
python -m bench.regression \
  --previous results/v4/stage01/baseline \
  --current results/v4/some_opt
```

WARN if GPU memory or latency ↑ >10%, or throughput ↓ >10%, or unexpected quality shift.

---

## Optimization report

Every completed optimization generates `optimization_report.md` via `bench.report.write_optimization_report` with Baseline / Optimized / Hardware / Memory / Latency / Throughput / Training / Accuracy / KV / Analysis / Trade-offs / Conclusion.

No subjective labels (“excellent”, “amazing”, “huge”) unless tied to a number.

---

## Experiment history

`comparison/optimization_history.csv` columns:

experiment · component · version · baseline_version · parameters · active_parameters · gpu_memory_mb · cpu_memory_mb · prefill_ms · decode_ms · total_inference_ms · training_step_ms · training_tokens_per_sec · inference_tokens_per_sec · kv_bytes_per_token · loss · validation_loss · perplexity · max_abs_error · mean_abs_error · timestamp · git_commit

---

## Commands

```bash
# Record Stage-1 baseline (B0) — required before any Stage-1+ optimization
python scripts/run_stage01_baseline.py

# Demo delta loop on RoPE formulations (A0 vs A1)
python scripts/run_rope_delta_demo.py

# Compare two bundles
python -m bench.compare \
  --baseline results/v4/rope_formulation/A0_complex \
  --optimized results/v4/rope_formulation/A1_rotate_half \
  --report results/v4/rope_formulation/optimization_report.md

# Regression gate
python -m bench.regression \
  --previous results/v4/stage01/baseline \
  --current results/v4/some_opt
```

---

## README dashboards

README.md must keep continuously updated **measured-only** tables:

| Optimization | Memory Δ | Prefill Δ | Decode Δ | Throughput Δ | KV Δ | Quality Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

V4.1 rows (CED / CSA2 / FP4 KV / SWA Replay) appear only after B0+B1 JSON exists under `results/v4_1/`.
