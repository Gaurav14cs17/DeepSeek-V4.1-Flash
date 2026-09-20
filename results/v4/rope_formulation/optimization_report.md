# Optimization: rope_formulation

## Baseline

Implementation: `A0_complex`
Component: `rope`
Configuration: `v4_tiny`
Commit: `3454681`

## Optimized Version

Implementation: `A1_rotate_half`
Component: `rope`
Configuration: `v4_tiny`
Commit: `3454681`

## Hardware

CPU: x86_64
GPU: None
VRAM: 0.0 MB
Platform: Linux-7.2.4-100.fc43.x86_64-x86_64-with-glibc2.42

## Software

Python: 3.12.11
PyTorch: 2.13.0+cu130
CUDA: 13.0

## Memory

### Before
```
==================================================
MEMORY REPORT (BASELINE)
==================================================
Model parameters:      0
Trainable parameters:  0
Parameter memory:      0.000 MB
Gradient memory:       0.000 MB
Optimizer memory:      0.000 MB
Activation memory:     0.016 MB
KV cache:              0.000 MB
Temporary tensors:     0.000 MB
Peak GPU memory:       0.000 MB
Allocated GPU:         0.000 MB
Reserved GPU:          0.000 MB
Peak CPU RAM:          632.605 MB
==================================================
```

### After
```
==================================================
MEMORY REPORT (OPTIMIZED)
==================================================
Model parameters:      0
Trainable parameters:  0
Parameter memory:      0.000 MB
Gradient memory:       0.000 MB
Optimizer memory:      0.000 MB
Activation memory:     0.016 MB
KV cache:              0.000 MB
Temporary tensors:     0.000 MB
Peak GPU memory:       0.000 MB
Allocated GPU:         0.000 MB
Reserved GPU:          0.000 MB
Peak CPU RAM:          632.871 MB
==================================================
```

```
MEMORY IMPROVEMENT
----------------------------------------
peak_gpu_memory_mb: +0.000  (+0.00% vs baseline; lower is better)
kv_cache_mb: +0.000  (+0.00% vs baseline; lower is better)
activation_memory_mb_est: +0.000  (+0.00% vs baseline; lower is better)
param_memory_mb: +0.000  (+0.00% vs baseline; lower is better)
peak_cpu_memory_mb: +0.266  (-0.04% vs baseline; lower is better)
```

### Peak GPU

Before: `0.0`

After: `0.0`

Absolute change: `+0`

Percentage change (Improve %): `n/a`

### CPU RAM

Before: `632.605`

After: `632.871`

Absolute change: `+0.266`

Percentage change (Improve %): `-0.04%`

### KV cache

Before: `0.0`

After: `0.0`

Absolute change: `+0`

Percentage change (Improve %): `n/a`


## Latency

### Prefill

Not measured in this experiment.

### Decode

Not measured in this experiment.

### TTFT

Not measured in this experiment.

### Forward (microbench)

Before: `0.009511859389021993`

After: `0.06373704178258777`

Absolute change: `+0.0542252`

Percentage change (Improve %): `-570.08%`

### Step

Not measured in this experiment.


## Throughput

### Inference tokens/sec

Not measured in this experiment.

### Training tokens/sec

Not measured in this experiment.

### Generic tokens/sec

Before: `26913770.43435478`

After: `4016502.6935708253`

Absolute change: `-2.28973e+07`

Percentage change (Improve %): `-85.08%`


## Training

### Train loss

Not measured in this experiment.

### Validation loss

Not measured in this experiment.

### Perplexity

Not measured in this experiment.


## Accuracy

### Max abs error

Not measured in this experiment.

### Mean abs error

Not measured in this experiment.

### Cosine similarity

Not measured in this experiment.


## KV Cache

### KV bytes/token

Not measured in this experiment.


## Standard metric table

| Metric                       |     Baseline |    Candidate |   Absolute Δ |  Improve % | Better? |
| ---------------------------- | ------------: | ------------: | ------------: | ----------: | ------- |
| active_parameters            |            0 |            0 |           +0 |          — |       — |
| forward_ms                   |     0.009512 |      0.06374 |     +0.05423 |   -570.08% |      NO |
| p50_latency_ms               |     0.009017 |      0.05962 |     +0.05061 |   -561.22% |      NO |
| p95_latency_ms               |      0.01309 |      0.07686 |     +0.06377 |   -486.97% |      NO |
| parameters                   |            0 |            0 |           +0 |          — |       — |
| peak_cpu_memory_mb           |        632.6 |        632.9 |       +0.266 |     -0.04% |      NO |
| peak_gpu_memory_mb           |            0 |            0 |           +0 |          — |    SAME |
| tokens_per_second            |    2.691e+07 |    4.017e+06 |    -2.29e+07 |    -85.08% |      NO |
| trainable_parameters         |            0 |            0 |           +0 |          — |       — |

_Improve % > 0 means better; < 0 means worse. Absolute Δ = candidate − baseline._

## Verdict

VERDICT: REGRESSION — keep Baseline (A0). Candidate is not an optimization.

## Numerical difference

- max_abs_error: 0.0
- mean_abs_error: 0.0
- relative_error: 0.0
- cosine_similarity: 1.0

## Analysis

_Fill in: why each metric changed._

## Trade-offs

_List anything that became worse._

## Conclusion

_State only measured results. No subjective labels (excellent / amazing / huge)._
