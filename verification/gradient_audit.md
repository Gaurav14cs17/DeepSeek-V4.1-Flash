# Gradient Audit

Scanned `DeepSeekFlashV4-Mini` and `DeepSeekFlashV4.1-Mini` for `detach()`, `no_grad()`, `inference_mode()`, `.data`, `requires_grad=False`.

| Location | Pattern | Intentional? | Notes |
|----------|---------|--------------|-------|
| `V4/model/router.py` balance bias update | `torch.no_grad()` | Yes | Load-balancing bias must not pollute router score grads |
| `V4/model/model.py` `generate` | `@torch.no_grad()` | Yes | Inference |
| `V4/inference/generate.py` | `@torch.no_grad()` | Yes | Inference |
| `V4/training/stage01_train.py` eval helper | `@torch.no_grad()` | Yes | Val metrics |
| `V4/training/pretrain.py` | `@torch.no_grad()` | Yes | Eval |
| `V4/optimization/quantization.py` | `x.detach()` for scale | Yes | STE-style fake-quant (straight-through on rounded path partially) |
| `V4.1/model/router.py` | `torch.no_grad()` bias | Yes | Same as V4 |
| `V4.1/model/_transformer.py` `generate` | `@torch.no_grad()` | Yes | Inference |
| `V4.1/model/_dspark.py` | `@torch.no_grad()` | Yes | Drafting |
| `V4.1/model/_csa2.py` FP4 sim scale | `x.detach().abs().amax` | Yes | Scale not trained |
| `V4.1/model/_swa.py` FP8 sim scale | `t.detach()` | Yes | Scale not trained |
| `V4.1/model/_engram.py` | `no_grad` + `p.data` | ⚠️ Review | Table init / normalize uses `.data`; ensure training path still receives grads on table params when intended |
| `V4.1/inference/generate.py` | `@torch.no_grad()` | Yes | Inference |

## Accidental graph breaks found

No hard evidence of accidental `detach` on main residual stream in V4 decoder/MoE path during training.

**Caveat:** MoE uses in-place masked writes `out[mask] = ...`. Autograd still tracks selected expert paths in practice (overfit test passes), but this pattern is fragile vs a scatter-add formulation.

## Gradient evidence

- RoPE: `test_rope_gradient_finite` PASS
- Stage-1 modules: `test_stage01_modules_gradient` PASS
- Full V4 Mini overfit: loss decreases; parameters change; grads finite (`test_v4_overfit_tiny_batch`)
- Finite-difference gradient check for large modules: **not run** (CPU time); mark as unverified beyond autograd finiteness

## Status

⚠️ PASS WITH SIMPLIFICATION for gradient *existence*; full numerical gradcheck of CSA/CSA2/router not completed.
