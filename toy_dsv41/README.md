# Toy DeepSeek-V4.1-Flash

CPU toy mirroring DeepSeek-V4.1-Flash ([paper](../docs/DeepSeek_V41_Tech_Report.pdf)).

Flow: `docs/` → `demo` → `train` → `_artifacts/` → `generate`  
Parent: [../README.md](../README.md)

**Same ideas, totally different scale.** Educational clone of the paper architecture — not the 552B product.

### Big picture

| | **Original DeepSeek-V4.1-Flash** | **Our toy** |
|--|----------------------------------|-------------|
| Goal | Production multimodal agent model | Learn every major paper block on a laptop |
| Size | **552B** backbone (+ **196B** Engram) | **~few M** (`better` preset) |
| Activated | **8B** prefill / **16B** decode | Tiny full forward |
| Hardware | Multi-GPU (e.g. DGX Sparks) | CPU (`ml` env) |
| Data | **~45T** multimodal tokens | TinyStories subset (~4MB / 5k stories) |
| Context | up to **1M** tokens | **128** chars |

### Architecture (paper idea → toy)

| Idea | Original | Ours |
|------|----------|------|
| CED | 20+20, decoder KV from encoder | **4+4** + CED Full KV + prefill replay path |
| CSA2 Full / Reindex / Reuse | Yes | Yes (tiny) |
| Hierarchical indexer | Large candidate pool | Top-P pool passed to Reindex |
| SWA | window 128, FP8 KV | window **8**, FP8 sim + RoPE |
| MoE | 384 / top-6 + modality balance | **8 / 2** + aux-free text/image biases |
| Engram | 196B, layers 1 & 14 | Tiny tables ×2 (emb + mid), orders {2,3,4}, Sinkhorn |
| FP4 main KV | MXFP4 QAT | Per-16-channel E2M1-ish sim |
| Vision | DeepSeek-ViT + 3×3 unshuffle | `TinyDeepSeekViT` (optional `images=`) |
| DSpark | 3-block drafter, 5 drafts | `DSparkDrafter` in `generate()` |
| Single-Pass mHC | Eq. 6 + Mega-mHC | `SinglePassMHC` (n=2 streams) |
| RoPE | LLM + 2D-RoPE ViT | Yes in SWA/CSA2/ViT |
| Serving kernels | FlashMLA, SGLang, … | Plain PyTorch |

Full live table: `preset("better").comparison_table()` in `model/config.py`.

---

## Package structure

```
toy_dsv41/
├── README.md
├── requirements.txt
├── __init__.py
│
├── model/                 ← network (paper §2)
│   ├── config.py
│   ├── transformer.py     # CED prefill, Engram sites, Vision, DSpark generate
│   ├── blocks.py          # Encoder/Decoder + mHC
│   ├── csa2.py / swa.py / moe.py / engram.py
│   ├── rope.py / mhc.py / vision.py / dspark.py
│   ├── norms.py / kv_cache.py
│
├── data/                  ← training text (see data/README.md)
│   ├── README.md
│   ├── corpus_toy.txt     # builtin (--dataset toy)
│   ├── download_opensource.py
│   ├── tokenizer.py
│   ├── opensource/        # downloaded .txt (gitignored)
│   └── __init__.py
│
├── utils/                 ← plotting.py, plot_history.py
├── train.py / demo.py / tests/test_smoke.py
└── _artifacts/            # checkpoints + graphs (gitignored)
```

**Data default:** TinyStories (`--dataset tinystories`). Builtin = `--dataset toy`.  
Details: [data/README.md](data/README.md).

**Train note:** CE training uses full-length forward (`use_ced_prefill=False`). CED + SWA Bounded Replay is enabled in `demo` / `phase=prefill` for architecture tracing.

---

## Train + plots

```bash
conda activate ml
cd DeepSeek-V4.1-Flash
pip install -r toy_dsv41/requirements.txt

python -m toy_dsv41.demo
python -m toy_dsv41.tests
python -m toy_dsv41.train --preset better --steps 1000
xdg-open toy_dsv41/_artifacts/pc_run/loss_curves.png

python -m toy_dsv41.utils.plot_history
python -m toy_dsv41.train --generate-only --prompt "Once upon a time" --temperature 0.3
```

### Tests

```bash
python -m toy_dsv41.tests                 # full suite (smoke + components)
python -m toy_dsv41.tests.test_smoke      # end-to-end model/data
python -m toy_dsv41.tests.test_components # RoPE, mHC, CSA2, MoE, Vision, DSpark, …
# optional: pytest toy_dsv41/tests -q
```

`--no-plot` skips PNG. Blue = train, red dashed = val.

---

## Presets (`model/config.py`)

| Preset | Approx params | Use |
|--------|---------------|-----|
| `demo` | smallest | demo + feature flags on |
| `pc` | ~2–3M | quick train |
| `better` | ~few M | recommended + graphs |

Flags (all default **on**): `use_rope`, `use_mhc`, `use_vision`, `use_dspark`, `use_fp4_main_kv`, `use_fp8_swa_kv`.

---

## Paper → file (deep check)

Status key: **P** = present (toy-faithful shape) · **~** = partial stub · **✗** = missing · **N/A** = paper also omits

| Paper (§) | Status | Path | Honest note |
|-----------|--------|------|-------------|
| CED (§2.2) | ~ | `blocks.py`, `transformer.py` | Encoder→decoder KV yes; **no separate \(W^Z\)**; train disables prefill skip |
| CED Eq.1 \(W^{KV},W^Z\) | ~ | `kv_from_encoder` | One Linear only |
| SWA first 2 enc layers | P | `blocks.py` | Matches Fig. 3 |
| CSA2 Full / Reindex / Reuse | P | `csa2.py` | Modes work; naive gather attn |
| Hier. Sparse Indexer (§2.3.2) | ~ | `csa2.py` | Top-P pool → Reindex; **not** block-max hierarchy |
| SWA + RoPE + FP8 sim | P/~ | `swa.py`, `rope.py` | RoPE yes; FP8 = round sim |
| SWA Bounded Replay (§3.2.2) | ~ | `transformer.py`, `kv_cache.py` | Demo slice last `n_win`; **not** real cache system |
| DeepSeekMoE + modality / aux-free | P | `moe.py` | Shape right; default train is text-only |
| Single-Pass mHC (§2.4.1) | P | `mhc.py` | Eq. 6-ish, n=2 streams |
| Mega-mHC kernel | ✗ | — | Production only |
| Engram (§2.4.2) | ~ | `engram.py` | Orders {2,3,4}, gate, mid site; **no** layer-14 / FP8 / RDMA / tok-compress |
| Engram Sinkhorn Alg.1 | ~ | `sinkhorn_balance_` | Toy row/col norm, not momentum Sinkhorn |
| DSpark (§2.4.3) | ~ | `dspark.py` | 3-block + 5 drafts + conf; **no** train stage / throughput scheduler |
| FP4 / MXFP4 QAT (§2.4.4) | ~ | `csa2.py` | E2M1-ish sim; **no** real QAT / post-RoPE quant |
| Vision / DeepSeek-ViT (§2.1.1) | ~ | `vision.py` | Tiny ViT + 3×3 unshuffle + projector; **no** multimodal train |
| RMSNorm / SwiGLU | P | `norms.py`, `moe.py`, `vision.py` | |
| MLA / FlashMLA | ✗/~ | — | Shared KV expand only; no MLA / FlashMLA |
| Muon optimizer (§2.5) | ✗ | `train.py` | **AdamW** |
| Attention Sharing Training (§3.1.2) | ✗ | — | |
| Engram shard / RDMA prefetch (§3.1.3) | ✗ | pack-to-disk demo only | |
| Persistent KV / EPD / SGLang (§3.2) | ✗ | — | |
| MTP | N/A | — | Paper uses DSpark instead |
| 552B / 1M ctx / 45T data | ✗ | — | Scale intentionally tiny |

### Scorecard (paper §§1–3 named ideas)

| | Count |
|--|------:|
| Present (toy shape) | ~18 |
| Partial / stub | ~42 |
| Missing (infra / fidelity) | ~28 |
| N/A (paper omits too) | ~4 |

**Verdict:** Almost every **named §2 block** has a file or flag in `toy_dsv41/model/`. Almost none match paper **algorithm or infra fidelity**. README “Yes” means *educational presence*, not parity.

### Top gaps still open

1. Real MXFP4 QAT + quantize-after-RoPE  
2. True hierarchical block-max indexer  
3. Full CED Eq.1 (`W^Z`) + train-time prefill path  
4. Real SWA Bounded Replay / HBM–persistent KV  
5. DSpark training stages + confidence scheduler  
6. Engram layer-14, FP8, RDMA, Alg.1 Sinkhorn  
7. Muon (+ head-wise)  
8. Attention Sharing Training  
9. FlashMLA / Mega-mHC / SGLang / EPD  
10. MLA + long-context serving story  
