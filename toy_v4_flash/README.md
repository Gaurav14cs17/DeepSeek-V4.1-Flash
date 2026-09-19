# Toy DeepSeek-V4-Flash — older predecessor of V4.1-Flash

CPU educational clone of **DeepSeek-V4-Flash** (284B / 13B activated in the real model).

Sibling: [`../toy_dsv41/`](../toy_dsv41/) = DeepSeek-**V4.1**-Flash (CED + CSA2 + Engram + DSpark).

Paper context: [../docs/DeepSeek_V41_Tech_Report.pdf](../docs/DeepSeek_V41_Tech_Report.pdf) (compares V4-Flash → V4.1).  
V4 architecture details: [DeepSeek-V4 tech report / HF docs](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4).

---

## V4-Flash vs V4.1-Flash (why a separate folder)

| Aspect | **V4-Flash** (this toy) | **V4.1-Flash** (`toy_dsv41`) |
|--------|-------------------------|------------------------------|
| Layout | Full-depth stack | **CED** encoder + decoder |
| Attention | **CSA–HCA hybrid** | Pure **CSA2** Full/Reindex/Reuse |
| Activated | ~13B full depth | 8B prefill / 16B decode |
| Residuals | Original **multi-pass mHC** | **Single-Pass mHC** |
| Extra heads | **MTP** (multi-token pred.) | **DSpark** (no MTP) |
| Memory | — | **Engram** |
| SWA rebuild | Exact **L × n_win** | **Bounded Replay** (n_win only) |
| Main KV | FP8-era | **FP4** QAT |
| Vision | not in this toy | Tiny ViT path |

---

## Package

```
toy_v4_flash/
├── model/
│   ├── csa.py / hca.py / swa.py   ← hybrid attention zoo
│   ├── moe.py                     ← hash-MoE bootstrap + sqrt-softplus
│   ├── mhc.py                     ← multi-pass mHC
│   ├── mtp.py                     ← multi-token prediction
│   ├── blocks.py / transformer.py
│   └── config.py
├── data/   utils/   demo.py   train.py   tests/
└── _artifacts/pc_run/
```

Layer schedule (toy 8 layers): `SWA, SWA, CSA, HCA, CSA, HCA, CSA, HCA`.

---

## Run

```bash
conda activate ml
cd DeepSeek-V4.1-Flash

python -m toy_v4_flash.demo
python -m toy_v4_flash.tests
python -m toy_v4_flash.train --preset better --steps 500 --dataset toy
python -m toy_v4_flash.train --generate-only --prompt "Once upon a time"
xdg-open toy_v4_flash/_artifacts/pc_run/loss_curves.png
```

TinyStories (shared with `toy_dsv41` via symlink): `--dataset tinystories`.

---

## Paper → file

| Paper idea | Path | Toy status |
|------------|------|------------|
| SWA (first layers) | `swa.py` | Yes |
| CSA (m=4, indexer Top-K, overlap compress) | `csa.py` | Educational |
| HCA (m'=128→16 toy, dense over compress) | `hca.py` | Educational |
| DeepSeekMoE + hash bootstrap | `moe.py` | Yes (tiny) |
| Sqrt(Softplus) router + SwiGLU clamp | `moe.py` | Yes |
| Multi-pass mHC | `mhc.py` | Yes (n=2) |
| MTP | `mtp.py`, `train.py --mtp-weight` | Yes |
| Exact SWA replay L×n_win | `kv_cache.py` | Documented |
| CED / CSA2 / Engram / DSpark / FP4 | — | **Not in V4-Flash** (see `toy_dsv41`) |
