# DeepSeek-V4.1-Flash — Local Learning Workspace

Learn the **DeepSeek-V4.1-Flash** paper on your laptop with a small toy model you can **train, plot, test, and sample**.

> This is **not** the real 552B model. It is a CPU-friendly classroom for **CED / CSA2 / MoE / Engram**.

**Flow:** `docs/` (paper) → `demo` → `train` → `graph` → `generate` → `test`

Every major step below is written as **Input → Method → Output → Result**.

---

## Table of contents

1. [What this repo is](#1-what-this-repo-is)
2. [Setup](#2-setup)
3. [Folder map](#3-folder-map)
4. [About the data](#4-about-the-data)
5. [Training method](#5-training-method)
6. [Training — input / output / result](#6-training--input--output--result)
7. [Graph — input / output / result](#7-graph--input--output--result)
8. [Generate — input / output / result](#8-generate--input--output--result)
9. [Demo — input / output / result](#9-demo--input--output--result)
10. [Test — input / output / result](#10-test--input--output--result)
11. [Presets](#11-presets)
12. [Paper → code](#12-paper--code)
13. [Quick path](#13-quick-path)
14. [Cheat sheet](#14-cheat-sheet)
15. [Troubleshooting](#15-troubleshooting)
16. [More links](#16-more-links)

---

## 1. What this repo is

| This repo **is** | This repo **is not** |
|------------------|----------------------|
| A readable toy of DeepSeek-V4.1-Flash ideas | The production 552B Flash model |
| Runnable on **CPU** | Multi-GPU serving |
| Char-level LM on open text | DeepSeek’s 45T multimodal pretrain |
| Full loop: train → graph → sample | A chat API product |

**End-to-end picture**

```text
  INPUT (text file)
       │
       ▼
  tokenizer (characters → ids) ──► train / val split
       │
       ▼
  toy model (CED + CSA2 + MoE + Engram)
       │
       ▼
  OUTPUT files
    ├── toy_pc.pt          (checkpoint)
    ├── history.json       (losses)
    └── loss_curves.png    (graph)
       │
       ▼
  RESULT
    ├── lower train/val loss
    └── sample text from --generate-only
```

---

## 2. Setup

### Input

| Need | Notes |
|------|--------|
| Python | 3.10+ |
| Conda env | e.g. `ml` |
| Packages | `toy_dsv41/requirements.txt` (torch, matplotlib, datasets) |

### Method

```bash
conda activate ml
cd ~/Downloads/DeepSeek-V4.1-Flash
pip install -r toy_dsv41/requirements.txt
```

### Output / result

```bash
python -c "import torch, matplotlib; print('ok', torch.__version__)"
python -m toy_dsv41.tests.test_smoke
```

```text
ok 2.x.x
all tests passed
```

---

## 3. Folder map

```
DeepSeek-V4.1-Flash/
├── README.md                          ← this guide
├── docs/                              ← paper (PDF + searchable .txt)
├── toy_dsv41/                         ← V4.1-Flash toy (CED + CSA2 + Engram + DSpark)
└── toy_v4_flash/                      ← older V4-Flash toy (CSA–HCA + mHC + MTP)
    ├── demo.py / train.py / tests/
    ├── model/                         ← CSA / HCA / SWA / MoE / mHC / MTP
    ├── data/                          ← corpus (+ symlink to toy_dsv41 opensource)
    └── _artifacts/pc_run/
```

| Path | Role |
|------|------|
| `docs/` | Read paper |
| `toy_dsv41/` | **V4.1-Flash** educational toy |
| `toy_v4_flash/` | **V4-Flash** (older) educational toy — see [toy_v4_flash/README.md](toy_v4_flash/README.md) |
| `toy_dsv41/data/` | **Training data** |
| `toy_dsv41/model/` | Network |
| `toy_dsv41/train.py` | **Training method** |
| `toy_dsv41/utils/` | **Graph** code |
| `toy_dsv41/_artifacts/` | **Results** (checkpoint + curves) |

```bash
# Older V4-Flash toy
python -m toy_v4_flash.demo
python -m toy_v4_flash.tests
python -m toy_v4_flash.train --preset better --steps 500 --dataset toy
```

---

## 4. About the data

**Not** DeepSeek’s 45T mix. We use small **open** UTF-8 text on your PC.

### 4.1 Dataset choices

| Dataset | Flag | File | About | Default? |
|---------|------|------|-------|----------|
| **TinyStories** | `--dataset tinystories` | `data/opensource/tinystories.txt` | Synthetic children’s stories (Eldan & Li). Good for tiny LMs. HF: [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories). Example train repo: [tiny-stories-hf](https://github.com/SauravP97/tiny-stories-hf) | **Yes** |
| Tiny Shakespeare | `--dataset tinyshakespeare` | `data/opensource/tinyshakespeare.txt` | Classic char-LM corpus (~1.1M chars) | No |
| Builtin educational | `--dataset toy` | `data/corpus_toy.txt` | Short notes about CED/CSA2 + practice sentences (~3–4k chars) | No |
| Custom | `--corpus PATH` | your `.txt` | Any clean UTF-8 text | No |

### 4.2 TinyStories (default) — input / output

| | |
|--|--|
| **Input** | Hugging Face dataset `roneneldan/TinyStories` (streamed) |
| **Method** | Export first **20,000** train stories to plain text (not full ~2GB dump) |
| **Output** | `toy_dsv41/data/opensource/tinystories.txt` |
| **Result (typical)** | ~**4.19M characters**, vocab ≈ **90** chars after tokenize |

```bash
python -m toy_dsv41.data.download_opensource --list
python -m toy_dsv41.data.download_opensource tinystories
python -m toy_dsv41.data.download_opensource tinystories --max-stories 50000 --force
```

**Download output example**

```text
already exists: .../tinystories.txt (4,197,330 bytes)
license: TinyStories synthetic stories (Eldan & Li). ...
```

### 4.3 Builtin toy corpus — what is inside

`corpus_toy.txt` is educational English about the paper ideas, e.g.:

```text
the causal encoder builds global memory.
the decoder reads keys from the encoder.
sparse attention keeps the cache small.
```

Use it for **offline** smoke trains (`--dataset toy`). It overfits quickly because it is tiny.

### 4.4 Data pipeline (how text becomes model input)

```text
  .txt file
     │  load UTF-8
     ▼
  CharTokenizer.from_text(text)     ← vocab = unique characters
     │  encode every char → int id
     ▼
  full id tensor
     │  split by --val-ratio (default 0.1)
     ▼
  train_ids (≈90%)   val_ids (≈10%)
     │  random windows of length --seq-len
     ▼
  batch x[B, T]  →  predict y[B, T] = next characters
```

| Stage | Input | Output |
|-------|-------|--------|
| Load | path to `.txt` | Python `str` |
| Tokenize | `str` | `CharTokenizer` + `ids` |
| Split | `ids`, `val_ratio=0.1` | `train_ids`, `val_ids` |
| Batch | `train_ids`, `batch_size`, `seq_len` | tensors `x`, `y` shape `(B, T)` |

**Real TinyStories split (laptop run)**

| Item | Value |
|------|--------|
| Corpus chars | **4,186,740** |
| Vocab | **90** |
| Train ids | **3,768,066** |
| Val ids | **418,674** |

### 4.5 Why TinyStories?

Small models (~2–5M params) learn fluent-looking English more easily on simple synthetic stories than on random web text. That matches this toy’s size.

More notes: [toy_dsv41/data/README.md](toy_dsv41/data/README.md).

---

## 5. Training method

Implemented in `toy_dsv41/train.py`.

### 5.1 Task

**Next-character language modeling**

- Given characters `x[0..T-1]`, predict `y[i] = x[i+1]`
- Loss = **cross-entropy** over the character vocab

### 5.2 Model (toy DeepSeek-V4.1-Flash)

| Piece | Role in training forward |
|-------|--------------------------|
| CED blocks | Encoder builds memory; decoder reads it (`phase="prefill"`) |
| SWA | Local window attention on early encoder layers |
| CSA2 | Sparse attention modes: `full` / `reuse` / `reindex` |
| MoE | 8 routed experts, 2 activated per token |
| Engram | Extra hash memory tables |
| FP4 KV | Simulated low-precision main KV |

Forward call used in train:

```text
logits, _ = model(x, phase="prefill", use_bounded_replay=False, trace=False)
loss = cross_entropy(logits, y)
```

### 5.3 Optimizer loop (one step)

```text
1. sample random batch (x, y) from train_ids
2. forward → logits
3. cross-entropy loss
4. backward
5. clip grad norm to 1.0
6. AdamW step  (lr default 3e-3, weight_decay 0.01)
```

Every `--log-every` steps (default 25):

- evaluate mean loss on **10 random val batches**
- append `{step, train_loss, val_loss}` to history

Every `--sample-every` steps (default 100):

- print a short generated sample from `--prompt`

At the end:

- save `history.json`
- plot `loss_curves.png` (unless `--no-plot`)
- save checkpoint `toy_pc.pt` (+ `toy_pc.json`)

### 5.4 Device

| Item | Value |
|------|--------|
| Device | **CPU only** |
| Threads | `--threads` (default 8) |
| Precision | float32 train; FP4 only simulated for KV ideas |

### 5.5 Flags that control the method

| Flag | Default | Effect |
|------|---------|--------|
| `--preset` | `pc` | model size (`demo` / `pc` / `better`) |
| `--steps` | `500` | number of optimizer steps |
| `--batch-size` | `8` | sequences per step |
| `--seq-len` | `64` | character window `T` |
| `--lr` | `0.003` | AdamW learning rate |
| `--val-ratio` | `0.1` | held-out fraction |
| `--dataset` | `tinystories` | which corpus |
| `--corpus` | none | override with a file path |
| `--log-every` | `25` | val + print frequency |
| `--sample-every` | `100` | sample print frequency |
| `--plot` / `--no-plot` | plot on | write PNG |

---

## 6. Training — input / output / result

### 6.1 Command

```bash
conda activate ml
cd ~/Downloads/DeepSeek-V4.1-Flash

python -m toy_dsv41.train --preset better --steps 2000
# same as:
python -m toy_dsv41.train --dataset tinystories --preset better --steps 2000
```

### 6.2 INPUT

| Kind | What you provide |
|------|------------------|
| **Data** | TinyStories `.txt` (auto-download if missing) |
| **Model preset** | `better` (~4.76M params, `d_model=96`) |
| **Hyperparams** | steps=2000, batch=8, seq_len=64, lr=0.003 |
| **Prompt for mid-train samples** | auto → `"Once upon a time"` on TinyStories |

Other inputs:

```bash
# offline builtin
python -m toy_dsv41.train --dataset toy --preset better --steps 1000

# Shakespeare
python -m toy_dsv41.train --dataset tinyshakespeare --preset better --steps 2000

# your file
python -m toy_dsv41.train --corpus /path/to/book.txt --preset better --steps 1500
```

### 6.3 OUTPUT (printed)

Header from a real TinyStories `better` run:

```text
================================================================
Toy DeepSeek-V4.1-Flash — structured PC training
================================================================
  preset     : better
  device     : cpu  threads=8
  dataset    : opensource:tinystories
  corpus     : 4,186,740 chars  vocab=90
  split      : train=3,768,066  val=418,674
  params     : 4,764,672
  batch/seq  : 8 / 64
  steps      : 2000  lr=0.003
```

Progress lines:

```text
  step     1/2000  train=4.4675  val=4.1322  tok/s≈1372  t=0.4s
  step   100/2000  train=2.2207  val=2.2894  tok/s≈4267  t=12.0s
  sample@100: 'Once upon a time holl ax ss da osend ...'
  step   500/2000  train=1.6410  val=1.7346  ...
  sample@500: 'Once upon a time ept thoes a and crummy ...'
  step  1000/2000  train=1.5044  val=1.5647  ...
  sample@1000: 'Once upon a time waae for a pare ...'
  step  1600/2000  train=1.4084  val=1.4606  ...
  sample@1600: 'Once upon a time, there was a little bee ...'
```

End of run:

```text
  history  → toy_dsv41/_artifacts/pc_run/history.json
  plot     → toy_dsv41/_artifacts/pc_run/loss_curves.png
  saved    → toy_dsv41/_artifacts/pc_run/toy_pc.pt
```

### 6.4 OUTPUT (files)

| File | Contents |
|------|----------|
| `toy_pc.pt` | `model` state_dict + `tokenizer` + `config` + step/losses |
| `toy_pc.json` | short summary: step, train_loss, val_loss, params, preset |
| `history.json` | `meta` + list of `{step, train_loss, val_loss}` |
| `loss_curves.png` | train/val graph (see §7) |

**`history.json` shape**

```json
{
  "meta": {
    "preset": "better",
    "steps": 2000,
    "batch_size": 8,
    "seq_len": 64,
    "lr": 0.003,
    "params": 4764672,
    "corpus_chars": 4186740,
    "vocab": 90
  },
  "history": [
    {"step": 1, "train_loss": 4.4675, "val_loss": 4.1322},
    {"step": 25, "train_loss": 2.9807, "val_loss": 2.9920}
  ]
}
```

### 6.5 RESULT (how to read success)

| Signal | Good result | Bad / expected caveat |
|--------|-------------|------------------------|
| Train loss | falls from ~4.5 → ~1.4 | stuck near random (~4.5) |
| Val loss | follows train down | rises while train falls = overfit |
| Samples | start as noise; later get story-ish words | still gibberish after many steps → need more steps / data |
| Files | all 4 artifacts exist | missing PNG → install matplotlib |

**Real TinyStories `better` loss table (excerpt)**

| Step | Train | Val | Sample quality (rough) |
|------|-------|-----|------------------------|
| 1 | 4.47 | 4.13 | noise |
| 100 | 2.22 | 2.29 | letter soup |
| 500 | 1.64 | 1.73 | some English fragments |
| 1000 | 1.50 | 1.56 | words appear |
| 1600 | 1.41 | 1.46 | short story-like phrases |

**Builtin `--dataset toy` overfit example** (tiny file)

| Step | Train | Val | Note |
|------|-------|-----|------|
| 1 | ~3.9 | ~3.7 | start |
| 200 | ~1.2 | ~2.8 | val rising |
| 1000 | ~0.34 | ~4.4 | memorized train |

### 6.6 Recommended recipes

| Goal | Command |
|------|---------|
| Fast smoke | `--preset pc --steps 50 --dataset toy` |
| First real run | `--preset better --steps 2000` (TinyStories) |
| Longer CPU | `--preset better --steps 5000 --seq-len 96` |
| Shakespeare | `--dataset tinyshakespeare --steps 3000` |

---

## 7. Graph — input / output / result

Example result from a real laptop run (`--preset better --steps 2000`, TinyStories):

![Train / val loss curves](toy_dsv41/_artifacts/pc_run/loss_curves.png)

*Blue = train loss · Red dashed = val loss · Path: `toy_dsv41/_artifacts/pc_run/loss_curves.png`*

### 7.1 Automatic graph (during train)

| | |
|--|--|
| **Input** | in-memory history list written to `history.json` |
| **Method** | `toy_dsv41/utils/plotting.py` → matplotlib Agg backend |
| **Output** | `toy_dsv41/_artifacts/pc_run/loss_curves.png` |
| **Result** | visual train vs val curve (image above) |

```bash
xdg-open toy_dsv41/_artifacts/pc_run/loss_curves.png
```

### 7.2 What the graph shows

| Series | Style | Meaning |
|--------|-------|---------|
| **Train loss** | blue solid | loss on the latest train batch at log time |
| **Val loss** | red dashed | mean loss on 10 random val batches |

Optional annotation: **“overfit risk (val↑ train↓)”** when val rises while train falls.

### 7.3 How to read the graph (result)

```text
Both curves down          → learning (good)
Train down, val flat/up   → overfitting
Both flat high            → not learning (lr / steps / data)
Train << val late         → memorization (especially --dataset toy)
```

### 7.4 Replot without retraining

| | |
|--|--|
| **Input** | `history.json` |
| **Method** | `python -m toy_dsv41.utils.plot_history` |
| **Output** | `loss_curves.png` |
| **Result** | same graph regenerated |

```bash
python -m toy_dsv41.utils.plot_history

python -m toy_dsv41.utils.plot_history \
  --history toy_dsv41/_artifacts/pc_run/history.json \
  --out    toy_dsv41/_artifacts/pc_run/loss_curves.png
```

```text
wrote .../loss_curves.png
```

Skip plot during train: `--no-plot`.

---

## 8. Generate — input / output / result

### 8.1 Command

```bash
python -m toy_dsv41.train --generate-only \
  --prompt "Once upon a time" \
  --temperature 0.3 \
  --max-new 80
```

### 8.2 INPUT

| Input | Meaning |
|-------|---------|
| `toy_pc.pt` | trained checkpoint (weights + tokenizer + config) |
| `--prompt` | starting characters |
| `--temperature` | randomness (lower = safer) |
| `--max-new` | how many new characters to append |
| `--out-dir` | folder that contains `toy_pc.pt` (default `pc_run`) |

### 8.3 Method

```text
1. load checkpoint
2. encode prompt with saved CharTokenizer
3. autoregressive generate (temperature + top-k)
4. decode ids → string
```

### 8.4 OUTPUT

```text
loaded step=2000 train=1.xxxx val=1.xxxx
prompt : 'Once upon a time'
output : 'Once upon a time, there was a little ...'
```

### 8.5 RESULT tips

| Want | Try |
|------|-----|
| Cleaner text | `--temperature 0.2` or `0.3` |
| More variety | `--temperature 0.8` |
| Longer text | `--max-new 120` |
| After TinyStories | prompt `"Once upon a time"` |
| After Shakespeare | prompt `"First Citizen:\n"` |
| After builtin toy | prompt `"the causal encoder"` |

```bash
python -m toy_dsv41.train --generate-only \
  --prompt "Once upon a time there was" \
  --temperature 0.25 --max-new 120
```

If missing checkpoint:

```text
missing .../toy_pc.pt; train first
```

---

## 9. Demo — input / output / result

### Command

```bash
python -m toy_dsv41.demo
python -m toy_dsv41.demo --preset better
```

| | |
|--|--|
| **Input** | `--preset` only (no corpus / checkpoint required) |
| **Method** | build `ToyDSV41`, pack Engram, run short prefill with `trace=True` |
| **Output** | console: folder flow, original-vs-toy table, KV math, CSA2 mode logs |
| **Result** | you see architecture behavior; also writes `_artifacts/engram_demo.pt` |

Useful for learning paper acronyms before training.

---

## 10. Test — input / output / result

### Command

```bash
python -m toy_dsv41.tests.test_smoke
```

| | |
|--|--|
| **Input** | random tensors + builtin `--dataset toy` corpus (offline) |
| **Method** | shape checks, corpus batching, all presets, short generate |
| **Output** | `all tests passed` |
| **Result** | package is healthy before a long train |

| Test | Checks |
|------|--------|
| `test_forward_shapes` | logits `(B,T,V)`, KV length |
| `test_corpus_and_batch` | builtin load + `(B,T)` batches |
| `test_presets` | `demo` / `pc` / `better` build |
| `test_generate` | sequence grows by `max_new` |

### Manual checklist

| Check | Command | Expected result |
|-------|---------|-----------------|
| Demo | `python -m toy_dsv41.demo` | trace prints, no crash |
| Short train | `--preset pc --steps 50 --dataset toy` | train loss drops |
| Graph | `ls .../loss_curves.png` | file exists |
| Generate | `--generate-only --prompt "hello"` | prints `output :` |
| Unit tests | `python -m toy_dsv41.tests.test_smoke` | `all tests passed` |

---

## 11. Presets

From `toy_dsv41/model/config.py`.

| Preset | Approx params | `d_model` | When to use |
|--------|---------------|-----------|-------------|
| `demo` | smallest | 64 | architecture demo |
| `pc` | ~2.1M | 64 | fast CPU train (CLI default) |
| `better` | ~4.8M | 96 | better samples + graphs |

All presets share: **4 enc + 4 dec** layers, **4** heads, MoE **8/2**, SWA + CSA2 + Engram.

---

## 12. Paper → code

| Paper idea | Meaning | Toy file |
|------------|---------|----------|
| CED | causal encoder–decoder | `model/blocks.py`, `transformer.py` |
| CSA2 | compressed sparse attention | `model/csa2.py` |
| SWA | sliding window | `model/swa.py` |
| MoE | sparse experts | `model/moe.py` |
| Engram | hash memory | `model/engram.py` |
| FP4 KV / bounded replay | KV ideas | `kv_cache.py`, `csa2.py` |
| Size table | 552B vs toy | `model/config.py` |

```bash
xdg-open docs/DeepSeek_V41_Tech_Report.pdf
rg -n "CSA2|Causal Encoder|Engram|MoE" docs/DeepSeek_V41_Tech_Report.txt
```

---

## 13. Quick path

```bash
conda activate ml
cd ~/Downloads/DeepSeek-V4.1-Flash

# data (once)
python -m toy_dsv41.data.download_opensource tinystories

# see architecture
python -m toy_dsv41.demo

# train → files + graph
python -m toy_dsv41.train --preset better --steps 2000

# open graph
xdg-open toy_dsv41/_artifacts/pc_run/loss_curves.png

# sample
python -m toy_dsv41.train --generate-only \
  --prompt "Once upon a time" --temperature 0.3

# test
python -m toy_dsv41.tests.test_smoke
```

**Offline only**

```bash
python -m toy_dsv41.train --dataset toy --preset pc --steps 500
```

---

## 14. Cheat sheet

| Goal | Command |
|------|---------|
| Setup | `pip install -r toy_dsv41/requirements.txt` |
| About data / download | `python -m toy_dsv41.data.download_opensource tinystories` |
| Train (method + results) | `python -m toy_dsv41.train --preset better --steps 2000` |
| Graph | `xdg-open toy_dsv41/_artifacts/pc_run/loss_curves.png` |
| Replot graph | `python -m toy_dsv41.utils.plot_history` |
| Generate | `python -m toy_dsv41.train --generate-only --prompt "Once upon a time"` |
| Demo | `python -m toy_dsv41.demo` |
| Test | `python -m toy_dsv41.tests.test_smoke` |
| Paper | `xdg-open docs/DeepSeek_V41_Tech_Report.pdf` |

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No module named torch` | wrong env | `conda activate ml` + pip install requirements |
| `Need datasets for TinyStories` | missing HF lib | `pip install datasets` or `--dataset toy` |
| `missing toy_pc.pt` | no train yet | run train, or fix `--out-dir` |
| Plot skipped | no matplotlib | `pip install matplotlib` |
| Val explodes on `--dataset toy` | tiny corpus | use TinyStories; overfit is expected on builtin |
| Samples still garbage | early training | more `--steps`, lower `--temperature` |
| Prompt weird chars | vocab mismatch | use characters from the trained corpus |
| Slow CPU | large preset | `--preset pc`, smaller batch/seq |

---

## 16. More links

- Data details: [toy_dsv41/data/README.md](toy_dsv41/data/README.md)
- Package tree: [toy_dsv41/README.md](toy_dsv41/README.md)
- Paper folder: [docs/README.md](docs/README.md)
- Real multi-GPU serve (not this toy): [MiaAI DGX Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)
