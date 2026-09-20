# Dataset ladder

Never download a huge corpus by default. Every script supports:

```text
--num-samples --max-tokens --max-seq-len --streaming --seed
```

## Levels

| Level | Source | Purpose |
|------:|--------|---------|
| **0** | Synthetic tensors / tiny char strings | Shapes, numerics, unit tests |
| **1** | TinyStories (few samples) | First LM training, RoPE, attention, MoE |
| **2** | TinyStories-scale prepared shard (~15M tokens max opt-in) | Longer train, perplexity |
| **3** | FineWeb / FineWeb-Edu **small subset** | Throughput / memory scaling |
| **4** | Small image–text pairs | V4.1 multimodal path |
| **5** | Synthetic long-context | KV, CSA2, FP4 cache, SWA replay |

## Defaults (safe)

- Level 0 is always available offline.
- Level 1 downloads **at most a few thousand samples** unless you raise `--num-samples`.
- Levels 2–3 require explicit larger `--num-samples` / `--max-tokens`.

## Scripts

| Script | Role |
|--------|------|
| `download.py` | Generic HF pull with caps |
| `prepare_tinystories.py` | Level 1–2 text shards |
| `prepare_fineweb.py` | Level 3 subset |
| `prepare_multimodal.py` | Level 4 |
| `synthetic_long_context.py` | Level 5 |

## Stage-1 note

Stage 1 training uses **Level 0** (bundled / synthetic char corpus) so the lab runs with zero network.
