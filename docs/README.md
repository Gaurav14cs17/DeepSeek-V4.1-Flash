# docs/ — paper (flow step ①)

Workspace flow: **`docs/` → `toy_dsv41/demo` → `train` → `_artifacts/` → generate**  
See [../README.md](../README.md) for the full tree.

## Files in this folder

```
docs/
├── README.md                      ← this file
├── DeepSeek_V41_Tech_Report.pdf   # official paper
└── DeepSeek_V41_Tech_Report.txt   # searchable extract
```

```bash
xdg-open docs/DeepSeek_V41_Tech_Report.pdf
rg -n "CSA2|Causal Encoder" docs/DeepSeek_V41_Tech_Report.txt
```

## Read with the toy (`../toy_dsv41/model/`)

| Paper section | Toy path |
|---------------|----------|
| §2.2 CED | `toy_dsv41/model/blocks.py`, `transformer.py` |
| §2.3 CSA2 | `toy_dsv41/model/csa2.py` |
| §2.4 Engram / FP4 | `engram.py`, `csa2.py`, `kv_cache.py` |
| §3.2 SWA Bounded Replay | `toy_dsv41/model/kv_cache.py` |
| §4.2 sizes (552B, …) | `toy_dsv41/model/config.py` → `ORIGINAL_VS_TOY` |

Next step in the flow: `python -m toy_dsv41.demo`
