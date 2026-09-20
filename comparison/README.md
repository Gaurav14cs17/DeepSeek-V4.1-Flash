# Comparison suite (activated after both Minis pass their stages)

Identical workloads across V4-Mini and V4.1-Mini. Outputs go to:

- `comparison/results.json`
- `comparison/results.csv`
- `comparison/report.md`

Do **not** treat tiny-model numbers as production DeepSeek performance.

Scripts:

| Script | Status |
|--------|--------|
| `compare_architecture.py` | available (high-level) |
| `compare_parameters.py` | available |
| `compare_memory.py` | available |
| `compare_latency.py` | available |
| `compare_training.py` | available |
| `compare_kv_cache.py` | stub until KV stages land |
| `compare_accuracy.py` | stub until longer trains land |
