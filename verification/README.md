# Verification suite

Independent audit of DeepSeekFlashV4-Mini / DeepSeekFlashV4.1-Mini.

## Commands

```bash
python verification/run_fast_verification.py
python verification/run_numerical_verification.py
python verification/run_benchmark_verification.py
python verification/run_full_verification.py
# or
python -m pytest -q verification/tests
```

## Reports

| File | Contents |
|------|----------|
| `IMPLEMENTATION_INVENTORY.md` | Component × file × status table |
| `paper_to_code.md` | Paper → math → code → tests |
| `gradient_audit.md` | detach/no_grad review |
| `PAPER_CLAIM_AUDIT.md` | A–F claim classes |
| `FINAL_VERIFICATION_REPORT.md` | Full report + scoreboard + Q&A |
| `results/` | JSON machine outputs |

## Design notes

- `pkg_import.py` isolates the two Mini packages (shared top-level names).
- `reference/` holds slow correctness-first kernels.
- Audit rule: **report before mass fixes**; deferred Stage gates are not silently unlocked.
