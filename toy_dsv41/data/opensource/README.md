# opensource/ — downloaded training text

Plain `.txt` corpora for CPU toy training. **Not** checked into git (see `.gitignore`).

| File | Download |
|------|----------|
| `tinystories.txt` | `python -m toy_dsv41.data.download_opensource tinystories` |
| `tinyshakespeare.txt` | `python -m toy_dsv41.data.download_opensource tinyshakespeare` |

Then train:

```bash
python -m toy_dsv41.train --dataset tinystories --preset better --steps 2000
```

Details: [../README.md](../README.md)
