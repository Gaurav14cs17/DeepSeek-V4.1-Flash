# Training data for the toy model

**Not** DeepSeek’s 45T multimodal mix. We use small **open** texts on your PC.

## Datasets

| Dataset | Path | Source | Default? |
|---------|------|--------|----------|
| **TinyStories** | `opensource/tinystories.txt` | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — paper *TinyStories*; example train repo [SauravP97/tiny-stories-hf](https://github.com/SauravP97/tiny-stories-hf) | **Yes** |
| Tiny Shakespeare | `opensource/tinyshakespeare.txt` | karpathy/char-rnn | `--dataset tinyshakespeare` |
| Builtin educational | `corpus_toy.txt` | this repo | `--dataset toy` |
| Custom | any `.txt` | you | `--corpus PATH` |

TinyStories full dump is large (~2GB). We export a **subset** (default **20,000** train stories) as plain UTF-8 for CPU char-LM training.

```bash
python -m toy_dsv41.data.download_opensource --list
python -m toy_dsv41.data.download_opensource tinystories
python -m toy_dsv41.data.download_opensource tinystories --max-stories 50000 --force

python -m toy_dsv41.train --dataset tinystories --preset better --steps 2000
python -m toy_dsv41.train --generate-only --prompt "Once upon a time" --temperature 0.3
```

## Why TinyStories?

From the TinyStories paper (via [tiny-stories-hf](https://github.com/SauravP97/tiny-stories-hf)): small models trained on simple synthetic stories can still produce fluent English. That matches this toy’s size (~2–5M params) better than random web text.
