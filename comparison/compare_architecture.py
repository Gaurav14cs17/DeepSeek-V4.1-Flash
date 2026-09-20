#!/usr/bin/env python3
"""Print architecture diffs."""
from _load import load

V4, p4 = load("DeepSeekFlashV4-Mini", "DeepSeekFlashV4Mini")
V41, p41 = load("DeepSeekFlashV4.1-Mini", "DeepSeekFlashV41Mini")
c4, c41 = p4("pc"), p41("pc")
rows = [
    ("layout", f"{c4.n_layers} full-depth", f"{c41.n_encoder_layers}+{c41.n_decoder_layers} CED"),
    ("attention", "SWA + CSA + HCA", "SWA + CSA2 Full/Reindex/Reuse"),
    ("mHC", "multi-pass", "single-pass"),
    ("extra heads", "MTP", "Engram + DSpark"),
    ("KV", "exact SWA replay L*n_win", "Bounded Replay n_win + FP4 sim"),
]
print(f"{'aspect':16s} | {'V4-Mini':40s} | {'V4.1-Mini'}")
print("-" * 100)
for a, b, c in rows:
    print(f"{a:16s} | {b:40s} | {c}")
