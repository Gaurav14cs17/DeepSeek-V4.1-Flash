#!/usr/bin/env python3
"""One-step train loss smoke for both Minis."""
import torch
from _load import load

def step(model, x, y):
    import torch.nn.functional as F
    if hasattr(model.cfg, "n_encoder_layers"):
        logits, _ = model(x, trace=False, use_bounded_replay=False, use_ced_prefill=False)
    else:
        logits, _ = model(x, trace=False)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    return float(loss)

V4, p4 = load("DeepSeekFlashV4-Mini", "DeepSeekFlashV4Mini")
m4 = V4(p4("demo"))
x = torch.randint(0, m4.cfg.vocab_size, (4, 16))
y = torch.randint(0, m4.cfg.vocab_size, (4, 16))
print("V4-Mini loss", step(m4, x, y))

V41, p41 = load("DeepSeekFlashV4.1-Mini", "DeepSeekFlashV41Mini")
cfg = p41("demo"); cfg.use_vision = False
m41 = V41(cfg)
x = torch.randint(0, m41.cfg.vocab_size, (4, 16))
y = torch.randint(0, m41.cfg.vocab_size, (4, 16))
print("V4.1-Mini loss", step(m41, x, y))
