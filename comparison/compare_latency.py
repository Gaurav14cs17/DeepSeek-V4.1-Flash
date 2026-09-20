#!/usr/bin/env python3
import time
import torch
from _load import load

def bench(model, x, n=10):
    model.eval()
    for _ in range(3):
        model(x, trace=False) if "use_bounded" not in model.forward.__code__.co_varnames else model(x, trace=False, use_bounded_replay=False, use_ced_prefill=False)
    t0 = time.perf_counter()
    for _ in range(n):
        if hasattr(model.cfg, "n_encoder_layers"):
            model(x, trace=False, use_bounded_replay=False, use_ced_prefill=False)
        else:
            model(x, trace=False)
    return (time.perf_counter() - t0) * 1000 / n

V4, p4 = load("DeepSeekFlashV4-Mini", "DeepSeekFlashV4Mini")
m4 = V4(p4("demo"))
x = torch.randint(0, m4.cfg.vocab_size, (2, 32))
print(f"V4-Mini   {bench(m4, x):.2f} ms")

V41, p41 = load("DeepSeekFlashV4.1-Mini", "DeepSeekFlashV41Mini")
cfg = p41("demo"); cfg.use_vision = False
m41 = V41(cfg)
x = torch.randint(0, m41.cfg.vocab_size, (2, 32))
print(f"V4.1-Mini {bench(m41, x):.2f} ms")
