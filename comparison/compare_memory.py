#!/usr/bin/env python3
import sys
from pathlib import Path
from _load import REPO, load

sys.path.insert(0, str(REPO / "DeepSeekFlashV4-Mini"))
from optimization.kv_cache import estimate_kv as est4
from optimization.memory import param_bytes

V4, p4 = load("DeepSeekFlashV4-Mini", "DeepSeekFlashV4Mini")
m4 = V4(p4("pc"))
print("V4", param_bytes(m4), est4(m4.cfg, 64).summary())

sys.path.insert(0, str(REPO / "DeepSeekFlashV4.1-Mini"))
for name in list(sys.modules):
    if name.startswith("optimization"):
        del sys.modules[name]
from optimization.kv_cache import estimate_kv as est41
from optimization.memory import param_bytes as pb41

V41, p41 = load("DeepSeekFlashV4.1-Mini", "DeepSeekFlashV41Mini")
cfg = p41("pc"); cfg.use_vision = False
m41 = V41(cfg)
print("V4.1", pb41(m41), est41(m41.cfg, 64).summary())
