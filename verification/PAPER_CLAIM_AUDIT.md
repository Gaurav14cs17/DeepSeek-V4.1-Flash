# Paper Claim Audit

Classification key:

| Code | Meaning |
|------|---------|
| A | Faithfully implemented (miniaturized) |
| B | Simplified |
| C | Scale-adapted |
| D | Simulated |
| E | Missing |
| F | Incorrect |

Never upgrade C/D/E into A.

| Claim / mechanism | Class | Evidence |
|-------------------|-------|----------|
| Char tokenizer vs production BPE | B | Explicit `[SIMPLIFIED]` |
| Token embedding | A | Standard nn.Embedding |
| RMSNorm | A | Matches Zhang & Sennrich form |
| RoPE after which KV quant would apply | A (RoPE) / D (quant) | RoPE faithful; quant is fake |
| V4 CSA–HCA hybrid | B/C | Provisional in `mla.py`; public DSA deferred (E) |
| Overlap / non-overlap compression | B | Mean-pool toys |
| DeepSeekMoE shared + routed + hash bootstrap | B/C | Tiny expert counts; bias balancer toy |
| Exact SWA replay cost \(L\times n_{\mathrm{win}}\) | D | Accounting string only |
| Incremental KV cache decode | E | Not implemented |
| V4.1 CED encoder–decoder | B (provisional) / E (public API) | ToyDSV41 vs deferred `ced.py` |
| CSA2 Full/Reindex/Reuse | B | `_csa2.py` |
| Hierarchical indexer pool | B | Tiny pool size |
| FP4 main KV | D | Float round-trip; no pack; public stub E |
| FP8 SWA KV | D | Float round-trip |
| SWA Bounded Replay persist-global-only | B/D | Tail truncate + zero-pad logits |
| Vision / Engram / DSpark | B/C | Optional toys; multimodal public stubs E |
| Stage-1 “attention verified via train” | F if overclaimed | Train uses `TemporaryCausalBlock`, not `attention.py` |
| Deferred tests “passing” | F | Soft `assert True` |
| RoPE A1 faster than A0 | F if claimed | Measured **REGRESSION** |
| Any FP4/DSA/CSA2 practical latency win | E/D | No BEFORE/AFTER measurement except Stage-1/RoPE |

## Soft-pass / deferred inventory (code audit)

18 findings in `verification/results/code_audit.json`: 10 deferred modules, 7 soft-pass tests, 1 bare `pass` in DSpark accept loop.
