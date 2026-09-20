# DeepSeek-V4-Flash architecture (Mini paper notes)

**Status:** Stage-1 scaffolding. Later modules (CSA/HCA, MoE, …) will deepen these notes.

**Primary sources**

- DeepSeek-V4.1-Flash tech report (`docs/DeepSeek_V41_Tech_Report.pdf`) — contrasts V4-Flash vs V4.1 (Table 1, §1–2).
- Cited predecessor: DeepSeek-V4 / V4-Flash (DeepSeek-AI, 2026b) as described in that report.

## Production snapshot (Table 1 of V4.1 report)

| Setting | DeepSeek-V4-Flash |
|---------|-------------------|
| Architecture | MoE |
| Backbone params | **284B** |
| Activated params | **~13B** |
| Attention (reported contrast) | CSA–HCA hybrid (vs V4.1 pure CSA2) |
| SWA | Present in every layer (deployment); first layers SWA-heavy in layout descriptions |
| Replay | Exact-style SWA rebuild needing ~\(L \times n_{\mathrm{win}}\) tokens (report §1) |
| Multimodal native (Table 1) | blank / not the V4.1 story |

## Conceptual stack (what Mini will teach)

```text
tokens
  → Embedding
  → [RMSNorm]
  → Layers: SWA | CSA | HCA  +  DeepSeekMoE (shared + routed)
  → (+ multi-pass mHC in V4)
  → RMSNorm → LM head
  → (+ MTP in V4; replaced/augmented in V4.1)
```

## Labels for Mini components

| Component | Label | Note |
|-----------|-------|------|
| Char tokenizer | **[SIMPLIFIED]** | Production uses large BPE (~129k in V4.1 table); Mini uses char vocab |
| Token embedding | **[FAITHFUL]** | Standard learned embedding |
| RMSNorm | **[FAITHFUL]** | Used throughout DeepSeek stacks (Zhang & Sennrich, 2019; cited in report) |
| RoPE | **[FAITHFUL]** | Report: RoPE on attention; quantize KV *after* RoPE (§2.4-ish discussion) |
| CSA / HCA | later | **[SIMPLIFIED]** educational compress + Top-K |
| MoE | later | **[SIMPLIFIED]** tiny expert counts |
| Exact SWA replay cost | later | **[SIMULATED]** accounting, not cluster replay |

## Stage-1 scope

Implement and measure only: tokenizer, embedding, RMSNorm, RoPE.

Do **not** claim a TinyStories char-LM matches Flash quality.
