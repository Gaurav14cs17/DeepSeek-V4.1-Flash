# DeepSeek-V4.1-Flash architecture (Mini paper notes)

**Status:** Documented now; **implementation deferred** until V4-Mini stage-1+ passes (`docs/progress.md`).

**Primary source:** `docs/DeepSeek_V41_Tech_Report.pdf` / `.txt`.

## Production snapshot (abstract + Table 1 + Fig. 3)

| Setting | DeepSeek-V4.1-Flash |
|---------|---------------------|
| Architecture | Multimodal MoE + **CED** |
| Backbone | **552B** |
| Activated | **8B prefill / 16B decode** (CED) |
| Attention | Pure **CSA2** (Full / Reindex / Reuse) |
| KV precision | **FP4** main/global KV (+ FP8 SWA path discussed) |
| Deployment | **SWA Bounded Replay** (persist global KV; rebuild SWA from last \(n_{\mathrm{win}}\)) |
| Memory vs V4-Flash | ~1/4 runtime KV; ~1/8 persistent KV (report §1) |
| Extras | Engram, Single-Pass mHC, DSpark, native ViT |

## Prefill vs decode (must stay visible in Mini code)

**Prefill**

```text
input → Causal Encoder → compressed / shared global state → Decoder (CED) → logits
```

**Decode**

```text
previous token → Decoder → compressed/global KV (+ SWA local) → next token
```

Do not hide CED behind one opaque `forward` without logs.

## Labels for Mini (planned)

| Component | Label |
|-----------|-------|
| CED tiny enc/dec | **[SIMPLIFIED]** |
| CSA2 modes | **[SIMPLIFIED]** / **[SIMULATED]** cross-layer reuse |
| FP4 KV | **[SIMULATED]** fake-quant / MXFP4-ish |
| SWA Bounded Replay | **[SIMPLIFIED]** last-\(n_{\mathrm{win}}\) trim |
| Vision / Engram | **[SIMPLIFIED]** toys |

## Relation to V4-Mini

V4.1 Mini must **copy only reusable primitives** (embedding, RMSNorm, RoPE, MoE pieces) and document diffs in `comparison/` and these notes.
