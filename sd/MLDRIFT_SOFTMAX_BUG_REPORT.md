# MLDrift fp16 Softmax/Attention — Numerical Overflow Producing Gray Images

## Summary

A fully-fp16 ONNX graph (SD-family VAE decoder) renders a **flat gray image** when run on the
**MLDrift backend**, but renders **correctly** on the ORT CPU EP and the ORT WebGPU/JSEP backend.
We have root-caused this to MLDrift's **fp16 softmax / fused-attention kernel not subtracting the
row maximum before `exp`** ("unsafe" softmax), causing `exp` overflow → `inf`/`NaN` → collapsed
attention. The defect is inside MLDrift's kernel/EP; it is **not** expressible or fixable at the
ONNX-graph level.

## Where it happens

The offending node is the VAE decoder's **mid-block spatial self-attention**
(`mid_block.attentions.0`, a diffusers-style `AttnBlock`), a single-head attention at the
bottleneck:

```
GroupNorm → Q,K,V projections ([B, N, 512])
          → scores = Q·Kᵀ ([B, N, N])
          → scores × (1/√512)          # scale ≈ 0.0442
          → Softmax(axis=-1)
          → · V → proj_out → residual
```

- `N = H·W ≈ 4096` spatial tokens (64×64 latent)
- `head_dim = 512`, single head
- softmax is over `N ≈ 4096` keys per row

## Root cause

- **Pre-softmax logits reach ≈ 15** (measured on random input; likely larger on real latents).
- The fp16 `exp` overflow threshold is `ln(65504) ≈ 11.09`.
- A **numerically-naive softmax** (no row-max subtraction) computes `exp(logit)` directly:
  - `exp(15) ≈ 3.3e6 > 65504` → `inf`
  - the denominator, a sum over ~4096 keys, also overflows → `inf`
  - `inf / inf = NaN` → attention weights become NaN/uniform → feature-mixing pass collapses →
    **flat gray output**.
- The standard fix is the **safe softmax**: subtract the row max first, `exp(x − max) ≤ 1`, which
  never overflows. Softmax is **shift-invariant** (`softmax(x) = softmax(x − c)`), so this is
  numerically exact. Note it is **not scale-invariant**, so rescaling/temperature cannot
  substitute for max-subtraction.

## Evidence it is the backend kernel, not the graph math

Same ONNX model, three backends:

| Backend | fp16 softmax result | Why |
|---|---|---|
| ORT **CPU** EP | ✅ correct | kernel subtracts row-max / decomposes to a stable subgraph |
| ORT **WebGPU / JSEP** | ✅ correct | stable fp16 softmax kernel |
| **MLDrift** | ❌ gray | fp16 softmax/attention kernel appears to omit row-max subtraction |

The model grays **only** on MLDrift → the differentiator is MLDrift's fp16 softmax/attention kernel.

## Graph-level rewrites we tried — all still gray on MLDrift

We rewrote the softmax in the ONNX graph four different ways. All four still produce a gray image
on MLDrift:

| Graph form | Description | Result |
|---|---|---|
| `Softmax` (naive) | fused fp16 Softmax op | gray (expected — overflow) |
| `ReduceMax → Sub → Softmax` | max-subtract upstream, keep fused Softmax | gray |
| `ReduceMax → Sub → Exp → ReduceSum → Div` | fully decomposed, **no** Softmax op | gray |
| `Exp(z − Log(Σ exp z))` (logsumexp) | **no** Softmax op, **no** Div, **no** softmax-shaped subgraph | gray |

**The decisive result is the logsumexp form.** It is provably overflow-free at the graph level and
contains nothing that resembles a softmax pattern to re-fuse — yet it still grays. This means
MLDrift is **not** merely re-fusing a `Softmax` op. It is **substituting its own numerically-naive
fused attention/softmax kernel for the entire `MatMul → normalize → MatMul` block, regardless of
how the normalization is expressed in the ONNX graph.**

We also verified that ORT's own graph optimizer does **not** re-fuse these decomposed primitives
back into a `Softmax` (a `Softmax` op count of 0 survives at BASIC/EXTENDED/ALL optimization levels
on CPU). Therefore the substitution happens **inside MLDrift's EP graph construction / kernel
selection**, which cannot be influenced from the ONNX file.

## What we need from MLDrift

1. **Confirm the fusion**: Does MLDrift fuse the `MatMul → (softmax/normalize) → MatMul` block at
   `mid_block.attentions.0` into a native attention / SDPA / softmax kernel? (Debug logs /
   operator-mapping dump for that block would confirm.)
2. **Confirm the kernel math**: Does that fused fp16 kernel **subtract the row maximum** before
   `exp`? Our evidence says it does not.
3. **The fix**: Apply the standard **safe softmax** — subtract the per-row max before `exp`
   (equivalently, use a FlashAttention-style online-softmax with running max) in the fp16
   attention/softmax kernel. This is the same correction the ORT CPU and WebGPU backends already
   implement. With logits ≈ 15 and an fp16 overflow point of ≈ 11.09, an unsafe kernel is
   guaranteed to overflow here; a safe kernel keeps every `exp` argument ≤ 0.

## Why this matters (motivation for the fp16 path)

MLDrift applies a **whole-graph fp16 acceleration path** when the graph is uniformly fp16, giving a
**measured ~3× speedup** on this decoder. A single fp32 op disqualifies the whole-graph fp16 path.
Today we are forced to keep this one Softmax in fp32 (wrapped in
`Cast(fp16→fp32) … Softmax(fp32) … Cast(fp32→fp16)`) to get correct images — which forfeits the 3×.
**The gray image and the 3× speedup are two faces of the same fused fp16 path**: it is fast
*because* it fuses aggressively, and it is wrong *because* the fused fp16 attention/softmax kernel
does not subtract the row max. Fixing the kernel unlocks both correctness and the 3× on this and
every similar large-`N` single-head attention.
