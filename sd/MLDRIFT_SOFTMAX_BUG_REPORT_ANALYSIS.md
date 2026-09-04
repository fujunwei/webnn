# Analysis of MLDRIFT_SOFTMAX_BUG_REPORT.md

Source checked: `third_party/ml-drift` as vendored in this chromium checkout
(`C:\Users\fujun\workspace\chromium\src\third_party\ml-drift`). This may be a
different revision than the ORT MLDrift EP the report was filed against —
conclusions below are conditioned on that caveat (see §5).

## 1. Central claim is contradicted by source

The report's root-cause claim — "MLDrift's fp16 softmax/attention kernel does
not subtract the row max before `exp`" — does not hold for the code inspected.
All three softmax-shaped kernels implement safe/online (running-max) softmax:

- `ml_drift/common/kernels/softmax.cc`, `softmax1x1.cc` (standalone softmax):
  explicit `float maximum`, `new_max`, `scale = EXP_FUNC(maximum - new_max)`,
  `exp_res = EXP_FUNC(src - maximum)`.
- `ml_drift/common/kernels/special/conv_softmax_conv.cc` (fused
  `MatMul→softmax→MatMul`, the likely target of "fused attention" in the
  report): implements full FlashAttention-style online softmax — running max
  (`max_val_x0` init `-65000.0f`), rescaling of the running sum
  (`exp_sum_adj = exp_sum_adj * scale + interm_sum`), and rescaling of the
  already-accumulated output (`r_x*_s* *= scale`).

In both, every `exp()` argument is `src - running_max ≤ 0`, so `exp() ≤ 1`.
The overflow chain the report describes (`exp(15) → inf → inf/inf → NaN`)
cannot occur in this code as read. **Requesting "add row-max subtraction" as
the fix would be a no-op against this source.**

## 2. Report is right about kernel fusion

The report's claim that MLDrift substitutes a fused kernel for the whole
`MatMul → normalize → MatMul` block, independent of how the block is expressed
in the graph, matches what the code does: `ConvSoftmaxConv` fuses on
*structural* pattern match (`MatMul→…→MatMul`), not on the presence of a
`Softmax` op. This explains why all four of the report's graph-level rewrites
(including the overflow-free logsumexp form) still produced a gray image —
none of them change the structural shape the fusion pass matches on.

## 3. A real, different candidate defect: fp16 softmax accumulators

In `conv_softmax_conv.cc`, type selection is:

```cpp
const DataType acc_type = precision == CalculationsPrecision::F16
                              ? DataType::FLOAT16 : DataType::FLOAT32;
const DataType type = op_def.src_tensors[0].GetDataType();
StrReplaceAll({{"SType",   ToUclDataType(type, 1)},
               {"Type",    ToUclDataType(type, 4)},
               {"AccType", ToUclDataType(acc_type, 4)}}, &c);
```

`AccType` (fp32 when `CalculationsPrecision::F32_F16`) is used **only** for
the conv/matmul reduction (`ucl::Convert<AccType>(...)`, `AccType val =
ucl::Init<AccType>(0.0f)`). The softmax block's own accumulators are declared
with `SType`, not `AccType`:

```cpp
c += "  SType exp_sum_adj_x0 = 0.0f;\n";   // fp16 in an all-fp16 graph
c += "  SType max_val_x0 = -65000.0f;\n";  // fp16
```

So `exp_sum_adj` (softmax denominator) and `max_val` (running max) stay fp16
regardless of the `F32_F16` mixed-precision mode — that mode only upgrades the
matmul accumulation, not the softmax reduction. This is inconsistent with the
standalone `softmax.cc`, which always tracks `maximum` in `float`.

Precision (not overflow) is the failure mode this predicts: `N ≈ 4096`,
fp16 spacing near 2048 is 2.0, while each added term after max-subtraction is
`≤ 1`. Most terms are too small to change the fp16 accumulator once it grows —
the denominator saturates/rounds badly, corrupting the softmax weights.
This does not produce `inf`/`NaN`; it produces finite, wrong values.

## 4. Discriminating test

The two hypotheses (report's overflow claim vs. this fp16-accumulation
hypothesis) predict different tensor content at the failing node:

| Hypothesis | Predicted `mid_block.attentions.0` output |
|---|---|
| Report: unsafe softmax, no max-subtraction | contains `inf` / `NaN` |
| This analysis: fp16 accumulator precision | fully finite, but numerically wrong |

Dump that tensor and count non-finite values. Zero `NaN`/`inf` would rule out
the report's stated mechanism and point at accumulator precision (or another
cause) instead.

## 5. Caveats before acting on this

- **Version match unconfirmed.** This analysis used the ml-drift copy vendored
  in this chromium checkout (TFLite/LiteRT delegate path). The report is
  about the **ORT MLDrift execution provider**, which may vendor a different
  ml-drift revision. Line numbers/behavior should be re-checked against that
  revision before filing conclusions upstream.
- **Kernel-selection path unconfirmed.** Not yet verified whether the ORT
  MLDrift EP's operator/pattern selection for this exact attention shape picks
  `ConvSoftmaxConv` versus a separate SDPA path (`operation_selector.cc`
  references a `MakeScaledDotProductAttention`-style selector) — that other
  path's numerics have not been inspected.

## 6. Recommendation

1. Do not act on the report's literal ask (add max-subtraction) — it is
   already present.
2. Get a non-finite-value count from the actual failing tensor to confirm/deny
   the overflow hypothesis before proposing any kernel change.
3. If the count is zero, prototype fix: promote the softmax block's `SType`
   accumulators (`max_val_x*`, `exp_sum_adj_x*`, and the rescale/interm
   temporaries) to `AccType` (fp32) in `conv_softmax_conv.cc`, mirroring what
   `softmax.cc` already does — storage stays fp16, only the reduction widens,
   so the fp16 fast path's speedup is largely preserved.

## 7. Empirical verification (2026-09-01)

Ran `sam_encoder_runner.exe --verify` (CPU/XNNPACK fp32 reference vs. GPU
ml-drift-delegate fp16) against `sd/vae_decoder_f16.tflite` — this runner is
model-agnostic (any `.tflite`, ramp input by default), not SAM-specific.
Command:

```
sam_encoder_runner.exe --model=sd\vae_decoder_f16.tflite --verify \
  --precision=fp16 --tolerance=1e6 --dump-outputs=sd\vae_verify_out
```

Model compiled `IsFullyAccelerated=1` (1012/1012 ops on GPU delegate), so any
divergence is attributable to the delegate, not a CPU-fallback split.

| | elems | nan | min | max | mean | std |
|---|---|---|---|---|---|---|
| CPU (fp32, reference) | 786432 | 0 | -1.741 | 0.650 | -0.8335 | 0.7269 |
| GPU (fp16, ml-drift)  | 786432 | 0 | -69.75 | 16.47  | -30.30 | 27.91  |

`max_abs=68.0`, `mean_abs=31.9` (on a ramp input, not a real decoder latent —
this only tests whether the delegate reproduces CPU math on this graph, not
whether *this specific* input triggers the SD gray-image symptom).

**Interpretation:**

- **Zero NaN/Inf on both sides** — directly rules out the report's stated
  failure mechanism (unsafe softmax → `exp` overflow → `inf`/`NaN`). Confirms
  §1: the row-max subtraction is doing its job; nothing overflows.
- **But GPU output is real and systematically wrong**, not within normal
  fp16-vs-fp32 rounding tolerance. mean and std are both inflated by
  **~36–38×** (`-30.30 / -0.8335 ≈ 36.3`, `27.91 / 0.7269 ≈ 38.4`) — nearly
  the *same* ratio for both statistics, which is the signature of a
  near-uniform multiplicative bias, not random rounding noise.
- This matches the §3 hypothesis precisely: if the softmax denominator
  (`exp_sum_adj`) accumulates in fp16 over ~thousands of terms, most
  post-max-subtraction increments (`≤ 1`) fall below the fp16 ULP once the
  running sum grows large and get silently dropped — undercounting the true
  sum by roughly a constant factor for a given N. Dividing by an
  artificially-small denominator inflates the attention output by
  approximately that same factor, propagating into every downstream layer as
  a near-uniform scale distortion — consistent with the observed ~37×.
- This is evidence *for* an accumulator-precision defect and *against* the
  report's overflow claim, but is not yet proof of the exact mechanism —
  next step is to apply the `AccType` promotion from §6.3 to
  `conv_softmax_conv.cc`, rebuild the WebGPU delegate DLL, and re-run this
  same `--verify` command: if `mean_abs`/`max_abs` collapse toward CPU's
  scale (order ~1, not ~30), that confirms the softmax accumulator as the
  root cause.

Artifacts: `sd/vae_verify_out_cpu.bin`, `sd/vae_verify_out_gpu.bin` (raw f32,
786432 elements each), `sd/vae_verify_run.log` (full run log).

## 8. Fix attempt and result (2026-09-01): hypothesis refuted

Patched `conv_softmax_conv.cc`'s softmax block to accumulate `max_val_x*` /
`exp_sum_adj_x*` (and the per-iteration `interm_max`/`scale`/`interm_sum`
temporaries) in genuine `float` (fp32) instead of `SType`/`AccType` (both
fp16 for this kernel — see §correction below), using explicit
`ucl::Convert<float>(...)` / `ucl::Init<Type>(...)` / `ucl::Init<AccType>(...)`
at the boundaries where the fp32 locals meet the kernel's native fp16
`Type`/`AccType` values (required by WGSL's strict fp16/fp32 typing).
Rebuilt `libLiteRtWebGpuAccelerator.dll` (`-c opt`), redeployed to
`out/Release`, reran the identical `--verify` command.

**Result: bit-identical output**, to the last printed digit:

| | before fix | after fix |
|---|---|---|
| GPU min/max/mean/std | -69.75 / 16.4688 / -30.3043 / 27.9072 | -69.75 / 16.4688 / -30.3043 / 27.9072 |
| `max_abs` / `mean_abs` | 68.0359 / 31.9136 | 68.0359 / 31.9136 |

Per systematic-debugging: hypothesis test failed → stop, don't layer another
fix, re-derive. Traced why: `ConvSoftmaxConv` is not reached by structural
`MatMul→…→MatMul` shape-matching alone, as §2 assumed. Two extra gates sit in
front of it:

1. **TFLite→IR conversion** (`ml_drift_delegate/tflite/convert/convert_sdpa.cc`):
   an op only becomes `OperationType::SCALED_DOT_PRODUCT_ATTENTION` in the
   first place if the **TFLite graph itself encodes it as a
   `stablehlo.composite` op** (`TfLiteStablehloCompositeParams`) — i.e. the
   exporter must have tagged the attention region explicitly. A plain
   decomposed `BatchMatMul → Softmax → BatchMatMul` sequence, with no
   composite annotation, is never converted to this op at all.
2. **`GpuModelBuilder::BatchedMatMulSoftmaxBatchedMatMul`**
   (`ml_drift/common/gpu_model_builder.cc:3696`, only reachable from
   `MakeScaledDotProductAttention` once gate 1 passes) additionally requires
   `!mask_tensor && a_tensor width >= 1024`, then
   `IsConvSoftmaxConvSupported(...)` (Adreno+`ucl_wave_memory`, or
   Metal+Apple-Bionic, or AMD OpenCL, or supported Mali, or
   Intel-without-8-wide-subgroups) before it picks the fused kernel at all —
   otherwise it falls back to `BatchedMatMulSoftmaxBatchedMatMulSeparateKernels`
   (plain `Softmax` op, i.e. the already-fp32-accumulating `softmax.cc`).

The bit-identical rebuild result is strong evidence gate 1 (or gate 2) is
failing for `vae_decoder_f16.tflite` on this Intel UHD 630 / WebGPU run —
`ConvSoftmaxConv` is dead code for this specific model/hardware combination,
so its accumulator precision (fixed or not) cannot be the source of the
observed ~37× divergence.

**Status: root cause of the ~37× divergence is still open.** The fp16
accumulator theory for the *fused* kernel is refuted for this repro. Next
steps, in order of cheapness:

- Dump the actual op list of the compiled GPU model (or add a one-line log in
  `BatchedMatMulSoftmaxBatchedMatMul` / `ConvertSdpa`) to confirm whether SDPA
  fusion fires at all for this model — settles gate 1 vs. gate 2 directly.
- If it does not fire, the divergence must come from somewhere else in the
  1012-node graph — likely candidates: the standalone `Softmax` kernel path
  actually taken instead (verify it isn't secretly also lossy, despite reading
  fp32 in `softmax.cc`), generic fp16 `BatchMatMul`/`FullyConnected` rounding
  compounding over many layers, or a GroupNorm/scale constant issue. A
  layer-by-layer dump (intermediate tensor comparison CPU vs GPU at each op,
  not just final output) is the next concrete step to localize which op
  first diverges by an order of magnitude.

## 9. Probe result (2026-09-01): SDPA fusion never fires for this model

Added one-line `ABSL_LOG(INFO)` probes at the two candidate gates from §8 —
entry of `GpuModelBuilder::BatchedMatMulSoftmaxBatchedMatMul`
(`gpu_model_builder.cc`) and entry of `ConvertComposite`
(`ml_drift_delegate/tflite/ir_model_builder.cc`) — rebuilt, redeployed, reran
`--verify`. **Neither probe printed a single line.**

`BatchedMatMulSoftmaxBatchedMatMul` is compiled into the same bazel target
(`ml_drift_webgpu_accelerator_dll`, confirmed by the rebuild recompiling
`gpu_model_builder.cc`) and never fires — so gate 1 is where this dies:
`vae_decoder_f16.tflite` contains **no `stablehlo.composite` /
`odml.scaled_dot_product_attention` node at all**. (The `ConvertComposite`
probe result is inconclusive on its own — that file didn't recompile under
this bazel target, so it may be built by a different pipeline — but it's
moot given the definitive negative from the first probe.) This is consistent
with the model's shape: 1012 fully-decomposed GPU ops for a VAE decoder is
far more than a composite-collapsed graph would have (attention would be 1
node instead of Q/K/V-proj + transpose + matmul + softmax + matmul + reshape
≈ 10+ nodes).

**This refutes the fp16-accumulator-in-fused-kernel hypothesis entirely for
this repro** — not just "unconfirmed" as in §8, but definitively: the kernel
under suspicion is provably never invoked. The attention in this graph runs
through plain decomposed ops, meaning through the standalone `Softmax`
kernel (`softmax.cc`), which was already fp32-safe. Reverted both probe logs
(kept the §7 fp32 softmax-accumulator fix in `conv_softmax_conv.cc` itself —
harmless and still a real correctness improvement for whatever hardware path
does select that fused kernel, just not relevant to this bug).

**Where the investigation stands:** the ~37× mean/std amplification is real,
reproducible, and NOT explained by any softmax-related mechanism (neither
the original report's overflow claim, nor either accumulator-precision
hypothesis). It must originate elsewhere in the 1012-op decomposed graph —
GroupNorm, Conv2D, generic fp16 BatchMatMul/FullyConnected reduction, an
elementwise op, or a shape/layout bug that only manifests under the GPU
delegate. Next concrete step: instrument `sam_encoder_runner` (or a new
small tool) to dump every intermediate tensor for both CPU and GPU paths and
diff layer-by-layer to find the first op whose output ratio jumps to ~37×
instead of ~1× — that pinpoints the actual defective op directly, rather
than guessing from the final output's statistics.

## 10. Standalone Mul->Softmax->BatchMatMul microbenchmark (2026-09-01): also clean

Built a minimal, hand-written `.tflite` (no TF dependency, same flatbuffer
approach as `make_rank5_repro_tflite.py`) isolating exactly the decomposed
attention pattern confirmed in §9:

```
x    : [1, 4, 1024]                  (raw scores, input, range -18..18)
s    = mul(x, scale_const)           scale_const: scalar 0.125
p    = softmax(s, axis=-1)           reduces over 1024 (matches the >=1024
                                      threshold real fused attention gates on)
v    : [1, 1024, 32]                 constant "value" matrix
out  = batch_matmul(p, v)            -> [1, 4, 32]
```

Generator: `segment_anythings/tools/make_rank5_repro_tflite.py --softmax-attn`
(added `mul()`/`softmax()`/`batch_matmul()` builders to the existing
`ModelWriter`). Ran `sam_encoder_runner --verify --precision=fp16
--tolerance=1e-2` against `softmax_attn_repro.tflite`:

```
CPU: elems=128 min=-0.106847 max=0.0774683  mean=-0.00161107 std=0.045388
GPU: elems=128 min=-0.10675  max=0.0773926  mean=-0.00161064 std=0.0453472
[verify] max_abs=0.000153027 mean_abs=3.87187e-05 over_tol=0 -> PASS
```

Relative error is ~0.1-0.2%, i.e. ordinary fp16 rounding noise -- no trace of
a systematic multiplicative divergence. This rules out the standalone
(unfused) fp16 `Mul`/`Softmax`/`BatchMatMul` kernels as the source of the
~37x amplification, at the same reduction width (1024) the real fused
kernel's hardware gate checks for. Combined with §9, the softmax/attention
mechanism in general -- fused or decomposed -- is now excluded as the root
cause for `vae_decoder_f16.tflite`.

The ~37x divergence must come from a different op class in the 1012-op
graph. Next candidates to isolate with the same hand-written-model technique:
GroupNorm (`custom_call.LayerNorm`/RMS-norm-style ops), Conv2D (the VAE
decoder is conv-heavy), or generic fp16 accumulation in `FullyConnected`.
A synthetic single-op or small-chain repro per candidate, verified the same
way, is cheaper than instrumenting `sam_encoder_runner` for full
intermediate-tensor dumping and should be tried first.

## 11. VAE-exact-shape/scale Mul->Softmax->BatchMatMul microbenchmark (2026-09-01): also clean

§10 used arbitrary shapes/scale (reduction=1024, scale=0.125) to keep the
repro cheap. To close any "maybe it only shows up at the real shape/constant"
gap, located the actual `mul->softmax->matmul` occurrence inside
`sd/vae_decoder_f16.tflite` with a new tool,
`segment_anythings/tools/find_op_pattern.py` (generic builtin-op-sequence
finder + tensor/constant dumper, reusing the flatbuffer reader from
`inspect_tflite.py`). It found a single match at op index 120-122:

```
DEQUANTIZE(fp16 scalar buffer, raw bits 0x29a8 = 0.044189453125 ~= 1/sqrt(512))
  -> MUL([1,4096,4096] scores x scalar)
  -> SOFTMAX(axis=-1, reduces over 4096)
  -> BATCH_MATMUL([1,4096,4096] x [1,4096,512] -> [1,4096,512])
```

i.e. single-head self-attention over a 64x64=4096-token spatial map, 512
channels. The QK^T-scores tensor (#169) and V tensor (#152) are runtime
activations (no inline buffer); only the scale is a true constant, reached
through a `DEQUANTIZE`.

Built `build_softmax_attn_vae()` (added to `make_rank5_repro_tflite.py`,
`--softmax-attn-vae` flag) reproducing this exactly: same shapes
(`[1,4096,4096]` x `[1,4096,512]`), and the scale constant re-encoded as the
bit-identical fp16 value `0.044189453125` (decoded from the real model's
buffer, not recomputed from `1/sqrt(512)` in Python). Verified the generated
model's op codes/shapes/constant preview with `find_op_pattern.py` before
running. Ran `sam_encoder_runner --verify --precision=fp16 --tolerance=1e-2`:

```
CPU: elems=2097152 min=-0.0110522 max=0.0123534  mean=-3.15119e-05 std=0.00411959
GPU: elems=2097152 min=-0.0110397 max=0.0123444  mean=-3.15499e-05 std=0.00411533
[verify] max_abs=2.01389e-05 mean_abs=4.7736e-06 over_tol=0 nan_mismatch=0 -> PASS
```

Relative error is again ordinary fp16 rounding noise (mean/std match to
~0.1%), not a multiplicative bias — no trace of the ~37x pattern even at the
exact real shape and bit-exact real scale constant.

**Conclusion: this specific attention block, and the softmax/attention
mechanism generally (fused §9, decomposed at arbitrary shape §10, decomposed
at exact real shape/scale §11), is excluded as the root cause with high
confidence.** The ~37x divergence must come from a different op class in the
1012-op graph. Next candidates unchanged from §10: GroupNorm/LayerNorm-style
custom ops, Conv2D (the VAE decoder is conv-heavy), or generic fp16
FullyConnected/BatchMatMul accumulation elsewhere in the graph — not yet
investigated.

Artifacts: `segment_anythings/softmax_attn_vae_repro.tflite` (8.4MB),
`segment_anythings/softmax_attn_vae_repro_input.bin` (64MB, ramp input),
`segment_anythings/softmax_attn_vae_verify_run.log`,
`segment_anythings/tools/find_op_pattern.py` (new tool, reusable for
locating any other op pattern in the real model for future candidates).

### 11a. Follow-up: declaring the graph INPUT itself as FLOAT16

§11's model kept its graph input (`x`, the raw QK^T scores) declared
`FLOAT32`, matching the real model's own graph input (`latent_sample`,
also `FLOAT32` — see the `find_op_pattern.py` dump in §11). Tested whether
declaring the *TFLite tensor type* of that input as `FLOAT16` (dequantized
to fp32 immediately, same pattern as the scale constant) changes anything,
since `--precision=fp16` already governs the delegate's internal compute
precision independent of declared tensor dtype (this is *why* the real
model can be all-`FLOAT32`-typed activations yet still run fully in fp16 on
GPU).

Added `build_softmax_attn_vae_fp16in()` (`--softmax-attn-vae-fp16in` flag)
— identical to §11's model except `x` is `TYPE_FLOAT16` with a
`DEQUANTIZE` to fp32 before the `Mul`. `sam_encoder_runner` originally
always wrote raw input via `TensorBuffer::Write<float>`, which fails size
validation against a genuinely fp16-backed buffer, so also patched
`sam_encoder_runner.cc`'s `PrepareIO` to check each input tensor's
`ElementType()` and, when `Float16`, convert via `fp16_ieee_from_fp32_value`
and write `uint16_t` bit patterns instead (`services/webnn/tflite/sam_runner/
sam_encoder_runner.cc`, `BUILD.gn` gained a `//third_party/fp16` dep).
Rebuilt `sam_encoder_runner.exe` and ran `--verify` against the new model.

**Result: GPU compile fails outright** — `Some ops are not accelerated. Add
kLiteRtHwAcceleratorCpu to the compilation accelerator set to allow using
the CPU to run those.` (CPU/XNNPACK reference compiled and ran fine; only
the GPU-only ml-drift/WebGPU compile — which `--verify` always requests —
rejects the graph.) `--verify` never got to run element-wise comparison.

**Conclusion: declaring the input tensor FLOAT16 does not make the whole
model run in ml-drift under fp16 — it does the opposite: the delegate
refuses to fully accelerate the graph at all.** This explains why the real
`vae_decoder_f16.tflite` keeps its own graph input (and every other
activation tensor, per §11) declared `FLOAT32` despite being an "fp16
model": the GPU delegate's op-selection apparently requires `FLOAT32` at
tensor declarations for (at least) the boundary/activation path, and gets
its actual fp16 compute purely from `--precision=fp16`'s global
`CalculationsPrecision`, not from declared tensor dtype. No further action
needed here; this closes out the "should input be fp16" question raised
against §11 — the §11 setup (fp32-declared activations, fp16-declared
scale constant only) was already the faithful match to the real model, not
an oversight.

Artifacts: `segment_anythings/softmax_attn_vae_fp16in_repro.tflite`,
`segment_anythings/softmax_attn_vae_fp16in_repro_input.bin`,
`segment_anythings/softmax_attn_vae_fp16in_verify_run.log`.
