# Emit WebNN instanceNormalization as an `odml.group_norm` StableHLO composite

## Context

We just replaced `custom_call.LayerNorm` with a `STABLEHLO_COMPOSITE` op named
`odml.group_norm` (`sub_type = 1`), which the ML Drift WebGPU delegate lowers to a fused
kernel. `SerializeInstanceNormalization` (`graph_builder_tflite.cc:7594`) still emits
~12 primitive ops (MEAN, SUB, SQUARE, MEAN, 2×RESHAPE, SUB, ADD, SQRT, DIV, MUL, ADD)
into the main subgraph, so it gets no fused-kernel acceleration on GPU.

Goal: give instanceNormalization the same fused fast path, falling back to the existing
primitive emulation when the delegate's preconditions aren't met.

## Why the mapping is sound (verified in ML Drift / LiteRT sources)

Instance normalization is **group normalization with `num_groups == channels`**, and ML
Drift treats that as a first-class, separately-optimized case:

- `ml-drift/ml_drift/common/gpu_model_builder.cc:4088-4090` — `HWCGroupNorm()` computes
  `group_size = c / groups`; the `|| group_size == 1` clause exists solely to route this
  case to the fused kernel instead of the generic Reshape/Reduce fallback.
- `ml-drift/.../kernels/mean_stddev_normalization.cc:939-941, 952-954` —
  `group_size == 1` gets a dedicated code generator, `GetHWCNormalizationCodeGroupSize1`.
- That kernel (`mean_stddev_normalization.cc:382-452`) loops only over height/width,
  accumulating `float4` per-channel sums with `inv_elements_count = 1/(H*W*1)`, and
  applies `gamma`/`beta` per channel slice. Lanes never mix — 4 channels normalized
  independently, per batch. Exactly instance-norm semantics.
- Test coverage exists and is non-degenerate: `mean_stddev_normalization_test.cc:65`
  runs `group_size == 1` on `BHWC(1,6,32,24)` (so `groups == C == 24`) against a CPU
  reference, batched and unbatched.

The delegate path WebNN actually uses (legacy `model_builder.cc`, since `use_ir_model`
defaults false) supports it:

- `model_builder.cc:7186-7203` — dispatch on `odml.group_norm`; `sub_type == 0` selects
  `CompositeGroupNormParser`.
- `model_builder.cc:6429-6519` — that parser requires input rank ≤ 4; gamma/beta 1-D
  with length == innermost dim; `channel_axis` (if present) == rank-1; `num_groups` and
  `epsilon` present; and when `sub_type` is set,
  `_TENSOR_V1_reduction_axes == [1, 2, …, rank-1]`.

WebNN's limits already line up: `graph_builder_tflite.cc:1029-1032` declares
`instance_normalization_input` as `Exactly(4)` and `..._scale` as `Exactly(1)`,
float16-to-32. The mojom struct (`webnn_graph.mojom:378-395`) has no layout field —
NCHW→NHWC transposes are inserted in Blink
(`layout_transformer.cc:608-641`), and the serializer already `CHECK`s NHWC. So the
channel axis is always 3 and the spatial axes are `{1, 2}`.

## Reuse assessment (the explicit question)

**Yes — reuse it. The change is one extra parameter.**

`SerializeLayerNormalizationDecompositionSubgraph` hardcodes exactly one thing that
differs between the two operators:

| Aspect | LayerNorm (current) | InstanceNorm (needed) | Same? |
| --- | --- | --- | --- |
| Reduction axes | `{rank-1}`, hardcoded at `.cc:7720-7721` | `{1, 2}` | **differs** |
| scale/bias broadcast shape | all-1s, last = `dims.back()` (`.cc:7693-7695`) | `[1,1,1,C]` | same |
| scale/bias param shape | `{dims.back()}` | `{C}` | same |
| Subgraph signature | `[input, scale?, bias?] -> [output]` | identical | same |
| Body | `ComputeMeanAndVarianceForNormalization` + `SerializeNormalizationOperation` | identical | same |

Everything but the reduction axes coincides because the backend is NHWC, so the channel
axis *is* the innermost axis for both. `ComputeMeanAndVarianceForNormalization`
(`.cc:7482`) already takes an arbitrary axes span — only its caller is hardcoded.

Duplicating would copy ~70 lines to change one. Parameterizing costs ~6.

**Estimated size: ~+120 new lines, ~20 modified.** Low risk; the compiler catches the
single existing call site.

## Blocking issue found: always emit scale *and* bias

`MakeGroupNorm` (`operation_selector.cc:366-380`) and `MakeLayerNorm`
(`.cc:329-343`) contain a latent out-of-bounds write. When the attribute is absent they do:

```cpp
gamma.shape = Linear(input_shape.c);
for (int i = 0; i < input_shape.c; ++i) { gamma.data[i] = 1.0; }   // data is EMPTY
```

`Tensor::shape` and `Tensor::data` are independent members (`ml-drift/.../tensor.h:134,136`);
`data` is a `std::vector<float>` that is never resized. Writing `data[i]` on it is UB /
heap corruption. It is reachable: both composite parsers populate the attributes only
when the operand is present (`model_builder.cc:6530-6541` for GroupNorm,
`:6627-6638` for LayerNorm), and WebNN makes scale/bias optional for both operators.

**Mitigation (applies to both operators):** always pass three operands to the composite,
synthesizing a constant ones-vector for a missing scale and a zeros-vector for a missing
bias via the existing `SerializeTensorWithBuffer<float>` helper. Cost is `2 × C` floats
per op; it keeps ML Drift entirely out of its buggy default path.

This also **removes the bias-without-scale hazard** (ML Drift reads beta from
`inputs[2]`), so no special-case bail is needed for that.

Companion fix: apply the same always-emit-both treatment to the already-landed
`SerializeLayerNormalizationAsComposite`, which today can reach the UB path. Flagging
explicitly because it touches code outside the literal request.

## Implementation

### 1. `graph_builder_tflite.h` (~20 lines)

Generalize the decomposition builder and declare the new emitter:

```cpp
// Appends a decomposition subgraph normalizing over `reduction_axes` with
// primitive operators and returns its subgraph index, which composite
// operators reference as a fallback when no delegate claims them.
base::expected<int32_t, std::string>
SerializeNormalizationDecompositionSubgraph(
    base::span<const int32_t> input_dimensions,
    base::span<const int32_t> reduction_axes,
    ::tflite::TensorType tensor_type,
    float epsilon,
    const char* subgraph_name);

// Emits `odml.group_norm` with `sub_type = 0` and `num_groups = channels`,
// which ML Drift lowers to its fused group-size-1 normalization kernel.
std::optional<OperatorOffset> SerializeInstanceNormalizationAsComposite(
    const mojom::InstanceNormalization& instance_normalization);
```

The `has_scale`/`has_bias` params drop out, since both are now always present.

### 2. `graph_builder_tflite.cc` — generalize the decomposition builder (~10 lines)

In the body (`.cc:7659-7751`): replace the hardcoded `normalized_axis = {rank-1}` with
the `reduction_axes` parameter; always add the scale and bias subgraph inputs; use
`subgraph_name` for the `CreateSubGraph` name. Keep the `tensors_`/`operators_` swap
logic untouched.

### 3. `graph_builder_tflite.cc` — new `SerializeInstanceNormalizationAsComposite` (~90 lines)

Mirror `SerializeLayerNormalizationAsComposite` (`.cc:7753-7881`). Preconditions, each
returning `std::nullopt` to fall through to primitive emulation:

- `context_device_ != mojom::Device::kGpu`
- input rank != 4
- present scale/bias operands that are not **constant**, 1-D, float, length `C` — the
  legacy parser reads them via `ObjectReader::ReadTensor`, which needs constant data.
  (WebNN itself permits non-constant scale/bias — `graph_validation_utils.cc:1702-1758`
  imposes no constant requirement, and tests build them as graph inputs — so this check
  is load-bearing, not defensive.)
- after `SerializeInputTensorInfo`, `data_type != FLOAT32` (fp16 is cast to fp32 since
  `operation_supports_float16` defaults false, and `SerializeNormalizationOperation`
  has `CHECK_EQ(..., FLOAT32)` at `.cc:3836`)

Synthesize missing scale/bias as constants, then emit attributes:

```cpp
fbb.Map([&] {
  fbb.Int("sub_type", 0);              // 0 = GroupNorm
  fbb.Int("num_groups", channels);     // == C  =>  group_size 1  =>  instance norm
  fbb.Float("epsilon", instance_normalization.epsilon);
  fbb.Int("channel_axis", 3);
  fbb.Map("_TENSOR_V1_reduction_axes", [&] {
    fbb.Vector("TENSOR_DATA", [&] { fbb.Add(1); fbb.Add(2); fbb.Add(3); });
  });
});
```

Then emit `BuiltinOperator_STABLEHLO_COMPOSITE` with
`BuiltinOptions2_StableHLOCompositeOptions`, as the LayerNorm path does.

> **Subtlety to capture in a code comment.** `_TENSOR_V1_reduction_axes` must be
> `[1,2,3]` because that is the form `CompositeGroupNormParser` validates
> (`model_builder.cc:6492-6503`); the *actual* reduction is narrowed to the spatial axes
> by `num_groups == C`. The decomposition subgraph must therefore be built with
> `reduction_axes = {1, 2}`, **not** `{1,2,3}` — otherwise the CPU fallback computes
> different statistics than the GPU kernel and the two silently disagree.

### 4. `graph_builder_tflite.cc` — hook it in (~7 lines)

At the top of `SerializeInstanceNormalization` (`.cc:7594`), after the existing `CHECK`:

```cpp
if (auto fused =
        SerializeInstanceNormalizationAsComposite(instance_normalization);
    fused.has_value()) {
  return *fused;
}
```

The remainder of the function is unchanged and stays the fallback.

### 5. Harden the landed LayerNorm path (~15 lines)

In `SerializeLayerNormalizationAsComposite`, synthesize ones/zeros for missing
scale/bias instead of emitting a 1- or 2-operand composite, and drop the now-redundant
bias-without-scale bail.

## Verification

1. Build: `autoninja -C out/upstream_bots_debug services/webnn:webnn_service`.
2. `autoninja -C out/upstream_bots_debug services_unittests`, then run
   `--gtest_filter=*WebNNGraphImplTest.InstanceNormalization*` (validation) — these
   exercise the CPU path and must stay green.
3. **Numerical equivalence is the main risk.** Run the same instanceNormalization case on
   a `device: 'gpu'` context (fused) and a `device: 'cpu'` context (primitive) and compare
   outputs within float tolerance. A mismatch most likely means the decomposition's
   reduction axes are wrong — see the subtlety above.
   Note `webnn_graph_impl_backend_test.cc` has only one instanceNormalization case
   (`FuseStandaloneActivationIntoInstanceNormalization`, `:2153`) and it is currently
   **skipped** on this path (commented out of `kSupportedTests` at `:414`), so existing
   e2e coverage will not catch a regression — verify manually.
4. Confirm the node is actually delegated rather than silently CPU-executed: check that
   `IsFullyAccelerated` still holds for a graph containing instanceNormalization
   (skippable decomposition subgraphs are excluded by
   `litert/runtime/compiled_model.cc:1219`), or enable ML Drift delegate logging.
5. Dump a model and confirm the extra subgraph plus a `STABLEHLO_COMPOSITE` node with
   the attributes above.

## Out of scope

Generalizing the decomposition subgraph to the full WebNN operator surface (arbitrary
axes, non-constant scale/bias) and deleting the primitive fallback — we explicitly
decided to keep the narrow-composite + general-fallback split.

## Post-implementation notes (added after landing)

- `use_ir_model` was found locally flipped to `true` (uncommitted) in
  `third_party/litert/src/ml_drift_delegate/delegate/delegate_options.h`, meaning the
  build actually exercises the **IR** parser path (`support_group_norm.cc` /
  `convert_group_norm.cc`) rather than the legacy `model_builder.cc` path described
  above. Both paths impose the same constraints (rank ≤ 4 / exactly 4 for WebNN,
  1-D gamma/beta matching channel count, required `num_groups`/`epsilon`), so the
  implementation is unaffected, but this should be re-checked if `use_ir_model` reverts
  to `false` upstream.
- Build (`autoninja -C out/upstream_bots_debug services/webnn:webnn_service`) succeeded.
  `services_unittests` (457 WebNN tests) all passed with no regressions, but **no
  existing test actually exercises the new composite path**: the passing
  instanceNormalization/layerNormalization backend tests all fail one of the
  preconditions (rank 0/1/6, non-innermost/multi axes, or non-constant scale/bias
  built as graph inputs), so they only confirm the primitive fallback still works.
  `FuseStandaloneActivationIntoInstanceNormalization`, the one 4-D case, remains
  disabled in `kSupportedTests` — and separately its expected values assume NCHW
  while the TFLite backend is NHWC, so it would fail regardless of this change.
- **Outstanding verification**: a real GPU-vs-CPU numerical comparison for
  instanceNormalization has not been run. This is the one place a silent divergence
  could hide (composite advertises reduction axes `[1,2,3]`; decomposition subgraph
  reduces over `{1,2}` only), so it should be done before relying on this path in
  production.
