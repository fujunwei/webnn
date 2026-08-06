# Fused `custom_call.LayerNorm` — WebNN → ml-drift WebGPU

Implementation notes for routing WebNN `MLGraphBuilder.layerNormalization` to
the ml-drift WebGPU delegate's fused `custom_call.LayerNorm` kernel instead
of the primitive `sub`/`rsqrt`/`mul`/`add` emulation.

## Motivation

`LiteRtCompiledModelT::InitializeRuntime`'s
`interpreter_options.OptimizeMemoryForLargeTensors(1 << 20)` was disabled by
[route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch](../route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch)
because it interacts with ml-drift's static-shape delegation. As a result
every intermediate tensor stays in the single `SimpleMemoryArena`, and on SAM
encoder the packed high-water-mark trips PartitionAlloc's ~2 GiB single-alloc
cap in `SimpleMemoryArena::Commit`. See [analysis.md](analysis.md).

Picking a per-model `OptimizeMemoryForLargeTensors(N)` threshold is fragile.
A more principled fix is to stop producing large CPU-arena intermediates in
the first place. Each WebNN `layerNormalization` currently expands into ~5
primitive ops with 2–3 full-shape temporaries; fusing it into a single
`custom_call.LayerNorm` node lets ml-drift keep the whole computation on the
GPU, and the temporaries never enter the CPU arena.

## Prerequisites in LiteRT / ml-drift

- LiteRT registers `custom_call.LayerNorm` as a stub kernel **only when the
  GPU accelerator is selected** (see
  [`third_party/litert/src/litert/runtime/compiled_model.cc`](../../chromium/src/third_party/litert/src/litert/runtime/compiled_model.cc#L371-L381)):
  ```cpp
  if (hardware_accelerators & kLiteRtHwAcceleratorGpu) {
    const char* accelerator_supported_custom_ops[] = {
        ..., "custom_call.LayerNorm", ...};
    resolver->AddCustom(op_name, &sStubRegistration);
  }
  ```
  The CPU / NPU branches do **not** register the stub. Emitting the fused op
  when the WebNN context is not on GPU makes `AllocateTensors` fail with
  "Encountered unresolved custom op".

- ml-drift's delegate parses the op through
  [`LayerNormParser`](../../chromium/src/third_party/litert/src/ml_drift_delegate/tflite/model_builder.cc#L5844-L5900)
  and validates through
  [`IsLayerNormSupported`](../../chromium/src/third_party/litert/src/ml_drift_delegate/tflite/support/support_layer_norm.cc).
  The contract we must satisfy:

  | Constraint | Source |
  |---|---|
  | 1–3 inputs (`x`, optional `scale`, optional `bias`) | `support_layer_norm.cc` L44-L56 |
  | 1 output | L52-L56 |
  | dtype ∈ {bf16, f16, f32} | L84-L94 |
  | input rank 2..4 | L123-L128 |
  | scale/bias rank 1, size == innermost dim | L131-L152 |
  | scale/bias constant (mmap'd) | `CheckPopulateTensor<Linear, FLOAT32>` L99-L114 |
  | attributes flexbuffer: `{"epsilon": float}` | L172-L176, `convert_layer_norm.cc` L60 |
  | normalizes over the last axis (channel) | implicit from the `LayerNormAttributes` layout |

## Changes in Chromium

Four files:

1. [services/webnn/tflite/graph_builder_tflite.h](../../chromium/src/services/webnn/tflite/graph_builder_tflite.h)
2. [services/webnn/tflite/graph_builder_tflite.cc](../../chromium/src/services/webnn/tflite/graph_builder_tflite.cc)
3. [services/webnn/tflite/graph_impl_litert.h](../../chromium/src/services/webnn/tflite/graph_impl_litert.h)
4. [services/webnn/tflite/graph_impl_litert.cc](../../chromium/src/services/webnn/tflite/graph_impl_litert.cc)

### 1. Thread `mojom::Device` through `GraphBuilderTflite`

`GraphBuilderTflite` did not know which accelerator the compiled model was
targeting. Since `custom_call.LayerNorm` is GPU-only, the builder must know.

- `CreateAndBuild(...)` and the private constructor gain a
  `mojom::Device context_device` parameter.
- A new `const mojom::Device context_device_` member stores it.
- `graph_impl_litert.cc` passes `context.options().device` at both call
  sites:
  - `CreateAndBuildOnBackgroundThread` (incognito path)
  - `BuildGraphOnBackgroundThread` (weights-session path — also updated in
    the header to declare the new parameter).

### 2. Custom-op registration helper

Existing `GetOperatorCodeIndex(BuiltinOperator)` handles builtin opcodes.
The fused op requires `BuiltinOperator_CUSTOM` plus a non-null `custom_code`
string. New helper:

```cpp
GraphBuilderTflite::OperatorCodeIndex
GraphBuilderTflite::GetCustomOperatorCodeIndex(std::string_view custom_code,
                                               int32_t version) {
  auto operator_code_index =
      base::checked_cast<OperatorCodeIndex>(operator_codes_.size());
  operator_codes_.push_back(::tflite::CreateOperatorCode(
      builder_,
      base::checked_cast<int8_t>(::tflite::BuiltinOperator_CUSTOM),
      builder_.CreateString(std::string(custom_code)), version,
      ::tflite::BuiltinOperator_CUSTOM));
  return operator_code_index;
}
```

The op-code table dedup story is intentionally the same as the builtin one
(register on every call). Deduplication can be added later if profiling shows
it matters.

### 3. `SerializeLayerNormalizationAsCustomCall`

Located after `SerializeLayerNormalization`. Structure:

1. Guard on `context_device_ == mojom::Device::kGpu`; otherwise
   `return base::unexpected(...)`.
2. Validate the input operand's rank (2..4), dtype (fp32/fp16), and that the
   WebNN `axes` describe a single innermost-axis reduction. Fall back on any
   mismatch — the primitive path in `SerializeLayerNormalization` can still
   handle arbitrary axes and higher ranks.
3. Validate scale/bias operands (rank-1, length == channels, float,
   constant). Constant-ness is checked against `constant_operands_->find(...)`.
4. Serialize input / scale / bias tensor infos, emit the output tensor info.
5. Build the flexbuffer attributes `{ "epsilon": float }`.
6. Call `::tflite::CreateOperator` with `BuiltinOptions_NONE`,
   `custom_options` set, and `CustomOptionsFormat_FLEXBUFFERS`.

Falls back cleanly by returning `base::unexpected`; the caller
(`SerializeLayerNormalization`) checks `has_value()` and, if false, runs the
existing emulation.

Special case: `bias` without `scale`. ml-drift indexes bias at `inputs[2]`,
so an inputs vector of `[x, bias]` would be misinterpreted. Rather than
fabricating a synthetic identity `scale` constant here, this path bails to
emulation. WebNN callers rarely produce this shape today.

### 4. Dispatch in `SerializeLayerNormalization`

```cpp
if (auto fused = SerializeLayerNormalizationAsCustomCall(layer_normalization);
    fused.has_value()) {
  return fused;
}
```

Placed right after the `CHECK(...Supports(...))` line. When the fast path is
taken, none of the existing emulation code runs, so no temporary
`mean` / `variance` / rsqrt / mul-broadcast tensors are appended to the
subgraph.

## Attributes wire format

`CreateOperator(..., custom_options, CustomOptionsFormat_FLEXBUFFERS)` stores
the flexbuffer bytes directly. On the delegate side
`convert_layer_norm.cc` reads them via:

```cpp
const flexbuffers::Map flexbuffer_map =
    flexbuffers::GetRoot(params->attributes, params->attributes_size).AsMap();
attr.epsilon = flexbuffer_map["epsilon"].AsFloat();
```

Only `epsilon` is used; no other fields are required. WebNN's `axes` field
does not travel over the wire — ml-drift's kernel is hardcoded to the
innermost axis, which is why the pre-check in step 2 rejects other axis
selections.

## Fallback matrix

| Situation | Path taken |
|---|---|
| Device != GPU (CPU/NPU) | Emulation |
| Rank 5+ | Emulation |
| `axes` not `[rank-1]` | Emulation |
| Non-float dtype | Emulation |
| Non-constant scale/bias | Emulation |
| Scale/bias size mismatch | Emulation |
| Bias present, scale absent | Emulation |
| All checks pass | Fused `custom_call.LayerNorm` |

## Verification plan

1. Rebuild Chrome. Load Segment Anything encoder via WebNN with the ml-drift
   WebGPU delegate.
2. Enable subgraph logging: `--enable-logging=stderr --vmodule=*subgraph*=1`.
   Confirm no `"only supports static-sized tensors"` warning appears.
3. Inspect `ArenaPlanner::arena_.underlying_buffer_.data_size_` (or
   `high_water_mark_`) at `Commit()`. Expect it to drop below 2 GiB on
   models that previously crashed.
4. Compare an output tensor against a CPU LiteRT reference. LayerNorm is
   deterministic to within fp16 rounding, so element-wise close-enough
   (`|a-b| / (|b|+eps) < 1e-2` on fp16) is sufficient.
5. Non-GPU regression check: run any LayerNorm-containing model on the CPU
   accelerator and confirm the graph still compiles (fallback path).

## Follow-ups (not implemented)

- `custom_call.RmsNorm` — same shape of change, WebNN counterpart is under
  discussion; the delegate already accepts it
  ([compiled_model.cc:379](../../chromium/src/third_party/litert/src/litert/runtime/compiled_model.cc#L379)).
- `custom_call.GroupNorm` — same pattern, benefits Stable Diffusion UNet.
- `odml.scaled_dot_product_attention` composite — larger savings on SAM /
  SD attention blocks, but requires emitting a `StablehloComposite` op
  rather than a `Custom` op. Interface in
  [`model_builder.cc:6832+`](../../chromium/src/third_party/litert/src/ml_drift_delegate/tflite/model_builder.cc#L6832).
- Op-code deduplication in `GetOperatorCodeIndex` / `GetCustomOperatorCodeIndex`
  once the flatbuffer grows large enough to notice.
