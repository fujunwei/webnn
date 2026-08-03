# WebGPU delegate — Segment Anything bias descriptor fix

## Symptom

Running Segment Anything (MobileSAM) through the LiteRT WebGPU delegate on an
 iGPU crashed at graph compilation with:

```
INVALID_ARGUMENT: Read selector with single argument can be used only with
linear storage types(BUFFER or IMAGE_BUFFER); selector object=biases,
selector=Read, args=[DST_S + 0], template_args=[], descriptor=float16,
TensorStorageType::TEXTURE_2D, layout: hwc, shape: {bhwc, {1, 1, 1, 128}}
```

Stack (top frames):

```
ml_drift::TensorDescriptor::PerformReadSelector
ml_drift::ResolveSelectorsPass
ml_drift::GPUOperation::AssembleCode
ml_drift::GpuModelBuilder::GetGpuModel
ml_drift::GraphToGpuModel
litert::ml_drift::DelegateKernel::InitInferenceContextFromGraph
```

After the first fix attempt, a follow-up crash appeared, this time in the
elementwise Copy op the fix inserted:

```
std::vector::operator[]  → ForceCrashOnSigAbort
ml_drift::TensorDescriptor::Write                          (tensor_desc.cc:1311)
ml_drift::TensorDescriptor::PerformWriteSelector           (tensor_desc.cc:968)
ml_drift::PerformSelector                                  (tensor_desc.cc:684)
ml_drift::ResolveSelectorsPass                             (tensor_desc.cc:942)
```

## Environment

- Chromium `third_party/litert` delegate wrapping ml-drift.
- ml-drift external repo at
  `C:\Users\junweifu\workspace\chromium\src\third_party\ml-drift` (main,
  commit 0e2092a), bound via `local_repository` in `src/WORKSPACE:370`.
- Backend: WebGPU on  iGPU (`GetFastestStorageType` returns
  `TensorStorageType::TEXTURE_2D`, see
  `ml_drift/webgpu/environment.cc:271-276`).

## Root cause

1. Every tensor produced by `GetTensorDescForValue`
   (`ml_drift/common/gpu_model_util.cc:187-240`) is initialized as
   `Layout::HWC` (or `BHWC` for `b>1`) with `create_info.storage_type`
   (TEXTURE_2D on  WebGPU). This is fine for spatial tensors but wrong
   for a bias tensor of shape `{1,1,1,C}`.
2. LiteRT's `convert_fully_connected.cc` weights-runtime branch
   (`src/ml_drift_delegate/tflite/convert/convert_fully_connected.cc:275-281`)
   unconditionally forwards the bias as a graph consumer, even when the
   TFLite bias tensor is constant. So the bias reaches ml-drift as a
   graph tensor with the default HWC + TEXTURE_2D descriptor.
3. The `FULLY_CONNECTED` handler in ml-drift
   (`ml_drift/common/selectors/operation_selector.cc:840-870` and its
   duplicate at `1425-1460`) fetches the bias with
   `GetTensor(inputs[2]->id)` and passes the `TensorHandle` straight into
   `Convolution(src, weights, bias, conv_attr)`.
4. `Convolution` forwards the descriptor into
   `SelectConvolutionWithExternalWeights` →
   `CreateConvGenericExternalWeights`, whose shader emits
   `args.biases.Read(DST_S + sind)` (`conv_generic.cc:902`).
5. `PerformReadSelector` (`tensor_desc.cc:699-745`) with
   `args.size() == 1` falls into the branch at `731-738`, which rejects
   the descriptor because layout is HWC and storage is TEXTURE_2D.

The constant-bias / constant-weights path does not hit this: it flows
through the LINEAR `AddConstantTensor(const Tensor<Linear, FLOAT32>&, …)`
overload in `gpu_model_builder.cc:247-252`, which builds the descriptor
via `CreateConstantLinearTensorDescriptor` → LINEAR + BUFFER.

### Why the first fix attempt crashed on Write

The first version of `MakeBiasLinear` allocated the destination via
`AddLinearTensor(c, data_type)`. That helper starts with the default
storage (TEXTURE_2D on WebGPU) and relies on
`UpdateToSupportedStorageType` (`tensor_desc.cc:2083`) to fall back if the
combo is invalid. On this configuration `CanCreateTensorWithShape` accepts
`LINEAR + TEXTURE_2D`, so no fallback happens. The elementwise Copy op then
called `PerformWriteSelector` on a LINEAR descriptor; the generic
`GetPhysicalCoords` returns `{""}` for LINEAR layout, and the WebGPU
TEXTURE_2D write branch (`tensor_desc.cc:1311`) accesses `coords[1]` →
`std::vector::operator[]` out-of-bounds.

Elementwise ops are not designed to write into `Layout::LINEAR`
descriptors: LINEAR writes require the dedicated
`PerformWriteLinearSelector`.

## Fix

Add a `GpuModelBuilder::MakeBiasLinear(TensorHandle)` helper that:

- Returns `src` unchanged when the descriptor already satisfies the
  single-arg Read selector (LINEAR or HW layout, or BUFFER / IMAGE_BUFFER
  storage).
- Otherwise builds a new `TensorDescriptor` with the **same layout and
  shape** as `src` but `TensorStorageType::BUFFER`, adds a graph tensor
  for it, and inserts an elementwise `Copy(src, dst)`.

Keeping the layout constant across the Copy means both endpoints agree on
coord generation (HWC read on TEXTURE_2D, HWC write on BUFFER, both use
`GetPhysicalCoordsWHS`). The resulting bias descriptor satisfies the
`PerformReadSelector` requirement at
`tensor_desc.cc:731-738` because storage is now BUFFER.

Call sites: both `FULLY_CONNECTED` handlers in
`ml_drift/common/selectors/operation_selector.cc` invoke `MakeBiasLinear`
immediately after `GetTensor(inputs[2]->id)`.

### Why we did not fix `CONVOLUTION_2D`

The `CONVOLUTION_2D` handler in `operation_selector.cc` never fetches
runtime bias via `inputs[2]` — its runtime-weights branch only reads
`attr.bias.data` (already the LINEAR path). If a runtime bias arrives from
LiteRT's convert layer it is silently dropped; that is a separate,
pre-existing bug not required for Segment Anything.

### Alternative fixes considered

- **Fix at `GetTensorDescForValue`**: change `{1,1,1,C}` tensors to
  LINEAR + BUFFER globally. Too invasive; other ops (e.g. broadcasts,
  reshapes) rely on the default HWC descriptor.
- **Fix at the CONSTANT node in `ConvertOperations`**: only helps constant
  biases. Runtime biases still break.
- **Fix in LiteRT `convert_fully_connected.cc`** (leave `attr.bias`
  populated on the weights-runtime path when the TFLite bias is constant):
  Also viable and even lower-risk for the constant-bias case, but does not
  cover pure runtime-bias FCs, and it fixes only the LiteRT delegate — an
  ml-drift consumer that produces the same graph shape hits the same
  crash.
- **`use_buffer_storage_type = true`** in delegate options: works around
  the issue by forcing BUFFER everywhere. Ships a perf regression on
   iGPU and is not a real fix.

The chosen ml-drift-side helper covers both constant and runtime bias and
is contained to one function plus two call sites.

## Files changed

All paths inside `C:\Users\junweifu\workspace\chromium\src\third_party\ml-drift\`:

- `ml_drift/common/gpu_model_builder.h` — declaration of
  `TensorHandle MakeBiasLinear(const TensorHandle&)`.
- `ml_drift/common/gpu_model_builder.cc` — implementation.
- `ml_drift/common/selectors/operation_selector.cc` — `MakeBiasLinear`
  call in both `FULLY_CONNECTED` handlers (v1 uses
  `node.operation.attributes`, v2 uses `node.attr`).

See `ml-drift-webgpu-bias-fix.patch`.

## Verification plan

1. Rebuild `libLiteRtWebGpuAccelerator.dll`.
2. Load Segment Anything through the WebGPU delegate on  iGPU.
3. Confirm `InitInferenceContextFromGraph` no longer aborts.
4. Optional: verify output correctness against a CPU reference. Expect
   bit-identical results since `Copy` is a pure element-wise identity in
   FP16; the only added overhead is one buffer-sized copy per FC layer
   with a runtime bias.

---

# OOM at `SimpleMemoryArena::Commit` on SAM encoder

## Symptom

After the bias-descriptor fix, loading
`segment_anything_encoder.tflite` through the ml-drift WebGPU delegate
still aborts, this time inside PartitionAlloc:

```
KernelBase!RaiseException
partition_alloc::OnNoMemoryInternal
partition_alloc::TerminateBecauseOutOfMemory
partition_alloc::PartitionExcessiveAllocationSize      ← single-alloc > ~2 GiB
partition_alloc::PartitionBucket::SlowPathAlloc
partition_alloc::PartitionRoot::AlignedAlloc<16>
tflite::SimpleMemoryArena::Commit
tflite::ArenaPlanner::ExecuteAllocations
tflite::Subgraph::AllocateTensors
tflite::Subgraph::ModifyGraphWithDelegateImpl
LiteRtCompiledModelT::Create
webnn::litert::GraphImplLiteRt::CreateAndBuild
```

`IsFullyDelegated` in the stack is a neighboring symbol from
`Subgraph::AllocateTensors`, not the actual caller of `Commit`.

The same model runs fine on the LiteRT CPU path.

## Background: how TFLite allocates tensor memory

TFLite classifies every tensor by `allocation_type`:

| type | who owns it |
|---|---|
| `kTfLiteMmapRo` | mmap'd flatbuffer weights |
| `kTfLiteArenaRw` | packed into `ArenaPlanner::arena_` (reusable) |
| `kTfLiteArenaRwPersistent` | packed into `persistent_arena_` |
| `kTfLiteDynamic` | plain `malloc/free` on every kernel invocation |
| delegate-owned | opaque to TFLite (e.g. ml-drift GPU buffers) |

Per subgraph there are exactly **two** `SimpleMemoryArena` instances
(`arena_`, `persistent_arena_` — see
[arena_planner.h](../../chromium/src/third_party/tflite/src/tensorflow/lite/arena_planner.h#L140-L146)),
not one per tensor.

`ArenaPlanner::PlanAllocations` walks the execution plan, records for
every arena tensor an interval `[first_node, last_node]`, then packs
them into a single contiguous buffer using a first-fit-by-offset greedy
so tensors with disjoint intervals share bytes. `Commit()` performs
**one** `AlignedAlloc<16>(high_water_mark)`. That single allocation is
what hits PartitionAlloc's ~2 GiB cap.

### Effect of a delegate on the arena

`ModifyGraphWithDelegate` fuses supported ops into a single kernel node.
Two consequences on the CPU-side arena:

1. Tensors entirely internal to the delegate disappear from
   `ArenaPlanner`'s view (they become GPU buffers owned by ml-drift).
2. Boundary tensors (delegate inputs/outputs, graph inputs/outputs) keep
   `kTfLiteArenaRw`. Their intervals now collapse onto the single fused
   node, so many of them are alive **simultaneously** and can no longer
   share bytes with each other. The high-water mark of `arena_` goes
   **up**, not down, for graphs with large activations.

### Effect of `OptimizeMemoryForLargeTensors(threshold)`

Upstream LiteRT's `InitializeRuntime` calls
`interpreter_options.OptimizeMemoryForLargeTensors(1 << 20)` — every
tensor larger than 1 MiB is moved out of the arena and re-tagged
`kTfLiteDynamic` (allocated on demand per invocation). That keeps the
single Commit allocation well below 2 GiB on models like SAM / SD-turbo.

## Root cause

[route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch](../route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch)
comments out that call:

```cpp
// interpreter_options.OptimizeMemoryForLargeTensors(1 << 20);
```

Why the patch was written: `kTfLiteDynamic` tensors set
`Subgraph::has_dynamic_tensors_ = true`, which
`Subgraph::ModifyGraphWithDelegateImpl` rejects for delegates that only
support static shapes (ml-drift is one) with:

```
Attempting to use a delegate that only supports static-sized tensors
with a graph that has dynamic-sized tensors (tensor#%d is a
dynamic-sized tensor).
→ kTfLiteApplicationError
```

So disabling `OptimizeMemoryForLargeTensors` is required for the
delegate to apply at all. The cost is that every large activation stays
in `arena_`, and on SAM's encoder the packed high-water-mark exceeds
2 GiB, tripping `PartitionExcessiveAllocationSize`.

CPU path does not hit this because it never applies a static-shape
delegate: LiteRT ships `OptimizeMemoryForLargeTensors(1 << 20)` intact
for CPU users.

## Worked example — `Add → Mul → Sub`

Tensor sizes all `S` bytes; nodes indexed `0/1/2`.

Graph:

```
in0 ─┐
     Add ── A ─┐
in1 ─┘        Mul ── M ─┐
in2 ──────────┘        Sub ── out
in3 ────────────────────┘
```

Lifespans (graph inputs/outputs are pinned to `[-1, ∞]`):

| tensor | first | last |
|---|---|---|
| in0,in1 | -1 | 0 |
| in2 | -1 | 1 |
| in3 | -1 | 2 |
| A | 0 | 1 |
| M | 1 | 2 |
| out | 2 | ∞ |

### CPU (no delegate)

Arena packing can reuse `in0` with `A`, `in1` with `M`, etc.
`high_water_mark ≈ 5·S`, one `AlignedAlloc<16>(5·S)`.

### `Mul` unsupported → CPU-Add | Delegate{Mul} | CPU-Sub

Execution plan still 3 nodes; every tensor still crosses the CPU/GPU
boundary and stays in `arena_`. Layout ≈ same 5·S.

### `Add`+`Mul` delegated, `Sub` on CPU

Execution plan collapses to 2 nodes: `Delegate{Add,Mul}` (idx 0),
`Sub` (idx 1).

- `A` is now internal to the delegate → out of the CPU arena (owned by
  ml-drift as a GPU buffer).
- Boundary set on the arena: `in0, in1, in2, in3, M, out`.
- Intervals: `in0/in1/in2 [-1,0]`, `in3 [-1,1]`, `M [0,1]`,
  `out [1,∞]`. The three delegate inputs are all alive at node 0
  together with `M`'s allocation — reuse across them is impossible.
- `high_water_mark` typically ≥ CPU case; one fewer packed tensor does
  not offset the loss of interval overlap.

### Fully delegated

Single node `Delegate{Add,Mul,Sub}`. `A` and `M` leave the arena; only
`in0..in3` and `out` remain, but all boundary inputs are alive during
that one node ⇒ still simultaneous. On SAM encoder this is the shape
of the crash: a small number of very large boundary/activation tensors
share the arena high-water at the same instant.

`IsFullyDelegated` is nothing more than a query used after
`AllocateTensors`; the arena Commit still runs whenever any
`kTfLiteArenaRw` tensor exists (graph IO always does).

## Fix options (ordered by effort)

1. **Raise the threshold instead of removing the call.**
   Restore `OptimizeMemoryForLargeTensors(N)` in
   [patches/06](../route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch)
   with `N` chosen so that ml-drift's required-static tensors remain in
   the arena but SAM's few multi-hundred-MB activations get dynamic
   allocation. Verify with `--enable-logging=stderr
   --vmodule=*subgraph*=1`: no `"only supports static-sized tensors"`
   warning **and** the encoder loads.
2. **Teach the delegate/subgraph check to distinguish
   `allocation_type == kTfLiteDynamic` (shape known) from truly
   dynamic-shape tensors.** Modify
   `Subgraph::has_dynamic_tensors_` accounting or ml-drift's
   `IsNodeSupported` so dynamically-allocated but statically-shaped
   tensors do not disqualify delegation. Medium risk, permanent fix.
3. **Partition SAM encoder into sub-graphs on the WebNN side** so no
   single arena crosses 2 GiB. Highest effort.

Option 1 is the recommended first step: it is a one-line change in the
existing patch stack and directly addresses the failing allocation
without changing delegate semantics.

## Verification hooks

- Log `ArenaPlanner::arena_.underlying_buffer_.data_size_` (or
  `high_water_mark_`) right before/after `Commit()`; failing case is
  ≥ `2 * 1024 * 1024 * 1024`.
- Toggle `OptimizeMemoryForLargeTensors(64 << 20)` and observe the
  arena size drop and the crash disappear.
- If dynamic tensors reappear in the delegated subgraph, TFLite logs
  `"only supports static-sized tensors (tensor#N …)"` — that identifies
  which large tensor still needs to be kept in arena (or in a supported
  op).
