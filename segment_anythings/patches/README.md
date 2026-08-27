# Patches for `third_party/litert/src` and `third_party/ml-drift`

Small, individually applicable patches carrying the local changes to the two
LiteRT third-party checkouts, in commit order.

Regenerate them from the working trees with:

```powershell
py ..\tools\gen_patches.py          # --check to preview without writing
```

`gen_patches.py` splits per *logical change*, not per file: two files
(`operation_selector.cc`, `delegate_kernel.cc`) carry hunks belonging to
different fixes, and the generator selects hunks by index. Edit the `PATCHES`
table there when adding a patch.

## Which repo does each patch apply to?

| Patches | Repo |
|---|---|
| `01`–`13`, `28`–`30` (prefix `litert`) | `third_party\litert\src` |
| `22`–`27` (prefix `mldrift`) | `third_party\ml-drift` |

Both are their own git checkouts. Apply and commit in numeric order **within
each repo**.

## Status

`10-litert-delegate-diagnostic-logging.patch` is **already committed** as
`c666480ac` ("litert: add GPU delegate diagnostic logging"); that commit also
absorbed the `model_builder.cc` and `delegate_webgpu.cc` portions. Do not
re-apply it.

Uncommitted in `litert/src` at the time of writing: `WORKSPACE`,
`ml_drift_delegate/delegate/BUILD`, `ml_drift_delegate/delegate/delegate_kernel.cc`,
`ml_drift_delegate/tflite/model_builder.cc`,
`tflite/tools/versioning/gpu_compatibility.cc`,
`third_party/dawn/workspace.bzl`.

Uncommitted in `ml-drift`: everything covered by `24`–`27`.

## How to apply / commit one patch at a time

Patches are standard unified diffs with a subject/body preamble above the
`diff --git`. `git apply` ignores the preamble:

```powershell
cd C:\Users\fujun\workspace\chromium\src\third_party\ml-drift
git apply ..\..\..\..\webnn\segment_anythings\patches\24-mldrift-5d-reshape-transpose.patch
git add -A
git commit          # paste the subject/body from the top of the patch file
```

To confirm a patch still matches the working tree without touching anything:

```powershell
git apply --check --reverse <patch>
```

## Patch list

### litert/src

| # | File(s) | Change |
|---|---|---|
| 01 | `WORKSPACE` | point `@ml_drift` at the local chromium ml-drift checkout |
| 02 | `third_party/dawn/workspace.bzl` | load Dawn from the prebuilt tree |
| 03 | `third_party/odml/litert/weight_loader/BUILD` (new) | stub alias for internal `//third_party/odml/...` label |
| 04 | `ml_drift_delegate/delegate/BUILD` | use `//weight_loader:external_weight_loader` |
| 05 | `ml_drift_delegate/delegate/shared_memory_manager` BUILD | weight-loader dep |
| 06 | logging | always-log toggle |
| 07 | `litert/runtime/subgraph` | fix `HasDynamicTensor` |
| 08 | `litert/runtime/subgraph` | optimize memory bytes from dims |
| 09 | `ml_drift_delegate/tflite/model_builder.cc`, `tflite/tools/versioning/gpu_compatibility.cc` | 0-runtime-input + 5D RESHAPE/TRANSPOSE, constant/>=5D inputs, rank-5 tensors |
| 10 | *(committed as `c666480ac`)* | diagnostic logging |
| 11 | `litert/runtime/compiled_model.cc` | disable + document `OptimizeMemoryForLargeTensors` toggle |
| 13 | `ml_drift_delegate/delegate/delegate_webgpu.cc` | farmhash include + disable pipeline-cache callback |
| 28 | `ml_drift_delegate/delegate/BUILD`, `delegate_kernel.cc` | recover real rank-5 shapes into `CreateGpuModelInfo::rank5_shapes` |
| 29 | `ml_drift_delegate/tflite/model_builder.cc` | **actually permute constant data for rank-2/3 TRANSPOSE** |
| 30 | `ml_drift_delegate/delegate/delegate_kernel.cc` | log the nodes the delegate really takes |

### ml-drift

| # | File(s) | Change |
|---|---|---|
| 22 | `ml_drift/common/gpu_info.h` | `UINT32_MAX` instead of `-1` for Vulkan api_version |
| 23 | `ml_drift/common/gpu_model_builder.{cc,h}`, `selectors/operation_selector.cc` | BUFFER-storage bias tensors (`MakeBiasLinear`) |
| 24 | `selectors/operation_selector.cc`, `transformations/remove_noop.cc` | 5D `Reshape3DAttributes` / `Transpose3DAttributes` |
| 25 | `common/kernels/elementwise.cc` | **guard `pow()` against negative bases off the OpenCL path** |
| 26 | `common/BUILD`, `gpu_model.h`, `gpu_model_util.cc`, `selectors/operation_selector.cc` | allocate rank-5 graph values with their real shape |
| 27 | `ml_drift/webgpu/webgpu_api_util.cc` | `MLD_WEBGPU_READBACK_TIMEOUT_SECONDS` |
| 31 | `common/kernels/winograd.cc` | **bake Winograd Bt/At constants as FLOAT32** — f16 WGSL on devices without shader-f16 silently kills the whole dispatch (SAM fp16 all-zeros) |
| 32 | `webgpu/webgpu_api_util.cc` | debug hook: `MLD_WEBGPU_SHADER_DUMP_DIR` dumps WGSL sources that mention f16 |

## Notes

- `01` hard-codes an absolute path to the local ml-drift checkout, and `02`
  references the local prebuilt Dawn tree (overridable via `DAWN_PREBUILT_DIR`).
  Both are local-machine dev changes; keep that in mind before sending upstream.
- **`24` no longer demotes the `UpdateOutputTensors` shape-mismatch warning to
  `VLOG(1)`.** An earlier revision did, on the theory that the mismatch was a
  harmless axis-relabeling artifact and "execution is unaffected". That was
  wrong: the *truncated* shape is the one installed by `SetOutputDescriptor()`,
  so the warning marks real data loss — on the SAM encoder it fired 100 times
  and every one was a tensor allocated 14x–768x too small. `26` removes the
  mismatch at the source; the warning should stay loud so a regression is
  visible.
- `29` fixes a silent-corruption bug, not a crash: a rank-2 constant TRANSPOSE
  used to relabel the shape and copy the data unpermuted. Worth prioritizing if
  you are cherry-picking.

See [../gpu_op_bisect.zh.md](../gpu_op_bisect.zh.md) for how these were found.
