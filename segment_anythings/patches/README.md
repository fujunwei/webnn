# Patches for `third_party/litert/src` and `third_party/ml-drift`

This directory contains the **complete, current uncommitted working-tree changes**
of the two LiteRT third-party checkouts, split into small, individually
applicable patches, in commit order.

## Which repo does each patch apply to?

| Patches | Repo |
|---|---|
| `01`–`11`, `13` (prefix `litert`) | `C:\Users\junweifu\workspace\chromium\src\third_party\litert\src` |
| `22`–`24` (prefix `mldrift`) | `C:\Users\junweifu\workspace\chromium\src\third_party\ml-drift` |

Both are their own git checkouts (they have their own `.git`). Apply and commit
in numeric order **within each repo**.

## How to apply / commit one patch at a time

Patches are standard unified diffs (with a subject/body preamble above the
`diff --git`). `git apply` ignores the preamble, so either of these works:

```powershell
# litert/src patches, in order:
cd C:\Users\junweifu\workspace\chromium\src\third_party\litert\src
git apply C:\Users\junweifu\workspace\webnn\segment_anythings\patches\01-litert-workspace-local-ml-drift.patch
git add -A
git commit -F -   # then paste the subject/body from the top of the patch file

# ...repeat for 02..13
```

```powershell
# ml-drift patches, in order:
cd C:\Users\junweifu\workspace\chromium\src\third_party\ml-drift
git apply C:\Users\junweifu\workspace\webnn\segment_anythings\patches\22-mldrift-gpu-info-uint32max.patch
git add -A
git commit
# ...repeat for 23..24
```

To commit with the patch's message automatically, convert to `git am` format is
not required — copy the top lines of each patch into the commit message.

## Notes

- `03-litert-odml-weight-loader-stub.patch` creates a **new file**
  `third_party/odml/litert/weight_loader/BUILD` (a stub alias package). It is
  intentionally untracked in the current tree; `git apply` creates it, then
  `git add -A` stages it.
- `01` (`WORKSPACE`) hard-codes an absolute path
  `C:/Users/junweifu/workspace/chromium/src/third_party/ml-drift`. This is a
  local-machine dev change; keep it in mind if committing upstream.
- `02` (`third_party/dawn/workspace.bzl`) likewise references the local prebuilt
  Dawn tree `C:\Users\junweifu\workspace\webnn\_dawn_prebuilt_win` (overridable
  via the `DAWN_PREBUILT_DIR` env var).
- `09`–`26` were consolidated into coarser commits: former `09+16+17+18+19` →
  `09` (0-runtime-input + 5D support), `10+12+14+20+21` → `10` (diagnostic
  logging), `11+15` → `11` (memory-optimize toggle), `24+25+26` → `24`
  (ml-drift 5D support11`, `13 `13`, `22`, `23` are unchanged.

## Patch list

### litert/src (`01`–`21`)

| # | File(s) | Change |
|---|---|---|
| 01 | `WORKSPACE` | point `@ml_drift` at the local chromium ml-drift checkout |
| 02 | `third_party/dawn/workspace.bzl` | load Dawn from the prebuilt tree |
| 03 | `third_party/odml/litert/weight_loader/BUILD` (new) | stub alias for internal `//third_party/odml/...` label |
| 04 | `ml_drift_delegate/delegate/BUILD` | use `//weight_loader:external_weight_loader` |
| 05 | `ml_drift_delegate/delegate/shared_memory_mana, `ml_drift_delegate/tflite/model_builder.cc` | 0-runtime-input + 5D RESHAPE/TRANSPOSE, constant/>=5D inputs, rank-5 tensors (merged 09+16+17+18+19) |
| 10 | `ml_drift_delegate/delegate/delegate_kernel.cc`, `litert/runtime/compiled_model.cc`, `ml_drift_delegate/delegate/delegate_webgpu.cc`, `ml_drift_delegate/tflite/model_builder.cc` | diagnostic logging (merged 10+12+14+20+21) |
| 11 | `litert/runtime/compiled_model.cc`, `ml_drift_delegate/delegate/delegate_webgpu.cc` | disable + document OptimizeMemoryForLargeTensors toggle (merged 11+15) |
| 13 | `ml_drift_delegate/delegate/delegate_webgpu.cc` | farmhash include + disable pipeline-cache callback|
| 18 | `ml_drift_delegate/tflite/model_builder.cc` | 0-runtime-input + 5D + rank-2 TRANSPOSE support |
| 19 | `ml_drift_delegate/tflite/model_builder.cc` | allow rank-5 tensors (IsAllAllowedTensors) |
| 20 | `ml_drift_delegate/tflite/model_builder.cc` | log BuildFinalModel build/transform progress |
| 21 | `ml_drift_delegate/tflite/model_builder.cc` | log per-opcode unsupported-node counts |

### ml-drift (`22`–`26`)

| # | File(s) | Change |
|---|---|---|
| 22 | `ml_drift/common/gpu_info.h` | `UINT32_MAX` instead of `-1` for Vulkan api_version |
| 23 | `ml_drift/common/gpu_model_builder.{cc,h}`, `ml_drift/common/selectors/operation_selector.cc` | BUFFER-storage bias tensors (MakeBiasLinear) |
| 24 | `ml_drift/common/selectors/operation_selector.cc` | 5D Reshape3DAttributes/Transpose3DAttributes |
| 25 | `ml_drift/common/gpu_model_builder.cc` | demote 5D shape-mismatch warning to VLOG(1) |
| 26 | `ml_drift/common/transformations/remove_noop.cc` | skip 5D identity reshape |
4, `ml_drift/common/gpu_model_builder.cc`, `ml_drift/common/transformations/remove_noop.cc` | 5D Reshape3D/Transpose3D attrs + VLOG(1) demotion + skip 5D identity reshape (merged 24+25+26)