# Bug: WebNN LiteRT GPU uses TEXTURE_2D storage on Intel GPUs, inflating GPU memory

Related analysis: `SD1.5_reduce_memory_usage.md` (problem A).
Companion bug (separate root cause): `BUG_tflite_decomposition_arena_oom.md`.

---

## Title

**WebNN LiteRT GPU: Intel GPUs default to TEXTURE_2D storage for intermediate tensors, inflating GPU memory (SD 1.5 VAE decoder: 5.1 GB vs 1.8 GB with buffers)**

## Summary

`GraphImplLiteRt` only sets the GPU precision on the LiteRT GPU options and leaves the tensor storage type at its default. In ML Drift's WebGPU backend the default is chosen per vendor (`GetFastestStorageType`): buffer for NVIDIA, AMD and Apple, and `TEXTURE_2D` for everything else, including Intel. The two storage types use different memory-reuse strategies:

| Storage | Strategy (`ml_drift/webgpu/memory_manager.cc`) | Reuse condition |
|---|---|---|
| buffer | `kGreedyBest` (line 296) | lifetimes do not overlap and the size fits |
| texture | `kEquality` (line 351) | tensor descriptor and BHWDC shape must be **identical** |

Networks with many distinct activation shapes get almost no reuse with textures. The SD 1.5 VAE decoder goes from 64x64 to 512x512 and changes channel counts between 512, 256 and 128, so it is the worst case.

## Impact (measured)

Environment: Windows 11, Intel Arc 140T (Arrow Lake), driver 32.0.101.6452, 63 GB RAM. Models: SD 1.5 text encoder, UNet (batch 1) and VAE decoder, as produced by `GraphBuilderTflite`. A standalone runner linked against the same LiteRT loads the TFLite models and measures peak process counters, including GPU shared memory.

| Model | GPU shared memory, texture (default) | buffer | Change |
|---|---|---|---|
| text_encoder | 0.11 GB | n/a | n/a |
| UNet | 5.67 GB | 4.85 GB | -14% |
| **VAE decoder** | **5.10 GB** | **1.83 GB** | **-64%** |
| SD total | ~10.9 GB | ~6.8 GB | -38% |

- Inference time, 5-run average: VAE 8.2 s -> 8.2 s; UNet 1.20 s -> 1.25 s (about +4%).
- In Chrome (peak memory per process, measured by the reporter): GPU process peak 13.3 GB before and 8.5-8.9 GB after (-33% to -36%).
- VAE private commit in the runner dropped from 12.32 GiB to 8.85 GiB.
- Outputs were compared by statistics only (min, max, mean, std) and agree within fp16 error. They were not compared element by element yet (see "Validation").

## Proposed fix

Set the buffer storage type explicitly in `GraphImplLiteRt::ComputeResources::GetCompilationOptions` for GPU:

```cpp
gpu_options->SetPrecision(...);
if (auto status = gpu_options->SetBufferStorageType(
        ::litert::GpuOptions::BufferStorageType::kBuffer);
    !status) {
  return base::unexpected(mojom::Error::New(
      mojom::Error::Code::kUnknownError,
      base::StringPrintf("Unable to set GPU buffer storage type: %s",
                         status.Error().Message())));
}
```

The setting flows through `ml_drift_delegate_create.cc` to `use_buffer_storage_type`, which `DelegateKernel::GetStorageType()` honors before falling back to `GetFastestStorageType()`.

## Scope and behavior change

- This is a **no-op** for GPUs that already default to buffer: NVIDIA, AMD and Apple.
- It **changes behavior** for every GPU that currently defaults to texture. That is Intel and any other vendor not in that list (for example Qualcomm or Arm GPUs).
- Measurements are from **one Intel machine only**. Performance on other texture-default GPUs is unknown, and textures may be faster there.
- If reviewers prefer a conservative rollout, the alternative is to apply `kBuffer` only for Intel adapters. The adapter vendor would need to be available at this point in the code (not checked).

## Reference point

ORT's WebGPU EP allocates all intermediate tensors as storage buffers, with runtime size-bucketed reuse. It uses no textures.

## What this does not fix

- The renderer-process OOM. The renderer *working set* peak stayed at 15.7-16.7 GB before and after. (That metric cannot show this effect either way: see the note in `BUG_tflite_decomposition_arena_oom.md`.) The cause of the renderer OOM is TFLite arenas for composite decomposition subgraphs, tracked separately in that bug.
- GPU hangs (TDR / `DEVICE_HUNG`). No effect has been shown either way.
- This change has **not been validated on the low-memory (18 GB) machine** where the user-visible failures were reported. Whether it removes those failures is unconfirmed.

## Validation done and pending

Done: runner measurements above (VAE, UNet, text encoder, texture vs buffer); one Chrome run with the change (SD generated a correct image on the second attempt; the first attempt hit the separate renderer OOM).

Pending before landing:
1. Element-wise GPU-vs-CPU comparison of VAE and UNet outputs with buffer storage (`--verify --precision=fp32`).
2. Run on the 18 GB machine: single tab, then with an extra empty tab, then idle.
3. Repeat the measurement on at least one non-Intel GPU to confirm the no-op claim, and ideally one texture-default non-Intel GPU to check performance.

---

## Short CL description

```
WebNN LiteRT: use buffer storage for GPU intermediate tensors

When no storage type is set, ML Drift's WebGPU backend picks TEXTURE_2D
on Intel GPUs (GetFastestStorageType returns buffer only for NVIDIA, AMD
and Apple). Textures are only reused between tensors with identical
shapes (MemoryStrategy::kEquality), while buffers use greedy size-based
reuse (kGreedyBest). Models with many distinct activation shapes get
almost no reuse with textures.

Set BufferStorageType::kBuffer explicitly in the LiteRT GPU options.

Measured with a standalone runner on an Intel Arc 140T (peak GPU shared
memory, texture -> buffer):
  SD 1.5 VAE decoder: 5.10 GB -> 1.83 GB
  SD 1.5 UNet:        5.67 GB -> 4.85 GB
  SD 1.5 total:       ~10.9 GB -> ~6.8 GB
Per-inference time is unchanged for the VAE and about 4% slower for the
UNet. In Chrome, the GPU process peak dropped from 13.3 GB to 8.5-8.9 GB.

This is a no-op on GPUs that already default to buffers (NVIDIA, AMD,
Apple) and a behavior change on every other GPU that defaults to
texture. It has only been measured on Intel.

This does not address renderer process memory (TFLite arenas, tracked
separately), nor GPU hangs.

Bug: <bug id>
Tested: Stable Diffusion 1.5 (text encoder, UNet, VAE decoder) on Intel
  Arc 140T; output statistics match within fp16 error.
```

## Notes for the submitter

- Fill in `Bug:` with the real bug id.
- Update `Tested:` once the element-wise verification is done.
- Decide and state in the description whether the change is global (as above) or Intel-only.
- Do not claim this fixes the PTL (18 GB) failures; that is unconfirmed.
