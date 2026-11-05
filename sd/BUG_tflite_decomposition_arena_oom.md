# Bug: TFLite allocates never-used FP32 arenas for StableHLO composite decomposition subgraphs, causing renderer OOM

Related analysis: `SD1.5_reduce_memory_usage.md` (problem B).
Companion fix (separate root cause, already proposed): `BUG_buffer_storage_type.md`.

---

## Title

**WebNN/LiteRT GPU path: TFLite allocates never-used FP32 arenas for StableHLO composite decomposition subgraphs (InstanceNorm), which PartitionAlloc then rounds up 2x. This exhausts the renderer's address pool and causes OOM on Stable Diffusion 1.5.**

## Summary

When Stable Diffusion 1.5 runs through WebNN with the LiteRT/ML Drift GPU backend, the renderer process dies with "Aw, Snap! Error code: Out of Memory". The failing allocation is only 2 MiB and the system still has 24-31 GiB of commit available. Allocation tracing shows that the largest identifiable host-memory consumer is TFLite's `ArenaPlanner`, which allocates arenas for the decomposition subgraphs of `STABLEHLO_COMPOSITE` ops. On the GPU path these subgraphs are never executed, because ML Drift implements the composite natively. The arenas are sized from the FP32 tensor types declared in the graph and are committed but never touched. PartitionAlloc's aligned-allocation path then rounds each request up to the next power of two. For the SD 1.5 VAE decoder this turns 3.4 GiB of requested arena into 6.8 GiB of committed memory.

## Environment

- Windows 11 (build 26200), Intel Arc 140T (Arrow Lake), driver 32.0.101.6452, 63 GB RAM. The same behavior is reported on a PTL machine with 18 GB (not verified with dumps there).
- Chromium, official release build, `symbol_level=0`.
- Flags: `--enable-features=WebMachineLearningNeuralNetwork[,WebNNLiteRTGpuInRenderer] --disable-features=WebNNDirectML,WebNNOnnxRuntime --enable-unsafe-webgpu`.
- Local changes in the tree: `SetBufferStorageType(kBuffer)` in `graph_impl_litert.cc`, and enlarged `kMaxInlineBufferSize` / `kFlatbufferSafetyThreshold` in `graph_builder_tflite.cc`. Neither affects the arena allocation described here.

## Steps to reproduce

1. Launch Chrome with the flags above.
2. Open the SD 1.5 demo (ONNX Runtime Web, WebNN EP, GPU) with `unetbatch=1` or `2`, and generate an image.
3. The renderer's private commit stays high after generation. Eventually (or after a `readTensor` failure) the renderer crashes with an OOM.

Standalone reproduction without the browser: a runner that links the same LiteRT/TFLite and loads the TFLite model produced by `GraphBuilderTflite` for the VAE decoder shows the same allocations.

## Observed

**Renderer OOM crash dumps** (Crashpad, exception `0xE0000008`; the exception parameters are `{failed size, ullTotalPageFile, ullAvailPageFile}`, and per the source comment they are sampled after the OOM):

| Date | Failed allocation | Renderer private commit | Avail. commit (sampled after) | LiteRT in the process |
|---|---|---|---|---|
| 10-29 | 2 MiB | 17.94 GiB | 24.7 GiB | yes |
| 10-30 | 2 MiB | 19.87 GiB | 31.4 GiB | yes |
| 10-30 | 2 MiB | 17.92 GiB | 30.2 GiB | yes |
| 11-02 | 2 MiB | 16.79 GiB | 23.9 GiB | no |
| 11-04 | 2 MiB | 19.85 GiB | 30.0 GiB | yes |

- All five fail on a 2 MiB request (the PartitionAlloc super page size) while at least 24 GiB commit is available, so this is not system-wide memory exhaustion.
- In the 11-04 dump, the PartitionAlloc reservation is 32 GiB (two 16 GiB pools). One pool has 0 GiB committed and the other has **13.65 GiB (85%)** committed. `kPoolMaxSize` is 16 GiB on Windows x64.
- User-measured Chrome peaks: with buffer storage the GPU process peak dropped from 13.3 GB to 8.5-8.9 GB, but the **renderer peak working set stayed at 15.7-16.7 GB**. Note: that script reports `PeakWorkingSet64` (resident pages). The arenas are committed but never touched, so they count toward commit only, while file-mapped weights (about 3.6 GiB) count toward the working set only. The working set is therefore the wrong metric here; use commit (`PeakPagedMemorySize64` / `PrivateMemorySize64`).

**Allocation tracing** (a runner option that records the call stack of every allocation >= N MiB through the allocator dispatcher and reports the ones still alive after inference). All large allocations share one call chain:

```
aligned_realloc
 <- tflite::SimpleMemoryArena::Commit
 <- tflite::ArenaPlanner::ExecuteAllocations
 <- tflite::Subgraph::AllocateTensors
 <- ... ModifyGraphWithDelegate
 <- LiteRtCompiledModelT::Create
```

The PDB has public symbols only, so function names in the middle frames are unreliable. Only `aligned_realloc`, `SimpleMemoryArena::Commit`, `ArenaPlanner::ExecuteAllocations` and `LiteRtCompiledModelT::Create` are trusted.

| Model | Allocations >= 16 MiB | Requested | Committed (`VirtualQuery`) | Ratio |
|---|---|---|---|---|
| text_encoder | 0 | 0 | 0 | - |
| unet (b1) | 3 | 70 MiB | 96 MiB | 1.37x |
| **vae_decoder** | **30** | **3504 MiB** | **7008 MiB** | **2.00x** |

The blocks are committed but their content is all zero and about 0% resident, so they are never written. They are identical across GPU precision (fp16/fp32), storage type (texture/buffer), and accelerator set (GPU+CPU vs GPU-only).

## Expected

Host memory that no execution path will touch should not be committed on the GPU path, and certainly not at twice the requested size. SD 1.5 should not OOM the renderer.

## Root cause analysis

1. **The graph contains 30 decomposition subgraphs.** The VAE TFLite model has 31 subgraphs: the main graph plus 30 `instance_norm_decomposition` subgraphs. `SerializeInstanceNormalizationAsComposite` emits a `STABLEHLO_COMPOSITE` op for GPU contexts, and ML Drift lowers it to its group-norm kernel. The decomposition body is never run.
2. **The decomposition is declared in FP32.** `SerializeInstanceNormalizationAsComposite` calls `SerializeInputTensorInfo` with the default `operation_supports_float16=false`, so FP16 operands are upcast. It also has `CHECK_EQ(input_tensor_info.data_type, FLOAT32)`. The decomposition subgraph is built with that same type, so its tensors are FP32, for example `[1,2097152,1,32]` = 256 MiB.
   - This is a declared type only. `graph_requires_fp32_precision` is 0 for all three SD graphs (logged), so the GPU runs in fp16.
3. **TFLite allocates the decomposition anyway.** The composite kernel's `Prepare` (`kernels/stablehlo_composite.cc`) calls `decomposition_subgraph->AllocateTensors()`. That runs `Prepare` on every node of the decomposition and plans and commits its arena. This happens on the host before the delegate takes over. Running with `--gpu-only` (no CPU accelerator) gives byte-identical memory, so the CPU fallback is not the source.
4. **The arena size matches the graph.** Each arena is about 2x the largest tensor of its subgraph: seven subgraphs with a 128 MiB largest tensor give 7x256 MiB, and one with a 256 MiB largest tensor gives 512 MiB. This matches the traced 7x256 + 1x512. The remaining five 128 MiB arenas are consistent with 64 MiB subgraphs but were not individually matched.
5. **PartitionAlloc doubles it.** The arena request is `power-of-two + 512 bytes` with alignment 64 (the origin of the 512 bytes was not found). The shim's `AlignedRealloc` goes to `PartitionRoot::AlignedAllocInline`. `GetAdjustedSizeForAlignment` rounds up to the next power of two when `16 < alignment <= PartitionPageSize()` and `use_tighter_aligned_alloc_bound` is false. That option defaults to `kDisabled`, and no code in `base/allocator`, `components`, `chrome` or `content` enables it. The result is a committed region of exactly 2x the request (256 MiB + 512 B becomes a 512 MiB region).
6. **Ruled out:**
   - fp16-to-fp32 weight conversion for conv/FC. Weights total 1.6 GiB (unet) and 0.09 GiB (VAE), and ML Drift's partitioner removes the `DEQUANTIZE` ops on the GPU path.
   - The CPU fallback (XNNPACK).
   - ML Drift host-mapped GPU buffers. The blocks are not resident.
   - Dawn D3D12 buddy-allocator heaps. The blocks come from TFLite through PartitionAlloc.

## Proposed solutions

**Option A: do not allocate arenas for decomposition subgraphs that will never run** (largest benefit: all 6.84 GiB for VAE). **Validated in the standalone runner.**
- Patch in `kernels/stablehlo_composite.cc`: at the end of `Prepare`, `if (!this_subgraph->ShouldPreserveAllTensors()) TF_LITE_ENSURE_OK(context, decomposition_subgraph->ReleaseMemory());`, and at the start of `Eval_static`, `TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());`. If the delegate takes the composite, `Eval` never runs; if not, the CPU fallback allocates lazily.
- Runner results: VAE live arenas 30 -> 0, private commit 8.85 -> 2.00 GiB, output identical; text_encoder (which does run the composite `Eval` path) passes `--verify`.
- Chrome, before vs after (renderer, `chrome-peak-commit.ps1`, one run each, not a strict A/B; whether both used the same unetbatch is not verified): peak commit 15312 -> 12504 MB, **current commit after generation 12942 -> 4387 MB**, peak working set 16500 -> 15414 MB. GPU process peak commit 12116 -> 7252 MB (includes the buffer-storage change). No 512/256 MiB arena blocks in the live renderer after the fix. The 8.6 GB drop in current commit is the clearest signal; the peak dropped by only 2.8 GB, less than the arena total, and why is not established (the peak may come from another phase). It is not proven that the Chrome binary contains the patch.
- Not tested: the `Eval_dynamic` path. A VAE CPU-reference `--verify` crashed with OOM (0xE0000008); the cause is presumed to be the CPU reference path itself, not shown to be independent of the patch.

**Option B: declare the decomposition in FP16** (predicted: VAE committed 7008 to 3504 MiB, halved; not yet validated). Required changes:
- `graph_builder_tflite.cc`, `SerializeInstanceNormalizationAsComposite`:
  - add `CAST` FP32->FP16 before the composite and FP16->FP32 after it (the composite's own inputs and outputs must be FP16, because the kernel copies the op's input types onto the decomposition inputs);
  - read scale/bias with `operation_supports_float16=true` to keep them as FP16 constants;
  - pass `TensorType_FLOAT16` to `SerializeNormalizationDecompositionSubgraph`;
  - remove the `CHECK_EQ(..., FLOAT32)`.
- `SerializeNormalizationOperation`: remove its `CHECK_EQ(input_tensor_type, FLOAT32)` and write epsilon as Float16.
- `third_party/tflite/.../kernels/reduce.cc`, `InitializeTemporaries`: add `case kTfLiteFloat16:` to the type switch. Today MEAN's `Prepare` returns `kTfLiteError` for FP16, which would make `CompiledModel::Create` fail. This holds even on a GPU-only graph, because `Prepare` runs regardless of device. The other ops' `Prepare` (`sub`, `div`, `add`, `mul`, elementwise) do not reject FP16.
- Trade-offs: a small TFLite patch to maintain; 2 extra `CAST` nodes per InstanceNorm (60 for VAE) with unknown GPU cost; if ML Drift ever declines the composite, the CPU fallback would fail at `Eval` (expected as an error, not silent corruption, not tested); whether ML Drift accepts an FP16 group-norm input was not checked.

**Option C: enable PartitionAlloc `tighter_aligned_alloc_bound`** (removes the 2x rounding, about 3.4 GiB for VAE). It is a process-wide allocator setting that affects all aligned allocations, so it needs evaluation.

**Option D: find the source of the extra 512 bytes** in the arena request. If requests did not land just above a power of two, the 2x amplification would mostly disappear for these sizes.

**Suggested order:** land A (validated in the runner), keep C as a fallback, and treat B as the longer-term option.

## Validation plan

- Re-run the allocation trace on the VAE and compare committed arena memory (baseline 7008 MiB).
- Compare VAE output statistics before and after, then element-wise (`--verify` against CPU).
- Chrome: measure renderer peak **commit** with `peak_memory\chrome-peak-commit.ps1` (not the working set), and run on the 18 GB PTL machine (single tab, extra empty tab, idle).
- Find what makes up the remaining ~12.5 GB renderer peak commit (timeline sampling).

## Known gaps and limitations

- The renderer dump does not match the runner exactly: its largest pool blocks are 512 MiB x1 and 256 MiB x7, half of the runner's. Chrome's actual graph may differ from the dumped TFLite model. So the mechanism is established, but "the renderer's VAE arenas total 6.8 GiB" is not.
- Pool exhaustion as the final trigger is inferred (16 GiB pool, 85% committed, 2 MiB request failing). It was not observed directly.
- The PartitionAlloc rounding was verified from source and from the runner, not inside the renderer.
- The renderer holds about 7 GiB more in the pool and about 6 GiB outside it that are not attributed yet.
- About 13 minutes passed between a `readTensor` failure and the OOM in the 11-04 session; the cause of the delay is unknown.
- Not checked on the PTL machine's `Crashpad\reports`.

## Related issues (suggest filing separately)

1. **Intel TEXTURE_2D storage inflates GPU memory.** See `BUG_buffer_storage_type.md`. The fix is a one-line `SetBufferStorageType(kBuffer)` in `graph_impl_litert.cc`, already applied locally, pending validation on PTL.
2. **Intermittent GPU hangs (TDR).** `DEVICE_HUNG` and `ReadTensor ... failed: 4` coincide with WER entries `LKD_0x141_Tdr ... igdkmdn64.sys_ARL_DX12COMPUTE_BBHANG/RINGHANG`: 34 entries on this machine (33 between 10-27 and 10-30, one on 11-04). It is independent of memory size and the root cause is not found. LiteRT submits 256 nodes per command encoder, versus 16 dispatches per submit in the ORT WebGPU EP, which is a candidate worth testing. Dawn has no DRED support, so the hanging command cannot be identified directly.

## Attachments to include

The five Crashpad dumps (or the summary table), the allocation traces (`trace_vae*.log`), the per-subgraph analysis for `vae_decoder.tflite`, the user-side peak-memory logs, and the doc `SD1.5_reduce_memory_usage.md`.

## Notes for the submitter

- The allocation-trace option (`--trace-big-allocs`) currently exists only in the local, `git add`-ed `sam_runner/sam_encoder_runner.cc`. Share that patch if reviewers want to reproduce the trace.
- Statements marked as "predicted", "inferred" or "not tested" above should stay worded that way; Options A and B have not been validated.
