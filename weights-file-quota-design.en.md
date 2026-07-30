# WebNN Weights File: Disk Usage and Quota Design

## 1. Background

When WebNN compiles a graph, it serializes constant operands (weights) into a separate temporary file so TFLite can `mmap` them directly, avoiding an extra in-memory copy. The current implementation:

- Entry point: [`services/webnn/host/weights_file_provider.cc`](../chromium/src/services/webnn/host/weights_file_provider.cc) — `CreateTemporaryFile()` calls `base::CreateTemporaryFile()` directly.
- Path: the system temp directory (`/tmp`, `%TEMP%`), **not** under the profile's storage partition.
- Flags: `FLAG_DELETE_ON_CLOSE` — the file is unlinked once the fd closes.
- Only existing safeguard: incognito mode skips file creation entirely and uses the in-memory Flatbuffer path.

### 1.1 Problems

1. **No size check.** A malicious site can craft a huge model and exhaust `/tmp`.
2. **No per-origin accounting.** A single origin can build graph after graph and fill the disk.
3. **No graceful fallback.** A full disk causes the build to fail outright.

### 1.2 Threat model

- **Trust boundary:** the renderer is treated as an untrusted compromise boundary. The WebNN service inside the GPU process and the broker inside the browser process are trusted.
- **Attack surface.** A compromised renderer / malicious page can:
  1. Concurrently build many large models within a single origin to exhaust `/tmp` (DoS).
  2. Share the same origin across multiple frames / workers to bypass per-frame caps.
  3. Request `required_bytes` and then write more (the **fd-level over-write** issue — see §5 followup).
  4. Coordinate across origins (multiple malicious origins each filling their own quota).
- **Out of scope:** disk consumption from other Chrome subprocesses or other programs is handled by OS-level disk pressure; the browser can only cushion it with headroom.

---

## 2. Design choices

### 2.1 Why not plug into `storage::QuotaManager`

`storage::QuotaManager` (`//storage/browser/quota/`) tracks quota per `StorageKey` (origin + top-level site) and is shared by IndexedDB, Cache API, File System Access, Service Worker, and similar APIs. WebNN should **not** plug into it directly:

- The WebNN service runs in the GPU process; `QuotaManager` lives in the Storage Service / browser process. Querying it requires a Mojo round-trip that would block the graph-build critical path.
- Weights are `DELETE_ON_CLOSE` ephemeral compute artifacts, not "persistent user data" — semantically they don't belong to the category Quota manages.
- The files land in the system temp directory, not under the profile, so they aren't on Quota's books in the first place.
- The real risk is **`/tmp` DoS**, which is a different concern from per-origin storage quota.

### 2.2 The in-memory fallback already exists

The TFLite backend already accepts an invalid `base::File` and falls back to the in-memory Flatbuffer path (this is exactly what incognito does today). The fallback channel exists; we only need `CreateTemporaryFile()` to:

- Check free space with `base::SysInfo::AmountOfFreeDiskSpace()` before creating the file.
- Return an empty `base::File()` on write failure (`ENOSPC`, or `Initialize()` returning a non-valid file).

The renderer falls back to in-memory automatically; no new protocol is required.

Caveat: the in-memory Flatbuffer is bounded by **< 2 GiB** (the Flatbuffer offset is `int32`), so "fall back to in-memory" cannot scale to arbitrarily large graphs.

### 2.3 Layered defense

| Layer | Mechanism | Problem solved |
|---|---|---|
| **A** | `AmountOfFreeDiskSpace()` before file creation; on failure return invalid file so TFLite uses in-memory | Disk-full causing build failure / crash |
| **B** | `WebNNContextImpl` enforces a **per-context cap** (4 GiB), summing weights-file sizes across graphs; over-cap requests are rejected or fall back | Single-tab DoS of `/tmp` |
| **C** | Browser process enforces a **per-origin total cap** (8 GiB) via the `WeightsFileCreatorImpl` broker, which maintains an origin tracker | Origin-level abuse across contexts |
| D (not done) | If we ever wanted to plug into `QuotaManager`, it would be a separate storage type (e.g. `kWebNNWeights`) brokered through the browser process | Alignment with the Web Storage quota model |

A + B + C handle ~80% of the practical problem and are what's implemented.

---

## 3. Implementation

### 3.1 Mojo interface

[`services/webnn/public/mojom/webnn_context_provider.mojom`](../chromium/src/services/webnn/public/mojom/webnn_context_provider.mojom)

```mojom
interface WebNNWeightsFileCreator {
  // Creates a temporary file for storing model weights. `required_bytes` is
  // the size the caller intends to write; it is used by the browser to
  // enforce free-disk-space and per-origin budgets. Returns null if the
  // request is denied (over quota / not enough disk space) or the file
  // cannot be created; the caller is expected to fall back to keeping the
  // weights embedded in the in-memory model.
  CreateWeightsFile(uint64 required_bytes) => (mojo_base.mojom.File? file);

  // Returns `bytes` previously granted by a successful `CreateWeightsFile`
  // call back to the per-origin budget. Calling this for more bytes than
  // were ever granted (or before any successful `CreateWeightsFile`) is a
  // protocol violation and results in `mojo::ReportBadMessage` on the
  // browser side. The pipe is also self-tracking: any reservation still
  // outstanding when the pipe is dropped is released by the destructor as a
  // backstop against crashed / killed render processes.
  ReleaseWeights(uint64 bytes);
};
```

### 3.2 §A — Disk-space check

[`services/webnn/host/weights_file_provider.{h,cc}`](../chromium/src/services/webnn/host/weights_file_provider.cc)

`CreateWeightsFile(required_bytes, cb)` calls `AmountOfFreeDiskSpace()` and `AmountOfTotalDiskSpace()` before creating the file. The headroom shape mirrors `storage::QuotaSettings::must_remain_available` (`min(fixed, ratio)`), but the values are **considerably more aggressive** than the storage subsystem's:

```cpp
inline constexpr uint64_t kWeightsFileMustRemainAvailableBytes = 10ull << 30;  // 10 GiB
inline constexpr double   kWeightsFileMustRemainAvailableRatio = 0.10;         // 10%

// headroom = min(fixed reserve, total disk × ratio)
const uint64_t headroom = std::min<uint64_t>(
    kWeightsFileMustRemainAvailableBytes,
    static_cast<uint64_t>(static_cast<double>(*total_bytes) *
                          kWeightsFileMustRemainAvailableRatio));
```

If `free < required_bytes + headroom`, return an invalid `base::File`, which triggers the existing in-memory fallback.

#### 3.2.1 Headroom derivation

The §A disk check runs **fully in parallel** on the ThreadPool: N concurrent calls each see the same `free` snapshot, each one only checks `free ≥ rᵢ + H`, but the actual write is `W = Σrᵢ`. For all N calls to pass without hitting `ENOSPC`, we need

```
H ≥ W − max(rᵢ)
```

Plugging in the upper bound `W ≤ kMaxBytesPerOrigin = 8 GiB`, and noting that `max(rᵢ)` has no lower bound (the renderer can split a single origin's 8 GiB into arbitrarily many small graphs), the worst case is `max(rᵢ) → 0`, so

```
H ≥ kMaxBytesPerOrigin = 8 GiB
```

Concrete examples: a single origin with 4 concurrent contexts each at 2 GiB → `W = 8 GiB`, `max = 2 GiB` → headroom must be ≥ 6 GiB; eight 1-GiB graphs → ≥ 7 GiB; many tiny graphs → approaches 8 GiB.

The 10-GiB constant adds a 2-GiB buffer above the 8-GiB lower bound to absorb cross-process disk consumption (other Chrome subprocesses, other programs) and the window between fd handoff and write completion. **Dropping to 2 GiB / 2% would cause `ENOSPC` whenever a single origin splits into multiple ≥ 2-GiB models** — a failure mode the design must avoid because the renderer has no retry path.

The cross-origin worst case (two origins each filling their cap → `W = 16 GiB`, `max = 4 GiB` → `H_min = 12 GiB`) exceeds 10 GiB. This is a known gap; the per-origin cap forces "two concurrent malicious origins each at 8 GiB" into a narrow window where the attacker must precisely synchronize multiple frames / workers — normal usage will not trigger it.

Effective headroom on different devices:

| Total disk | min(10 GiB, 10% × total) |
|---|---|
| 1 TiB | 10 GiB |
| 256 GiB | 10 GiB |
| 100 GiB | 10 GiB |
| 64 GiB | 6.4 GiB |
| 16 GiB | 1.6 GiB |
| 8 GiB | 800 MiB |

#### 3.2.2 Concurrency model and TOCTOU

`AmountOfFreeDiskSpace` is a **point-in-time snapshot**. Two concurrent `CreateWeightsFile` calls can both observe the same free space, both pass the check, and the second renderer's `mmap` / `write` can then hit `ENOSPC` mid-write:

```
Context A: free=3 GB, need=2 GB+headroom -> pass, get fd
Context B: free=3 GB, need=2 GB+headroom -> pass, get fd
After both fds reach the renderer, the second hits ENOSPC mid-write.
```

Two alternatives were considered and rejected:

**True preallocation (`base::AllocateFileRegion` / `SetLength`).**

1. **Breaks existing caller semantics.** [`graph_builder_tflite.cc`](../chromium/src/services/webnn/tflite/graph_builder_tflite.cc) treats `weights_file_.GetLength()` as "bytes written so far" to compute the next buffer's append offset. After preallocation, `GetLength()` returns `required_bytes` (e.g. 2 GiB) immediately, so the first weights buffer is written **past** 2 GiB, and TFLite reads zero-filled preallocated space at the metadata-recorded offset — the model is corrupt.
2. **High fix cost.** `GraphBuilderTflite` would need a new `weights_bytes_written_` member, and every `weights_file_.GetLength()` / `SetLength` call site would have to be rewritten or guarded (calling `SetLength(64)` on a 2-GiB preallocated file is `ftruncate(64)`, which immediately releases the entire reservation). Large blast radius, high regression risk.

**In-process serialization (single `SequencedTaskRunner` for all `CreateTemporaryFile` calls).**

- Even if the sequenced runner only runs metadata-level syscalls (millisecond scale), distinct origins / contexts would visibly become "previous page's weights must finish before next page can even start checking", which is a confusing UX.
- The single-process TOCTOU window is roughly the duration of one `mkstemp` call (sub-millisecond); concurrent collisions are meaningful only in attacker-crafted microsecond-scale races. The worst-case concurrent exhaustion requires ≥ 16 GiB outstanding, which is already capped by per-origin / per-context limits.
- With headroom raised to 10 GiB / 10%, cross-process consumption is absorbed by the same mechanism; serialization adds no further protection.

**Chosen approach: default ThreadPool concurrency + large headroom + upper-layer caps.**

[`weights_file_provider.cc`](../chromium/src/services/webnn/host/weights_file_provider.cc) uses the default `base::ThreadPool::PostTaskAndReplyWithResult`:

```cpp
void CreateWeightsFile(uint64_t required_bytes,
                       CreateWeightsFileCallback callback) {
  base::ThreadPool::PostTaskAndReplyWithResult(
      FROM_HERE,
      {base::TaskPriority::USER_BLOCKING,
       base::TaskShutdownBehavior::CONTINUE_ON_SHUTDOWN, base::MayBlock()},
      base::BindOnce(&CreateTemporaryFile, required_bytes),
      std::move(callback));
}
```

Concurrency model:

- `CreateTemporaryFile` calls from different origins / contexts run **fully in parallel** on the thread pool, with no shared mutex.
- Each call independently invokes `AmountOfFreeDiskSpace` + `AmountOfTotalDiskSpace` + `mkstemp`.
- Upper-layer [`WeightsFileCreatorImpl`](../chromium/src/services/webnn/host/weights_file_creator_impl.cc) (per-origin 8-GiB cap) and [`WebNNContextImpl`](../chromium/src/services/webnn/webnn_context_impl.cc) (per-context 4-GiB cap) bound outstanding writes per origin / per context.
- The 10 GiB / 10% headroom absorbs the gap between "snapshot seen by `AmountOfFreeDiskSpace`" and "renderer actually finishing the write", plus cross-process consumption.

Worst cases:

- One fully-saturated malicious origin = 8 GiB outstanding; requires `free ≥ 8 GiB + 10 GiB headroom = 18 GiB`. Trivially satisfied on a 1-TiB disk.
- Two concurrent fully-saturated origins = 16 GiB outstanding; requires `free ≥ 26 GiB`. Still reasonable.
- 8-GiB small disk: headroom is 800 MiB, so a full 8-GiB origin request is rejected because `required + headroom > free`. This is the right behavior — small disks can't run large models anyway.

#### 3.2.3 Residual risks

| Risk | Mitigation |
|---|---|
| Cross-process disk consumption (other Chrome subprocesses, other programs) | Absorbed by the 10 GiB / 10% headroom. This is an OS-level concern; the browser cannot defend perfectly. |
| Seconds-scale window between handing the `base::File` to the renderer and the write completing | Same headroom. |
| Filesystem / OS rejects the write | Renderer write fails → existing in-memory fallback path takes over. |
| Writable fd handed to the renderer has no fd-level hard cap | See §5 followup. |

### 3.3 §B — Per-context cap

[`services/webnn/webnn_context_impl.{h,cc}`](../chromium/src/services/webnn/webnn_context_impl.cc)

```cpp
static constexpr uint64_t kMaxWeightsBytesPerContext = 4ull << 30;  // 4 GiB
uint64_t weights_bytes_granted_ = 0;
```

`WebNNContextImpl::CreateWeightsFile(required_bytes, cb)` performs an overflow check before forwarding to `ContextProviderTflite` / `WebNNContextProviderImpl`; the reservation is returned when the file ends up invalid.

### 3.4 §C — Per-origin cap

[`services/webnn/host/weights_file_creator_impl.{h,cc}`](../chromium/src/services/webnn/host/weights_file_creator_impl.cc)

```cpp
static constexpr uint64_t kMaxBytesPerOrigin = 8ull << 30;  // 8 GiB
```

A process-wide `OriginUsageTracker` (a lock-guarded `std::map<url::Origin, uint64_t>`) is shared across `WeightsFileCreatorImpl` instances:

- A `WeightsFileCreatorImpl` lifetime equals one self-owned mojo pipe (per-frame / per-worker).
- `CreateWeightsFile` invokes `TryReserve()`; on file failure or instance destruction, `Release()` undoes it.
- Frames / workers within the same origin share the budget.

Why 8 GiB: the storage quota system grants each `StorageKey` `total_disk × 0.6 × 0.75 = total_disk × 45%` (see Appendix A). WebNN weights are ephemeral compute artifacts (`DELETE_ON_CLOSE`), so chasing 45% would be absurd. A fixed 8-GiB ceiling is far more conservative than the quota default and is safe as an anti-abuse hard cap.

### 3.5 Reservation-release lifecycle

Three release paths, in decreasing order of immediacy:

1. **Per-context (intra-process, immediate).** When `WebNNGraphImpl` is destroyed, it synchronously calls `WebNNContextImpl::ReleaseWeightsBytes(weights_bytes_granted_)`, which subtracts §B's `weights_bytes_granted_`. The `weights_bytes_granted` value is plumbed from the `CreateWeightsFile` callback's `granted_bytes` parameter into `GraphImplTflite` / `GraphImplLiteRt` constructors and into the `WebNNGraphImpl` base class — no Mojo IPC.
2. **Per-origin (cross-process, explicit mojom on graph destruction).** Also at graph destruction, `WebNNContextImpl::ReleaseWeightsBytes` posts a `WebNNWeightsFileCreator::ReleaseWeights(bytes)` Mojo call on the main task runner to the browser-side `WeightsFileCreatorImpl`, which decrements §C's `OriginUsageTracker`. Compared with "wait for the pipe to close and release everything at once", this lets **the budget be reused immediately by the next graph from the same origin**, which avoids long-lived pipes (Service Worker / shared worker) holding onto budget they no longer use.
3. **Pipe-disconnect backstop.** When `WeightsFileCreatorImpl` is destroyed, it returns any remaining `reserved_bytes_` to `OriginUsageTracker` in a single shot. This covers three abnormal cases: (a) renderer crashes before sending `ReleaseWeights`; (b) the browser kills the renderer; (c) a future code path that forgets to call `ReleaseWeights`.

#### 3.5.1 Defending against over-release

`WeightsFileCreatorImpl::ReleaseWeights(bytes)` validates `bytes ≤ reserved_bytes_`. On overflow it calls `mojo::ReportBadMessage("WebNN: ReleaseWeights over-release")` to terminate the renderer and **never** lets `OriginUsageTracker` go negative (it cannot — going negative would mean some other frame's budget gets "refunded", actually letting a malicious origin exceed 8 GiB).

`WebNNContextImpl::ReleaseWeightsBytes(bytes)` similarly uses `CHECK_GE(weights_bytes_granted_, bytes)` — this layer is inside the GPU process and cannot be driven directly by the renderer, so a `CHECK` is more appropriate than `BadMessage`.

#### 3.5.2 Race-free path on graph-build failure

After `GraphImplTflite::DidCreateAndBuild` / `GraphImplLiteRt::DidCreateAndBuild` receives the background-thread `base::expected`:

- If `compute_resources.has_value()`: build the `GraphImpl`, plumb `weights_bytes_granted` into the base-class member, and let destruction handle release.
- If `!compute_resources.has_value()`: **no `GraphImpl` is constructed**, so the destructor release path is unreachable; release explicitly in place via `context->ReleaseWeightsBytes(weights_bytes_granted)`. Otherwise the reservation stays pinned in `weights_bytes_granted_` until context destruction.

The `if (!context) return` branch does not release explicitly: a destroyed context means §B's accounting is destroyed with it, and §C falls through to the pipe-disconnect backstop.

#### 3.5.3 Why renderer-driven release is rejected

> *(Internal-only note. The renderer is the untrusted compromise boundary.)*

A tempting "simpler" approach: have the renderer send `ReleaseWeights` itself when the graph is destroyed. Rejected because:

1. **A compromised renderer can over-release at will**, draining `OriginUsageTracker` so it (or another frame in the same origin) bypasses the 8-GiB cap. `ReportBadMessage` is after-the-fact remediation — there's still a window.
2. **`OriginUsageTracker` is shared across origins / frames.** A single renderer's release affects *other frames'* budgets. Untrusted code must not write shared state.
3. **Disk is a process-global resource.** Only the browser process can authoritatively decide when to reclaim.

Therefore release must be triggered by **`WebNNGraphImpl` destruction inside the GPU process**; renderer-initiated RPC is not accepted.

### 3.6 End-to-end call flow

Reservation:

```
ContextImplTflite::CreateGraphImpl
  -> sum constant_operands ByteSpan().size() to get required_bytes
  -> WebNNContextImpl::CreateWeightsFile(required_bytes, cb)         [§B check]
       -> ContextProviderTflite::CreateWeightsFile(required_bytes, cb)
            -> mojom::WebNNWeightsFileCreator.CreateWeightsFile(required_bytes)  [Mojo]
                 -> WeightsFileCreatorImpl::CreateWeightsFile         [§C check]
                      -> webnn::CreateWeightsFile(required_bytes, cb) [§A check]
                           -> CreateTemporaryFile()
                 cb(file, granted_bytes)  // granted_bytes flows back to GraphImpl
```

Release (triggered by graph destruction):

```
~WebNNGraphImpl
  -> WebNNContextImpl::ReleaseWeightsBytes(weights_bytes_granted_)   [§B decrement]
       -> if (is_context_provider_in_renderer_):                      [in-renderer path]
            WebNNContextProviderInRenderer::ReleaseWeights(bytes)
              -> mojom::WebNNWeightsFileCreator.ReleaseWeights(bytes) [Mojo]
                   -> WeightsFileCreatorImpl::ReleaseWeights          [§C decrement]
                        -> OriginUsageTracker::Release(origin, bytes)
       -> else:                                                       [GPU-process path]
            // TODO: viz::mojom::GpuHost::CreateWebNNWeightsFile does not carry
            // origin, so the browser side has no per-origin reservation to
            // release. The TFLite/LiteRT MLDrift GPU delegate is planned to
            // move out of the GPU process into the renderer, after which this
            // branch goes away. Until then, weights files on the GPU path are
            // reclaimed only when their fds close (unlink-on-close).
```

Pipe-disconnect backstop: `~WeightsFileCreatorImpl` returns any remaining `reserved_bytes_` to `OriginUsageTracker` in one shot.

### 3.7 Origin source (content/browser)

[`content/browser/browser_interface_binders.cc`](../chromium/src/content/browser/browser_interface_binders.cc)

```cpp
// RenderFrame:
host->GetLastCommittedOrigin()

// Worker hosts (template specializations):
DedicatedWorkerHost / SharedWorkerHost -> GetWorkerStorageKey().origin()
ServiceWorkerHost                       -> GetBucketStorageKey().origin()
```

---

## 4. Known limitations

1. **Per-origin cap is fixed**, not coupled to disk size. We could compute `min(8 GiB, total_disk × N%)`, but that would force `OriginUsageTracker` to become async or to cache the disk-size lookup at `Create` time. Given the complexity trade-off, we kept the constant.
2. **GPU path bypasses the per-origin cap.** LiteRT GPU inference runs in the GPU process and creates files via `viz::GpuHost::CreateWebNNWeightsFile` → `webnn::CreateWeightsFile`, **bypassing `WeightsFileCreatorImpl` and not carrying origin** (see [`gpu_host.mojom`](../chromium/src/services/viz/privileged/mojom/gl/gpu_host.mojom) `CreateWebNNWeightsFile(uint64 required_bytes)` and [`gpu_host_impl.cc`](../chromium/src/components/viz/host/gpu_host_impl.cc) `CreateWebNNWeightsFile`). Net effect: an origin using the GPU backend is bounded only by §A (disk headroom) + §B (per-context 4 GiB); §C (per-origin 8 GiB) does not apply.
   - Fix direction: add origin to `gpu_host.mojom::CreateWebNNWeightsFile`; promote `OriginUsageTracker` to a browser-process-wide singleton so that `GpuHostImpl::CreateWebNNWeightsFile` and `WeightsFileCreatorImpl::CreateWeightsFile` share the same per-origin ledger.
   - Why we don't fix it now: the GPU path's effective concurrent contexts per origin are implicitly bounded by the mojo channel and browser-side throttling (≤ 2, so per-context 4 GiB × 2 = 8 GiB matches §C). This is only an implicit invariant; **§C must be added explicitly if we ever relax that concurrency limit**.
3. **Writable fd is handed to the renderer with no fd-level hard cap** (reviewer-flagged). Today `WebNNWeightsFileCreator::CreateWeightsFile` ships the browser-created `base::File` handle back to the renderer. A compromised in-renderer TFLite/LiteRT can request `required_bytes = 1` and then `write()` arbitrary GBs. The current mitigations (`must_remain_available` headroom + `DELETE_ON_CLOSE` + per-origin/per-context caps) are not enforced at the fd level; in the worst case a single origin can write up to its allotted disk slice minus headroom.
   - Fix direction: see §5 [Followup CL](#5-followup-cl-cross-platform-fd-level-hard-cap).
   - **The GPU-process path is unaffected**: the fd stays in the GPU process and the renderer never sees it; `required_bytes` is summed inside the GPU process by `WebNNContextImpl` from `WebNNConstantOperand::ByteSpan().size()`, which is the same source as the actual write — they cannot diverge.
4. **TODO**: `crbug.com/507502295` — Use file manager for weights files.

### 4.1 Comparison with `storage::QuotaManager`

| Dimension | `QuotaManager` | WebNN weights |
|---|---|---|
| Concurrency arbitration | Single-sequence serialization + in-memory pending-reservation table | No serialization; concurrency is bounded by per-origin (8 GiB) + per-context (4 GiB) caps, with 10 GiB / 10% headroom absorbing snapshot drift |
| Cross-process consistency | Single browser-instance storage service | Single browser process; cross-process pressure is absorbed by headroom |
| Release timing | Explicit commit/release on the in-memory table | Immediate on graph destruction (§B sync; §C explicit `ReleaseWeights` mojom); pipe close releases any remaining §C reservation |
| Failure semantics | Returns a quota error; caller decides | Returns an invalid file; auto-fallback to in-memory |

WebNN does not use true preallocation (`fallocate` / `SetEndOfFile`) for the reasons in §3.2.2: it would break the existing `GraphBuilderTflite` `GetLength()` append semantics. Large headroom + upper-layer caps is the smaller change.

---

## 5. Capacity tracker: renderer holds writable fd, sync IPC before each extension

> **Design evolution (2026-06-17 update)**
>
> This section originally chased a "kernel-enforced fd-level hard cap", first via chunked SharedMemory (commit `2881a74236`, now abandoned). The motivation was to elevate §A/§B/§C from bookkeeping to kernel enforcement.
>
> That starting premise was wrong: **the FSA picker is not a security boundary**. File System Access gives the renderer a writable fd directly to a user-chosen file. Security comes from two *independent* layers: (1) the user authorizes through the picker which byte-streams this origin may touch — a *privacy* boundary; (2) `FileSystemAccessCapacityTracker` does **incremental accounting** in the browser process — a *resource-exhaustion* boundary. The two layers are orthogonal: the picker stops a malicious page from reading your private files; the capacity tracker stops it from filling the disk. Neither substitutes for the other.
>
> The key observation: **OPFS (Origin Private File System) follows exactly the same picker-less path.** A `FileSystemSyncAccessHandle` requires no user authorization dialog; the origin gets its private filesystem automatically, the renderer still holds a writable fd, and it still goes through the same [`FileSystemAccessCapacityTracker`](https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/modules/file_system_access/file_system_access_capacity_tracker.cc) for capacity accounting. OPFS security comes 100% from the browser-side capacity tracker — **no picker required**.
>
> Mapping this to WebNN: the weights tempfile is browser-created, origin-private, lifetime-bound to the context, never exposed to the user or other origins — the same shape as an OPFS sandbox file. WebNN doesn't need a picker; it needs **the OPFS-equivalent capacity tracker** — renderer holds the writable fd, calls `RequestCapacityChange(new_size)` before each extension, and the browser re-runs §A disk headroom + §B per-context cap + §C per-origin cap + §D `fstat` anti-tamper on every IPC. Safety comes from *incremental, mandatory, un-bypassable* enforcement at the IPC boundary, not from the fd type.
>
> The old chunked-SHM implementation (commit `2881a74236`) is replaced by this design.

### 5.1 Problem

The original `CreateWeightsFile(required_bytes)` was one-shot bookkeeping: the browser runs §A/§B/§C once before handing out the fd, and after that the fd is in the renderer with no per-fd limit. A compromised renderer can call `CreateWeightsFile(required_bytes=1)`, pass all upper-layer checks, and then `write()` arbitrary bytes to the fd — up to ENOSPC — without ever calling `RequestCapacityChange`.

A secondary problem (raised by reillyg@): streaming weights require that `build()` can start without knowing the total weight size in advance, but `CreateWeightsFile(required_bytes)` requires that total upfront.

### 5.2 Security model: picker and capacity tracker are independent layers

| Dimension | FSA picker file | OPFS sandbox file | WebNN weights tempfile |
|---|---|---|---|
| User authorization layer | Picker selects the specific file (privacy) | No picker (origin gets it automatically) | No picker (service creates it) |
| Who holds the writable fd | Renderer | Renderer | Renderer (in-renderer TFLite/LiteRT) |
| Sync IPC check before each extension | ✅ `FileSystemAccessCapacityTracker.RequestFileCapacityChangeSync` | ✅ Same tracker class | ✅ `WeightsFileCapacityHost.RequestCapacityChange` |
| Resource-exhaustion defense layer | Browser-process capacity tracker | Browser-process capacity tracker | Browser-process §A/§B/§C/§D (this design) |
| File lifetime | User file, survives origin unload | Origin-private, cleared with origin | Context-bound, unlinked on pipe close |
| Cross-origin read exposure | Picker prevents it | OPFS isolation prevents it | fd never leaves service↔renderer channel |

WebNN resembles OPFS more than FSA picker files: picker-less, origin-internal, user-agent-created. Safety comes entirely from browser-side incremental accounting.

> **GPU-process path is out of scope here.** The LiteRT GPU delegate runs in the GPU process, uses `viz::mojom::GpuHost::CreateWebNNWeightsFile` to create its fd, and never hands that fd to the renderer — no fd out-of-bounds write threat. Its limits are §A disk headroom + §B per-context self-check. The capacity tracker only covers the in-renderer TFLite/LiteRT path.

### 5.3 Design

#### 5.3.1 Mojo interface

```mojom
// Browser-process self-owned receiver, one creator per `navigator.ml`
// (= one per renderer ExecutionContext). The creator is a thin factory:
// it owns no per-file state, so concurrent `OpenWeightsFile` calls from
// the same `navigator.ml` are independent.
interface WebNNWeightsFileCreator {
  // Browser creates a deferred-unlink tempfile, dups a writable fd to the
  // renderer, and self-owns a paired `WeightsFileSession` whose
  // `PendingRemote` is returned. Fails (incognito, disk error) → returns
  // null fd + null session; renderer falls back to in-memory Flatbuffer.
  OpenWeightsFile()
      => (mojo_base.mojom.File? writable_fd,
          pending_remote<WeightsFileSession>? session);
};

// Browser-side self-owned receiver, one per tempfile (= per `build()` call).
// Combines incremental capacity accounting and finalization in a single
// interface so all per-file state — tempfile, path, granted_bytes, origin —
// is local to one object. Pipe disconnect (renderer crash / mid-build abort /
// post-`Finalize` cleanup) triggers `~WeightsFileSessionImpl`, which returns
// any accumulated granted_bytes_ to the process-wide `OriginUsageTracker`
// and unlinks the tempfile (POSIX) / closes via `DeleteOnClose` (Windows).
interface WeightsFileSession {
  // Sync IPC before each write that would extend the file.
  // Browser re-runs §D/§B/§C on every call (see §5.3.2).
  [Sync]
  RequestCapacityChange(uint64 new_size) => (bool granted);

  // Called after all weights are written. Browser:
  //   1. fstat the tempfile; size > granted_bytes_ → ReportBadMessage.
  //   2. Reopen the tempfile read-only by path; unlink the path (POSIX) /
  //      set DeleteOnClose (Windows). Returns the sealed RO fd for mmap.
  // The session pipe is implicitly closed after this reply, which
  // self-destructs the browser-side session and releases per-origin budget.
  Finalize() => (mojo_base.mojom.ReadOnlyFile? sealed_file);
};
```

> **Option A considered and rejected**: a single `WebNNWeightsFileCreator`
> interface keyed by a per-file token in every method (`Write(token, …)`,
> `Finalize(token)`) would also support concurrent files, but it spreads
> per-file state across the creator and forces every call to validate the
> token. Modeling each file as its own self-owned session matches existing
> Mojo idioms (cf. `Tensor`, `Graph`) and keeps the renderer-side code
> path identical to a single-build scenario.

#### 5.3.2 Browser-side core logic (implementation summary)

`WeightsFileCreatorImpl` is a thin factory: it owns only `origin_` and
`is_incognito_`. `WeightsFileSessionImpl` is self-owned via
`MakeSelfOwnedReceiver` and owns `tempfile_`, `tempfile_path_`,
`granted_bytes_`, and `origin_` — every piece of per-file state lives in
one object. Multiple concurrent sessions per creator are fully independent.

```
OpenWeightsFile() on WeightsFileCreatorImpl:
  if is_incognito_: return (null fd, null pipe)
  (tempfile, path) = webnn::CreateWeightsFileWithPath()
            // §A headroom check only (free >= headroom); unrelated to capacity grants
  if !tempfile.IsValid(): return (null fd, null pipe)
  renderer_fd = tempfile.Duplicate()
  // Hand the writable fd to the session for the lifetime of this build.
  WeightsFileSessionImpl::Create(session_remote, std::move(tempfile),
                                 std::move(path), origin_)
  return (renderer_fd, session_remote)

RequestCapacityChange(new_size) on WeightsFileSessionImpl:
  // §D anti-tamper — re-fstat on every IPC
  current = tempfile_.GetLength()
  if current < 0 || uint64_t(current) > granted_bytes_:
    ReportBadMessage("renderer wrote past granted capacity"); return false
  if new_size <= granted_bytes_: return true        // shrink / no-op
  delta = new_size - granted_bytes_
  // §B per-session cap (4 GiB) — see §5.3.3 for naming
  if granted_bytes_ + delta > kMaxWeightsBytesPerContext: return false
  // §C per-origin (8 GiB) — process-global OriginUsageTracker
  if !OriginUsageTracker::TryReserve(origin_, delta): return false
  granted_bytes_ += delta
  return true

Finalize() on WeightsFileSessionImpl:
  current = tempfile_.GetLength()
  if current < 0 || uint64_t(current) > granted_bytes_:
    ReportBadMessage(...); return  // destructor cleans up tempfile + budget
  // mojo_base.mojom.ReadOnlyFile traits CHECK O_RDONLY on POSIX; dup(O_RDWR)
  // fails the check, so we reopen the path read-only and then unlink it.
  ro_fd = base::File(tempfile_path_, FLAG_OPEN | FLAG_READ | FLAG_WIN_NO_EXECUTE)
#if POSIX
  unlink(tempfile_path_)            // open fds keep the inode alive
#else  // Windows
  tempfile_.DeleteOnClose(true)     // reclaimed when both handles close
#endif
  tempfile_.Close()
  return ro_fd  // self-owned receiver tears down after this reply

~WeightsFileSessionImpl:
  if granted_bytes_ > 0:
    OriginUsageTracker::Release(origin_, granted_bytes_)
  if !tempfile_path_.empty():
    // Backstop: pipe disconnected before Finalize ran.
    base::ThreadPool::PostTask(BEST_EFFORT, base::DeleteFile(tempfile_path_))
```

Note: §A is not re-run inside `RequestCapacityChange`. The `OpenWeightsFile` call already ran a headroom check at creation time; subsequent extensions are bounded by §B/§C caps plus the global headroom. Adding a per-IPC `statvfs` would cost hundreds of µs × N constants per build for marginal benefit given §B/§C upper bounds.

#### 5.3.3 Accounting ownership and concurrent-build support (§B semantics)

In the in-renderer path, §B's "per-context 4 GiB" is enforced by `WeightsFileSessionImpl::granted_bytes_` **only** — `WebNNContextImpl::weights_bytes_` is never incremented (it is only touched by the GPU-process `WebNNContextImpl::CreateWeightsFile` path). As a result, `GraphImpl{Tflite,LiteRt}::DidCreateAndBuild` always passes `weights_bytes=0`, and there is no per-graph destructor release in the in-renderer path — the session pipe handles everything atomically.

Practical consequence: §B granularity is *per-session* (= per-build), not *per-context*. A single `navigator.ml` running N concurrent graph builds gets N independent 4 GiB ceilings; the cross-build aggregate is bounded by §C's 8 GiB per-origin cap.

**Concurrent-build correctness**: making the session interface self-owned per file (instead of multiplexing all files through a single `WeightsFileCreator` pipe) is what makes concurrent `build()` calls within one `navigator.ml` safe. Earlier iterations of this design used a single creator-bound `WeightsFileCapacityHost` and rejected overlapping `OpenWeightsFile` with `ReportBadMessage`, which would have killed the whole WebNN pipe for the frame on the second parallel `build()`. The current design makes each build's `OpenWeightsFile` spawn a fresh `WeightsFileSessionImpl`, so parallel builds never collide. Tightening §B to a true context-level aggregate would require cross-pipe accounting in `WebNNContextImpl`; this is intentionally not done — §C already caps the worst case at 8 GiB per origin, and per-build independence is the more useful invariant for graph compilation parallelism.

`FLAG_DELETE_ON_CLOSE` note: on POSIX the tempfile is `unlink`'d immediately after `Finalize` (open fds keep the inode alive until the renderer drops its mmap); on Windows `DeleteOnClose(true)` reclaims on close. Net result is identical — no on-disk persistence after the last fd closes — but the mechanism differs.

#### 5.3.4 Renderer-side (`GraphBuilderTflite::SerializeBuffer`)

```cpp
// graph_builder_tflite.cc:3287
auto GraphBuilderTflite::SerializeBuffer(base::span<const uint8_t> buffer)
    -> base::expected<BufferInfo, std::string> {
  // ... seek + align + SetLength (trim padding)

  if (session_.is_bound()) {
    base::CheckedNumeric<uint64_t> new_size_checked = offset;
    new_size_checked += buffer.size();
    uint64_t new_size = 0;
    if (!new_size_checked.AssignIfValid(&new_size))
      return base::unexpected("Weights file size overflow.");
    bool granted = false;
    if (!session_->RequestCapacityChange(new_size, &granted) || !granted)
      return base::unexpected("Weights file capacity request denied by browser.");
  }
  if (!weights_file_.WriteAtCurrentPosAndCheck(buffer))
    return base::unexpected("Failed to write weights file.");
  ...
}
```

**Streaming weights compatibility**: `RequestCapacityChange` only takes `new_size` — no global total needed. Each constant can be serialized and immediately dropped (`WebNNConstantOperand::Drop()`).

#### 5.3.5 TFLite / LiteRT side

No changes. The `weights_file` source changes from "writable fd returned by one-shot `CreateWeightsFile`" to "RO fd returned by `WeightsFileSession::Finalize`"; TFLite/LiteRT `mmap`s it transparently. `GraphImpl{Tflite,LiteRt}::CreateAndBuild` binds the `WeightsFileSession` `SharedRemote` on the context sequence and keeps two copies — one moved into the background-thread builder closure for `RequestCapacityChange` calls, the other into the reply closure (`DidBuildGraph`) so it can call `session->Finalize` once the worker returns. The second SharedRemote is moved into the `Finalize` reply closure as a keep-alive; once the reply fires and the closure drops, the pipe closes and the browser-side session self-destructs (releasing per-origin budget).

### 5.4 Security analysis

| Attack | Old `CreateWeightsFile` | SHM chunked (abandoned) | Capacity tracker (this design) |
|---|---|---|---|
| Out-of-bounds fd write | ❌ bookkeeping only | ✅ kernel SIGBUS | ✅ per-IPC `fstat` + ReportBadMessage |
| Requires total size upfront | Yes | Yes | No |
| Streaming weights compatible | ❌ | ❌ | ✅ |
| Renderer peak memory | All weights resident | All weights resident in SHM | Can drop each constant after write |
| Extra memory copy | — | ❌ SHM→tempfile full copy | ✅ None |
| Cross-platform | ✅ | ✅ | ✅ |

> **Why fd out-of-bounds write is no longer the threat it was**: the renderer holds a writable fd — same as FSA / OPFS — and the kernel will accept any `write()` syscall. The defense is not "renderer can't call `write()`"; it is "bytes written without a matching `RequestCapacityChange` can never reach the graph". See §5.4.1 for the full compromised-renderer threat model.

#### 5.4.1 Compromised renderer threat model

A compromised renderer holds a writable fd. The kernel has no knowledge of our IPC protocol — `write()` will succeed. This section enumerates *what the attacker can do / what happens / what the upper bound is*.

| Attacker action | Succeeds? | Consequence / upper bound |
|---|---|---|
| `write(fd, ...)` directly, skip `RequestCapacityChange` | ✅ Bytes really land in the file | Bytes can't enter the graph: `FinalizeWeightsFile` / next `RequestCapacityChange` runs §D `fstat`; `current_length > granted_bytes_` → `ReportBadMessage` + renderer killed. If renderer never calls either again, build never completes (self-DoS). |
| Repeatedly `write()` until ENOSPC | ✅ Single renderer can fill the disk during its lifetime | **Persistent damage = 0**: `FLAG_DELETE_ON_CLOSE` (POSIX unlinks at creation; Windows on close) reclaims the inode when the renderer exits. **Blast radius = contained**: once free disk < §A 10 GiB / 10% headroom, all subsequent `OpenWeightsFile` calls (any origin) return invalid → WebNN globally degrades to in-memory; GPU process does not crash. Same posture as OPFS / FSA — no web file API can prevent active writes by a compromised renderer. Hard upper bound = `min(physical free disk, sandbox RLIMIT_FSIZE)`. |
| `dup(fd)` or `SCM_RIGHTS` clone fd across processes | ✅ Kernel allows it | All clones point to the same inode; §D `fstat` measures the inode's true size regardless of which fd wrote. Consequences same as row above. |
| Clone the session pipe | ❌ | `pending_remote<WeightsFileSession>` is a one-shot Mojo endpoint issued by the browser; it cannot be cloned. Legitimate capacity growth still requires a sync IPC. |
| Open many concurrent sessions to bypass per-session §B (4 GiB) | ⚠️ Partial — each session legitimately gets its own 4 GiB ceiling | Bounded by §C: the same `OriginUsageTracker` covers all sessions for the origin, so total outstanding across N sessions is still ≤ 8 GiB. "Many concurrent sessions" is the legitimate code path for parallel `build()` calls in the same `navigator.ml`. |
| Never call `RequestCapacityChange` or `Finalize` | ✅ | Build never completes — pure self-DoS. Disk occupancy bounded by §A headroom + session-disconnect unlink. |
| Write past `granted_bytes_`, then call `RequestCapacityChange(new_size <= granted_bytes_)` to smuggle it | ❌ | §D `fstat` is independent of `new_size` — it checks `current_length` vs `granted_bytes_` first; any overrun → `ReportBadMessage`. |
| Smuggle tampered bytes into the final graph | ❌ | `Finalize` runs the same `fstat ≤ granted_bytes_` check; no RO fd is issued on failure → no `mmap` → no graph. |

**Net attacker payoff**: a compromised renderer can only achieve "DoS the disk and WebNN during renderer lifetime" — no data-integrity violation, no arbitrary code execution, no cross-origin impact. Disk is reclaimed on renderer exit. This matches the defense depth of OPFS / FSA under the same threat model; WebNN cannot and does not need to offer stronger guarantees within the web platform framework.

### 5.5 Costs

- **One sync IPC per weights extension**: ~N calls for N constants, ~50 µs each; negligible relative to graph-build time.
- **One extra `fstat` per IPC**: ~µs.
- **No extra memory copy**: saves up to 4 GiB of SHM→tempfile copying vs. the chunked-SHM design.

### 5.6 Out of scope

- **Plugging `storage::QuotaManager` in as the §C per-origin backend.** Long-term direction; `RequestCapacityChange` is designed as a frontend that can swap backends without touching renderer code.
- **A LiteRT in-memory API (`SetExternalWeightFromMemory`).** Would eliminate the tempfile entirely; requires upstream LiteRT cooperation.
- **Coupling the per-origin cap to disk size**, **collapsing the GPU path under §C.** Tracked as §4 #1 / #2.

### 5.7 Tracking

`crbug.com/XXXXXXX` (TBD). Recommend tagging `Security>WebNN` and `Component:WebNN`, Hotlist-Security-Severity-Low.

The old chunked-SHM implementation (commit `2881a74236`) is replaced by this design.
---

## Appendix A: `storage::QuotaSettings` reference

### A.1 `quota_settings.cc::CalculateNominalDynamicSettings()`

```cpp
const double kDefaultPerStorageKeyRatio = 0.75;

// Pool size = total disk × kPoolSizeRatio (default 0.6)
int64_t pool_size = total * kTemporaryPoolSizeRatio;

// Per-StorageKey cap = 75% of the pool
settings.per_storage_key_quota = pool_size * kPerStorageKeyTemporaryRatio;

// Minimum free space (aggressive-eviction trigger)
settings.must_remain_available =
    std::min(kMustRemainAvailableFixed,           // 1 GiB
             total * kMustRemainAvailableRatio);  // 1%

// Desired free space (start-compressing-quota trigger)
settings.should_remain_available =
    std::min(kShouldRemainAvailableFixed,           // 2 GiB
             total * kShouldRemainAvailableRatio);  // 10%
```

WebNN's headroom reuses this `min(fixed, ratio)` shape but with much more aggressive values (10 GiB / 10% vs 1 GiB / 1%); rationale in §3.2.1.

`per_storage_key_quota` on common devices:

| Total disk | per_storage_key_quota |
|---|---|
| 1 TiB | ~450 GiB |
| 256 GiB | ~115 GiB |
| 64 GiB | ~28 GiB |
| 16 GiB | ~7.2 GiB |

WebNN's per-origin cap is fixed at 8 GiB — far more conservative than this default.

### A.2 Incognito path

`CalculateIncognitoDynamicSettings()`:

```cpp
pool_size = physical_memory × (15% ~ 20%)   // randomized
per_storage_key_quota = pool_size / 3
```

WebNN's incognito handling: `WeightsFileCreatorImpl::CreateWeightsFile` returns an invalid `base::File` directly → in-memory path; **no file is created**. This is even more aggressive than storage quota's incognito handling, because incognito should leave no disk trace.

---

## Appendix B: Files touched

| File | Change |
|---|---|
| `services/viz/privileged/mojom/gl/gpu_host.mojom` | `CreateWebNNWeightsFile()` gains `uint64 required_bytes` |
| `components/viz/host/gpu_host_impl.{h,cc}` | Plumb `required_bytes` through to `webnn::CreateWeightsFile` (GPU-process path covers §A) |
| `services/webnn/public/mojom/webnn_context_provider.mojom` | `CreateWeightsFile()` gains `uint64 required_bytes`; new `ReleaseWeights(uint64 bytes)` |
| `services/webnn/host/weights_file_provider.{h,cc}` | §A: disk-space check + adaptive headroom (10 GiB / 10%, no in-process serialization) |
| `services/webnn/host/weights_file_creator_impl.{h,cc}` | §C: thin per-`navigator.ml` factory; `OpenWeightsFile` creates a tempfile and spawns a self-owned `WeightsFileSessionImpl` (no per-file state retained) |
| `services/webnn/host/weights_file_session_impl.{h,cc}` | Per-file self-owned session: owns `tempfile_` / `tempfile_path_` / `granted_bytes_` / `origin_`; implements `RequestCapacityChange` (§B + §C + §D) and `Finalize` (anti-tamper fstat + RO reopen by path + unlink). Per-origin `OriginUsageTracker` lives in this TU as a process singleton, with the per-origin cap (`WeightsFileCreatorImpl::kMaxBytesPerOrigin`) and per-session cap (`kMaxWeightsBytesPerContext`) both enforced here |
| `services/webnn/host/BUILD.gn` | Add `//url` dependency |
| `services/webnn/webnn_context_impl.{h,cc}` | §B: per-context `weights_bytes_granted_`; `CreateWeightsFile` callback now `(File, uint64 granted_bytes)`; new public `ReleaseWeightsBytes(uint64)` |
| `services/webnn/webnn_context_provider_impl.{h,cc}` | Plumb `required_bytes`; GPU-process path has no per-origin tracker — `CreateWeightsFile` carries a TODO (the path goes away once MLDrift delegate moves back to the renderer) |
| `services/webnn/webnn_context_provider_in_renderer.{h,cc}` | New `ReleaseWeights` plumbed to `WebNNWeightsFileCreator` mojom |
| `services/webnn/webnn_graph_impl.{h,cc}` | Base ctor gains `uint64 weights_bytes_granted = 0`; destructor calls `context_->ReleaseWeightsBytes` |
| `services/webnn/tflite/context_provider_tflite.{h,cc}` | Plumb `required_bytes` |
| `services/webnn/tflite/context_impl_tflite.{h,cc}` | Sum `constant_operands` byte sizes; `DidCreateWeightsFile` gains `uint64 granted_bytes`, plumbed into `GraphImplTflite::CreateAndBuild` |
| `services/webnn/tflite/context_impl_litert.{h,cc}` | Same as the tflite mirror |
| `services/webnn/tflite/graph_impl_tflite.{h,cc}` | `CreateAndBuild` / ctor gain `uint64 weights_bytes_granted`; `DidCreateAndBuild` explicitly calls `context->ReleaseWeightsBytes` on build failure |
| `services/webnn/tflite/graph_impl_litert.{h,cc}` | Same as tflite, LiteRT mirror |
| `services/webnn/webnn_test_environment.{h,cc}` | Test entry passes `url::Origin()` and `required_bytes` |
| `content/browser/browser_interface_binders.cc` | Per-host origin extraction (template specializations) |

Build verification: `autoninja -C out/Debug services/webnn:webnn_service services_unittests -j 40` passes (1078 steps, 3 min 18 s, 0 errors); `services_unittests --gtest_filter="WebNN*"` passes 365/365.
