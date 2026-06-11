# WebNN Weights File 磁盘使用与配额设计

## 1. 背景

WebNN 在执行图编译时会把 constant operand（权重）单独写到一个临时文件里，然后让 TFLite 通过 `mmap` 直接加载，避免再做一次内存拷贝。当前实现：

- 入口：[`services/webnn/host/weights_file_provider.cc`](../chromium/src/services/webnn/host/weights_file_provider.cc) `CreateTemporaryFile()` 直接调 `base::CreateTemporaryFile()`
- 路径：系统 temp 目录（`/tmp`、`%TEMP%`），**不**在 profile 的 storage partition 下
- 标志：`FLAG_DELETE_ON_CLOSE`，关闭即销毁
- 唯一已有的限制：incognito 模式跳过文件创建，走 in-memory Flatbuffer

### 1.1 存在的问题

1. 没有任何大小检查 —— 恶意站点构造巨大模型可以撑爆 `/tmp`
2. 没有 per-origin 记账 —— 一个 origin 可以反复构图占满磁盘
3. 没有 graceful fallback —— 磁盘满会直接让 build 失败

### 1.2 威胁模型

- **信任边界**：renderer 被视作 untrusted compromise boundary。GPU process 内的 WebNN service 与 browser process 内的 broker 是 trusted。
- **攻击面**：被攻陷的 renderer / 恶意页面可以
  1. 在单个 origin 内并发构造大量大模型，目标是耗尽 `/tmp`（DoS）
  2. 在多个 frame / worker 间共享同一 origin，规避 per-frame 限制
  3. 申请 `required_bytes` 后实际写入更多数据（**fd 级 over-write**，详见 §5 followup）
  4. 跨 origin 协同（多个恶意 origin 同时占满各自配额）
- **不在范围**：跨进程磁盘消耗（其他 Chrome 子进程、其他程序）由 OS 级 disk pressure 处理，浏览器只能用 headroom 缓冲。

---

## 2. 设计要点

### 2.1 为什么不接 `storage::QuotaManager`

`storage::QuotaManager`（`//storage/browser/quota/`）按 `StorageKey`（origin + top-level site）管理配额，被 IndexedDB / Cache API / File System Access / Service Worker 等使用。WebNN **不应该**直接接入：

- WebNN service 跑在 GPU 进程，`QuotaManager` 在 Storage Service / Browser 进程，要查必须 Mojo 绕一圈，会阻塞 graph build 关键路径
- weights 是 `DELETE_ON_CLOSE` 的临时计算产物，不是"持久用户数据"，语义上不属于 Quota 管的那一类
- 文件落在系统 temp 而非 profile 目录，根本不在 quota 账本里
- 真正的风险是 **/tmp DoS**，跟 per-origin storage quota 关心的事情不是同一个问题

### 2.2 In-memory fallback 已经存在

TFLite 后端已经接受 invalid `base::File` 走 in-memory Flatbuffer 路径（incognito 就是这么做的）。fallback 通道已经存在，只需要在 `CreateTemporaryFile()` 里：

- 创建前用 `base::SysInfo::AmountOfFreeDiskSpace()` 检查剩余空间
- 写入失败（ENOSPC / `Initialize()` 返回非 valid）就返回空 `base::File()`

renderer 侧自然就退回 in-memory，无需新协议。

注意：in-memory Flatbuffer 受 **< 2 GiB** 限制（Flatbuffer offset 是 int32），所以"大图退回 in-memory"不能无限退。

### 2.3 分层防护方案

| 层 | 做法 | 解决的问题 |
|---|---|---|
| **A** | 写文件前 `AmountOfFreeDiskSpace()` + 失败返回 invalid file 让 TFLite 走 in-memory | 磁盘满导致 build 失败/崩溃 |
| **B** | `WebNNContextImpl` 加 **per-context 上限**（4 GiB），累加各 graph 的 weights 文件大小，超限直接拒绝或退 in-memory | 单 tab DoS `/tmp` |
| **C** | browser 进程加 **per-origin 总量上限**（8 GiB），通过 `WeightsFileCreatorImpl` broker 维护 origin tracker | 跨 context 的 origin 级滥用 |
| D（未做） | 真要接 `QuotaManager` 的话，应作为独立 storage type（如 `kWebNNWeights`），并通过 browser 进程 broker | 与 Web Storage 配额体系对齐 |

A + B + C 解决约 80% 的实际问题，已经实现。

---

## 3. 实现

### 3.1 Mojo 接口

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

### 3.2 §A — 磁盘空间检查

[`services/webnn/host/weights_file_provider.{h,cc}`](../chromium/src/services/webnn/host/weights_file_provider.cc)

`CreateWeightsFile(required_bytes, cb)` 在创建文件前调用 `AmountOfFreeDiskSpace()` 与 `AmountOfTotalDiskSpace()`。headroom 形状对齐 `storage::QuotaSettings::must_remain_available`（`min(固定, 比例)`），但**取值比 storage 子系统更激进**：

```cpp
inline constexpr uint64_t kWeightsFileMustRemainAvailableBytes = 10ull << 30;  // 10 GiB
inline constexpr double   kWeightsFileMustRemainAvailableRatio = 0.10;         // 10%

// headroom = min(固定 reserve, 总磁盘 × ratio)
const uint64_t headroom = std::min<uint64_t>(
    kWeightsFileMustRemainAvailableBytes,
    static_cast<uint64_t>(static_cast<double>(*total_bytes) *
                          kWeightsFileMustRemainAvailableRatio));
```

如果 `free < required_bytes + headroom`，返回 invalid `base::File`，触发现有 in-memory fallback。

#### 3.2.1 Headroom 推导

A 层 disk check 在 ThreadPool 上**完全并行**：N 个并发 call 看到同一个 `free` 快照，每个只检查 `free ≥ rᵢ + H`，但实际写入是 `W = Σrᵢ`。要让 N 个 call 全部通过又不撞 ENOSPC，必须满足

```
H ≥ W − max(rᵢ)
```

代入上层封顶 `W ≤ kMaxBytesPerOrigin = 8 GiB`，且 `max(rᵢ)` 没有下界（renderer 可把单 origin 的 8 GiB 切成任意多的小 graph），最坏情形 `max(rᵢ) → 0`，所以

```
H ≥ kMaxBytesPerOrigin = 8 GiB
```

具体例子：单 origin 同时开 4 个 context、每个 2 GiB → W=8 GiB, max=2 GiB → 至少需要 6 GiB headroom；切成 8 个 1 GiB → 至少 7 GiB；切成大量小 graph → 趋近 8 GiB。

10 GiB 在 8 GiB 下限上多留 2 GiB 缓冲，给跨进程磁盘消耗（其他 Chrome 子进程、别的程序）和 fd 移交到写完之间的窗口。**降到 2 GiB / 2% 会让单 origin 切多个 ≥2 GiB 模型时直接 ENOSPC**，这是设计上必须避免的失败模式（renderer 端没有重试逻辑）。

跨 origin 的 worst-case（2 origin 同时拿满 → W=16 GiB, max=4 GiB → H_min=12 GiB）超出了 10 GiB——这是已知 gap，由 per-origin cap 把"两个并发恶意 origin 各自达 8 GiB"压成攻击者必须在多 frame/worker 间精确同步的窄窗口；正常使用不会触发。

不同设备实际 headroom：

| 总磁盘 | min(10 GiB, 10% × total) |
|---|---|
| 1 TiB | 10 GiB |
| 256 GiB | 10 GiB |
| 100 GiB | 10 GiB |
| 64 GiB | 6.4 GiB |
| 16 GiB | 1.6 GiB |
| 8 GiB | 800 MiB |

#### 3.2.2 并发模型与 TOCTOU

`AmountOfFreeDiskSpace` 是**瞬时快照**。两个并发的 `CreateWeightsFile` 可能都看到同样的 free space、都通过检查、然后第二个 renderer mmap/write 到一半 `ENOSPC`：

```
Context A: free=3 GB, need=2 GB+headroom -> 通过, 拿到 fd
Context B: free=3 GB, need=2 GB+headroom -> 通过, 拿到 fd
两个 fd 都交给 renderer 后,第二个写到一半 ENOSPC
```

考虑过两个替代方案，均否决：

**真预分配（`base::AllocateFileRegion` / `SetLength`）**

1. **破坏现有 caller 语义**：[`graph_builder_tflite.cc`](../chromium/src/services/webnn/tflite/graph_builder_tflite.cc) 用 `weights_file_.GetLength()` 当作"已写入字节数"来计算下一个 buffer 的 append offset。预分配后 `GetLength()` 直接返回 `required_bytes`（如 2 GiB），第一个 weights buffer 就被写到 2 GiB **之后**，TFLite 加载时按 metadata offset 读到的是预分配的全零区，模型坏掉。
2. **修复成本高**：要给 `GraphBuilderTflite` 加 `weights_bytes_written_` 成员，所有 `weights_file_.GetLength()` / `SetLength` 调用都要改写或加守卫（直接 `SetLength(64)` 在 2 GiB 预分配文件上等于 `ftruncate(64)`，会立刻释放整个预留）。改动面大、回归风险高。

**进程内串行（单个 `SequencedTaskRunner` 跑所有 `CreateTemporaryFile`）**

- 即使 sequenced runner 上跑的只是 metadata 级 syscall（毫秒级），多个不同的 origin / context 在视感上变成"上一个网页的 weights 写完，下一个网页才能开始检查"，认知负担大。
- 单进程内 TOCTOU 窗口的实际宽度 ≈ 一次 `mkstemp` 调用的时长（亚毫秒），并发命中只在攻击者刻意构造的微秒级竞态下才有意义；同时段被 worst-case 并发耗尽的需要 ≥ 16 GiB outstanding，本身已被 per-origin / per-context 封顶。
- headroom 提到 10 GiB / 10% 后，跨进程消耗也由同一个机制吸收，串行不再贡献额外保护。

**实际方案：默认 ThreadPool 并发 + 大 headroom + 上层封顶**

[`weights_file_provider.cc`](../chromium/src/services/webnn/host/weights_file_provider.cc) 直接走默认 `base::ThreadPool::PostTaskAndReplyWithResult`：

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

并发模型：

- 多个 origin / context 各自的 `CreateTemporaryFile` 在线程池上**完全并行**，没有共享互斥
- 每个调用独立做 `AmountOfFreeDiskSpace` + `AmountOfTotalDiskSpace` + `mkstemp`
- 上层 [`WeightsFileCreatorImpl`](../chromium/src/services/webnn/host/weights_file_creator_impl.cc)（per-origin 8 GiB cap）和 [`WebNNContextImpl`](../chromium/src/services/webnn/webnn_context_impl.cc)（per-context 4 GiB cap）保证 outstanding 写入有硬上限
- 10 GiB / 10% headroom 吸收"`AmountOfFreeDiskSpace` 看到的快照"和"renderer 实际写完"之间的全部 outstanding 数据 + 跨进程消耗

最坏情形：

- 1 个完全占满的恶意 origin = 8 GiB outstanding；要求 free ≥ 8 GiB + 10 GiB headroom = 18 GiB。在 1 TiB 盘上轻易满足。
- 2 个并发完全占满 origin = 16 GiB outstanding；要求 free ≥ 26 GiB。仍然合理。
- 8 GiB 小盘：headroom = 800 MiB，单个完整 8 GiB origin 请求会因 `required + headroom > free` 被拒——这是合理结果，小盘本就跑不动大模型。

#### 3.2.3 残留风险

| 风险 | 处理 |
|---|---|
| 跨进程磁盘消耗（其他 Chrome 子进程、别的程序占空间） | 靠 10 GiB / 10% headroom 吸收。OS 级问题，浏览器无法完全防御。 |
| `base::File` 移交 renderer 到写入完成有秒级窗口 | 同样靠 headroom |
| 文件系统/OS 拒绝写入 | renderer 写失败 → 现有 in-memory fallback 路径接管 |
| 可写 fd 给 renderer 后无字节数硬上限 | 见 §5 followup |

### 3.3 §B — Per-context 上限

[`services/webnn/webnn_context_impl.{h,cc}`](../chromium/src/services/webnn/webnn_context_impl.cc)

```cpp
static constexpr uint64_t kMaxWeightsBytesPerContext = 4ull << 30;  // 4 GiB
uint64_t weights_bytes_granted_ = 0;
```

`WebNNContextImpl::CreateWeightsFile(required_bytes, cb)` 在转发给 `ContextProviderTflite` / `WebNNContextProviderImpl` 之前先做溢出检查；获得 invalid file 时归还配额。

### 3.4 §C — Per-origin 上限

[`services/webnn/host/weights_file_creator_impl.{h,cc}`](../chromium/src/services/webnn/host/weights_file_creator_impl.cc)

```cpp
static constexpr uint64_t kMaxBytesPerOrigin = 8ull << 30;  // 8 GiB
```

进程内 `OriginUsageTracker`（lock-guarded `std::map<url::Origin, uint64_t>`）在 `WeightsFileCreatorImpl` 实例间共享：

- `WeightsFileCreatorImpl` 的生命周期 = 单个 self-owned mojo pipe（per-frame / per-worker）
- `CreateWeightsFile` 时 `TryReserve()`，文件失败或实例销毁时 `Release()`
- 跨 frame / worker 共享同一 origin 的额度

`kMaxBytesPerOrigin` 选 8 GiB 的考量：storage quota 默认给单个 `StorageKey` 的额度是 `total_disk × 0.6 × 0.75 = total_disk × 45%`（参见附录 A）。WebNN weights 是临时计算产物（`DELETE_ON_CLOSE`），没必要追到 45%。固定 8 GiB 比 quota 默认保守得多，可放心作为防滥用硬上限。

### 3.5 配额释放生命周期

三类释放（按时机即时性递减）：

1. **Per-context（intra-process，立即）**：`WebNNGraphImpl` 析构时同步调用 `WebNNContextImpl::ReleaseWeightsBytes(weights_bytes_granted_)`，把 §B 的 `weights_bytes_granted_` 减回去。`weights_bytes_granted` 由 `CreateWeightsFile` 回调通过 `granted_bytes` 参数透传到 `GraphImplTflite` / `GraphImplLiteRt` 构造函数，再传到 `WebNNGraphImpl` 基类成员，无 mojo IPC。
2. **Per-origin（cross-process，graph 销毁触发显式 mojom）**：同样在 graph 析构时，`WebNNContextImpl::ReleaseWeightsBytes` 还会在 main task runner 上 post 一个 `WebNNWeightsFileCreator::ReleaseWeights(bytes)` Mojo 调用给 browser-side `WeightsFileCreatorImpl`，把 §C 的 `OriginUsageTracker` 减回去。和"等 pipe 关闭再统一退还"相比，**配额可以在 graph 一销毁就立刻被同 origin 的下一个 graph 复用**，避免长寿命 pipe（如 Service Worker / shared worker）持续占着已经不用的配额。
3. **Pipe-disconnect 兜底**：`WeightsFileCreatorImpl` 析构时把残余 `reserved_bytes_` 一次性归还给 `OriginUsageTracker`。覆盖三种异常：(a) renderer crash 没机会发 `ReleaseWeights`；(b) 浏览器 kill renderer；(c) 任何未来路径漏调 `ReleaseWeights` 的 bug。

#### 3.5.1 防御 over-release

`WeightsFileCreatorImpl::ReleaseWeights(bytes)` 校验 `bytes ≤ reserved_bytes_`，超出时调用 `mojo::ReportBadMessage("WebNN: ReleaseWeights over-release")` 杀掉 renderer，**绝不**把 `OriginUsageTracker` 减成负数（也不能减成负数，那意味着别的 frame 的额度被"还"掉，反而放行恶意 origin 占用 > 8 GiB）。

`WebNNContextImpl::ReleaseWeightsBytes(bytes)` 同理 `CHECK_GE(weights_bytes_granted_, bytes)`——这一层在 GPU 进程内，不可能被 renderer 直接驱动，CHECK 比 BadMessage 合适。

#### 3.5.2 Graph 构造失败的 race-free 路径

`GraphImplTflite::DidCreateAndBuild` / `GraphImplLiteRt::DidCreateAndBuild` 收到背景线程的 `base::expected` 后：

- 若 `compute_resources.has_value()`：构造 `GraphImpl`，把 `weights_bytes_granted` 灌进基类成员，等析构释放。
- 若 `!compute_resources.has_value()`：**没有 `GraphImpl` 会被构造**，析构释放走不通；在原地立刻 `context->ReleaseWeightsBytes(weights_bytes_granted)`。否则 reservation 就一直挂在 `weights_bytes_granted_` 里直到 context 销毁。

`if (!context) return` 分支不显式释放：context 已经销毁意味着 §B 账本随之销毁，§C 走 pipe-disconnect 兜底。

#### 3.5.3 为什么不让 renderer 主动 release

> *(Internal-only note. Renderer 是 untrusted compromise boundary。)*

可能想到的"省事"方案：让 renderer 在 graph 销毁时自己发 `ReleaseWeights`。否决理由：

1. **被攻陷 renderer 可任意 over-release**，把 `OriginUsageTracker` 减空，让自己（或同 origin 的其他 frame）越过 8 GiB 上限。`ReportBadMessage` 是事后补救，仍然有窗口。
2. **跨 origin/frame 共享 `OriginUsageTracker`**：单个 renderer 的 release 会影响 *其他 frame* 的配额。Untrusted 不能写共享状态。
3. **磁盘是进程全局资源**：browser 进程才能权威知道何时回收。

因此设计上 release 必须由 **GPU process 内的 `WebNNGraphImpl` 析构** 触发，不接受 renderer 主动 RPC。

### 3.6 调用链贯通

预留路径：

```
ContextImplTflite::CreateGraphImpl
  -> 求和 constant_operands 的 ByteSpan().size() 得到 required_bytes
  -> WebNNContextImpl::CreateWeightsFile(required_bytes, cb)         [§B 检查]
       -> ContextProviderTflite::CreateWeightsFile(required_bytes, cb)
            -> mojom::WebNNWeightsFileCreator.CreateWeightsFile(required_bytes)  [Mojo]
                 -> WeightsFileCreatorImpl::CreateWeightsFile         [§C 检查]
                      -> webnn::CreateWeightsFile(required_bytes, cb) [§A 检查]
                           -> CreateTemporaryFile()
                 cb(file, granted_bytes)  // granted_bytes 透传回 GraphImpl
```

释放路径（graph 销毁触发）：

```
~WebNNGraphImpl
  -> WebNNContextImpl::ReleaseWeightsBytes(weights_bytes_granted_)   [§B 减账]
       -> if (is_context_provider_in_renderer_):                      [in-renderer 路径]
            WebNNContextProviderInRenderer::ReleaseWeights(bytes)
              -> mojom::WebNNWeightsFileCreator.ReleaseWeights(bytes) [Mojo]
                   -> WeightsFileCreatorImpl::ReleaseWeights          [§C 减账]
                        -> OriginUsageTracker::Release(origin, bytes)
       -> else:                                                       [GPU-process 路径]
            // TODO: viz::mojom::GpuHost::CreateWebNNWeightsFile 不带 origin，
            // browser 端没有 per-origin reservation 可还。TFLite/LiteRT MLDrift
            // GPU delegate 计划迁出 GPU process 到 renderer 后此分支自然消失；
            // 在此之前 GPU 路径的 weights 文件只能等 fd 关闭被 unlink 回收。
```

Pipe-disconnect 兜底：`~WeightsFileCreatorImpl` 把残余 `reserved_bytes_` 一次性归还 `OriginUsageTracker`。

### 3.7 Origin 来源（content/browser）

[`content/browser/browser_interface_binders.cc`](../chromium/src/content/browser/browser_interface_binders.cc)

```cpp
// RenderFrame:
host->GetLastCommittedOrigin()

// Worker hosts (模板特化):
DedicatedWorkerHost / SharedWorkerHost -> GetWorkerStorageKey().origin()
ServiceWorkerHost                       -> GetBucketStorageKey().origin()
```

---

## 4. 已知遗留问题

1. **Per-origin cap 是固定值**：未跟磁盘大小联动。可以做 `min(8 GiB, 总磁盘 × N%)`，但 `OriginUsageTracker` 需改为异步或在 `Create` 时缓存查询结果，权衡复杂度后保留固定值。
2. **GPU 路径绕过 per-origin cap**：LiteRT GPU inference 跑在 GPU process 里，通过 `viz::GpuHost::CreateWebNNWeightsFile` → `webnn::CreateWeightsFile` 直接创建文件，**不经过 `WeightsFileCreatorImpl`，也没传 origin**（见 [`gpu_host.mojom`](../chromium/src/services/viz/privileged/mojom/gl/gpu_host.mojom) `CreateWebNNWeightsFile(uint64 required_bytes)` 与 [`gpu_host_impl.cc`](../chromium/src/components/viz/host/gpu_host_impl.cc) `CreateWebNNWeightsFile`）。结果是同一个 origin 走 GPU backend 时只受 §A (disk headroom) + §B (per-context 4 GiB) 限制，§C (per-origin 8 GiB) 不生效。
   - 修复方向：把 origin 加进 `gpu_host.mojom::CreateWebNNWeightsFile` 参数；把 `OriginUsageTracker` 抽成 browser-process 进程级单例，让 `GpuHostImpl::CreateWebNNWeightsFile` 和 `WeightsFileCreatorImpl::CreateWeightsFile` 共享同一份 per-origin 账本。
   - 当前不修是因为：GPU 路径单 origin 实际并发 context 数受 mojo channel 与 browser-side throttling 隐式封顶（≤ 2 时，per-context 4 GiB × 2 = 8 GiB 等价于 §C），这只是隐式不变式，**未来如放宽 context 并发应同步补上 §C**。
3. **可写 fd 直接交给 renderer，无 fd 级 hard cap**（reviewer 指出）：当前 `WebNNWeightsFileCreator::CreateWeightsFile` 把 browser 创建的 `base::File` 句柄发回 renderer，被攻陷的 in-renderer TFLite/LiteRT 可以请求 `required_bytes = 1` 然后 `write()` 任意 GBs。当前缓解（`must_remain_available` headroom + `DELETE_ON_CLOSE` + per-origin/per-context cap）不在 fd 级强制，最坏情况下单个 origin 可在自己被允许的磁盘片内写到 ENOSPC headroom 边界。
   - 修复方向见 §5 [Followup CL](#5-followup-cl跨平台-fd-级-hard-cap)。
   - **GPU-process 路径不受影响**：fd 留在 GPU 进程，renderer 拿不到；`required_bytes` 在 GPU 进程内由 `WebNNContextImpl` 自行从 `WebNNConstantOperand::ByteSpan().size()` 累加，与实际写入字节同源，无法分裂。
4. **TODO**：`crbug.com/507502295` — Use file manager for weights files。

### 4.1 与 `storage::QuotaManager` 的差异

| 维度 | `QuotaManager` | WebNN weights |
|---|---|---|
| 并发仲裁 | 单 sequence 串行 + 内存 pending reservation 表 | 不串行；并发由 per-origin (8 GiB) + per-context (4 GiB) 上限封顶，10 GiB / 10% headroom 吸收快照漂移 |
| 跨进程一致性 | 仅本浏览器 storage service | 仅本浏览器进程；跨进程靠 headroom 吸收 |
| 释放时机 | 内存表显式 commit/release | graph 销毁即时（§B 同步；§C 显式 `ReleaseWeights` mojom），pipe 关闭兜底释放 §C 残余 |
| 失败语义 | 返回 quota error，调用方决定 | 返回 invalid file，自动 fallback in-memory |

WebNN 没采用真预分配（`fallocate` / `SetEndOfFile`）的原因见 §3.2.2：会破坏 `GraphBuilderTflite` 现有的 `GetLength()` append 语义；改用大 headroom + 上层封顶是更小的改动。

---

## 5. Followup CL：跨平台 fd 级 hard cap

### 5.1 问题

§A/§B/§C 都是**簿记型**防护：browser 信任 in-renderer TFLite/LiteRT 不会写超过 `required_bytes`。被攻陷 renderer 可以 `CreateWeightsFile(required_bytes=1)` 通过所有上层检查，然后对返回的可写 fd `write()` 任意字节，最坏写满 §C cap 减 headroom 那么多。

POSIX 没有 per-fd 写入字节配额（`RLIMIT_FSIZE` 是进程级）。Linux 可以 `memfd_create` + `F_SEAL_GROW` 在 fd 上做 hard cap，但 **Windows / macOS 没有等价机制**——任何可写 file handle 都能 `SetEndOfFile`/`ftruncate` 扩大文件。

`base::WritableSharedMemoryRegion` 是天然的跨平台 hard cap：

| 平台 | 后端 | Cap 强制方 |
|---|---|---|
| Linux / Android / ChromeOS | `memfd_create` + `F_SEAL_SHRINK \| F_SEAL_GROW` | Kernel 拒绝 grow，越界写 `SIGBUS` |
| Windows | 命名 section（pagefile-backed），`CreateFileMapping(SECTION_QUERY \| FILE_MAP_WRITE)` | Section 大小创建时固定，越界访问 access violation |
| macOS | Mach VM region / POSIX shm | 大小创建时固定，越界访问 `EXC_BAD_ACCESS` |

跨平台**无任何代码分支**：`base::WritableSharedMemoryRegion::Create(N)` 在每个平台内部都做对应的 syscall；renderer 端 `WritableSharedMemoryMapping::GetMemoryAsSpan<uint8_t>()` 给出的 span 大小就是 `N`，越界由 kernel 强制。

### 5.2 SharedMemory 的 2 GiB 限制：必须分块

`base::PlatformSharedMemoryRegion::Create()` 在所有平台都硬编码 `INT_MAX` 上限：

| 平台 | 文件 | 检查 |
|---|---|---|
| POSIX (Linux/CrOS) | `base/memory/platform_shared_memory_region_posix.cc:183` | `if (size > std::numeric_limits<int>::max()) return {};` |
| Android | `..._android.cc:53,138` | 同上，rounded_size 也再查一次 |
| macOS | `..._apple.cc:30` | 同上 |
| Windows | `..._win.cc` | 同上 |

`std::numeric_limits<int>::max() = 2³¹ − 1 ≈ 2 GiB − 1 byte`。任何 `size ≥ 2 GiB` 的 `WritableSharedMemoryRegion::Create()` 返回 invalid region，无 fallback。

但 §B per-context 上限是 **4 GiB**、§C per-origin 是 **8 GiB**，覆盖到了 `required_bytes ≥ 2 GiB` 的合法场景：

| `required_bytes` | 单 region 设计 |
|---|---|
| < 2 GiB | ✅ 一个 region 装得下 |
| 2 – 4 GiB | ❌ 单个 `WritableSharedMemoryRegion::Create()` 直接失败 |
| ≥ 4 GiB | ❌ 同上（且超 §B） |

额外约束：`SharedMemorySecurityPolicy::kTotalMappedSizeLimit = 32 GiB`（`base/memory/shared_memory_security_policy.cc:27`）是 process-wide 映射上限，多个并发大 region 也会撞这个。

**解决：把请求拆成 N 个 ≤ `kChunkSize` 的 chunk，每个 chunk 由 kernel 独立强制大小。**

`kChunkSize` 选择 1 GiB：

- 在 INT_MAX 下方留 1 GiB 安全边距
- per-context 4 GiB → 最多 4 chunks，mojom array 开销可忽略
- 4 chunks × 1 GiB = 4 GiB ≪ 32 GiB 进程级上限，并发空间充裕
- page-aligned (`base::SharedMemorySecurityPolicy::AlignWithPageSize` 内部还会 round up)

### 5.3 设计：browser 持有 fd，renderer 通过分块的 size-capped SHM 写入

核心思路：**renderer 永远不持有可写文件句柄**。renderer 看到的是一组每块大小被 kernel 钉死的 `WritableSharedMemoryRegion`；写完后由 browser 把所有 chunk 顺序刷到只读临时文件，发回 RO 句柄给 LiteRT/TFLite mmap。

#### 5.3.1 Mojo 接口（替换当前 `CreateWeightsFile`）

```mojom
interface WebNNWeightsFileCreator {
  // Browser:
  //   1. 配额检查（§A/§B/§C），原 CreateWeightsFile 的语义
  //   2. 分配 ceil(required_bytes / kChunkSize) 个 region；前 N-1 个 size = kChunkSize，
  //      最后一个 size = required_bytes - (N-1) * kChunkSize
  //   3. 在 browser 进程创建 tempfile（FLAG_DELETE_ON_CLOSE），不发给 renderer
  //   返回 null 表示拒绝（over quota / not enough disk / SHM 分配失败）。
  AllocateWeightsBuffer(uint64 required_bytes)
      => (array<mojo_base.mojom.WritableSharedMemoryRegion>? regions,
          uint64 granted_bytes);

  // Renderer 写完所有 chunks 后调用。`bytes_per_chunk[i]` 是第 i 个 chunk 的实际写入字节数，
  // 必须满足 bytes_per_chunk[i] <= regions[i].GetSize()，否则 ReportBadMessage。
  // Browser:
  //   1. 顺序把每个 region 的前 bytes_per_chunk[i] 字节写入 tempfile（offset 累加）
  //   2. 关掉所有 region 句柄
  //   3. 返回 RO file handle，renderer 喂给 LiteRT::ScopedFile
  FinalizeWeightsBuffer(array<uint64> bytes_per_chunk)
      => (mojo_base.mojom.ReadOnlyFile? sealed_file);

  ReleaseWeights(uint64 bytes);
};
```

`AllocateWeightsBuffer` + `FinalizeWeightsBuffer` 的 mojo `AssociatedRemote` 状态由 browser-side `WeightsFileCreatorImpl` 持有：从 allocate 到 finalize 之间，browser 端保留 `tempfile_` 与 `regions_`，pipe 中途 disconnect 由析构兜底（参见 §3.5 Pipe-disconnect 兜底）。

#### 5.3.2 Renderer 侧改造（`GraphBuilderTflite`）

当前用 `weights_file_.WriteAtCurrentPosAndCheck` + `GetLength()` cursor + 偶尔 `SetLength` 缩容（[`graph_builder_tflite.cc#L3258-L3312`](../chromium/src/services/webnn/tflite/graph_builder_tflite.cc#L3258-L3312)）。改造后：

```cpp
class GraphBuilderTflite {
  std::vector<base::WritableSharedMemoryMapping> chunks_;
  uint64_t total_capacity_ = 0;       // = sum(chunks_[i].size())
  uint64_t weights_bytes_written_ = 0;

  bool AppendWeights(base::span<const uint8_t> buf) {
    auto end = base::CheckedNumeric<uint64_t>(weights_bytes_written_) + buf.size();
    uint64_t end_value;
    if (!end.AssignIfValid(&end_value) || end_value > total_capacity_) {
      return false;  // build 失败，无 ENOSPC
    }
    while (!buf.empty()) {
      const size_t chunk_idx = weights_bytes_written_ / kChunkSize;
      const size_t in_chunk_offset = weights_bytes_written_ % kChunkSize;
      auto chunk_span = chunks_[chunk_idx].GetMemoryAsSpan<uint8_t>();
      const size_t available = chunk_span.size() - in_chunk_offset;
      const size_t to_copy = std::min<size_t>(buf.size(), available);
      base::span_copy(chunk_span.subspan(in_chunk_offset, to_copy),
                      buf.first(to_copy));
      weights_bytes_written_ += to_copy;
      buf = buf.subspan(to_copy);
    }
    return true;
  }

  uint64_t WeightsBytesWritten() const { return weights_bytes_written_; }
};
```

这是 §3.2.2 "为什么不做真预分配"里识别出来的同一处改动（消除对 `GetLength()` 的依赖），所以不算新增 scope。`AppendWeights` 内部 while 循环处理跨 chunk 边界的写入；上层 metadata offset 仍然用 `WeightsBytesWritten()`，跟原 `GetLength()` 语义一致。

#### 5.3.3 LiteRT/TFLite 侧改造

零改动。`GraphImplLiteRt::ComputeResources::Create` 仍然走：

```cpp
self->weights_file_ = std::make_unique<::litert::ScopedFile>(
    build_graph_result.weights_file.TakePlatformFile());
compilation_options.SetExternalWeightScopedFile(*self->weights_file_, ...);
```

只是 `weights_file` 来源从"`CreateWeightsFile` 返回的可写 fd"变成"`FinalizeWeightsBuffer` 返回的 RO fd"。LiteRT 透明 mmap，行为不变。

### 5.4 代价

- **额外一次 in-browser 内存→磁盘拷贝**，量级 ≤ `kMaxWeightsBytesPerContext = 4 GiB`。NVMe 上 1 GiB 模型大约 +300 ms–1 s graph build 延迟。一次性成本，不影响推理。
- **Handoff 瞬间峰值内存**：所有 chunks（pagefile/swap，最多 4 GiB）+ tempfile（pagecache）同时存在；pagecache 共享缓解但不消除。
- **多一次 Mojo round-trip**：从一次 `CreateWeightsFile` 拆成 `AllocateWeightsBuffer` + `FinalizeWeightsBuffer`，array<region> 序列化 ≤ 4 个 fd，开销可忽略。
- **Renderer 端编排略复杂**：跨 chunk 的 cursor + per-chunk bytes_written 数组，约 30 行额外逻辑。

### 5.5 不在范围

- **LiteRT 加内存 API（`SetExternalWeightFromMemory(span<const uint8_t>)`）**：长期目标，可消除整个 tempfile 拷贝步骤——renderer 写完所有 chunks 转 RO mapping，指针直接喂给 LiteRT。需要 LiteRT 上游协作，单独 file feature request。先不阻塞这个 followup。
- **Per-origin cap 跟磁盘联动**、**GPU 路径合入 §C**：与 fd 级 hard cap 正交，保持现有 §4 #1/#2 的 followup 计划。

### 5.6 跟踪

`crbug.com/XXXXXXX`（待开）。建议挂 `Security>WebNN` 和 `Component:WebNN`，Hotlist-Security-Severity-Low（攻击边界仅限单 origin 自己的磁盘片，磁盘 headroom 兜底，无 cross-origin 影响）。

---

## 附录 A：`storage::QuotaSettings` 参考

### A.1 `quota_settings.cc::CalculateNominalDynamicSettings()`

```cpp
const double kDefaultPerStorageKeyRatio = 0.75;

// 池子大小 = 总磁盘 × kPoolSizeRatio (默认 0.6)
int64_t pool_size = total * kTemporaryPoolSizeRatio;

// 单 StorageKey 上限 = 池子的 75%
settings.per_storage_key_quota = pool_size * kPerStorageKeyTemporaryRatio;

// 必须留出的最小空闲（aggressive eviction 触发线）
settings.must_remain_available =
    std::min(kMustRemainAvailableFixed,           // 1 GiB
             total * kMustRemainAvailableRatio);  // 1%

// 期望留出的空闲（开始压缩 quota 触发线）
settings.should_remain_available =
    std::min(kShouldRemainAvailableFixed,           // 2 GiB
             total * kShouldRemainAvailableRatio);  // 10%
```

WebNN 的 headroom 直接复用这套 `must_remain_available` 的 `min(固定, 比例)` 模式，但取值激进得多（10 GiB / 10% vs 1 GiB / 1%），原因见 §3.2.1。

`per_storage_key_quota` 在不同设备的近似值：

| 总磁盘 | per_storage_key_quota |
|---|---|
| 1 TiB | ~450 GiB |
| 256 GiB | ~115 GiB |
| 64 GiB | ~28 GiB |
| 16 GiB | ~7.2 GiB |

WebNN per-origin cap 固定 8 GiB，比这套默认保守得多。

### A.2 Incognito 路径

`CalculateIncognitoDynamicSettings()`：

```cpp
pool_size = physical_memory × (15% ~ 20%)   // 随机化
per_storage_key_quota = pool_size / 3
```

WebNN 的 incognito 处理：在 `WeightsFileCreatorImpl::CreateWeightsFile` 直接返回 invalid `base::File` → 走 in-memory，**不创建任何文件**。比 storage quota 还激进，因为 incognito 不应有任何磁盘痕迹。

---

## 附录 B：改动文件清单

| 文件 | 改动 |
|---|---|
| `services/viz/privileged/mojom/gl/gpu_host.mojom` | `CreateWebNNWeightsFile()` 加 `uint64 required_bytes` |
| `components/viz/host/gpu_host_impl.{h,cc}` | 透传 `required_bytes` 给 `webnn::CreateWeightsFile`（GPU-process 路径覆盖 §A） |
| `services/webnn/public/mojom/webnn_context_provider.mojom` | `CreateWeightsFile()` 加 `uint64 required_bytes`；新增 `ReleaseWeights(uint64 bytes)` |
| `services/webnn/host/weights_file_provider.{h,cc}` | §A: 磁盘空间检查 + 自适应 headroom (10 GiB / 10%, 无进程内串行) |
| `services/webnn/host/weights_file_creator_impl.{h,cc}` | §C: per-origin `OriginUsageTracker`；`ReleaseWeights` 实现（over-release `ReportBadMessage`，析构兜底 `reserved_bytes_` 残余） |
| `services/webnn/host/BUILD.gn` | 加 `//url` 依赖 |
| `services/webnn/webnn_context_impl.{h,cc}` | §B: per-context `weights_bytes_granted_`；`CreateWeightsFile` 回调改为 `(File, uint64 granted_bytes)`；新增 public `ReleaseWeightsBytes(uint64)` |
| `services/webnn/webnn_context_provider_impl.{h,cc}` | 透传 `required_bytes`；GPU-process 路径无 per-origin tracker，`CreateWeightsFile` 上加 TODO（等 MLDrift delegate 迁回 renderer 后此路径整体消失） |
| `services/webnn/webnn_context_provider_in_renderer.{h,cc}` | 新增 `ReleaseWeights` 透传到 `WebNNWeightsFileCreator` mojom |
| `services/webnn/webnn_graph_impl.{h,cc}` | 基类 ctor 加 `uint64 weights_bytes_granted = 0`；析构调 `context_->ReleaseWeightsBytes` |
| `services/webnn/tflite/context_provider_tflite.{h,cc}` | 透传 `required_bytes` |
| `services/webnn/tflite/context_impl_tflite.{h,cc}` | 求和 constant_operands 字节数；`DidCreateWeightsFile` 加 `uint64 granted_bytes` 透传到 `GraphImplTflite::CreateAndBuild` |
| `services/webnn/tflite/context_impl_litert.{h,cc}` | 求和 constant_operands 字节数；`DidCreateWeightsFile` 同上 |
| `services/webnn/tflite/graph_impl_tflite.{h,cc}` | `CreateAndBuild` / ctor 加 `uint64 weights_bytes_granted`；`DidCreateAndBuild` 在 build 失败时显式 `context->ReleaseWeightsBytes` |
| `services/webnn/tflite/graph_impl_litert.{h,cc}` | 同 tflite，LiteRT 镜像 |
| `services/webnn/webnn_test_environment.{h,cc}` | 测试入口传 `url::Origin()`、`required_bytes` |
| `content/browser/browser_interface_binders.cc` | 各 host 提取 origin（模板特化） |

构建验证：`autoninja -C out/Debug services/webnn:webnn_service services_unittests -j 40` 通过（1078 steps，3 min 18 s，0 errors）；`services_unittests --gtest_filter="WebNN*"` 365/365 通过。
