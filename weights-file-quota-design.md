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

## 5. Capacity tracker：renderer 直写 fd + 每次扩展前 sync IPC 记账

> **方案演进说明（2026-06-17 更新）**
>
> 本节最初追求"kernel 强制的 fd 级 hard cap"，先后考虑过 chunked SharedMemory（commit `2881a74236`，已弃用）等方案，理由是把 §A/§B/§C "簿记型" 防护抬到 kernel 层。
>
> 这个出发点其实站不住：**FSA picker 不是一道安全边界**。File System Access 让 renderer 直接持有用户挑选文件的可写 fd，安全性来自两个**独立**层面：(1) 用户通过 picker 授权，决定 *哪些* 字节流可被这个 origin 访问，是 *用户隐私* 边界；(2) `FileSystemAccessCapacityTracker` 在 browser 进程对 quota / 磁盘空间做**增量记账**，决定 *能写多少*，是 *资源耗尽* 边界。这两层正交，缺一不可——picker 阻止恶意页面读你的私文件，但 picker 之后的写入量必须由 capacity tracker 守。
>
> 关键观察：**OPFS（Origin Private File System）走完全相同的 picker-less 路径**。OPFS sync access handle 不需要任何用户授权对话（origin 自动获得自己的私有文件系统），renderer 仍然直接拿到 writable fd，仍然走同一份 [`FileSystemAccessCapacityTracker`](https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/modules/file_system_access/file_system_access_capacity_tracker.cc) 做容量记账。OPFS 的安全性 100% 来自 browser-side capacity tracker——**没有 picker 也安全**。
>
> 把这个映射到 WebNN：weights tempfile 是 browser 自建的、生命周期严格绑定 context 的、不暴露给用户也不暴露给其他 origin 的私有文件，跟 OPFS 的 sandbox 文件本质同源。WebNN 也不需要 picker（用户从未"授权"weights 文件的存在），需要的只是**与 OPFS 等价的 capacity tracker**——renderer 持 writable fd，每次扩展前 sync IPC 调 `RequestCapacityChange(new_size)`，browser-side 在每次 IPC 上**重新执行** §A 磁盘 headroom + §B per-context cap + §C per-origin cap + §D `fstat` anti-tamper。安全保证不来自 fd 类型，而来自 IPC 边界上*增量、强制、不可绕开*的校验。
>
> 旧 chunked SHM 方案保留在 git 历史（commit `2881a74236`），由本设计取代。

### 5.1 问题

最初的 `CreateWeightsFile(required_bytes)` 是一次性簿记：browser 在创建 fd 之前根据 `required_bytes` 做一次 §A/§B/§C 检查，之后 fd 留在 renderer 手里没有任何 per-fd 上限。被攻陷 renderer 可以 `CreateWeightsFile(required_bytes=1)` 通过所有上层检查，再对可写 fd `write()` 任意字节，攻击者可在自己被允许的磁盘片内写到 §A headroom 边界（最坏情形 ENOSPC）。

附加问题（同样由 reillyg@ 指出）：streaming weights 是 open issue——按 build 进度逐步释放 renderer 端 `WebNNConstantOperand` 内存——但要求 `CreateWeightsFile` 时给出精确 `required_bytes` 与 streaming 不兼容，因为 streaming 路径下 build 启动时根本不知道总大小。

### 5.2 安全模型：picker 与 capacity tracker 的解耦

把 FSA / OPFS / WebNN 摊开对比：

| 维度 | FSA picker 文件 | OPFS sandbox 文件 | WebNN weights tempfile |
|---|---|---|---|
| 用户授权层 | picker 选具体文件（隐私） | 无 picker（origin 自动） | 无 picker（service 自建） |
| 谁拿到 writable fd | renderer | renderer | renderer（in-renderer TFLite/LiteRT） |
| 每次扩展前 sync IPC 校验 | ✅ `FileSystemAccessCapacityTracker.RequestFileCapacityChangeSync` | ✅ 同一 tracker 类 | ✅ `WeightsFileCapacityHost.RequestCapacityChange` |
| 资源耗尽防御层 | browser 进程 capacity tracker | browser 进程 capacity tracker | browser 进程 §A/§B/§C/§D（本设计） |
| 文件生命周期 | 用户文件，绕过 origin 卸载 | origin 私有，origin 删除即清 | context 绑定，pipe 关闭即 unlink |
| 跨 origin 读取暴露 | picker 阻止 | OPFS 隔离阻止 | 文件 fd 不离开 service↔renderer 通道 |

WebNN 与 OPFS 的相似度比与 FSA picker 路径更高：都是 picker-less、origin-内部、由 user agent 自建的文件，安全性完全来自 browser-side 增量记账。

> **GPU-process 路径不在本节范围**：GPU process 内的 LiteRT GPU delegate 通过 `viz::mojom::GpuHost::CreateWebNNWeightsFile` 创建 fd，fd 不离开 GPU 进程，renderer 拿不到，没有 fd 越界写问题；它的限制由 §A 进程级 headroom + §B per-context 自检负责。本节的 capacity tracker 只覆盖 in-renderer TFLite/LiteRT 路径。

### 5.3 设计

#### 5.3.1 Mojo 接口

```mojom
// Browser 端 self-owned receiver，per-`navigator.ml` 一个 creator
// （= per renderer ExecutionContext 一个）。Creator 是纯工厂：自身不持有任何
// per-file 状态，所以同一个 `navigator.ml` 内并发 `OpenWeightsFile` 互相独立。
interface WebNNWeightsFileCreator {
  // Browser 创建一个延迟 unlink 的 tempfile，dup 一个 writable fd 给 renderer，
  // 同时为这个 tempfile self-own 一个 `WeightsFileSession`，把 PendingRemote
  // 一并返回。失败（incognito、磁盘错误）返回 null fd + null session，
  // renderer 退回 in-memory Flatbuffer。
  OpenWeightsFile()
      => (mojo_base.mojom.File? writable_fd,
          pending_remote<WeightsFileSession>? session);
};

// Browser 端 self-owned receiver，per-tempfile 一个（= per `build()` 一个）。
// 把增量配额记账与 finalize 合在同一个接口里，所有 per-file 状态
// （tempfile、path、granted_bytes、origin）集中在一个对象。Pipe 断开
// （renderer crash / 中途放弃 / `Finalize` 完成后的析构）会触发
// `~WeightsFileSessionImpl`，把累计 granted_bytes_ 一次性归还进程级
// `OriginUsageTracker`，并 unlink tempfile（POSIX）/ `DeleteOnClose`（Windows）。
interface WeightsFileSession {
  // 每次扩展文件前 sync IPC。new_size = 计划写入后的文件大小。
  // Browser 在每次调用上**全部**重新执行 §A/§B/§C/§D 检查（见 5.3.2）。
  [Sync]
  RequestCapacityChange(uint64 new_size) => (bool granted);

  // Renderer 写完后调用。Browser:
  //   1. fstat tempfile，校验 size <= granted_bytes_（否则 ReportBadMessage）
  //   2. 按 path 重开为 RO；POSIX unlink path / Windows DeleteOnClose(true)。
  //      返回 sealed RO fd 给 renderer mmap。
  // Session pipe 在 reply 后随之关闭，触发浏览器端 session 自销毁
  // （归还 §C per-origin 配额）。
  Finalize() => (mojo_base.mojom.ReadOnlyFile? sealed_file);
};
```

> **Option A 已考虑、被否决**：另一种做法是把所有方法塞进同一个
> `WebNNWeightsFileCreator` 接口，每次调用都带一个 per-file token
> （`Write(token, …)`, `Finalize(token)`）。该方案在功能上同样支持并发文件，
> 但 per-file 状态会摊在 creator 实现里，每次调用都要校验 token。把每个文件
> 建模成自己的 self-owned session 与现有 Mojo 习惯一致（参见 `Tensor`、
> `Graph`），并且让 renderer 端代码路径与单 build 场景完全一样。

#### 5.3.2 Browser 侧核心逻辑（实现摘要）

`WeightsFileCreatorImpl` 是一个纯工厂：只持有 `origin_` 与 `is_incognito_`。
`WeightsFileSessionImpl` 通过 `MakeSelfOwnedReceiver` 自持有，独占
`tempfile_`、`tempfile_path_`、`granted_bytes_`、`origin_` ——所有 per-file
状态集中在一个对象内。同一个 creator 下并发存在多个 session 互不影响。

```
OpenWeightsFile() on WeightsFileCreatorImpl:
  if is_incognito_: return (null fd, null pipe)
  (tempfile, path) = webnn::CreateWeightsFileWithPath()
            // 内部跑 §A 磁盘 headroom 大致体检
            // (free >= headroom)，与"授予容量"无关
  if !tempfile.IsValid(): return (null fd, null pipe)
  renderer_fd = tempfile.Duplicate()
  // 把 writable fd 移交给 session，跟随本次 build 生命周期。
  WeightsFileSessionImpl::Create(session_remote, std::move(tempfile),
                                 std::move(path), origin_)
  return (renderer_fd, session_remote)

RequestCapacityChange(new_size) on WeightsFileSessionImpl:
  // §D anti-tamper —— 每次 IPC 前重新 fstat
  current = tempfile_.GetLength()
  if current < 0 || uint64_t(current) > granted_bytes_:
    ReportBadMessage("renderer wrote past granted capacity"); return false
  if new_size <= granted_bytes_: return true              // shrink / no-op
  delta = new_size - granted_bytes_
  // §B per-session（4 GiB）—— 命名见 5.3.3
  if granted_bytes_ + delta > kMaxWeightsBytesPerContext: return false
  // §C per-origin（8 GiB）—— 进程级 OriginUsageTracker
  if !OriginUsageTracker::TryReserve(origin_, delta): return false
  granted_bytes_ += delta
  return true

Finalize() on WeightsFileSessionImpl:
  current = tempfile_.GetLength()
  if current < 0 || uint64_t(current) > granted_bytes_:
    ReportBadMessage(...); return  // 析构清理 tempfile + 归还配额
  // mojo_base.mojom.ReadOnlyFile traits 在 POSIX 上 CHECK O_RDONLY；
  // dup(O_RDWR) 通不过，所以按 path 重开为 read-only 再 unlink。
  ro_fd = base::File(tempfile_path_, FLAG_OPEN | FLAG_READ | FLAG_WIN_NO_EXECUTE)
#if POSIX
  unlink(tempfile_path_)            // 已打开的 fds 维持 inode 存活
#else  // Windows
  tempfile_.DeleteOnClose(true)     // 两个句柄都关时回收
#endif
  tempfile_.Close()
  return ro_fd  // reply 后 self-owned receiver 析构

~WeightsFileSessionImpl:
  if granted_bytes_ > 0:
    OriginUsageTracker::Release(origin_, granted_bytes_)
  if !tempfile_path_.empty():
    // 兜底：pipe 在 Finalize 之前断开。
    base::ThreadPool::PostTask(BEST_EFFORT, base::DeleteFile(tempfile_path_))
```

注意：§A 不在 `RequestCapacityChange` 路径上重复执行，因为 §A 检查的是"剩余磁盘空间是否够本次新分配 + 系统级 headroom"。`OpenWeightsFile` 创建文件那一刻已经做过一次（仅校验 headroom），后续扩展由 §B/§C 上限封顶 + 全局 headroom 吸收。如果担心慢速磁盘填充，可以把 §A 也搬进 `RequestCapacityChange`，但每次 sync IPC 多 1 次 `statvfs`（数百 µs，对 N 个常量的 build 是 N×几百 µs），目前不做。

#### 5.3.3 簿记的归属与并发 build 支持（§B 语义）

In-renderer 路径下 §B 的"per-context 4 GiB"由 `WeightsFileSessionImpl::granted_bytes_` 承担，**不**走 `WebNNContextImpl::weights_bytes_`——后者只在 GPU-process 路径的 `WebNNContextImpl::CreateWeightsFile` 里被加减，in-renderer 路径全程为 0；`GraphImpl{Tflite,LiteRt}::DidCreateAndBuild` 因此固定传 `weights_bytes=0`，不走析构释放。

实际粒度：§B 的真实粒度是 *per-session*（= per-build）而非 *per-context*。同一个 `navigator.ml` 同时跑 N 个并发 graph build 时各拿一个独立 4 GiB 上限，互不约束；跨 build 累计仍由 §C 的 8 GiB per-origin 兜底。

**并发 build 正确性**：把 session 接口做成 per-file self-owned（而不是把所有文件复用到同一个 `WeightsFileCreator` pipe）是并发 `build()` 安全的根本原因。早期一版设计是单个 creator-bound `WeightsFileCapacityHost`，重叠 `OpenWeightsFile` 直接 `ReportBadMessage`，这会让 frame 上第二个并发 `build()` 触发整个 WebNN pipe 被杀。当前设计每次 `OpenWeightsFile` 都 spawn 一个全新的 `WeightsFileSessionImpl`，并发 build 永远不撞车。如果未来希望把 §B 收紧到真正的 context 级聚合，需在 `WebNNContextImpl` 里做 cross-pipe 累计；目前不做——§C 已经把最坏情形封到 8 GiB/origin，而 per-build 独立性对 graph 编译并行才是更有用的不变式。

`FLAG_DELETE_ON_CLOSE` 的语义同此处一并澄清：POSIX 上 tempfile 在 `Finalize` 后立即 `unlink`，但 open fds 维持 inode 存活直到 renderer 把 mmap 释放；Windows 上靠 `DeleteOnClose(true)`，关闭即销毁——结果一致（fd 关掉就没了），但路径不同。

#### 5.3.4 Renderer 侧（`GraphBuilderTflite::SerializeBuffer`）

```cpp
// graph_builder_tflite.cc:3287
auto GraphBuilderTflite::SerializeBuffer(base::span<const uint8_t> buffer)
    -> base::expected<BufferInfo, std::string> {
  // ... seek + align + SetLength（trim padding）

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

**Streaming weights 兼容性**：`RequestCapacityChange` 只看 `new_size`，不依赖任何全局总大小，每个常量序列化后可立刻 `WebNNConstantOperand::Drop()`。

#### 5.3.5 TFLite/LiteRT 侧

零改动。`weights_file` 来源从一次性 `CreateWeightsFile` 的可写 fd 变成 `WeightsFileSession::Finalize` 返回的 RO fd；TFLite/LiteRT 透明 mmap，运行时行为不变。`GraphImpl{Tflite,LiteRt}::CreateAndBuild` 在 context sequence 上把 `WeightsFileSession` `SharedRemote` 绑定，保留两份 ——一份移交给后台线程的 builder 闭包用于 `RequestCapacityChange`，另一份移交 reply 闭包（`DidBuildGraph`），等 worker 返回后调 `session->Finalize`。第二份 SharedRemote 作为 keep-alive 移入 `Finalize` 的 reply 闭包；reply 触发、闭包销毁后 pipe 自然关闭，浏览器端 session 自销毁（归还 per-origin 配额）。

### 5.4 安全分析

| 攻击 | 旧 `CreateWeightsFile` | SHM chunked（已弃用） | Capacity tracker（本设计） |
|---|---|---|---|
| 越界写 fd | ❌ 仅簿记 | ✅ kernel SIGBUS | ✅ 每次 IPC `fstat` + ReportBadMessage |
| 需提前知道总大小 | 是 | 是 | 否 |
| Streaming weights 兼容 | ❌ | ❌ | ✅ |
| Renderer 内存峰值 | 全部 weights 驻留 | 全部 weights 驻留 SHM | 可边写边释放 |
| 零额外内存拷贝 | — | ❌ SHM→tempfile 全量拷贝 | ✅ |
| 跨平台 | ✅ | ✅ | ✅ |

> **关于 fd 越界写为何不再是问题**：renderer 拿 writable fd 与 FSA / OPFS 一致；安全保证不来自 fd 类型，而来自每次扩展前的强制 sync IPC。最坏情况 renderer 在两次 `RequestCapacityChange` 之间偷写：下一次 `RequestCapacityChange` 的 §D `fstat` 检查会发现 `current_length > granted_bytes_`，立刻 `ReportBadMessage` 杀掉该 renderer；偷写的字节因 `FLAG_DELETE_ON_CLOSE` 在 fd 关闭时随 tempfile 一起 unlink。攻击者无法把"偷写的字节"提交进图（`FinalizeWeightsFile` 也跑同样的 `fstat` 校验），即使能写，写完了也马上被销毁。

#### 5.4.1 Compromised renderer 行为模型

Security review 反复出现的问题：被攻陷的 renderer 拿到 writable fd 之后能干什么？kernel 不知道我们的 IPC 协议——可写 fd + `write()` syscall 永远成功。所以这一节明确列出 *能干什么 / 干完会怎样 / 上界是什么*。

| 攻击者动作 | 是否成功 | 后果 / 上界 |
|---|---|---|
| 直接 `write(fd, ...)` 跳过 `RequestCapacityChange` | ✅ 写入会真的进文件 | 偷写字节进不了 graph：`FinalizeWeightsFile` / 下一次 `RequestCapacityChange` 跑 §D `fstat`，`current_length > granted_bytes_` 即 `ReportBadMessage` 杀渲染器；renderer 不再调这两个则 build 永不完成（自 DoS） |
| 不停 `write()` 直到 ENOSPC | ✅ 单 renderer 寿命内可填满磁盘 | **持久伤害 = 0**：`FLAG_DELETE_ON_CLOSE`（POSIX 创建时即 `unlink`，Windows 关 fd 触发）保证 renderer 一退磁盘归还。**爆炸半径 = 受控**：free disk 跌破 §A 10 GiB / 10% headroom 后，所有 origin 的下次 `OpenWeightsFile` 自动返回 invalid → WebNN 全局降级到 in-memory，GPU 进程不崩。同 OPFS / FSA 现状，无 web 文件 API 能在"渲染端已被攻陷"前提下阻止主动写。硬上限 = `min(物理 free disk, sandbox RLIMIT_FSIZE)` |
| `dup(fd)` 或 `SCM_RIGHTS` 跨进程克隆 fd | ✅ 内核允许 | 所有克隆指向同一 inode，§D `fstat` 看的是 inode 实际大小，与写入路径无关；偷写后果同上一行 |
| 克隆 session pipe | ❌ | `pending_remote<WeightsFileSession>` 是 browser 一次性下发的 Mojo 端点，不可克隆。合法扩容仍需走 sync IPC |
| 同时开很多 session 试图绕过 per-session §B（4 GiB）| ⚠️ 部分成立 ——每个 session 合法地各拿 4 GiB 上限 | 由 §C 兜底：同一 origin 所有 session 共享同一份 `OriginUsageTracker`，跨 session 累计 outstanding 仍 ≤ 8 GiB。"并发多 session"本身就是同一个 `navigator.ml` 并发 `build()` 的合法路径 |
| 永远不调 `RequestCapacityChange` 也不 `Finalize` | ✅ | Graph 永不完成，纯自 DoS；磁盘占用受 §A headroom + session pipe 断开兜底约束 |
| 偷写后调 `RequestCapacityChange(new_size <= granted_bytes_)` 试图蒙混 | ❌ | §D `fstat` 不依赖 `new_size`，先看 `current_length` vs `granted_bytes_`；越界即杀 |
| 把偷写字节注入最终 graph | ❌ | `Finalize` 同样跑 `fstat ≤ granted_bytes_` 才发 RO fd；不通过则不 mmap、不 build |

**净收益分析**：被攻陷 renderer 只能拿到"在自己进程寿命内 DoS 一块磁盘 + DoS WebNN 整体"，无任何数据完整性 / 任意代码 / 跨 origin 收益；renderer 一退即清。这与 OPFS / FSA 在同等威胁模型下的防御深度持平——WebNN 不需要、也无法在 web platform 框架内做更强保证。

### 5.5 代价

- **每次 weights 扩展一次 sync IPC**：约 N 个常量 N 次 IPC，单次 ~50 µs；graph build 总延迟可忽略
- **每次 IPC 多一个 fstat**：µs 级
- **零额外内存拷贝**：相比 SHM 方案省去最多 4 GiB 的 SHM→tempfile 搬运

### 5.6 不在范围

- **接入 `storage::QuotaManager` 替换自建 §C per-origin budget**：长期方向。Capacity tracker 的 `RequestCapacityChange` 接口设计为 frontend，未来可平滑替换 QuotaManager backend 而不改 renderer 端代码。
- **LiteRT 加内存 API（`SetExternalWeightFromMemory`）**：可消除 tempfile 整体，未来上游协作。
- **Per-origin cap 跟磁盘联动**、**GPU 路径合入 §C**：保持 §4 #1/#2 followup 计划。

### 5.7 跟踪

`crbug.com/XXXXXXX`（待开）。建议挂 `Security>WebNN` 和 `Component:WebNN`，Hotlist-Security-Severity-Low。

旧 SHM chunked 实现（commit `2881a74236`）将在新 CL 中被替换；新 CL 标题：`webnn: Stream weights via incremental capacity tracker`。
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
| `services/webnn/host/weights_file_creator_impl.{h,cc}` | §C: per-`navigator.ml` 纯工厂；`OpenWeightsFile` 创建 tempfile + 拉起 self-owned `WeightsFileSessionImpl`（自身不保留 per-file 状态） |
| `services/webnn/host/weights_file_session_impl.{h,cc}` | Per-file self-owned session：独占 `tempfile_` / `tempfile_path_` / `granted_bytes_` / `origin_`；实现 `RequestCapacityChange`（§B + §C + §D）与 `Finalize`（anti-tamper fstat + 按 path RO 重开 + unlink）。Per-origin `OriginUsageTracker` 作为进程单例落在此 TU 内，per-origin 上限（`WeightsFileCreatorImpl::kMaxBytesPerOrigin`）与 per-session 上限（`kMaxWeightsBytesPerContext`）都在这里执行 |
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
