# WebNN Weights File 磁盘使用与配额设计

## 背景

WebNN 在执行图编译时会把 constant operand（权重）单独写到一个临时文件里，然后让 TFLite 通过 `mmap` 直接加载，避免再做一次内存拷贝。当前实现：

- 入口：[`services/webnn/host/weights_file_provider.cc`](../chromium/src/services/webnn/host/weights_file_provider.cc) `CreateTemporaryFile()` 直接调 `base::CreateTemporaryFile()`
- 路径：系统 temp 目录（`/tmp`、`%TEMP%`），**不**在 profile 的 storage partition 下
- 标志：`FLAG_DELETE_ON_CLOSE`，关闭即销毁
- 唯一已有的限制：incognito 模式跳过文件创建，走 in-memory Flatbuffer

存在的问题：
1. 没有任何大小检查 —— 恶意站点构造巨大模型可以撑爆 `/tmp`
2. 没有 per-origin 记账 —— 一个 origin 可以反复构图占满磁盘
3. 没有 graceful fallback —— 磁盘满会直接让 build 失败

---

## 设计决策

### Q1：Chromium 有 per-origin 磁盘配额机制吗？

**有。** `storage::QuotaManager`（`//storage/browser/quota/`）按 `StorageKey`（origin + top-level site）管理配额，被 IndexedDB / Cache API / File System Access / Service Worker 等使用。

但 WebNN **不应该**直接接入这个系统：

- WebNN service 跑在 GPU 进程，`QuotaManager` 在 Storage Service / Browser 进程，要查必须 Mojo 绕一圈，会阻塞 graph build 关键路径
- weights 是 `DELETE_ON_CLOSE` 的临时计算产物，不是"持久用户数据"，语义上不属于 Quota 管的那一类
- 文件落在系统 temp 而非 profile 目录，根本不在 quota 账本里
- 真正的风险是 **/tmp DoS**，跟 per-origin storage quota 关心的事情不是同一个问题

### Q2：超过限制能 fallback 到 in-memory 吗？

**可以，且改动很小。** TFLite 后端已经接受 invalid `base::File` 走 in-memory Flatbuffer 路径（incognito 就是这么做的）。fallback 通道已经存在，只需要在 `CreateTemporaryFile()` 里：

- 创建前用 `base::SysInfo::AmountOfFreeDiskSpace()` 检查剩余空间
- 写入失败（ENOSPC / `Initialize()` 返回非 valid）就返回空 `base::File()`

renderer 侧自然就退回 in-memory，无需新协议。

注意：in-memory Flatbuffer 受 **< 2 GiB** 限制（Flatbuffer offset 是 int32），所以"大图退回 in-memory"不能无限退。

### Q3：分层防护方案

| 阶段 | 做法 | 解决的问题 |
|---|---|---|
| **A** | 写文件前 `AmountOfFreeDiskSpace()` + 失败时返回 invalid file 让 TFLite 走 in-memory | 磁盘满导致 build 失败/崩溃 |
| **B** | 在 `WebNNContextImpl` 加 **per-context 上限**（4 GiB），累加各 graph 的 weights 文件大小，超限直接拒绝或退 in-memory | 单 tab DoS /tmp |
| **C** | 在 browser 进程加 **per-origin 总量上限**（8 GiB），通过 `WeightsFileCreatorImpl` broker 维护轻量记账表 | 跨 context 的 origin 级滥用 |
| D（未做） | 真要接 QuotaManager 的话，应作为独立 storage type（如 `kWebNNWeights`），并通过 browser 进程 broker | 与 Web Storage 配额体系对齐 |

A + B + C 解决约 80% 的实际问题，已经实现。

---

## 实现概览

### Mojo 接口变更

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
};
```

### A — 磁盘空间检查

[`services/webnn/host/weights_file_provider.{h,cc}`](../chromium/src/services/webnn/host/weights_file_provider.cc)

- `CreateWeightsFile(required_bytes, cb)` 在创建文件前调用 `AmountOfFreeDiskSpace()` 与 `AmountOfTotalDiskSpace()`
- headroom 算法对齐 `storage::QuotaSettings::must_remain_available`：

```cpp
inline constexpr uint64_t kWeightsFileMustRemainAvailableBytes = 1ull << 30;  // 1 GiB
inline constexpr double   kWeightsFileMustRemainAvailableRatio = 0.01;        // 1%

// headroom = min(固定 reserve, 总磁盘 × ratio)
const uint64_t headroom = std::min<uint64_t>(
    kWeightsFileMustRemainAvailableBytes,
    static_cast<uint64_t>(static_cast<double>(*total_bytes) *
                          kWeightsFileMustRemainAvailableRatio));
```

不同设备实际 headroom（参考 `quota_settings.cc` 注释）：

| 总磁盘 | min(1 GB, 1% × total) |
|---|---|
| 1 TB | 1 GB |
| 64 GB | 640 MB |
| 16 GB | 160 MB |
| 8 GB | 80 MB |

如果 `free < required_bytes + headroom`，返回 invalid `base::File`，触发现有 in-memory fallback。

### B — Per-context 上限

[`services/webnn/webnn_context_impl.{h,cc}`](../chromium/src/services/webnn/webnn_context_impl.cc)

```cpp
static constexpr uint64_t kMaxWeightsBytesPerContext = 4ull << 30;  // 4 GiB
uint64_t weights_bytes_granted_ = 0;
```

`WebNNContextImpl::CreateWeightsFile(required_bytes, cb)` 在转发给 `ContextProviderTflite` / `WebNNContextProviderImpl` 之前先做溢出检查；获得 invalid file 时归还配额。

### C — Per-origin 上限

[`services/webnn/host/weights_file_creator_impl.{h,cc}`](../chromium/src/services/webnn/host/weights_file_creator_impl.cc)

```cpp
static constexpr uint64_t kMaxBytesPerOrigin = 8ull << 30;  // 8 GiB
```

进程内 `OriginUsageTracker`（lock-guarded `std::map<url::Origin, uint64_t>`）在 `WeightsFileCreatorImpl` 实例间共享：

- `WeightsFileCreatorImpl` 的生命周期 = 单个 self-owned mojo pipe（per-frame / per-worker）
- `CreateWeightsFile` 时 `TryReserve()`，文件失败或实例销毁时 `Release()`
- 跨 frame / worker 共享同一 origin 的额度

`kMaxBytesPerOrigin` 选 8 GiB 的考量：

storage quota 默认给单个 `StorageKey` 的额度是：
$$\text{per\_storage\_key\_quota} = \text{total\_disk} \times 0.6 \times 0.75 = \text{total\_disk} \times 45\%$$

| 总磁盘 | per_storage_key_quota |
|---|---|
| 1 TB | ~450 GB |
| 256 GB | ~115 GB |
| 64 GB | ~28 GB |
| 16 GB | ~7.2 GB |

WebNN weights 是临时计算产物（`DELETE_ON_CLOSE`），没必要追到 45%。固定 8 GiB 比 quota 默认保守得多，可放心作为防滥用硬上限。

### 调用链贯通

```
ContextImplTflite::CreateGraphImpl
  -> 求和 constant_operands 的 ByteSpan().size() 得到 required_bytes
  -> WebNNContextImpl::CreateWeightsFile(required_bytes, cb)         [B 检查]
       -> ContextProviderTflite::CreateWeightsFile(required_bytes, cb)
            -> mojom::WebNNWeightsFileCreator.CreateWeightsFile(required_bytes)  [Mojo]
                 -> WeightsFileCreatorImpl::CreateWeightsFile         [C 检查]
                      -> webnn::CreateWeightsFile(required_bytes, cb) [A 检查]
                           -> CreateTemporaryFile()
```

### Origin 来源（content/browser）

[`content/browser/browser_interface_binders.cc`](../chromium/src/content/browser/browser_interface_binders.cc)

```cpp
// RenderFrame:
host->GetLastCommittedOrigin()

// Worker hosts (模板特化):
DedicatedWorkerHost / SharedWorkerHost -> GetWorkerStorageKey().origin()
ServiceWorkerHost                       -> GetBucketStorageKey().origin()
```

---

## 配额来源参考

### `storage/browser/quota/quota_settings.cc::CalculateNominalDynamicSettings()`

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

WebNN 的 headroom 直接复用这套 must_remain_available 的 `min(固定, 比例)` 模式。

### Incognito 路径

`CalculateIncognitoDynamicSettings()`：

```cpp
pool_size = physical_memory × (15% ~ 20%)   // 随机化
per_storage_key_quota = pool_size / 3
```

WebNN 的 incognito 处理：在 `WeightsFileCreatorImpl::CreateWeightsFile` 直接返回 invalid `base::File` → 走 in-memory，**不创建任何文件**。比 storage quota 还激进，因为 incognito 不应有任何磁盘痕迹。

---

## 已知遗留问题

1. **GPU-process 路径**：`WebNNContextProviderImpl::CreateWeightsFile` → `viz::mojom::GpuHost::CreateWebNNWeightsFile` 目前不带 `required_bytes`，A/C 暂未覆盖该路径，B 仍然生效。需要扩 `viz::mojom::GpuHost` 接口（后续 work）。
2. **Per-origin cap 是固定值**：未跟磁盘大小联动。可以做 `min(8 GiB, 总磁盘 × N%)`，但 `OriginUsageTracker` 需改为异步或在 `Create` 时缓存查询结果，权衡复杂度后保留固定值。
3. **TODO**：`crbug.com/507502295` — Use file manager for weights files。

---

## 改动文件清单

| 文件 | 改动 |
|---|---|
| `services/webnn/public/mojom/webnn_context_provider.mojom` | `CreateWeightsFile()` 加 `uint64 required_bytes` |
| `services/webnn/host/weights_file_provider.{h,cc}` | A: 磁盘空间检查 + 自适应 headroom |
| `services/webnn/host/weights_file_creator_impl.{h,cc}` | C: per-origin `OriginUsageTracker` |
| `services/webnn/host/BUILD.gn` | 加 `//url` 依赖 |
| `services/webnn/webnn_context_impl.{h,cc}` | B: per-context `weights_bytes_granted_` |
| `services/webnn/tflite/context_provider_tflite.{h,cc}` | 透传 `required_bytes` |
| `services/webnn/tflite/context_impl_tflite.cc` | 求和 constant_operands 字节数 |
| `services/webnn/tflite/context_impl_litert.cc` | 求和 constant_operands 字节数 |
| `services/webnn/webnn_test_environment.cc` | 测试入口传 `url::Origin()`、`required_bytes=0` |
| `content/browser/browser_interface_binders.cc` | 各 host 提取 origin（模板特化） |

构建验证：`autoninja -C out/Debug chrome -j 40` 通过（28 min, 6264 steps, 0 errors）。
