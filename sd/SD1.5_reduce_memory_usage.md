# SD 1.5 在 WebNN(LiteRT) GPU 路径上的内存与稳定性分析

> 更新：2026-11-04　机器：WEBNNARL（Intel Arc 140T，63 GB 内存，驱动 32.0.101.6452）
> 模型：`C:\workspace\junwei\tflite-dump-model\unetbatch=1\`（`text_encoder` / `unet` / `vae_decoder`，各有 `.onnx` 和 dump 出来的 `.tflite`）
>
> 每条结论标了依据：**[实测]** 本机测出的数据；**[用户实测]** 你提供的数据；**[读码]** 只读了源码；**[推测]** 由现象推出、尚未验证。

---

## 0. 摘要：三个相互独立的问题

| # | 问题 | 目前的结论 | 状态 |
|---|---|---|---|
| A | **GPU 共享显存过大** | Intel 上 ML Drift 默认用 TEXTURE_2D 存中间张量，只有形状完全相同才复用。强制 BUFFER 存储后，SD 的 GPU 共享显存约 10.9 → 6.8 GB（runner）；Chrome GPU 进程峰值 13.3 GB → 8.5~8.9 GB [用户实测] | 已有修复（`SetBufferStorageType(kBuffer)`），待 PTL 验证 |
| B | **渲染进程 OOM**（"Aw, Snap! Out of Memory"） | 渲染进程 private 已提交 16.8~19.9 GiB 时，一次 **2 MiB** 的分配失败，而系统还有 24~31 GiB commit 可用，疑为触到 PartitionAlloc 单池 16 GiB 上限。其中可确认的一大块：**VAE 的 30 个 `instance_norm_decomposition` 子图被 TFLite 分配了从不使用的 arena，请求 3.4 GiB、实际提交 6.8 GiB（被 PartitionAlloc 取整放大 2 倍）** | 机制已定位；B.3 方案 2（TFLite composite 内核 `Prepare` 后释放分解体 arena）已打补丁，待真机验证 |
| C | **GPU 挂起（TDR）**：`DEVICE_HUNG`、`ReadTensor ... failed: 4` | 这是 Intel 驱动的 TDR（WER 里 `ARL_DX12COMPUTE_BBHANG/RINGHANG`），与内存大小无直接关系，本机在 10-27~10-30 出现过 33 次，11-04 又出现 1 次 | **根因未定位** |

几点要特别说明（避免误读）：

- **"内存不够"不能解释全部现象**：本机 63 GB 照样出过 TDR 和渲染进程 OOM；渲染进程 OOM 时系统 commit 并未耗尽。
- 渲染进程的 OOM 与 buffer 存储修复**无关**：5 个 OOM dump 里有 3 个发生在修复之前，1 个发生时进程里根本没有 LiteRT。
- 渲染进程 OOM 与 TDR 也**无关**：11-02 的 OOM 当天没有任何 TDR。

---

## 1. 现象

- 在 `chrome.exe --enable-features=WebMachineLearningNeuralNetwork[,WebNNLiteRTGpuInRenderer] --disable-features=WebNNDirectML,WebNNOnnxRuntime --enable-unsafe-webgpu` 下加载
  `https://10.239.115.25:8080/demos/stable-diffusion-1.5/?unetbatch=1&safetychecker=false&ort=latest`，偶发：
  - `graph_impl_litert.cc:544 Failed to compute: Failed to invoke the compiled model`，随后 `D3D12CreateDevice failed with DXGI_ERROR_DEVICE_HUNG`；
  - VAE 读回失败：`buffer_content_tflite.cc:222 Mapping the readback buffer of a tensor failed: 4`（`4 = WGPUMapAsyncStatus_Aborted`），页面报 `Tensor has been destroyed or context is lost`；
  - 页面 "Aw, Snap! Error code: Out of Memory"。
- PTL（18 GB）：多开一个空 Tab 时，另一个 Tab 加载 SD 就出错；单开能出图，但静置一段时间后 crash。
- NUC（48 GB）：开很多 Tab 都加载 SD 也能跑，静置不 crash。
- 本机（63 GB）：怎么加压都复现不了 ReadTensor 错误（见 5.3），但历史上出过 TDR 和渲染进程 OOM。

---

# 问题 A：GPU 共享显存过大

## A.1 根因 [读码]

1. **WebNN 没有指定存储类型**：`services/webnn/tflite/graph_impl_litert.cc` 的 GPU 选项只调了 `SetPrecision(...)`。
2. **LiteRT 按 GPU 厂商选存储**：`delegate_kernel.h:115` `GetStorageType()` → `GetFastestStorageType()`（`ml_drift/webgpu/environment.cc:272`）：
   ```cpp
   return !SupportsImages() || IsApple() || IsNvidia() || IsAMD()
              ? TensorStorageType::kBuffer
              : TensorStorageType::kTexture2D;      // Intel 走这里
   ```
3. **两种存储的复用策略不同**（`ml_drift/webgpu/memory_manager.cc`）：

   | 存储 | 位置 | 策略 | 复用条件 |
   |---|---|---|---|
   | buffer | 第 296 行 | `MemoryStrategy::kGreedyBest` | 生命周期不重叠且大小放得下即可共用 |
   | texture | 第 351 行 | `MemoryStrategy::kEquality` | 描述符与 BHWDC 形状**完全相同**才能共用 |

4. VAE 从 64×64 上采样到 512×512，通道数在 512/256/128 间变化，形状种类多 → texture 几乎无法复用。

## A.2 修复

在 `graph_impl_litert.cc` 的 GPU 选项里（`SetPrecision` 之后）加（已同步到本机源码，`chrome.dll` 为 11-03 15:56 编译，包含此改动）：

```cpp
gpu_options->SetBufferStorageType(
    ::litert::GpuOptions::BufferStorageType::kBuffer);
```

NVIDIA/AMD/Apple 本来就走 buffer，所以全局设置对它们是 no-op。

## A.3 实测数据

**runner，每个模型单独跑 [实测]**

| 模型 | texture（默认）GPU 共享显存 | buffer | 变化 | 单次推理耗时 |
|---|---|---|---|---|
| text_encoder | 0.11 GB | – | – | – |
| unet | 5.67 GB | 4.85 GB | -14% | 1.20 s → 1.25 s |
| **vae_decoder** | **5.10 GB** | **1.83 GB** | **-64%** | 8.2 s → 8.2 s |
| **SD 合计** | 约 10.9 GB | 约 6.8 GB | **-38%** | |

- unet 降得少：显存大头是 1.7 GB 的 fp16 权重。
- 输出只对比了统计量（min/max/mean/std 在 fp16 误差内），**没有做逐元素 `--verify`**。
- 耗时 = "N 次推理 + 最后一次读回"的总时间 ÷ N，不是逐次精确值。

**Chrome 里 1 个 SD Tab（修复前，未开 in-renderer）[实测]**：GPU 进程 private 17.42 GB、GPU 共享显存 11.76 GB；SD 的 renderer private 6.77 GB。

### A.3.1 用户实测的 Chrome 峰值内存（`chrome-peak-memory.ps1`）[用户实测]

**没有优化之前：**

```
Type               PID  PeakMB   CurMB
----               ---  ------   -----
renderer         23536 16104.6  7673.1
gpu-process       8024 13278.4 11427.3
browser           5444   230.9   221.9
renderer         15244   153.2   134.6
utility          23344    60.6    52.6
renderer         15848    58.5    58.5
renderer         19152    56.1    56.1
utility          26072    23.5    23.5
crashpad-handler  8740    11.8    11.8
crashpad-handler  1620    10.7    10.7


GPU process peak (sum)          : 13278.4 MB
Renderer peak (largest)         : 16104.6 MB (PID 23536)
Renderer peak (all, sum)        : 16372.4 MB
GPU + largest renderer peak     : 29383 MB
All chrome processes peak (sum) : 29988.3 MB
```

**优化之后（buffer 存储）：**

第一次 OOM：

```
Type               PID  PeakMB   CurMB
----               ---  ------   -----
renderer         19924 16728.8 12718.6
gpu-process      22384  8943.2    8257
browser          23088   224.8     215
renderer          4912   156.4   135.9
utility          20636    60.7    52.2
renderer          6060    59.6    59.6
renderer          2448    55.5    55.5
utility          24028    23.1    23.1
utility          24324    21.7    21.6
crashpad-handler  8808    10.8    10.7

GPU process peak (sum)          : 8943.2 MB
Renderer peak (largest)         : 16728.8 MB (PID 19924)
Renderer peak (all, sum)        : 17000.3 MB
GPU + largest renderer peak     : 25672 MB
All chrome processes peak (sum) : 26284.6 MB
```

第二次可以生成正确的图片：

```
Type               PID  PeakMB   CurMB
----               ---  ------   -----
renderer         23808 15733.8 11799.8
gpu-process      19884  8477.3  8076.1
browser           4472   221.4   212.9
renderer         25556   151.6   134.4
renderer         22540      90    87.6
utility          25216      60    49.6
renderer         21808    57.1    57.1
renderer          1376    55.6    55.6
utility           2076    23.4    23.4
utility           7732    21.7    21.7
crashpad-handler 21100    10.8    10.8

GPU process peak (sum)          : 8477.3 MB
Renderer peak (largest)         : 15733.8 MB (PID 23808)
Renderer peak (all, sum)        : 16088.1 MB
GPU + largest renderer peak     : 24211.1 MB
All chrome processes peak (sum) : 24902.7 MB
```

**读数（我的解读）**

| | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| GPU 进程峰值 | 13278 MB | 8477~8943 MB | **-4.3~-4.8 GB（-33%~-36%）** |
| 渲染进程峰值**工作集** | 16105 MB | 15734~16729 MB | 基本不变（**但这个口径看不出 arena 修复，见下**） |
| GPU + 最大渲染进程 | 29383 MB | 24211~25672 MB | -3.7~-5.2 GB |

- buffer 存储有效地降低了 **GPU 进程**的内存。
- **口径更正 [实测，2026-11-05]**：`chrome-peak-memory.ps1` 读的是 `PeakWorkingSet64`（常驻页）。TFLite arena 是"提交了但从未写入"的内存，**不进工作集，只占提交量（commit）**；而渲染进程里映射的权重文件（约 3.6 GiB）是文件映射页，进工作集、不进 commit、也不占 PartitionAlloc 池。所以**"渲染进程工作集没变"不能当作"buffer/arena 修复无效"或"问题 B 与 A 无关"的证据**，上面把它当证据的表述作废。
- OOM 的判据是 **commit**（dump 里 private 16.8~19.9 GiB，池上限 16 GiB），应当看 `PeakPagedMemorySize64` / `PrivateMemorySize64`。新增脚本 `peak_memory\chrome-peak-commit.ps1` 同时输出 commit 和工作集。
- 第一次 OOM 的渲染进程 pid 19924 正是 11-04 09:42 OOM dump 里的进程。

#### A.3.2 加入 `ReleaseMemory` 补丁（`stablehlo_composite.cc`）后的验证 [实测]

补丁：`Prepare` 末尾在 `!ShouldPreserveAllTensors()` 时对 decomposition subgraph 调 `ReleaseMemory()`；`Eval_static` 开头重新 `AllocateTensors()`。

| 项 | 结果 |
|---|---|
| runner：VAE 存活的大 arena | 30 个 → **0 个** |
| runner：VAE 私有 commit | 8.85 GiB → **2.00 GiB** |
| runner：VAE 输出 | 与补丁前一致 |
| runner：text_encoder（走 composite 的 Eval 路径） | `--verify` **PASS** |
| Chrome（unetbatch=2，生成成功）渲染进程 | **峰值 commit 12.5 GB**（`chrome-peak-commit.ps1`），当前 commit 4.4 GB，峰值工作集 15.4 GB；活的渲染进程里没有 512/256 MiB 的 arena 块 |

**对照（同一脚本 `chrome-peak-commit.ps1`，用户提供的"优化之前"一次运行）[用户实测]：**

| 渲染进程 | 优化前 (pid 22056) | 补丁后 (pid 10468) | 变化 |
|---|---|---|---|
| 峰值 commit | 15312 MB | 12504 MB | -2.8 GB |
| **生成完后的当前 commit** | **12942 MB** | **4387 MB** | **-8.6 GB** |
| 峰值工作集 | 16500 MB | 15414 MB | -1.1 GB（基本不变） |
| GPU 进程峰值 commit | 12116 MB | 7252 MB | -4.9 GB（这里含 buffer 存储的效果） |

- **当前 commit 少 8.6 GB 是最清楚的信号**：优化前 arena 一直占着（约 6.8 GiB 以上），补丁后已释放。这与 runner 的结果一致（VAE 8.85→2.00 GiB）。
- 峰值 commit 只降了 2.8 GB，**小于 arena 总量**：峰值可能出现在别的阶段（如 unet 编译 / 权重拷贝），或补丁下每个 composite 的 arena 仍会在 `Prepare` 里短暂存在（逐个释放，同一时刻只有一个）。具体原因**未验证**。
- 两次运行是否同配置（unetbatch、是否 buffer 存储）我没有核实，只当作一次对照，不是严格 A/B。
- 优化前 15.3 GB 的峰值 commit 已接近 16 GiB 池上限（OOM 的前提），补丁后余量约 3.9 GB。

- 12.5 GB 对比之前 5 个 OOM dump 的 16.8~19.9 GiB 私有提交，离 16 GiB 池上限多约 4 GB 余量。**这是一致的迹象，不是对照实验**（dump 是崩溃时，这次是一次正常跑完，条件不同；只跑了一次）。
- **没有直接证明 Chrome 二进制含补丁**（`chrome.dll` 是同步过来的，时间戳不可靠）；只有间接证据：没有 arena 大块，且 GPU 进程仍持有 ~6.5 GiB，模型未被销毁。
- **未解释**：渲染进程峰值 commit 12.5 GB 由什么组成、何时出现。推测主要是 unet 编译的主机侧内存加上 JS/ORT 的权重副本，**未验证**。需要时间线（`C:\temp\sd_runner\renderer_mem.ps1 -Seconds 2`）。
- 未测：`Eval_dynamic` 路径；VAE 在 CPU 参考（`--verify`）下曾 OOM 崩溃（0xE0000008），原因推断为 CPU 参考路径本身，**未证明与补丁无关**。
- 仍待：PTL（18 GB）上单 Tab / 多开空 Tab / 静置三种场景。

### A.4 验证步骤（待 PTL）

1. 单 Tab：看任务管理器"共享 GPU 内存"峰值和 GPU 进程 commit（预期比修复前少约 4 GB）。
2. 多开一个空 Tab 再加载 SD：之前必现错误的场景。
3. 静置一段时间看是否仍 crash（**注意：问题 B 与 A 无关，修了 A 不一定能解决静置 crash**）。
4. 数值：对比修复前后生成的图片，必要时用 runner 的 `--verify` 做逐元素对比。

---

# 问题 B：渲染进程 OOM

## B.1 现象与证据：5 个渲染进程 OOM dump [实测]

Crashpad 的 dump 在 `%LOCALAPPDATA%\Chromium\User Data\Crashpad\reports\`，异常码 `0xE0000008`（`kOomExceptionCode`），异常参数 = `{失败的分配大小, ullTotalPageFile, ullAvailPageFile}`（`partition_alloc/oom.cc`，注释说明这些值是 OOM **之后**采集的，可能与 OOM 瞬间不一致）：

| 时间 | 进程 | 失败的分配 | private 已提交 | 崩溃后采到的可用 commit | 进程里有 LiteRT |
|---|---|---|---|---|---|
| 10-29 17:29 | renderer | **2 MiB** | 17.94 GiB | 24.7 GiB | 是 |
| 10-30 19:04 | renderer | **2 MiB** | 19.87 GiB | 31.4 GiB | 是 |
| 10-30 20:02 | renderer | **2 MiB** | 17.92 GiB | 30.2 GiB | 是 |
| 11-02 20:25 | renderer | **2 MiB** | 16.79 GiB | 23.9 GiB | **否** |
| **11-04 09:42** | **renderer（pid 19924）** | **2 MiB** | **19.85 GiB** | 30.0 GiB | 是 |

- 每次失败的都是一次 **2 MiB** 的小分配（PartitionAlloc 的 super page 大小），此时系统还有 24~31 GiB commit，**不是整机内存耗尽**。
- 5 次崩溃时 private 都在 16.8~19.9 GiB。Windows x64 上 PartitionAlloc 单池 `kPoolMaxSize = 16 GiB`（`partition_alloc_public_constants.h:169`，开了指针压缩则为 8 GiB）。
- **11-04 dump 的 PartitionAlloc 保留区**（base `0x27c800000000`，共 32 GiB = 两个 16 GiB 池）：**一个池 0 GiB，另一个池已提交 13.65 GiB（85%）**；其余 private 内存（共 19.85 GiB）在池外。池内 ≥64 MiB 的块：512 MiB ×1、256 MiB ×7。
- **推测**：渲染进程的堆耗尽了这个 16 GiB 池（已提交 + 只保留未提交 + 超级页碎片），于是下一个 2 MiB 分配失败。这是推测，没有从 dump 里直接验证池耗尽。
- 11-04 的 OOM 发生在 ReadTensor 错误（09:29:15）**13.5 分钟之后**（09:42:49），同一个进程。

另有 4 个其他 dump：GPU 进程崩溃 3 个（10-29 15:53、15:55 异常码 `0x0517A7ED`；10-30 09:46 `0x80000003`）和渲染进程断点 1 个（11-02 10:24 `0x80000003`），与 OOM 无关。

## B.2 渲染进程那 17~20 GiB 里有什么

### B.2.1 不是 "conv2d / fully connected 的 fp16 权重被转成 fp32" [实测 + 读码]

- **权重总量太小**：unet 的 fp16 权重共 1.601 GiB，VAE 只有 0.092 GiB。按最终消费它的算子：unet 里 CONV_2D 1.097 GiB（69%）、FULLY_CONNECTED 0.375 GiB（23%）、BATCH_MATMUL 0.128 GiB（8%）。最大的单个权重只有 56 MiB。即使全转成 fp32 也只多 1.6 GiB。
- 图里确实声明了转换：unet 857 个、VAE 199 个 fp16 权重，**每一个**都经过 `DEQUANTIZE FLOAT16→FLOAT32`（`SerializeCastOperation` 的注释：TFLite 要求这样写）。
- 但 GPU 路径上不会真的转：ML Drift 的分区器（`FP16GraphPartitionHelper::RemapFp16InputTensors`）把 fp16 常量直接接到消费节点上，去掉前面的 DEQUANTIZE。主机侧的 fp16→fp32 转换 `AddFloat16ConstAsFloat32Input` **只被 CONCATENATION 用，且只在 fp32 精度下**，conv / FC 不走它。

### B.2.2 三个模型都是 fp16 精度 [用户实测]

你加的日志（`graph_builder_tflite.cc:805`）三张图全部输出 `graph_requires_fp32_precision = 0`。

- `RequiresFloat32Precision` 的规则：任一计算算子（除 cast / quantize / identity）的第一个输入 operand 是 WebNN `float32`，整图按 fp32 精度跑。
- **方法教训**：我曾从 ONNX 推断 text_encoder 会被判成 fp32（因为一个 `Trilu`），**结果是错的**。ORT 在转 WebNN 之前会做常量折叠等优化，ONNX 不等于 WebNN 图。tflite 里 text_encoder 也只有输入输出各 1 个 CAST，与日志一致。**精度以这条日志为准。**
- 图里声明 FLOAT32 ≠ GPU 在跑 fp32：builder 对"TFLite 内置算子不支持 fp16"的算子把张量降成 FLOAT32 并前后插 CAST / DEQUANTIZE（`graph_builder_tflite.cc:1390` 附近的注释：TFLite delegate 在 fp16 精度下会跳过 DEQUANTIZE）。GPU 仍按 fp16 精度执行。

### B.2.3 主要可识别的来源：TFLite 为分解体子图分配的 arena [实测]

**1. 分配者**：给 runner 加了 `--trace-big-allocs=<MiB>`（dispatcher 观察者，大块分配时抓调用栈，推理后打印仍存活的）。8~13 个大块的**调用链完全相同**：

```
aligned_realloc
 ← tflite::SimpleMemoryArena::Commit
 ← tflite::ArenaPlanner::ExecuteAllocations
 ← tflite::Subgraph::AllocateTensors
 ← ... ModifyGraphWithDelegate
 ← LiteRtCompiledModelT::Create
```

符号化只有公共符号（`symbol_level=0`），中间帧的函数名有错位；只信偏移量小的几帧（`aligned_realloc`、`SimpleMemoryArena::Commit [+26]`、`ArenaPlanner::ExecuteAllocations [+1d8]`、`LiteRtCompiledModelT::Create`）。这些分配只发生在 `Create` 时，推理 1 次和 3 次完全一样。

**2. 属于哪些子图（定量吻合，不是逐个分配的直接归属）**

- VAE 的 tflite 有 31 个子图：1 个主图 + **30 个 `instance_norm_decomposition`**（`STABLEHLO_COMPOSITE` 的分解体）。
- 追踪到 **30 个 ≥16 MiB 的分配**，一个子图一个 arena。arena 约等于同时存活的两个最大张量：最大张量 128 MiB 的 7 个子图（子图 18、25~30）→ 256 MiB ×7；最大张量 256 MiB 的子图 24 → 512 MiB ×1，与追踪到的 7×256 + 1×512 一致。其余 5 个 128 MiB 应对应最大张量 64 MiB 的子图（未逐个核对）。
- ML Drift 用 GPU 原生实现这个 composite，**分解体不会被执行**；但 TFLite 的 composite 内核 `Prepare`（`kernels/stablehlo_composite.cc`）会调用 `decomposition_subgraph->AllocateTensors()`，按图里声明的 **FP32** 大小给它分配 arena。这些块**提交了但内容全 0、常驻约 0%**。

**3. 被放大一倍：PartitionAlloc 的对齐分配取整** [读码 + 实测]

- arena 的请求大小是 `2 的幂 + 512 字节`，对齐 64（原因：`+512` 的来源**未查清**）。
- shim 的 `AlignedRealloc`（每次都是新分配 + 拷贝 + 释放旧块）走 PartitionAlloc 的对齐分配；对齐在 16 B 到 16 KiB 之间时，`GetAdjustedSizeForAlignment` **向上取整到下一个 2 的幂**，除非打开 `use_tighter_aligned_alloc_bound`（默认 `kDisabled`，我在 `base/allocator`、`components`、`chrome`、`content` 里没找到任何地方打开它）。
- 每个分配再查 `VirtualQuery`，实际提交区域**正好是请求的 2 倍**（128→256、256→512、512→1024 MiB）。

**4. 三个模型的数字（runner，buffer 存储）**

| 模型 | ≥16 MiB 的 arena | 请求 | 实际提交 | 倍数 |
|---|---|---|---|---|
| text_encoder | 0 | 0 | 0 | – |
| unet | 3 | 70 MiB | 96 MiB | 1.37× |
| **vae_decoder** | **30** | **3504 MiB** | **7008 MiB（6.84 GiB）** | **2.00×** |

→ 这是 **VAE 特有**的：它在 512×512 分辨率上有 30 个 InstanceNorm。VAE 的这 6.84 GiB 全是从未被使用的提交量。

**5. 排除过的假设**

| 假设 | 结果 |
|---|---|
| 来自 CPU 回退（XNNPACK） | **否**：`--gpu-only` 与 GPU+CPU 的内存逐项相同 |
| 来自 ML Drift 的 host-mapped GPU 缓冲 | **否**：这些块不常驻且内容全 0 |
| 与 GPU 精度有关 | **否**：`--precision=fp16/fp32` 切换时这批块一个字节不变 |
| 与存储类型有关 | **否**：texture / buffer 下这批块相同（buffer 只去掉 texture 相关的主机副本，VAE 私有提交 12.32 → 8.85 GiB） |
| Dawn D3D12 伙伴分配器的堆 | **否**：这些是 TFLite arena 经 PartitionAlloc 分配的 |

### B.2.4 还没解释的

- **请求里的 512 字节从哪来**：没查到。即使没有它，arena 大小是任意值，向上取整平均也浪费约 50%。
- **渲染进程 dump 与 runner 对不上**：OOM dump 里池内最大的块是 512 MiB ×1 和 256 MiB ×7，是 runner 里 VAE 的一半（runner 是 1024×1、512×7）。Chrome 里实际的图可能与我 dump 下来的 `vae_decoder.tflite` 不完全相同，我没有验证。所以不能说"渲染进程里的 VAE arena 就是 6.84 GiB"，只能说**机制相同、量级相近**。
- 渲染进程的取整行为（PartitionAlloc 开关）只是读源码和默认值推断，没有在渲染进程里直接验证。
- 渲染进程剩下的内存（池内另有约 7 GiB，池外约 6 GiB）还没分类：已知有 ~1.6 GiB 的权重、3.43 GiB 的单个大块等，尚未对应到具体来源。
- **报错后 13 分钟才 OOM** 的原因未知：log 里这段时间该进程没有输出，只看到 memory dump 请求在 09:42 开始超时（说明崩溃前它已无响应）。

## B.3 修复方向

### 方案 1：让分解体子图声明 FLOAT16

**预测收益（算术，未实测）**：张量大小都是 2 的幂，arena 请求和 PartitionAlloc 的取整都按比例减半 → VAE arena 请求 3504 → 1752 MiB，**实际提交 7008 → 3504 MiB**。

**不能只改 builder**，已查到的约束：

| 约束 | 依据 | 影响 |
|---|---|---|
| composite 内核 `Prepare` 把**算子输入张量的 type 复制到分解子图的输入张量** | `stablehlo_composite.cc` 的 `CopyTensorsShapeAndType`（随后才 `AllocateTensors`） | 分解体声明 FP16 而 composite 的输入是 FP32，type 会被覆盖成 FP32，子图内部 FP16 张量的类型就对不上。**composite 算子自己的输入输出也必须是 FP16** |
| **MEAN 的 `Prepare` 对 FP16 直接返回错误** | `kernels/reduce.cc` `InitializeTemporaries` 的 `switch (input->type)` 没有 `kTfLiteFloat16`，落到 `default: return kTfLiteError` | 分解体里有 MEAN（`ComputeMeanAndVarianceForNormalization`）。声明 FP16 后，composite 的 `Prepare` → 分解体 `AllocateTensors` 失败 → **`CompiledModel::Create` 失败**。**必须给 TFLite 打补丁** |
| builder 硬编码 FP32 | `SerializeInstanceNormalizationAsComposite` 里 `CHECK_EQ(input_tensor_info.data_type, FLOAT32)`；`SerializeNormalizationOperation` 开头 `CHECK_EQ(input_tensor_type, FLOAT32)`，epsilon 常量用 `SerializeTensorWithBuffer<float>` | 要去掉 CHECK，epsilon 按 Float16 写 |
| 周围张量是 FP32 | conv / FC 等不支持 fp16 的算子输出的是 FP32 张量 | composite 前后各要加一个 `CAST`（FP32→FP16、FP16→FP32），每个 InstanceNorm 多 2 个节点，30 个就是 60 个；这会不会增加 GPU 内存和耗时**未知** |
| **CPU 回退的 `Eval` 会坏（可接受）** | composite 只在 GPU context 下生成（`SerializeInstanceNormalizationAsComposite` 开头 `context_device_ != kGpu` 就返回）；分解体只有在 GPU 不接这个 composite 时才会被执行；`sub`/`div`/`mean` 等内置内核没有 FP16 的 `Eval`（我没读 `Eval` 代码，按 TFLite 惯例会返回错误） | GPU 正常接管时完全不受影响；GPU 不接时 `Invoke` 应是**报错而不是静默算错**（推断，未测）。**但 `Prepare` 不能忽略，见下** |

**`Prepare` 与 `Eval` 的区别：composite 只在 GPU 下生成，不能豁免 `Prepare`** [实测 + 读码]

- `Eval` 是实际计算；GPU 接管 composite 后，分解体永远不会被 `Eval`，所以 CPU 内核有没有 FP16 的 `Eval` 可以不关心（只剩"GPU 不接时报错"这个可接受的风险）。
- `Prepare` 是加载阶段在**主机上**对**每个子图的每个节点**做的类型 / 形状检查和内存规划，发生在 delegate 接管**之前**，与设备无关。证据：
  1. `--gpu-only`（没有任何 CPU 加速器）下，那 30 个分解体 arena 的内存与 GPU+CPU **逐项相同**（私有提交都是 8.85 GiB、大块完全一样）[实测]，说明 GPU 路径下分解体的 `Prepare` / `AllocateTensors` 照样执行；
  2. composite 内核的 `Prepare` 里直接调 `decomposition_subgraph->AllocateTensors()`，会对分解体里每个节点（包括 MEAN）调用 `Prepare` [读码]；
  3. 追踪到的分配调用链 `LiteRtCompiledModelT::Create → ModifyGraphWithDelegate → ... → AllocateTensors → ArenaPlanner → SimpleMemoryArena::Commit` [实测]。
- 因此 `reduce.cc` 里 MEAN 对 FP16 的 `return kTfLiteError` 在 GPU-only 下**同样会让 `CompiledModel::Create` 失败**。
- 好消息：**内核侧的硬阻塞只有这一处**（`reduce.cc` 的 `InitializeTemporaries` 加一个 `case kTfLiteFloat16`，约 2 行）；其余算子的 `Prepare` 都不拒绝 FP16。

**支持可行性的事实**：

- 分解体由一个函数 `SerializeNormalizationDecompositionSubgraph` 建出，用**一个 `tensor_type` 参数**统一决定所有张量类型，改动局部。
- ML Drift 的 group_norm 解析（`CompositeGroupNormParser`）**不读分解子图**：只把输入 0 当 value，gamma/beta 用 `ReadTensor` 当常量读。当前 fp16 精度下，分区器本来就把 FP16 的 gamma/beta 直接接到该节点，说明 FP16 常量已经在流通 [推测]。
- TFLite 的 `sub`/`div`/`add`/`mul`/`elementwise(sqrt、rsqrt、square)` 的 `Prepare` 不拒绝 FP16（`sub`/`div` 只要求两个输入 type 相同；`add`/`mul`/`elementwise` 明确支持 `kTfLiteFloat16`）。**只有 `reduce.cc` 是硬阻塞。**
- ML Drift 是否接受 **FP16 类型的 group_norm 输入张量**：我没有读这部分，未验证。

**大致改动（草图，未实现、未编译）**：

```cpp
// services/webnn/tflite/graph_builder_tflite.cc
// SerializeInstanceNormalizationAsComposite (GPU, operand 为 float16 时):
//  1. input:  FP32 张量(上游输出)  --CAST--> FP16 临时张量
//  2. scale/bias: SerializeInputTensorInfo(id, 0, /*operation_supports_float16=*/true)
//                 （保持 FP16 常量，不再 DEQUANTIZE）
//  3. composite 输出: FP16 临时张量 --CAST--> 下游期望的 FP32 张量
//  4. SerializeNormalizationDecompositionSubgraph(..., TensorType_FLOAT16, ...)
//  5. 去掉 CHECK_EQ(..., FLOAT32)
// SerializeNormalizationOperation:
//  - 去掉 CHECK_EQ(input_tensor_type, FLOAT32)
//  - epsilon 按 Float16 写入（SerializeTensorWithBuffer<Float16>）
// SerializeLayerNormalizationAsComposite 同理（unet 用，收益小：unet arena 仅 70 MiB）

// third_party/tflite/src/tensorflow/lite/kernels/reduce.cc
// InitializeTemporaries: switch 里加
//   case kTfLiteFloat16: temp_accum->type = kTfLiteFloat16 /* 或 Float32 */; break;
// （还要确认 MEAN 的其它 Prepare 路径没有别的 FP16 检查）
```

**风险汇总**：要改 TFLite（补丁很小，约 2 行，但需要维护）；builder 侧改动面不小（CAST、常量、两处 CHECK）；收益只有一半（VAE 约 3.5 GiB）；GPU 不接 composite 时回退会报错（仅在这种少见情况下）；ML Drift 是否接受 FP16 的 group_norm 输入未验证。

### 方案 2：不给不会被执行的分解体子图常驻 arena（**已打补丁；runner 与 Chrome 已验证，PTL 待验证**）

> **验证结果见 A.3.2**：runner 里 VAE arena 30→0、commit 8.85→2.00 GiB、输出一致、text_encoder `--verify` PASS；Chrome 渲染进程生成后当前 commit 12942→4387 MB（峰值 commit 15312→12504 MB）。未测：`Eval_dynamic`、CPU 回退连续两次 Invoke、PTL。

- 预期收益最大：runner 里 VAE 约 6.84 GiB（全部）；渲染进程按 dump 里的块大小估计约 3.4～6.8 GiB。
- **TFLite 本身已经在做"用完即放"，只是 `Prepare` 漏了** [读码]（`kernels/stablehlo_composite.cc`）：
  - `Eval` 结束时，若 `!ShouldPreserveAllTensors()`，会调 `decomposition_subgraph->ReleaseMemory()`（第 238-240 行）→ 设计意图就是分解体的 arena 不常驻；
  - `Eval_dynamic` 一开始就 `AllocateTensors()`（第 125 行）→ "释放后再按需分配"是已支持的路径；
  - `Subgraph::ReleaseMemory` → `ArenaPlanner::ReleaseNonPersistentMemory` → `SimpleMemoryArena::ReleaseBuffer`，会真的释放底层 buffer（`simple_memory_arena.cc:473`）；之后 state 为 `kStateUninvokable`，下次 `AllocateTensors()` 走完整重规划 + 重新提交（`core/subgraph.cc:1062-1097`）。
  - 但 `Prepare` 里 `AllocateTensors()`（第 98 行）之后没有释放 → 只要 GPU 接管了 composite，`Eval` 永远不会被调用，这 30 块 arena 就一直常驻。
- **补丁（TFLite，约 5 行）**：
  ```cpp
  // Prepare() 末尾：输出形状已算出，arena 不再需要
  if (!this_subgraph->ShouldPreserveAllTensors()) {
    TF_LITE_ENSURE_OK(context, decomposition_subgraph->ReleaseMemory());
  }
  // Eval_static() 开头：只有 CPU 真正执行分解体时才重新分配
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  ```
- **为什么不在 LiteRT 层调 `Interpreter::ReleaseNonPersistentMemory()`（原设想）**：① `CompiledModel` 不暴露 interpreter，要改 LiteRT 内部；② 释放后 `Eval_static` 不会重新分配，CPU 回退时会报 `Invoke called on model that is not ready`。
- **顺带修掉的潜在 bug** [读码，未实测]：原代码 `Eval` 末尾释放了 arena，但 `Eval_static` 不重新分配 → composite 在 CPU 上执行时，**第二次 Invoke 就可能失败**。补丁在 `Eval_static` 开头加的 `AllocateTensors()` 同时修了这个问题。
- 代价：仅在 CPU 真正执行分解体时，每次 `Eval` 多一次子图重规划（GPU 接管时为 0）。
- 风险 / 待验证：VAE 输出逐元素一致（CPU 参考 `--verify` 曾 OOM，只对比了输出一致/统计量）；CPU 回退下连续推理 ≥2 次正常；GPU 侧 `IsFullyAccelerated` 不变。

### 方案 3：打开 PartitionAlloc 的 `tighter_aligned_alloc_bound`

- 预期收益：去掉 2 倍取整，VAE 约 3.4 GiB。
- **作为产品修复不可行**：这是进程范围的分配器开关，影响渲染进程所有对齐分配，难以以 WebNN 的理由让上游接受。
- **可作为诊断实验**：临时打开，确认渲染进程里是否真有 2 倍取整。方案 2 落地后这块 arena 已不常驻，收益基本被覆盖。

### 方案 4：弄清 `+512` 字节的来源

- `+512` 不是 `ResizableAlignedBuffer::Resize` 加的（它按 `high_water_mark_` 原样分配），而是 planner 算出的 high water mark 本身就带着。
- [推测] 是与两个最大张量同时存活的小张量（如 MEAN / 方差的输出，128 个 fp32 = 512 字节），属于 arena 必需的部分，去不掉。**价值低**，方案 2 落地后无需再查。

### 建议顺序

1. **方案 2（已打补丁）**：WEBNNARL 上用 runner 验证 → Chrome 里验证渲染进程峰值与 OOM。
2. 方案 3 只作诊断实验，不作修复。
3. 方案 1：方案 2 生效后基本没必要（要改 TFLite `reduce.cc`、builder 改动面大、收益只有方案 2 的一半）。

### 方案 2 的真机验证步骤

1. **runner**（先确认 `out\Release\sam_encoder_runner.exe` 的时间是本次编译的）：
   ```powershell
   & $RUN --model=$M\vae_decoder.tflite --run --input=$IN --storage=buffer --trace-big-allocs=100   # 推理后不应再有 30 块分解体 arena
   & $RUN --model=$M\vae_decoder.tflite --run --runs=5 --input=$IN --storage=buffer                # 私有提交预期 8.85 → 约 2 GiB
   & $RUN --model=$M\vae_decoder.tflite --verify --precision=fp32 --input=$IN --storage=buffer     # 输出逐元素一致
   ```
   注意：编译机上的 runner 源码没有 `--storage` / `--trace-big-allocs`，编出来的 runner 不带这两个选项；需要用 WEBNNARL 上带这些选项的 runner 源码重编（TFLite 补丁在 `//third_party/tflite`，与 runner 源码无关）。
2. **CPU 回退**：让 composite 落在 CPU 上（例如屏蔽 GPU 的 group_norm 解析），连续推理 ≥2 次，确认不报 `not ready`。
3. **Chrome**：`chrome-peak-memory.ps1` 对比渲染进程峰值（修复前 15.7～16.7 GB），跑多次看 "Aw, Snap! Out of Memory" 是否消失；同时查 `Crashpad\reports` 有无新的 `0xE0000008`。

---

# 问题 C：GPU 挂起（TDR）

## C.1 证据 [实测]

- Windows Error Reporting 的 `LKD_0x141_Tdr:6_IMAGE_igdkmdn64.sys_ARL_DX12COMPUTE_BBHANG/RINGHANG`：Arrow Lake 上 Intel 驱动因 **DX12 compute 引擎挂起**触发 TDR（bugcheck 0x141）。系统日志里没有 Display 4101（Intel 的 TDR 记在 WER 里，**我最初只查 4101，得出"没有 TDR 记录"是错的**）。
- **本机共 34 条**：10-27 三条、10-29 二十一条、10-30 九条（最后一条 13:59:16）、11-04 一条（09:29:38）。**10-30 14:00 到 11-04 09:29 没有任何记录**，这期间包括我所有 runner 测试和 11-02 在 Chrome 里成功跑 SD。
- 对时：Chrome 报错与对应 WER 记录晚约 23~24 秒（11-04：09:29:15 → 09:29:38；10-30：13:58:52 → 13:59:16）。
- 10-31 08:53 那份 `chrome.txt`（现改名 `chrome_PTL.txt`）里的 ReadTensor 失败**不是这台机器的**：那天没有 TDR 记录，显示器分辨率也对不上（1280×800），应来自 PTL。
- `C:\Windows\LiveKernelReports\WATCHDOG\` 是空的，但目录修改时间 = 11-04 09:29：dump 生成过，之后被 WER 处理后清掉了。报告存档目录（`ReportArchive\Kernel_141_*`）有，但内容需管理员权限。
- **Dawn 没接 DRED**：设备被移除时只读 `GetDeviceRemovedReason` 并记 histogram，无法看到 GPU 卡在哪条命令。
- Dawn 默认的设备丢失回调会打印 `No Dawn device lost callback was set`，设备正常销毁时也会打，**不能单独当作设备丢失的证据**。

## C.2 不能下的结论

- **不能证明 buffer 存储让 TDR 变多或变少**：这台机器在 10-27~10-30（texture 默认）出现过 33 次，11-02 用 texture 成功跑过，buffer 下出现过 1 次。样本太少。11-04 那次还有其他变量（unetbatch=2、`WebNNLiteRTGpuInRenderer`）。
- 这台机器 63 GB，TDR 类型是 compute 引擎挂起，**不像是内存耗尽**；PTL 的现象和这台机器可能是不同机制。

## C.3 与 ORT WebGPU EP 的对照 [读码]

| | ORT WebGPU EP | LiteRT ML Drift |
|---|---|---|
| 每次 `queue.Submit` 的工作量 | 16 个 dispatch（`webgpu_context.h:201`） | 256 个 node（`gpu_backend_webgpu.cc:96`，注释：*Some Intel GPUs fail sometimes with 512 (or more) nodes per encoder*） |
| 中间张量 | `BufferManager` 的 storage buffer，运行时按大小分桶复用，超限的在每次运行结束释放 | texture（Intel 默认），编译期规划，只有形状完全相同才复用 |
| Intel 专用优化 | 有 `vendor/intel/`（subgroup-matrix MatMul/Gemm/Conv、im2col Conv、split-K 表、Transpose 驱动 bug 规避）；识别 `xe-2lpg`/`xe-3lpg` 等，**本机 `xe-lpg` 命中不了其中大部分** | 无 |

ORT 的 Intel 优化都是**算子性能**优化，与内存无关。

## C.4 下一步实验

1. **用 WER 的 TDR 记录当客观判据**（`C:\temp\sd_runner\tdr_since.ps1 "<起始时间>"`），不要只看 Chrome 报错。A/B：同一 URL、同一 flag，每次完全关掉 Chrome 重启，buffer 版和 texture 版各跑 5 次，对比 TDR 次数。
2. **`TdrDelay=30`**（`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`，管理员，需重启）：问题消失说明是某次提交执行太久；仍挂说明是真正的 hang（驱动或 shader）。
3. **减小 `kNumNodesPerCommandEncoder`**（256 → 16/32），只重编 `libLiteRtWebGpuAccelerator.dll`。
4. 在 PTL 上做同样的 A/B，**不要把这台机器的 TDR 和 PTL 的现象混在一起判断**。

---

## 5. 附录：问题 A 的补充分析

### 5.1 为什么 NUC / 本机没事，PTL 有事（GPU 共享显存角度）[推测]

| | PTL（18 GB） | NUC（48 GB） | 本机（63 GB） |
|---|---|---|---|
| 核显共享显存上限（约内存一半） | 约 9 GB | 约 24 GB | 约 32 GB（实测） |
| SD 需要 11.76 GB（修复前 Chrome 实测） | **放不下** | 放得下 | 放得下 |

- PTL 单开能出图：靠换页勉强跑通；多开一个 Tab 就换页抖动、提交超时或资源换入失败，设备被重置。
- 这个解释只针对**共享显存上限**；PTL 上"静置后 crash"很可能还叠加了**问题 B**（渲染进程触到 16 GiB 池），我之前把它归于 commit 耗尽，**证据不支持**。PTL 的 log 里 WebNN 的报错来自渲染进程，说明 PTL 也是 LiteRT 在渲染进程里的路径。**PTL 上的 `Crashpad\reports` 还没看过。**

### 5.2 加压实验（本机）[实测]

4 个 runner 进程空闲占 20 GB 共享显存（整机共享显存 33.4 GB，超过 32 GB 上限），Chrome 跑 SD 仍成功：空闲进程的显存被 Windows 换出（commit 上限从 78.6 涨到 94.8 GB），不影响正在推理的进程。**"别的进程占内存"触发不了，要有多个 SD Tab 同时在推理**（3 × 11.76 GB ≈ 35 GB > 32 GB）才可能复现。

### 5.3 其他未查的点

1. **VAE 单次 8.2 s**（浏览器里 8.5~9.1 s），unet 一步才 1.2 s，偏慢，且与存储类型无关。可能有算子落在很慢的 kernel（4096 token 的 attention softmax、GroupNorm）。执行越久，TDR 风险越高。
2. **dwm.exe 的 GPU 共享显存 12.92 GB**，与 Chrome（11.76 GB）接近，DWM 正常只有几百 MB。疑为 Chrome 的 GPU 分配被算到 DWM 头上，PTL 上会让预算更紧。
3. runner 里 VAE 稳态私有提交 buffer 模式下仍有 8.85 GiB（其中 ~5.9 GiB 即问题 B 的 arena）。
4. LiteRT / Dawn 的 device-lost 原因打到 **stderr**，不会写进 `--log-file`，下次请加 `2>&1 | Tee-Object` 保存。

---

## 6. 工具与改动

### 6.1 runner（`services/webnn/tflite/sam_runner/sam_encoder_runner.cc`）

| 选项 | 说明 |
|---|---|
| `ReadOutput` 支持 fp16 输出 | 原先一律按 f32 memcpy：SD 的输出都是 fp16，统计值是乱码，VAE 还越界读 1.5 MB 导致段错误 |
| `--storage=buffer\|texture` | 强制 `SetBufferStorageType`，默认让 delegate 自己选（Intel 为 texture） |
| `--hold-seconds=N` | 推理后不退出，空闲占着内存 N 秒 |
| `--gpu-only` | 只用 GPU，不加 CPU 回退加速器（chrome 总是两个都加） |
| `--trace-big-allocs=<MiB>` | dispatcher 观察者：记录大块分配的调用栈，推理后打印仍存活的，并附 `VirtualQuery` 的实际提交大小 |

注意：
- `--input` 是**一个 f32 bin，按各输入顺序拼接**，fp16 输入自动转换；INT32 输入（unet 的 `timestep`）要把 int32 位模式按 float 写入（`np.int32([500]).view(np.float32)`）。
- 默认 ramp 输入（`i*0.25`）在 fp16 下会溢出成 NaN/inf，只能用来复现"崩不崩"，**不能**用来判断数值对错。
- runner 日志里的 `Run #N OK in X ms` 只是提交耗时，不等 GPU 完成。
- **同步代码会覆盖 `sam_runner/` 和 `out\Release\sam_encoder_runner.exe`**：11-04 源码曾被覆盖回 09-21 的旧版本，丢了上面的改动，后来重新加回；之后 `out\Release` 里的 exe 又被换回 10-30 的旧版本（症状：VAE 读回时段错误、没有追踪输出）。用新选项前先确认 exe 时间，必要时关掉 Chrome 重编。
- 构建：Chrome 开着时 `dxcompiler.dll` 被占用，链接失败，先关掉 `out\Release` 下的 Chrome。需要 `services/webnn/BUILD.gn` 里引用 sam_runner。
  ```powershell
  $env:DEPOT_TOOLS_WIN_TOOLCHAIN = 0
  autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner
  ```

### 6.2 命令示例

```powershell
$M   = "C:\workspace\junwei\tflite-dump-model\unetbatch=1"
$RUN = ".\out\Release\sam_encoder_runner.exe"
$IN  = "C:\temp\sd_runner\vae_input.bin"

& $RUN --model=$M\vae_decoder.tflite --run --runs=5 --input=$IN                       # texture
& $RUN --model=$M\vae_decoder.tflite --run --runs=5 --input=$IN --storage=buffer      # buffer
& $RUN --model=$M\vae_decoder.tflite --run --input=$IN --storage=buffer --trace-big-allocs=100   # 大块分配的调用栈
& $RUN --model=$M\vae_decoder.tflite --run --input=$IN --storage=buffer --gpu-only    # 排除 CPU 回退
& $RUN --model=$M\vae_decoder.tflite --verify --precision=fp32 --input=$IN --storage=buffer       # GPU vs CPU 逐元素
```

### 6.3 `C:\temp\sd_runner\` 里的脚本

| 脚本 | 作用 |
|---|---|
| `make_inputs.py` | 生成 `unet_input.bin`、`vae_input.bin` |
| `peakmem2.ps1` | 每 0.3 s 采样进程 commit / 工作集 / GPU 共享显存，输出峰值 |
| `start_pressure.ps1`、`pressure_status.ps1` | 起 N 个空闲占显存的 runner / 查看当前共享显存 |
| `tdr_since.ps1` | 列出某时间之后的 TDR 次数；`ev2.ps1` 列最近 45 天全部 |
| `renderer_mem.ps1` | 每 5 s 记录各 chrome 进程的 private / 工作集 / GPU 共享显存到 CSV |
| `parse_dmp.py`、`dmp_summary.py`、`dmp_regions.py`、`pool_split.py` | 解析 Crashpad dump：异常、各 dump 汇总、已提交区域分布、PartitionAlloc 池占用 |
| `live_regions.py`、`live_sample.py` | 查看存活进程的大块已提交区域、常驻比例与内容采样 |
| `analyze_tflite.py`、`analyze_final_consumer.py`、`per_subgraph.py` | tflite 的权重按 dtype / 最终消费算子、最大激活张量、各子图的激活大小 |
| `onnx_dtype_scan.py` | ONNX 的 dtype 传播（**仅供参考，不能代替 `graph_requires_fp32_precision` 日志**） |

## 7. 相关文件位置

| 内容 | 位置 |
|---|---|
| WebNN 创建 LiteRT GPU 选项（含 `SetBufferStorageType`） | `services/webnn/tflite/graph_impl_litert.cc:606` |
| 是否需要 fp32 精度 | `graph_builder_tflite.cc` `RequiresFloat32Precision`（约 1772 行），日志在 805 行 |
| 权重外置 / 内联阈值 | `graph_builder_tflite.cc:88-89` `kMaxInlineBufferSize`、`kFlatbufferSafetyThreshold`（本地已改大） |
| InstanceNorm composite / 分解体 | `graph_builder_tflite.cc:8288`（composite）、`8496`（分解子图）、`3928`（`SerializeNormalizationOperation`） |
| TFLite composite 内核（分配分解体 arena） | `third_party/tflite/src/tensorflow/lite/kernels/stablehlo_composite.cc` |
| MEAN 对 FP16 的 `Prepare` 限制 | `third_party/tflite/src/tensorflow/lite/kernels/reduce.cc`（`InitializeTemporaries` 的 `switch`） |
| PartitionAlloc 对齐分配取整 | `base/allocator/partition_allocator/src/partition_alloc/internal/partition_root_internal.h`（`GetAdjustedSizeForAlignment`） |
| PartitionAlloc 池大小 | `partition_alloc_public_constants.h:169` |
| 存储类型选择 | `third_party/litert/src/ml_drift_delegate/delegate/delegate_kernel.h:115` |
| buffer / texture 复用策略 | `third_party/ml_drift/src/ml_drift/webgpu/memory_manager.cc:296 / 351` |
| ML Drift 的 group_norm composite 解析 | `third_party/litert/src/ml_drift_delegate/tflite/model_builder.cc`（约 6195 行、7312 行） |
| 每 encoder node 数 | `third_party/litert/src/ml_drift_delegate/delegate/gpu_backend_webgpu.cc:96` |
| host-mapped 指针开关 | `third_party/litert/src/ml_drift_delegate/delegate/delegate_webgpu.cc:346` |
| ORT WebGPU EP 提交粒度（对照） | `C:\Users\webnn\workspace\onnxruntime\onnxruntime\core\providers\webgpu\webgpu_context.h:201` |
| 渲染进程 OOM dump | `%LOCALAPPDATA%\Chromium\User Data\Crashpad\reports\` |
