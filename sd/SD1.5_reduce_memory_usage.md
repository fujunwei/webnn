# SD 1.5 在 WebNN(LiteRT) GPU 路径上的内存占用分析与降低方案

> 日期：2026-11-02　机器：WEBNNARL（Intel Arc 140T，63 GB 内存，驱动 32.0.101.6452）
> 模型：`C:\workspace\junwei\tflite-dump-model\unetbatch=1\`（`text_encoder.tflite` / `unet.tflite` / `vae_decoder.tflite`）
>
> 文中每条结论都标了依据：**[实测]** = 在本机跑出来的数据，**[读码]** = 只读了源码，**[推测]** = 由现象推出、尚未验证。

---

## 1. 现象

- 在 `chrome.exe --enable-features=WebMachineLearningNeuralNetwork --disable-features=WebNNDirectML,WebNNOnnxRuntime --enable-unsafe-webgpu` 下加载
  `https://10.239.115.25:8080/demos/stable-diffusion-1.5/?unetbatch=1&safetychecker=false&ort=latest`，偶发：
  - `graph_impl_litert.cc:544 Failed to compute: Failed to invoke the compiled model`，随后 `D3D12CreateDevice failed with DXGI_ERROR_DEVICE_HUNG`；
  - VAE 读回失败：`buffer_content_tflite.cc:222 Mapping the readback buffer of a tensor failed: 4`（`4 = WGPUMapAsyncStatus_Aborted`，即 device 已丢失，或 map 完成前 buffer 被 unmap/销毁）。
- PTL（18 GB 内存）：只要多开一个空 Tab，另一个 Tab 加载 SD 就出错；只开一个 Tab 能出图，但静置一段时间后 crash。
- NUC（48 GB）：开很多 Tab 都加载 SD 也能跑，静置很久不 crash。
- 本机（63 GB）：怎么试都复现不了（见第 5 节）。

→ 现象与**内存大小**强相关。

## 2. 结论（一句话）

**Intel GPU 上 LiteRT/ML Drift 默认用 TEXTURE_2D 存中间张量，而 texture 只能"形状完全相同才复用"，导致 VAE 这类形状种类很多的网络几乎没有复用，显存占用异常大。强制用 BUFFER 存储（greedy 按大小复用）可把 SD 的 GPU 共享显存从约 10.9 GB 降到约 6.8 GB，速度基本不变。**

## 3. 根因（读码）

1. **WebNN 没有指定存储类型** [读码]
   `services/webnn/tflite/graph_impl_litert.cc:606-615` 只调了 `gpu_options->SetPrecision(...)`，没有调 `SetBufferStorageType`。
2. **LiteRT 按 GPU 厂商选存储** [读码]
   `third_party/litert/src/ml_drift_delegate/delegate/delegate_kernel.h:115` `GetStorageType()`：
   `use_buffer_storage_type` 为 false 时调 `GetFastestStorageType()`
   （`third_party/ml_drift/src/ml_drift/webgpu/environment.cc:272`）：
   ```cpp
   return !gpu_info.SupportsImages() || IsApple() || IsNvidia() || IsAMD()
              ? TensorStorageType::kBuffer
              : TensorStorageType::kTexture2D;      // Intel 走这里
   ```
3. **两种存储的内存复用策略不同** [读码]（`ml_drift/webgpu/memory_manager.cc`）

   | 存储 | 位置 | 策略 | 复用条件 |
   |---|---|---|---|
   | buffer | 第 296 行 | `MemoryStrategy::kGreedyBest` | 生命周期不重叠，且大小放得下即可共用 |
   | texture | 第 351 行 | `MemoryStrategy::kEquality` | 描述符与 BHWDC 形状**完全相同**才能共用 |

4. VAE 从 64×64 上采样到 512×512，通道数在 512/256/128 间变化，形状种类很多 → texture 基本无法复用 → 每种形状各占一块。

## 4. 实测数据（runner，本机）

### 4.1 默认配置（texture）每个模型单独跑的峰值 [实测]

| 模型 | 模型文件 | 进程 commit 峰值 | 工作集峰值 | GPU 共享显存峰值 |
|---|---|---|---|---|
| text_encoder | 0.25 GB | 1.53 GB | 1.50 GB | 0.11 GB |
| unet | 1.72 GB | 13.78 GB | 12.83 GB | 5.67 GB |
| vae_decoder | **0.10 GB** | 12.39 GB | 5.51 GB | **5.10 GB** |

- 编译完这些内存就一直占着，推理结束也不释放（vae 时间序列：第 3.8 s ~ 11.8 s 共享显存一直是 5.10 GB）。
- 权重只有 100 MB 的 vae 占 5.1 GB 显存，说明几乎全是中间张量。

### 4.2 `--storage=buffer` 对比 [实测]（`--runs=5`，输出统计量一致）

| 模型 | GPU 共享显存 texture → buffer | 工作集 | 单次推理耗时 | 输出 min/max/mean/std |
|---|---|---|---|---|
| vae_decoder | **5.10 → 1.83 GB（-64%）** | 5.52 → 2.65 GB | 8.2 s → 8.2 s | -2.092/1.710/-0.1025/0.7216 → -2.104/1.694/-0.1025/0.7219 |
| unet | 5.67 → 4.85 GB（-14%） | 12.83 → 12.03 GB | 1.20 s → 1.25 s（约 +4%） | -3.979/3.848/0.0003/0.9325 → -3.973/3.854/0.0003/0.9327 |
| **SD 合计**（含 text_encoder 0.11 GB） | **约 10.9 → 约 6.8 GB（-38%）** | | | |

- unet 降得少：显存大头是 1.7 GB fp16 权重，中间张量本来就不多。
- 耗时 = "5 次推理 + 最后一次读回"的总时间 ÷ 5，不是逐次精确值。
- 输出差异在 fp16 误差范围内；只对比了统计量，**没有做逐元素 `--verify`**。

### 4.3 Chrome 里 1 个 SD Tab 的真实占用 [实测]

| 进程 | 已提交（Private） | 工作集 | GPU 共享显存 |
|---|---|---|---|
| GPU 进程 | 17.42 GB | 14.69 GB | 11.76 GB |
| SD 的 renderer | 6.77 GB | 6.18 GB | – |
| 合计 | 约 24 GB | 约 21 GB | 11.76 GB |

与 runner 估算（约 10.9 GB 显存）基本吻合，浏览器里多约 0.9 GB。

## 5. 为什么 NUC/本机没事，PTL 有事

| | PTL（18 GB） | NUC（48 GB） | 本机（63 GB） |
|---|---|---|---|
| 核显共享显存上限（约为内存一半） | 约 9 GB | 约 24 GB | 约 32 GB（实测） |
| SD 需要 11.76 GB（Chrome 实测） | **放不下** | 放得下 | 放得下 |
| SD 约 24 GB commit vs 物理内存 | **超过物理内存**，靠 pagefile | 放得下 | 放得下 |

- PTL 单开能出图：SD 本身已超上限，靠换页勉强跑通；多开一个 Tab 就换页抖动，提交超时/资源换入失败 → device 被重置 → `DEVICE_HUNG` / readback `Aborted`。VAE 一次用到的中间显存最多，所以常在 VAE 读回出错。 [推测，符合已有日志]
- PTL 静置后 crash：约 24 GB commit 常驻，物理内存只有 18 GB，系统内存紧张时 OOM 退出或被换出的 GPU 资源换不回来。 [推测]
- **本机做过加压实验**：4 个 runner 进程空闲占 20 GB 共享显存（整机共享显存用量 33.4 GB，已超 32 GB 上限），Chrome 跑 SD 仍成功。原因：空闲进程的显存被 Windows 换出（commit 上限自动从 78.6 涨到 94.8 GB），不影响正在推理的进程。→ **"别的进程占内存"触发不了，起作用的是 SD 这一个进程自身的工作集有没有超过显存上限。** 想在大内存机器上复现，需要同时有多个 SD Tab **都在推理**（3 × 11.76 GB ≈ 35 GB > 32 GB）。

## 6. 修复方案

在 `services/webnn/tflite/graph_impl_litert.cc` 的 GPU 选项里（`SetPrecision` 之后）加：

```cpp
gpu_options->SetBufferStorageType(
    ::litert::GpuOptions::BufferStorageType::kBuffer);
```

- 可以只对 Intel 生效，也可以全部厂商生效（NVIDIA/AMD 本来就走 buffer，所以全局设置对它们是 no-op）。
- 不建议改 ML Drift：texture 天生只能按形状相等复用。

### 验证步骤（待你在 PTL 上做）

1. 重编 chrome，PTL 上跑同样命令，**单 Tab**：记录任务管理器里"共享 GPU 内存"峰值和 Chrome GPU 进程的 commit（预期共享显存比修复前少约 4 GB）。
2. **多开一个空 Tab** + 加载 SD：之前必现错误的场景，看是否消失。
3. **静置**一段时间看是否仍 crash。
4. 数值：对比修复前后生成图片；必要时用 runner 的 `--verify`（见第 8 节）做逐元素对比。

## 7. 尚未解决 / 值得继续查

1. **commit 仍然很高**：buffer 模式下 vae 显存 1.83 GB 但 commit 仍 9.49 GB，unet 12.96 GB。这部分与中间张量无关，疑为主机侧"已提交未使用"内存（`delegate_webgpu.cc:346` 写死 `enable_host_mapped_pointer = true`、D3D12 堆、权重主机副本）。可能是 PTL 静置 crash 的另一个因素。[未查]
2. **VAE 单次 8.2 s**（浏览器里 8.5–9.1 s），而 unet 一步才 1.2 s，对 512×512 的 VAE 来说偏慢，且无论哪种存储都一样。可能有算子落到很慢的 kernel（如 4096 token 的 attention softmax、GroupNorm/InstanceNorm）。执行时间越长，TDR 风险越高。[未查]
3. **提交粒度**：LiteRT 每个 command encoder 装 256 个 node（`gpu_backend_webgpu.cc:96-98`，注释写明 *"Some Intel GPUs fail sometimes with 512 (or more) nodes per encoder"*）；ORT WebGPU EP 每 16 个 dispatch 就 `Submit`（`webgpu_context.h:201`）。显存吃紧、换页时，粒度粗的更容易超时。可作为第二个实验：把 256 改小再测。[读码，未实验]
4. **dwm.exe 的 GPU 共享显存 12.92 GB**，与 Chrome（11.76 GB）接近，DWM 正常只有几百 MB。疑为 Chrome 的 GPU 分配被算到 DWM 头上（共享句柄/合成表面），也可能是真的多了一份。PTL 上会让预算更紧。[未查]
5. `chrome.txt`（`C:\workspace\junwei\tflite-dump-model\unetbatch=1\chrome.txt`）里的显示器是 1280×800 缩放 1.5，与本机（1920×1080）不符，很可能是 PTL 抓的；log 里没有进程崩溃，是 JS 抛的 `readTensor` DOMException。LiteRT/Dawn 的 device-lost 原因打到 **stderr**，不会写进 `--log-file`，下次请加 `2>&1 | Tee-Object` 保存。

## 8. 复现 / 测量工具

### 8.1 runner（`services/webnn/tflite/sam_runner/sam_encoder_runner.cc`）

在原有功能上做了这些改动：

| 改动 | 说明 |
|---|---|
| `ReadOutput` 支持 fp16 输出 | 原先一律按 f32 memcpy：SD 三个模型输出都是 fp16，统计值是乱码，vae 还越界读 1.5 MB 导致段错误 |
| `--storage=buffer\|texture` | 强制 `SetBufferStorageType`，默认让 delegate 自己选（Intel 为 texture） |
| `--hold-seconds=N` | 推理后不退出，空闲占着内存 N 秒（制造内存压力，不增加 GPU 计算负载） |

构建（需要 `services/webnn/BUILD.gn` 里引用 sam_runner，否则 GN 报 unknown target）：

```powershell
$env:DEPOT_TOOLS_WIN_TOOLCHAIN = 0
autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner
```

注意：
- runner 的 `--input` 是**一个 f32 bin，按各输入顺序拼接**，fp16 输入自动转换；INT32 输入（unet 的 `timestep`）走 `Write<float>` 分支，所以要把 int32 位模式按 float 写入（`np.int32([500]).view(np.float32)`）。
- 默认 ramp 输入（`i*0.25`）在 fp16 下会溢出成 NaN/inf，只能用来复现"崩不崩"，**不能**用来判断数值对错。
- runner 日志里的 `Run #N OK in X ms` 只是提交耗时，不等 GPU 完成，不可信。要看真实耗时，用"N 次推理 + 最后读回"的总时间。

### 8.2 命令示例

```powershell
$M   = "C:\workspace\junwei\tflite-dump-model\unetbatch=1"
$RUN = ".\out\Release\sam_encoder_runner.exe"

# 对比 texture / buffer（输入 bin 见下）
& $RUN --model=$M\vae_decoder.tflite --run --runs=5 --input=C:\temp\sd_runner\vae_input.bin
& $RUN --model=$M\vae_decoder.tflite --run --runs=5 --input=C:\temp\sd_runner\vae_input.bin --storage=buffer

# 数值逐元素对比（GPU vs CPU，fp32 排除 fp16 舍入）
& $RUN --model=$M\vae_decoder.tflite --verify --precision=fp32 --input=C:\temp\sd_runner\vae_input.bin --storage=buffer
```

- `C:\temp\sd_runner\make_inputs.py` 生成 `unet_input.bin`（timestep=500 + 随机 sample + 随机 encoder_hidden_states）与 `vae_input.bin`。
- `C:\temp\sd_runner\peakmem2.ps1`：每 0.3 s 采样进程 commit/工作集和 `\GPU Process Memory(pid_*)\Shared Usage`，输出峰值。
  `-Extra "--runs=5 --storage=buffer" -Log <file>`。
- `C:\temp\sd_runner\start_pressure.ps1 -Count N -HoldSeconds S` / `pressure_status.ps1`：起 N 个空闲占显存的 vae 进程 / 查看当前共享显存。结束后 `Stop-Process -Name sam_encoder_runner`。

## 9. 相关文件位置

| 内容 | 位置 |
|---|---|
| WebNN 创建 LiteRT GPU 选项 | `services/webnn/tflite/graph_impl_litert.cc:606` |
| 存储类型选择 | `third_party/litert/src/ml_drift_delegate/delegate/delegate_kernel.h:115` |
| Intel → texture 的判断 | `third_party/ml_drift/src/ml_drift/webgpu/environment.cc:272` |
| buffer / texture 复用策略 | `third_party/ml_drift/src/ml_drift/webgpu/memory_manager.cc:296 / 351` |
| 每 encoder node 数 | `third_party/litert/src/ml_drift_delegate/delegate/gpu_backend_webgpu.cc:96` |
| host-mapped 指针开关 | `third_party/litert/src/ml_drift_delegate/delegate/delegate_webgpu.cc:346` |
| ORT WebGPU EP 提交粒度（对照） | `C:\Users\webnn\workspace\onnxruntime\onnxruntime\core\providers\webgpu\webgpu_context.h:201` |
| 测量日志与脚本 | `C:\temp\sd_runner\` |
