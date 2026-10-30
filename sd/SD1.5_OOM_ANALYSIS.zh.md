# Stable Diffusion 1.5 UNet OOM 问题分析

日期：2026-10-15

## 1. 问题现象

- `C:\Users\junwei\workspace\tflite-dump-model\stable_diffusion_turbo\no_weights_model` 下的模型可以在 Chrome 的 WebNN/LiteRT 后端正常跑。
- `C:\Users\junwei\workspace\tflite-dump-model\stable_diffusion_1.5\no_weights_model` 下的模型在同样的路径下会 OOM 崩溃，Chrome 崩溃栈如下：

```
chrome.dll!partition_alloc::TerminateBecauseOutOfMemory(unsigned __int64)
chrome.dll!partition_alloc::internal::OnNoMemory(unsigned __int64)
chrome.dll!partition_alloc::internal::PartitionExcessiveAllocationSize(unsigned __int64)
chrome.dll!partition_alloc::PartitionRoot::AlignedAlloc<16>(unsigned __int64,unsigned __int64)
...
chrome.dll!tflite::ResizableAlignedBuffer::Resize(unsigned __int64,bool *)
chrome.dll!tflite::ArenaPlanner::ExecuteAllocations(int,int)
chrome.dll!tflite::Subgraph::IsFullyDelegated(void)
chrome.dll!tflite::Subgraph::AllocateTensors(enum tflite::Subgraph::InliningStrategy)
chrome.dll!tflite::Subgraph::ReleaseMemory(void)
chrome.dll!tflite::Subgraph::ModifyGraphWithDelegateImpl(struct TfLiteDelegate *)
chrome.dll!tflite::impl::Interpreter::ModifyGraphWithDelegateImpl(...)
chrome.dll!LiteRtCompiledModelT::Create(...)
chrome.dll!LiteRtCreateCompiledModel()
chrome.dll!webnn::litert::GraphImplLiteRt::CreateAndBuild(...)
```

关键点：`PartitionExcessiveAllocationSize` 是 Chromium PartitionAlloc 对**单次分配请求**大小的硬上限检查（而不是系统内存耗尽的累积效应），说明 `ResizableAlignedBuffer::Resize` 在为 TFLite 的 arena 扩容时，单次请求的字节数就已经超过了 PartitionAlloc 允许的单次分配上限，直接被拒绝并终止进程。

## 2. 诊断方法

由于没有可用的 Python/flatc 环境，直接用 Node.js 手写了一个 FlatBuffers 读取器（按 TFLite 官方 schema.fbs 的字段顺序解析 vtable），对比了 SD1.5 和 Turbo 的 `unet.tflite` / `text_encoder.tflite` / `ave_decoder.tflite` 三个模型文件的 graph input/output shape 和各 tensor 的字节大小。后续又用仓库里已有的 `webnn\segment_anythings\sam_native_runner\inspect_tflite.py` 做了交叉验证，结果一致。

## 3. 关键发现：SD1.5 的 UNet 把 batch size 写死成了 2

| | SD1.5 `unet.tflite` | SD-Turbo `unet.tflite` |
|---|---|---|
| `timestep` | `[2]` INT32 | `[1]` FLOAT16 |
| `sample`（latent 输入） | `[2, 4, 64, 64]` FP16 | `[1, 4, 64, 64]` FP16 |
| `encoder_hidden_states` | `[2, 77, 768]` FP16 | `[1, 77, 1024]` FP16 |
| `out_sample` | `[2, 4, 64, 64]` FP16 | `[1, 4, 64, 64]` FP16 |
| tensor 总数 | 4613 | 4100 |
| 全图 tensor 字节数总和（忽略 arena 复用的粗略上限估算） | **≈28.9 GB** | **≈16.9 GB** |

SD1.5 的 UNet 把 classifier-free guidance（CFG）的 unconditional + conditional 两路 prompt 合并成了 **batch=2** 的一次前向推理，这是 SD1.5 官方 pipeline 的常见写法。而 SD/SDXL-Turbo 类模型通常 `guidance_scale=1`、不用 CFG，天然是 **batch=1**。

`text_encoder.tflite` 同样是 batch=2 vs batch=1：

- SD1.5 `text_encoder.tflite`：`input_ids [2,77]` → `last_hidden_state [2,77,768]`，`pooler_output [2,768]`。
- SD1.5 `ave_decoder.tflite`（VAE decoder）：`latent_sample [1,4,64,64]` → `sample [1,3,512,512]`，batch=1（VAE 解码只处理最终去噪结果，一次一张图，两边都一样）。

## 4. 放大因素（已用二进制直接验证并修正）：两边都已 fuse SDPA，但 SD1.5 的 attention score 矩阵比 Turbo 大 3.2 倍

**更正**：本节早先版本基于 `webnn\sd\SDPA_FUSION.zh.md`（2026-10-14，当时记录的融合补丁尚未提交/未验证）推测 SD1.5 的 32 个 attention block 未被融合。经直接解析 `.tflite` flatbuffer 的全部 142 个 subgraph 后发现**这个推测是错的**：SD1.5 和 Turbo 的 `unet.tflite` 实际上**都已经 fuse 了 SDPA**，现象如下：

- 主 subgraph 0 里**没有**裸的 `SOFTMAX` 节点，取而代之的是 141 个 `STABLEHLO_COMPOSITE`（builtin code 206）算子（其中 32 个对应 attention，其余是 GroupNorm/LayerNorm 等其他 composite）。
- 每个 SDPA composite 都带一个单独命名为 `scaled_dot_product_attention_decomposition` 的 fallback 分解子图——只有当 delegate 不支持原生执行该 composite 时才会走这个分解图，里面才有真正的 `SOFTMAX` + `BATCH_MATMUL`。
- SD1.5 和 Turbo 的 `unet.tflite` 里**各自都有 32 个**这样的分解子图，融合状态一致，不是两者的差异点。

真正的差异点是：**这 32 个分解子图里物化的 attention score 矩阵，SD1.5 比 Turbo 大得多**。对比两边最高分辨率（4096 token / 64×64）的 self-attention 分解子图里最大的张量：

| | shape | fp16 字节数 |
|---|---|---|
| SD1.5 | `[16, 4096, 4096]`（batch=2 × heads=8 折叠成一维） | **536,870,912（≈512MB）** |
| Turbo | `[1, 5, 4096, 4096]`（batch=1 × heads=5，分开存放） | 167,772,160（≈160MB） |

差距是 **3.2 倍**，不只是 batch 翻倍（2x）——SD1.5 这一层的 head 数（8）也比 Turbo（5）多，两者叠加。其余分辨率层级同样全面放大（实测全部 8 组 score shape 对比）：

| 分辨率/序列长度 | SD1.5 score shape | SD1.5 字节(fp16) | Turbo score shape | Turbo 字节(fp16) |
|---|---|---|---|---|
| 4096×4096 self-attn（5处） | `[16,4096,4096]` | 536,870,912 | `[1,5,4096,4096]` | 167,772,160 |
| 1024×1024 self-attn（5处） | `[16,1024,1024]` | 33,554,432 | `[1,10,1024,1024]` | 20,971,520 |
| 4096×77 cross-attn（5处） | `[16,4096,77]` | 10,092,544 | `[1,5,4096,77]` | 3,153,920 |
| 1024×80 / 1024×77 cross-attn（5处） | `[16,1024,80]` | 2,621,440 | `[1,10,1024,77]` | 1,576,960 |
| 256×256 self-attn（5处） | `[16,256,256]` | 2,097,152 | `[1,20,256,256]` | 2,621,440 |
| 256×160 / 256×77 cross-attn（5处） | `[16,256,160]` | 1,310,720 | `[1,20,256,77]` | 788,480 |

如果当前测试环境的 WebNN/LiteRT delegate 没有原生支持这个 STABLEHLO_COMPOSITE（即每次都要走 fallback 分解子图），`ArenaPlanner::ExecuteAllocations` 就需要为这些分解子图里高达 512MB 的单一张量（加上同一分解子图里另外 2~3 个同形状的中间张量）规划 arena 空间，叠加主图里本来就存在的大量百兆级卷积反量化临时张量，这是比单纯"batch 翻倍"更具体、更大的内存放大点，也比"未融合"的旧推测更站得住脚。

相关 feature flag 是 `kApplySdpaFusion`（`graph_builder_tflite.cc:73`），可以用 `--disable-features=...,ApplySdpaFusion` 做开关对比实验，但既然两模型融合状态相同，这个开关预计不会改变 SD1.5 对 Turbo 的相对劣势，只影响两者的绝对内存需求。

## 5. 附带发现（与本次 OOM 无直接关系，但影响排查，需留意）

`stable_diffusion_turbo\no_weights_model\` 目录下文件名与实际图内容对不上：

- `text_encoder.tflite` 实际内容是 **VAE decoder**（输入 `latent_sample [1,4,64,64]` → 输出 `sample [1,3,512,512]`）。
- `ave_decoder.tflite` 实际内容是 **text encoder**（输入 `input_ids [1,77]` → 输出 `last_hidden_state [1,77,1024]`，1024 维与 OpenCLIP 吻合）。

SD1.5 目录下命名是对的（`text_encoder.tflite` 就是文本编码器，`ave_decoder.tflite` 就是 VAE decoder）。如果有脚本按文件名加载 Turbo 模型，需要确认有没有因此加载反。

## 6. 结论与建议

**根因（高置信度）**：SD1.5 UNet/text_encoder 图里把 CFG 的两路 batch 写死成了 `batch=2`，且该分辨率下 head 数（8）也比 Turbo（5）多，两者叠加使得 32 个 SDPA composite 对应的 fallback 分解子图里，最大的 attention score 张量达到 `[16,4096,4096]`（≈512MB，fp16），是 Turbo 同位置张量（≈160MB）的 3.2 倍。如果当前 delegate 没有原生执行这个 composite（退回分解子图），TFLite arena planner 在 `AllocateTensors` 阶段就需要为这些巨型张量规划空间，叠加主图里本来就存在的大量百兆级卷积反量化临时张量，使单次 `ResizableAlignedBuffer::Resize` 请求的内存超过了 PartitionAlloc 单次分配的硬上限，被 `PartitionExcessiveAllocationSize` 直接拒绝并终止进程。两模型的 SDPA 融合状态本身是一致的（都融合了），**不是**融合与否的差异，而是融合后 fallback 路径里张量尺寸的差异。

**建议的验证/修复方向（按优先级）**：

1. **把 SD1.5 的 UNet / text_encoder 重新导出为 batch=1**，CFG 的 uncond/cond 两路改成分两次调用（跟 Turbo pipeline 的用法一致），而不是在图里写死 batch=2。这是最直接、最容易验证的改动，预期能让激活内存减半。
2. **确认当前测试用的 Chrome build / 硬件后端是否真的原生执行了这 32 个 STABLEHLO_COMPOSITE（SDPA）**，而不是退回到 `scaled_dot_product_attention_decomposition` fallback 子图。如果是退回到 fallback，512MB 级别的 score 矩阵就会被实际物化分配；可以用 `--disable-features=...,ApplySdpaFusion`（`graph_builder_tflite.cc:73`）对比两种路径下的内存表现，但注意这个开关不会改变 SD1.5 相对 Turbo 的劣势比例，只影响两者各自的绝对内存需求。
3. 以上两点可以分别验证、也可以叠加验证：即使 delegate 原生支持 SDPA composite（不走 fallback），batch=2 依然会让激活内存需求翻倍，仍建议同时处理第 1 点。

## 7. 本次分析未发现的内容

- 仓库中没有生成这些 `no_weights_model/*.tflite` 文件的独立导出脚本；它们是通过 Chrome 自带的 WebNN→TFLite dump 功能生成的（`--webnn-tflite-dump-model=<dir> --webnn-tflite-dump-model-skip-weights`），文档见 `webnn\sd\SDPA_FUSION.zh.md`。
- 没有找到加载这三个 `.tflite` 文件的完整 WebNN SD pipeline 测试页面（`webnn_sdpa_fusion_test.html` 只是一个独立的 SDPA 微基准测试，不跑完整 SD pipeline）。
- 没有找到专门记录本次 SD1.5 OOM 的既有日志文件。
