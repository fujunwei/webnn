# SDXL-Turbo text encoder：ML Drift GPU 算子支持与 DequantizeLinear 优化分析

> 范围：`C:\Users\junwei\workspace\tflite-dump-model\sdxl-turbo` 下的两个模型
> （`model0.tflite` / `model1.tflite`，分别由 `text_encoder_model_qdq_q4f16.onnx`、
> `text_encoder_2_model_qdq_q4f16.onnx` 经 Chrome WebNN→LiteRT 转换得到）。
>
> 所有结论来自本次会话的实测或读码。**标注"未验证"的地方没有实测，不要当作结论。**
> 所有代码改动**均未提交**。

---

## 1. 起点：GPU ML Drift 是否支持全部算子

用 `sam_native_runner`（`tflite-dump-model\sam_gpu_verify_package` 里的自包含副本）做编译检查，
结果（本机 adapter 是 WARP，只做编译检查，没有跑推理）：

| 模型 | 下沉 ML Drift 的算子 | `IsFullyAccelerated` |
|---|---|---|
| model0（旧版转换） | 467 / 744（约 63%） | 0 |
| model1（旧版转换） | 1293 / 2052（约 63%） | 0 |

不支持的算子：

- `CAST: inputs[0] should not be a constant tensor`（IR 前端的支持检查）
- `DEQUANTIZE:`（原因为空，见 §4.2）

运行环境上遇到的两个坑（已绕过，原目录未改动）：

- `out\upstream_bots_debug\sam_encoder_runner.exe` 缺 `webgpu_dawn.dll` 启动失败；
- 包内自带的 `dxcompiler.dll` 加载失败（错误 87），在临时目录
  `tflite-dump-model\_sdxl_run` 里换成 Windows SDK Redist 的 DXC 后跑通。

## 2. 模型结构：ONNX 里全是同一种模式

用 `tflite-dump-model\_tools\onnx_dq_analyze.py`（无第三方依赖，直接解析 protobuf）统计：

| 模型 | DequantizeLinear 个数 | 模式 |
|---|---|---|
| `text_encoder_model_qdq_q4f16.onnx` | 72 | 全部相同 |
| `text_encoder_2_model_qdq_q4f16.onnx` | 193 | 全部相同 |

统一模式：

- 输入：uint4 常量 `[K,N]`（如 `[3072,768]`、`[1280,1280]`）
- scale：fp16 `[K/32, N]`；零点：uint4 `[K/32, N]`
- `axis=0`、`block_size=32`
- 每个 DQ 只有一个消费者，是 `MatMul` 的第 2 个输入（权重）
- **所有零点都是 8**（uint4 零点 8 等价于对称 int4 零点 0）

## 3. Chromium 侧原有实现的问题

`GraphBuilderTflite::SerializeDequantizeLinear` 的原逻辑：

- 常量输入一律走**模拟路径**（`input_operand.kind != kConstant` 判断，对应
  TFLite issue #78748 的 TODO）；
- GPU 上 int4 也走模拟路径（ML Drift 没有运行时 4-bit 张量路径）；
- 不支持 uint4（`dequantize_linear_input` 与 `OperandDataTypeToTFLite` 都排除）。

模拟路径生成的算子（截图中的结构）：

```
CAST(输入) , CAST(零点)
RESHAPE + RESIZE_NEAREST_NEIGHBOR + RESHAPE   （block-wise 展开 scale 和零点，×32）
SUB , MUL
```

其中常量 `CAST` 被 ML Drift 拒绝，回退到 CPU，图被切成 GPU/CPU 交替。

## 4. 已验证的发现

### 4.1 常量 DEQUANTIZE（TFLite issue #78748）

- issue 在 2026-01-22 被关闭，**没有关联修复提交**，只能实测。
- **CPU（XNNPACK）**：常量 int8 / uint8 / uint4 的融合 `DEQUANTIZE` 现在可以正常工作，
  问题不再复现 → 可以去掉常量判断。
- **GPU（ML Drift）**：常量 int8 / uint8 的融合 `DEQUANTIZE` 失败
  （`Failed to modify graph with delegate`）→ GPU 上必须保留常量走模拟路径。
- 最终实现：`is_constant_on_gpu`（GPU 且输入为常量）与 `is_int4_on_gpu`
  （GPU 且 int4 / uint4）两个条件，CPU 上常量走融合。

### 4.2 ML Drift 的 `DEQUANTIZE:`（原因为空）

来自只读代码梳理（`litert/tflite/delegates/utils.cc:253`）：fp16 常量 `DEQUANTIZE`
由 `FP16GraphPartitionHelper` 折叠进消费者，该处返回 false 但不填
`unsupported_details`，日志就是 `DEQUANTIZE: ` 加空串。**这是有意的、无害的**，
节点仍会被全量委托采用。

### 4.3 ML Drift 的一个独立 bug：常量 `[2,4]` 作为二元算子的第二输入

- 现象：`ADD(运行时 [2,4], 常量 [2,4])` 第二行复用了第一行的常量值；常量 `[1,8]` 正常。
- 对照实验确认与 dequantize 无关（页面里的 `control: float32 constant [2,4]`）。
- 原因（读码）：运行时 `[2,4]` 映射成 `BHWC(2,1,1,4)`，常量 `[2,4]` 被映射成
  `BHWC(1,1,2,4)`（`model_builder.cc:2858`、IR 前端的 `GetRightAlignedBHWC`），
  kernel（`ml-drift/.../kernels/elementwise.cc:629-630`）看到常量 `b==1` 把 B 坐标固定为 0。
- **未修复**。建议的修法：只有常量秩低于运行时操作数时才右对齐，秩相同时用与运行时
  一样的 `ExtractTensorShape`。可参考 `constant_layout_issue.patch`。

### 4.4 ML Drift 旧前端的常量 DEQUANTIZE 形状 bug（读码，未实测）

`DequantizeOperationParser::ReadConstTensor`（`model_builder.cc:2099-2109`）把 2D / 3D
常量标成 `BHWC(1,1,H,W)`，而输出值是 `BHWC(N,1,1,C)`，`CheckShapes` 报 shape mismatch。
和补丁 09、29 修过的常量 `TRANSPOSE` 是同一类 bug。**未修复**。

## 5. 做过的实现与改动

### 5.1 Chromium：`services/webnn/tflite/graph_builder_tflite.{h,cc}`

| 改动 | 说明 |
|---|---|
| uint4 `DequantizeLinear` | `dequantize_linear_input` 与 `constant` 允许 uint4；uint4 序列化为 int4（每个半字节异或 `0x8`，即减 8），零点同样减 8，反量化结果不变；非常量 uint4 用 `CHECK` 兜底 |
| `TryFoldConstantDequantizeLinear`（仅 GPU） | 输入、零点、scale 全为常量且元素数 ≤ `kMaxFoldedConstantElements`（1<<20）时，构建期直接算成 float32 常量，不生成算子；支持 broadcast 与 block-wise。折叠整个算子而不是只折叠 `CAST`，是因为只折叠 `CAST` 会让 `SUB` 两个输入都是常量，GPU delegate 对此会拒绝或算错 |
| **int4 全连接**（CPU 和 GPU） | 见 §5.2 |

另外：曾尝试过给 `SerializeElementWiseUnary` 的 `cast` 加常量折叠，以及把 `where`
的常量条件直接转成 bool 常量，**已按要求撤销**（`git checkout`）。

### 5.2 int4 全连接（主要优化）

思路借鉴 ORT 的 `DQMatMulNBitsFusion`（`onnxruntime/core/optimizer/qdq_transformer/
selectors_actions/qdq_actions.h`）：不展开成 float，让消费者直接吃 4-bit 权重。

条件（`IsInt4FullyConnectedWeights`）：

- DQ 的输入、scale、零点都是常量；输入是 int4 / uint4 的 2D 权重，零点类型与输入相同；
- scale 是 fp16，全为正的正常值（XNNPACK 要求）；
- 沿 K 分块，`block_size % 32 == 0`；
- **零点必须对称**（uint4 为 8，int4 为 0），因为 TFLite 的 block-wise 量化是对称的，
  XNNPACK 会**静默忽略零点张量**；
- DQ 的唯一消费者是 `matmul` 或 `gemm` 的权重输入，输出不是 graph output。

满足时：DQ 不生成任何算子；消费者序列化为
`FULLY_CONNECTED`（`keep_num_dims=true`），权重为 INT4 `[N,K]`
（uint4 每个半字节减 8），scale 转置成 fp16 `[N,K/bs]`，量化参数用
`BlockwiseQuantization{scales, zero_points=-1, block_size}`。

**为什么必须支持 `gemm`**：ORT 在分区之前用 `MatMulAddFusion` 把 `MatMul + Add(bias)`
融合成 `Gemm`，3D 输入两边各插一个 `Reshape`。WebNN 图里权重的消费者是 `gemm`，
只匹配 `matmul` 时全部失配。证据：旧 `model0` 里 66 个 FULLY_CONNECTED 都带偏置、
权重是运行时 `TRANSPOSE` 的输出（float32），INT4 权重个数为 0。

`gemm` 的两种方向都支持：`b_transpose=false` 时 B 为 `[K,N]`，`true` 时为 `[N,K]`。

**偏置**：fp16 偏置经 `DEQUANTIZE` 解包后，XNNPACK 的 block-wise int4 全连接不接受
（CPU 上报 `failed to delegate FULLY_CONNECTED`），所以这条路径里偏置序列化为
fp32 常量。该原因是推断，修复后 CPU 用例通过，可视为得到验证。

### 5.3 ml-drift：`gpu_model_builder.cc`（补丁 39）

`GpuModelBuilder::GetWeights` 的 int4 重载把打包后缓冲区大小算成
`GetTotalElementsCountForLayout(...) / 4`（从 int2 重载照搬），int4 每字节两个权重，
应为 `/ 2`。后果：`RearrangeWeights` 的 XNNPACK transpose 写越界，GPU 进程访问冲突崩溃
（用 cdb 从 crash dump 定位到 `xnn_x8_transposec_ukernel`，调用链为
`GetWeights → RearrangeWeightsInt8AsUint4 → RearrangeWeights`）。

- 补丁：`webnn\segment_anythings\patches\39-mldrift-fc-int4-weights-buffer-size.patch`
  （`patches/README.md` 未更新，需要时补一行）。
- 已用 `build_accelerator_dll.ps1 -Mode opt` 重新编译 `libLiteRtWebGpuAccelerator.dll`，
  部署到 `out\Release`，旧 DLL 备份为 `libLiteRtWebGpuAccelerator.dll.bak_20261028`。

## 6. 测试结果

测试页：

- `webnn\route-a-webgpu\webnn_uint4_dequantize_test.html`：小张量的 uint4 / int8 / uint8
  `dequantizeLinear`，含 `a + float32 常量` 对照用例。
- `tflite-dump-model\sdxl-turbo\webnn_blockwise_dequantize_test.html`：block-wise，
  `?device=cpu|gpu` 可单独跑一个设备。

`webnn_blockwise_dequantize_test.html` 在 CPU（XNNPACK）和 GPU（ML Drift，
需加 `--no-sandbox`）上 **12 个用例全部通过**：

| 用例类别 | CPU | GPU |
|---|---|---|
| int4 全连接（matmul，uint4 / int4，`[768,3072]`、`[3072,768]`、`[1,77,768]` 输入） | OK | OK，`devices=["gpu"]` |
| int4 全连接（gemm 加偏置，含 `bTranspose`） | OK | OK，`devices=["gpu"]` |
| 零点不对称，小张量（GPU 上折叠） | OK | OK，`devices=["gpu"]` |
| 零点不对称，`[768,3072]`（GPU 上模拟路径） | OK | OK，`devices=["gpu","cpu"]`（常量 `CAST` 回退 CPU） |

相对误差（最大误差 / 输出范围）：int4 全连接路径 CPU 约 0.4%、GPU 约 0.1% 到 0.9%
（XNNPACK 动态量化激活、GPU fp16 计算所致）；其余路径远小于此。

用 `--webnn-tflite-dump-model` 导出模型并解析确认，int4 全连接用例生成的是
`CAST → FULLY_CONNECTED(INT4, blockwise, fp32 const bias) → CAST`，
没有 `DEQUANTIZE`、`TRANSPOSE` 和模拟算子。

## 7. 尚未完成 / 未验证

1. **没有用真实的 sdxl-turbo 模型验证**。需要用现在 `out\Release` 的 Chrome 重新转换两个
   ONNX，覆盖 `model0.tflite` / `model1.tflite`，再用 `_tools\tflite_ops.js` 检查：
   - model0 应有 66 个 INT4 FULLY_CONNECTED，model1 应有 192 个；
   - 不应再有权重相关的 `RESIZE_NEAREST_NEIGHBOR` / `SUB` 模拟；
   - 模型体积应明显变小（权重回到 4-bit）。
   - 当前目录里的 `model0/1.tflite`（20:15 生成）是**旧版本**，INT4 权重个数为 0。
2. 剩余的 `BATCH_MATMUL`（model0 的 22 个、model1 的 65 个，注意力的 Q·Kᵀ 与 ·V）
   两个输入都是运行时张量，不在这次优化范围内。
3. 整个 text encoder 是否全部下沉 GPU、输出与 CPU 参考的整体一致性，**未验证**。
   本机 adapter 是 WARP 时只能做编译检查。
4. §4.3、§4.4 的 ML Drift bug 没有修复。大于 1<<20 个元素的零点不对称 4-bit 常量
   仍走模拟路径，其中常量 `CAST` 在 CPU 上执行。
5. 折叠上限 `kMaxFoldedConstantElements`（1<<20）的取值没有经过权衡：放大会让 4-bit
   权重展开成 float32，模型体积变成 8 倍。
6. 没有检查 `webnn` 目录下是否有测试断言「uint4 不支持」或旧的常量行为，需要同步改。
7. 所有改动均**未提交**：Chromium（`graph_builder_tflite.{h,cc}`）、
   ml-drift（`gpu_model_builder.cc`）都在工作区。

## 8. 相关文件索引

| 文件 | 作用 |
|---|---|
| `chromium\src\services\webnn\tflite\graph_builder_tflite.{h,cc}` | 所有 Chromium 侧改动 |
| `chromium\src\third_party\ml-drift\ml_drift\common\gpu_model_builder.cc` | int4 权重缓冲区大小修复 |
| `webnn\segment_anythings\patches\39-mldrift-fc-int4-weights-buffer-size.patch` | 上一项的补丁 |
| `tflite-dump-model\_tools\onnx_dq_analyze.py` | 统计 ONNX 里 DequantizeLinear 的模式 |
| `tflite-dump-model\_tools\tflite_ops.js` | 统计 tflite 算子与 FC / BMM 权重类型 |
| `tflite-dump-model\_analyze_cast.js`、`_sdxl_json\` | 早期用 flatc 导出的 JSON 与分析脚本（可删） |
| `tflite-dump-model\sdxl-turbo\webnn_blockwise_dequantize_test.html` | block-wise 测试页 |
| `webnn\route-a-webgpu\webnn_uint4_dequantize_test.html` | uint4 / int8 / uint8 小张量测试页 |

## 9. 复现要点

- 强制走 LiteRT（否则会用 ORT 后端，改动不会生效）：
  `--disable-features=WebNNOnnxRuntime`。
- 跑 GPU 需要 `--no-sandbox --enable-unsafe-webgpu`（见
  `route-a-webgpu-windows\README.md` §3）。
- 导出 tflite：`--webnn-tflite-dump-model=<目录>`，文件名为 `model0.tflite`、
  `model1.tflite` …… 按创建顺序编号。
- 重新编译 `chrome` 前要先关掉所有 `out\Release\chrome.exe`，否则会因文件被占用在
  `v8_context_snapshot.bin` 这一步失败。
- 重新编译加速器 DLL：`route-a-webgpu-windows\scripts\build_accelerator_dll.ps1 -Mode opt`，
  再复制到 `out\Release`。
