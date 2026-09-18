# 全常量输入算子被 LiteRT GPU 分区器拒绝 + SAM 的 fp16/fp32 精度归因

> 背景：WebNN 走 LiteRT / ML Drift GPU 时出现两条同源报错：
> `MAXIMUM: Op can only handle 1 or 2 operand(s).`（gather indices clamp）和
> `SUB: Op can only handle 1 or 2 operand(s).`（decoder 开头的
> `(dequantize - dequantize) * dequantize`）。
> 顺着这条线还查清了「为什么 WebNN 把几乎所有张量都转成 fp32」，以及
> **encoder 拿 FP16、decoder 拿 FP32** 的真正原因。
>
> 本文档是 [analysis.zh.md](analysis.zh.md) `§B3`（`DIV: Op can only handle
> 1 or 2 operand(s).`，24 个，当时标记为「需要 dump 出这 24 个 DIV 节点…看具体
> 是哪种」）的答案。
>
> 相关文档：[analysis.zh.md](analysis.zh.md)、[gpu_op_bisect.zh.md](gpu_op_bisect.zh.md)
> （其 §「`graph_requires_fp32_precision`」只给了 bisect 手法，未做归因）、
> [layer_norm_fused_impl.md](layer_norm_fused_impl.md)

---

## 0. 结论速览（2026-09-08）

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | 报错根因是**二元 elementwise 算子的两个输入全是常量**，不是「不支持常量输入」 | 已读码确认 |
| 2 | fp16 DEQUANTIZE 的 fuse **不是 bug 本身**，它只是把问题暴露出来 | 已读码确认 |
| 3 | 「把数据转成 fp32」主要是 **WebNN 自己干的**（`SerializeInputTensorInfo` 默认参数），不是 ML Drift | 已读码确认 |
| 4 | ML Drift 的 fp16 由**图级 precision 选项**决定，与张量原始类型无关 | 已读码确认 |
| 5 | decoder 判定为 fp32 是**正确**的：它 5 个图输入全是 fp32 | 已静态分析模型确认 |
| 6 | decoder `op#0..#4` 是一个全常量子表达式，是报错的靶子 | 静态分析推断，**未经 delegate 日志确认** |

**没有实跑过任何模型。** 下面所有模型结构结论均来自静态分析 `.tflite` 文件
（脚本见 §6.0）；所有代码结论均来自读源码。

---

## 1. 根因：分区器只接受 (2 runtime) 或 (1 runtime + 1 const)

`third_party/litert/src/tflite/tools/versioning/gpu_compatibility.cc:1219-1230`：

```cpp
case kTfLiteBuiltinMaximum:
case kTfLiteBuiltinMinimum:
case kTfLiteBuiltinSub:
// ...所有二元 elementwise
{
  if (!CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/2,
                                /*required_const_inputs=*/0,
                                /*required_outputs=*/1).ok() &&
      !CheckInputsConstsOutputs(op_sig, /*required_runtime_inputs=*/1,
                                /*required_const_inputs=*/1,
                                /*required_outputs=*/1).ok()) {
    return absl::InvalidArgumentError(
        "Op can only handle 1 or 2 operand(s).");
  }
```

只有两种组合通过：**(2 runtime, 0 const)** 或 **(1 runtime, 1 const)**。
**(0 runtime, 2 const) 被拒。**

`is_const` 的判据在 `tflite/tools/versioning/op_signature.cc:47`：

```cpp
tensor_spec.is_const = (tfl_tensor->allocation_type == kTfLiteMmapRo);
```

即凡是带 flatbuffer buffer 的张量都算 const。

ML Drift 侧的根本原因在
`third_party/ml-drift/ml_drift/common/selectors/operation_selector.cc:629-661`：
它把常量输入折进 `ElementwiseAttributes`，`inputs.size() == 2` 走双 runtime 路径，
`inputs.size() == 1 && attributes.has_value()` 走单 runtime + 常量属性路径，
**零 runtime 输入时没有可 lower 的东西**，落到 `absl::UnimplementedError`。

> 所以措辞要准确：不是「ML Drift 不支持 constant input」，而是
> **「不支持所有输入都是 constant」**。单个常量输入是它最喜欢的形态。

同一条消息也出现在 `gpu_compatibility.cc:876`（GATHER 的检查），
GATHER 同样接受 `(1 runtime + 1 const)` —— 这一点是 §3 修法的基础。

> 注意区分：`analysis.zh.md` 里记录的
> `LESS/MAXIMUM: Can't parse inputs with const tensors.` 是**另一条**消息，
> 来自 parser 阶段而非 versioning 检查阶段。

### 1.1 RESHAPE 也有同样的约束

`gpu_compatibility.cc:986-990`：

```cpp
case kTfLiteBuiltinReshape:
  RETURN_IF_ERROR(CheckInputsOutputs(op_sig,
                                     /*required_runtime_inputs=*/1,
                                     /*required_outputs=*/1));
```

**恰好 1 个 runtime 输入。** 一个数据输入是常量的 RESHAPE 有 0 个 → 被拒。
CAST 同理（`:667-669`）。这决定了折叠时必须**直接产出目标形状的常量**，
不能「折叠 + 再 reshape」。

---

## 2. fp16 DEQUANTIZE 的 fuse 机制：它只是揭穿，不是元凶

WebNN 对**常量** fp16→fp32 转换刻意发 `DEQUANTIZE` 而非 `CAST`
（`services/webnn/tflite/graph_builder_tflite.cc:3563-3573`）：

```cpp
if (constant_input_tensor &&
    input_tensor_type == ::tflite::TensorType_FLOAT16 &&
    output_tensor_type == ::tflite::TensorType_FLOAT32) {
  // TFLite expects the DEQUANTIZE operator to be used to pass float16
  // weights to float32 operators, but WebNN represents this with the cast
  // operator.
  return ::tflite::CreateOperator(builder_, GetOperatorCodeIndex(
      ::tflite::BuiltinOperator_DEQUANTIZE), ...);
}
```

delegate 侧 `third_party/litert/src/tflite/delegates/utils.cc:236-277`
（`FP16GraphPartitionHelper::IsNodeSupported`）：

1. **`:239-254`** 遇到「输入是 fp16 **常量**」的 DEQUANTIZE 时，记入
   `constant_dequant_map_` 并 `return false`（该 DEQUANTIZE 本身不进分区）。
2. **`:262-265`** 检查**其它**节点前，先 `RemapFp16InputTensors(node, &orig_inputs)`
   把节点输入**临时改指回原始 fp16 常量张量**，再调基类 `IsNodeSupported`。
3. **`:270-275`** 检查完把输入还原。

所以 `gpu_compatibility.cc` 评估 SUB 时，看到的输入**已经是 fp16 常量**，
不是 DEQUANTIZE 的输出 → 2 const / 0 runtime → 报错。

`BuildModelEnforceIO`（`delegates/gpu/common/model_builder.cc:3500-3508`）里
真正 build 时也会 `continue` 跳过这些 DEQUANTIZE 节点。

### 2.1 关键推论：去掉 DEQUANTIZE 没用

把 fp16 常量在 build 期直接序列化成 fp32 常量、不发 DEQUANTIZE，
SUB 依然是两个常量输入，依然被拒。保持 fp16 也一样。
**DEQUANTIZE 那层间接曾经让输入「看起来像 runtime 张量」，fp16 remap 只是撕掉了这层伪装。**

反过来也给出一个诊断线索：**能触发这个错，说明该算子的两个输入必然都是常量。**
如果有一侧来自图输入，`utils.cc:242-243` 的 `IsConstantTensor` 不成立、不进 remap 表，
就是 1 runtime + 1 const，直接通过。见 §6.3 的 op#2 vs op#8 对照。

---

## 3. 案例一：gather indices 的 clamp（MAXIMUM）—— 已改

`SerializeGatherIndices()`（`graph_builder_tflite.cc:5910` 附近）为了防止运行时
indices 越界读，发 `maximum(indices, -N)` / `minimum(indices, N-1)`，再用
`less/add/select` 把负索引搬正，共 5~6 个算子。
当 WebNN 的 `indices` 操作数本身是**常量**时，`MAXIMUM` 两个输入全常量 → 报错。

**修法（方案 A，已落地并编译通过，未测）**：新增
`GraphBuilderTflite::ClampConstantIndices()`，在 build 期用 CPU 把 clamp 和
负索引归一化算完，产出**一个 INT32 常量张量**。

改动点：
- `graph_builder_tflite.h` — 新增 private 声明
- `graph_builder_tflite.cc` — 新增实现 + `SerializeGather()` /
  `SerializeGatherND()` / `SerializeScatterND()` 三处加常量分支

要点：
- 用 `GetConstantInt64Value()` 统一吃下 int32 / uint32 / int64
- 折叠结果恒为 INT32（受限于合法 WebNN 维度），因此原本被
  `gpu_compatibility.cc:881`（GATHER 只收 INT32 indices）挡回 CPU 的
  int64/uint32 indices 现在也能上 GPU
- 折叠时**直接序列化成扁平 1-D 形状**，不发 RESHAPE（理由见 §1.1）
- 三处常量分支都不调用 `SerializeInputTensorInfo(indices_operand_id)`，
  避免往 flatbuffer 塞死常量张量。这跟随
  `SerializeGatherElements()`（`:6069`）/ `SerializeScatterElements()`（`:9626`）
  的既有约定

**已知未验证项**：见 §8。

---

## 4. ML Drift 怎么支持 float16 计算

### 4.1 精度是图级显式选项，不是从张量类型推断的

`ml_drift/common/precision.h:25-31`：

```cpp
enum class CalculationsPrecision { F32, F32_F16, F16 };
// F32     - 所有数据和数学运算都用 F32
// F16     - 所有数据和数学运算都用 F16
// F32_F16 - 同 F16，但 Conv / DepthwiseConv / FullyConnected /
//           ConvolutionTransposed 的累加器用 F32
```

GPU 侧张量存储类型由 `ml_drift/common/precision.cc:32-38` 单独决定：

```cpp
DataType DeduceDataTypeFromPrecision(CalculationsPrecision precision) {
  if (precision == CalculationsPrecision::F32) return DataType::FLOAT32;
  else                                          return DataType::FLOAT16;
}
```

**只看 precision。**

WebNN 在 `services/webnn/tflite/graph_impl_litert.cc:481-483` 显式设定：

```cpp
gpu_options->SetPrecision(graph_requires_fp32_precision
                              ? ::litert::GpuOptions::Precision::kFp32
                              : ::litert::GpuOptions::Precision::kFp16);
```

`:480` 还有一行现成的 debug 日志可以直接看当前图拿到什么：

```cpp
LOG(ERROR) << "==== WebNN Setting GPU precision: "
           << (graph_requires_fp32_precision ? "FP32" : "FP16");
```

### 4.2 完整数据流：fp32 只是宿主侧中转

```
flatbuffer 里的 fp16 常量
   │  CreateVectorCopyData<float>()
   │  litert/src/tflite/delegates/gpu/common/model_builder_helper.cc:258-268
   ▼
宿主侧 fp32 暂存（GraphFloat32 / TensorFloat32）   ← 类型名里就写着 Float32
   │  TensorDescriptor::UploadData()
   │  ml-drift/ml_drift/common/task/tensor_desc.h:169-192
   │  data_type_ = DeduceDataTypeFromPrecision(precision) = FLOAT16
   ▼
GPU buffer / texture：fp16（字节数减半）
   │
   ▼
shader 里 half4 算术 / write_imageh
```

转换发生在 `tensor_desc.h:539-561` 最内层 —— `dst` 是 `Span<half>`，
`value` 是 `float`，逐元素隐式转换：

```cpp
FromType value;                    // float
value = src[cpu_index];
int gpu_index = desc.GetLinearIndex(shape, b, x, y, d, s, c);
dst.at(gpu_index) = value;         // Span<half>
```

`data_.resize(GetSizeInBytesForShape(shape_))` 按 `data_type_` 算尺寸，
所以 **GPU 显存确实减半**。

**这个来回是无损的**：fp16→fp32 精确，fp32→fp16 对原本就是 fp16 的值返回同一位模式。
真实代价只有初始化期的宿主峰值内存（2×）和一趟逐元素转换耗时。

### 4.3 ML Drift 本身能直接吃 fp16

`ml_drift/common/task/tensor_desc.cc:2187-2196` 和 `:2209-2219` 是专门的 fp16 重载，
直接 `memcpy`、零转换：

```cpp
TensorDescriptor CreateConstantLinearTensorDescriptor(
    DataType data_type, TensorStorageType storage_type,
    const Tensor<Linear, DataType::FLOAT16>& src) {
  ...
  tensor_desc.data_.resize(src.shape.v * sizeof(half));
  std::memcpy(&tensor_desc.data_[0], src.Data(), src.shape.v * sizeof(half));
```

**fp32 暂存是 TFLite delegate 前端（`GraphFloat32`）的性质，不是 ML Drift 的。**
经 delegate 进来就绕不开。这不是 WebNN 侧能改的。

---

## 5. 「为什么把所有数据都转成 fp32」—— 是 WebNN 自己干的

`graph_builder_tflite.h:221-225`：

```cpp
base::expected<TensorInfo, std::string> SerializeInputTensorInfo(
    OperandId operand_id,
    QuantizateParametersOffset quantize_params = 0,
    bool operation_supports_float16 = false,      // ← 默认 false
    bool fuse_dequantize_quantize = false);
```

该参数为 false 且操作数是 fp16 时，`graph_builder_tflite.cc:1319-1334`
插一个 CAST/DEQUANTIZE 转 fp32。

全文件 30+ 个调用点，**只有这几处传 `true`**：

| 位置 | 算子 |
|---|---|
| `:5820` | gather（且仅 `context_device_ == kGpu`） |
| `:7644` `:7651` | LayerNorm custom call |
| `:9546` `:9558` | （scatter 相关） |
| `:5363` `:5368` | elementwise unary，按 `op.kind` 条件判断 |

**其余全部传 false。** 所以 WebNN 现在的策略是：TFLite 图里几乎一律用 fp32 张量，
真正的 fp16 计算**完全依赖 delegate 那个全局 `Precision::kFp16` 开关**。

这解释了 encoder 里 229 个、decoder 里 178 个 DEQUANTIZE 的来源。

---

## 6. 实测：encoder vs decoder

### 6.0 分析脚本

`C:\Users\junwei\workspace\tflite-dump-model\inspect_precision.py`（只读，打 stdout）：

```
python3 inspect_precision.py <model.tflite> [<model2.tflite> ...]
```

复用了 `segment_anything_verify/verify_cpu_gpu/dump_opcodes.py` 里那个手写
flatbuffer Table walker（`tflite` pip 包在本机没装；注意从
`services/webnn/` 下 `import tflite` 会被同名目录误命中而假成功）。
枚举名从 `third_party/tflite/.../schema.fbs` 现场解析，不会和 schema 漂移。

### 6.1 对比数据

| | encoder | decoder |
|---|---|---|
| WebNN 报的 precision | **FP16** | **FP32** |
| 图输入 | 1 个，FLOAT32 | 5 个，**全是 FLOAT32** |
| 图输出 | 1 个 FLOAT32 | 3 个 FLOAT32 |
| FLOAT32 张量 | **1270** | 635 |
| FLOAT16 张量 | 184 | 166 |
| DEQUANTIZE | 229 | 178 |
| CAST | 3 | 36 |
| 算子总数 | 1148 | 666 |
| 非豁免且带 FLOAT32 输入的算子 | 916 | 410 |

decoder 图输入：

```
[8]   has_mask_input     FLOAT32  [1]
[17]  mask_input         FLOAT32  [1, 1, 256, 256]
[158] image_embeddings   FLOAT32  [1, 256, 64, 64]
[186] point_labels       FLOAT32  [1, 1]
[239] point_coords       FLOAT32  [1, 1, 2]
```

### 6.2 两个模型的 TFLite 图**都**是 fp32 的

encoder 拿到 FP16，可它 1270 个 FLOAT32 张量 vs 184 个 FLOAT16。
所以「TFLite 张量类型」和「WebNN 判定的精度」是脱钩的两套东西 ——
`RequiresFloat32Precision()` 读 **mojom 操作数**，TFLite 张量类型只是序列化产物（§5）。

decoder 为什么是 FP32：`RequiresFloat32Precision()`（`:1667`，末行 `:1830-1831`）

```cpp
return (GetOperand(input_operand_id).descriptor.data_type() ==
        OperandDataType::kFloat32);
```

它的注释写明了期待的模式：*"A graph is considered a fp16 graph if it casts input
float32 to float16 and performs all other operations in float16."*

- encoder 符合：1 个 fp32 输入 + 3 个 CAST，之后 mojom 操作数全 fp16
- decoder 不符合：5 个图输入全 fp32，36 个 CAST 在 fp32/fp16 之间来回倒

**判定是正确的，不是误判。** decoder 的 mojom 图本身就是 fp32 激活 + fp16 权重。

### 6.3 案例二：decoder 最开头就是报错靶子

```
op#0  DEQUANTIZE  in([0]FLOAT16/const)          -> out([1]FLOAT32)
op#1  DEQUANTIZE  in([2]FLOAT16/const)          -> out([3]FLOAT32)
op#2  SUB         in([1]FLOAT32, [3]FLOAT32)    -> out([4]FLOAT32)   ← 两输入都源自 fp16 常量
op#3  DEQUANTIZE  in([5]FLOAT16/const)          -> out([6]FLOAT32)
op#4  MUL         in([4]FLOAT32, [6]FLOAT32)    -> out([7]FLOAT32)
```

`op#0..#4` 是一个**完全常量的子表达式** `(A-B)*C`，三个叶子都是 fp16 常量。

紧邻的 op#8 是同样的 SUB 但**不报错**，因为一侧来自图输入：

```
op#5  CAST        in([8]FLOAT32)                -> out([9]FLOAT16)    ← has_mask_input
op#6  DEQUANTIZE  in([10]FLOAT16/const)         -> out([11]FLOAT32)
op#7  CAST        in([9]FLOAT16)                -> out([12]FLOAT32)
op#8  SUB         in([11]FLOAT32, [12]FLOAT32)  -> out([13]FLOAT32)   ← 1 const + 1 runtime，通过
```

这组对照同时验证了 §2.1 的推论。

---

## 7. 两个独立问题，别混在一起

### 问题 A — 全常量输入算子被拒

**修法**：build 期常量折叠。gather indices 侧已做（§3）；elementwise 侧
（decoder `op#0..#4`）待做。

**收益边界**：只解除分区拦截（那几个算子不再掉回 CPU、少几个 GPU/CPU 同步点），
外加删掉 5 个算子。**不改变精度。**

### 问题 B — decoder 整图跑 fp32

666 个算子本该 fp16 却跑 fp32。收益远大于 A，修法与 A 完全无交集：

- 根源是 decoder 的 **mojom 图**用 fp32 激活。要么改模型（像 encoder 那样入口
  cast 到 fp16），要么放宽 `RequiresFloat32Precision()` 的全图一刀切 ——
  现在**一个** fp32 算子就把 666 个算子全拖到 fp32，这个启发式过于保守。
- 正交优化：让更多算子传 `operation_supports_float16=true`（GPU 上），
  少发几百个 DEQUANTIZE/CAST。但这只减算子数和宿主转换开销，
  **不改变** delegate 的全局精度选择。

### A 不会影响 B 的判定

常量折叠**不会**把 `RequiresFloat32Precision()` 的结果从 FP16 翻成 FP32，
也不会把 FP32 翻回 FP16。三条依据：

1. 它读 mojom 操作数 descriptor；折叠改的是 flatbuffer 发什么，两者不相交。
2. 判定循环（`:743-750`）遍历 mojom 算子列表，且**先判定后序列化**；
   即使折叠让某算子一个 TFLite op 都不发，那条 mojom 算子依然被访问过。
   标志是 sticky 的，被 `fused_ops_to_skip_` 跳过序列化的算子也照样计入。
3. `graph_builder_tflite.h:1051` 是
   `base::raw_ref<const mojom::GraphInfo> graph_info_;` —— const，物理上改不了。

具体到已落地的 gather 折叠：`kGather` 检查的是 `input_operand_id`
（被索引的数据，`:1694-1696`），不是 indices；且 indices 的 mojom 类型只能是
uint32/int32/int64，永远不是 `kFloat32`。`kElementWiseBinary` 检查
`lhs_operand_id`（`:1754-1755`），折叠也不改其类型。

---

## 8. 未验证项 / 待办

- [ ] **§3 的 gather 折叠只编译通过（`out/upstream_bots_debug` 的 `chrome`，exit 0），
      没跑过任何测试。** `webnn_unittests` 和 gather / gatherND / scatterND 的 WPT
      都没验证，尤其
      `third_party/blink/web_tests/external/wpt/webnn/conformance_tests/gatherND.https.any.js:271`
      那条依赖 clamp 语义的越界用例（`crbug.com/366412395` 仍未定案）。
- [ ] **§6.3 认定 op#2 就是报错算子，是静态分析下唯一符合特征的候选，
      但没有 delegate 日志确认。**
- [ ] 常量操作数能否同时是 gather indices 和**图输出**？
      若能，`FinishAndTakeResult()`（`:3194-3196`）用 `.at()` 取
      `operand_to_tensor_info_map_`，缺键会 CHECK 失败。
      这是既有代码（`SerializeGatherElements` / `SerializeScatterElements`）
      就有的模式，我的改动把它扩到了 gather/gatherND/scatterND。**未确认。**
- [ ] `(A-B)*C` 这个全常量子表达式的语义来源未查（不是 `dequantizeLinear`
      的 emulation —— 那条路径的 input 应是 int8/int4，而这里三个叶子都是 fp16）。
- [ ] `analysis.zh.md §B3` 的 24 个 `DIV` 是否也是同一根因（全常量输入），未逐个确认。
- [ ] 问题 B 里 decoder 的 fp32 激活是模型本身的设计，还是 WebNN 前端某处
      该 cast 却没 cast，未查。

---

## 9. 备选方案与为什么不选

| 方案 | 为什么不选 |
|---|---|
| 改发 CAST 替代 DEQUANTIZE，避开 `utils.cc:239` 的匹配 | CAST 自己的输入是常量，`gpu_compatibility.cc:667-669` 要求恰好 1 runtime 输入 → CAST 被拒 → 掉回 CPU，制造 GPU/CPU 同步点，还丢掉 fp16 权重直传。把问题挪走而非解决。 |
| 把 fp16 常量直接序列化成 fp32 常量、不发 DEQUANTIZE | 无效。SUB 依然两个常量输入（§2.1）。 |
| 折叠结果回舍成 fp16 再发 DEQUANTIZE，以保住 fp16 存储 | 多余。GPU 侧上传时一律按 precision 转（§4.2），两条路进显存的是同一份 fp16 位模式；只差 flatbuffer 体积，而折叠已经删掉 5 个算子，净体积是降的。 |
| 让 `gpu_compatibility` 接受 (0 runtime, 2 const)，或在 delegate 里做常量折叠 | 正确的长期解，但不在本仓库，周期长。 |
| 跳过 GPU 上的 gather clamp，靠 ML Drift 兜底 | `ml_drift/common/kernels/gather.cc:82-87` 直接 `args.src_tensor.Read(gather_index, ...)`，无边界检查、不处理负索引，`gather_test_util.cc` 也无越界用例。不能依赖。 |

### 顺带记录：`FLOOR_MOD` 可以把 clamp 链从 5 个算子压到 3 个

对 **runtime** indices（常量已折叠，不走这条路）：

```
MAXIMUM(indices, -N) → MINIMUM(., N-1) → FLOOR_MOD(., N)
```

语义与现有 `less/add/select` 严格等价：`idx=-N-5` → clamp 到 `-N` →
`floor_mod(-N,N)=0`；`idx=N+5` → clamp 到 `N-1` → `floor_mod=N-1`。
且不再需要 BOOL 张量和 SELECT_V2。

- ML Drift 整型 FLOOR_MOD 实现正确（欧几里得取模）：
  `ml_drift/common/kernels/elementwise.cc:389-403`
- `gpu_compatibility.cc:1204` 放行
- TFLite CPU kernel 支持 int32/int64：`litert/src/tflite/kernels/floor_mod.cc:74`

**注意不要为了省算子而去掉 MAXIMUM/MINIMUM** —— 只留 `FLOOR_MOD` 会把越界值变成
wrap 而非 clamp，踩 `gatherND.https.any.js:271` 的既有期望。
