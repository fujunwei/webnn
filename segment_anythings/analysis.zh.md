# WebGPU delegate — Segment Anything 偏置描述符修复

## 现象

在  集显上通过 LiteRT WebGPU delegate 运行 Segment Anything（MobileSAM），
在图编译阶段崩溃：

```
INVALID_ARGUMENT: Read selector with single argument can be used only with
linear storage types(BUFFER or IMAGE_BUFFER); selector object=biases,
selector=Read, args=[DST_S + 0], template_args=[], descriptor=float16,
TensorStorageType::TEXTURE_2D, layout: hwc, shape: {bhwc, {1, 1, 1, 128}}
```

调用栈（顶部帧）：

```
ml_drift::TensorDescriptor::PerformReadSelector
ml_drift::ResolveSelectorsPass
ml_drift::GPUOperation::AssembleCode
ml_drift::GpuModelBuilder::GetGpuModel
ml_drift::GraphToGpuModel
litert::ml_drift::DelegateKernel::InitInferenceContextFromGraph
```

第一次尝试修复后又冒出新的崩溃，这次发生在补丁插入的 elementwise Copy 算子里：

```
std::vector::operator[]  → ForceCrashOnSigAbort
ml_drift::TensorDescriptor::Write                 (tensor_desc.cc:1311)
ml_drift::TensorDescriptor::PerformWriteSelector  (tensor_desc.cc:968)
ml_drift::PerformSelector                         (tensor_desc.cc:684)
ml_drift::ResolveSelectorsPass                    (tensor_desc.cc:942)
```

## 环境

- Chromium `third_party/litert` delegate 包装 ml-drift。
- ml-drift 外部仓库位于
  `C:\Users\junweifu\workspace\chromium\src\third_party\ml-drift`
  （main 分支，commit 0e2092a），通过 `src/WORKSPACE:370` 的 `local_repository`
  绑定。
- 后端：WebGPU on  iGPU（`GetFastestStorageType` 返回
  `TensorStorageType::TEXTURE_2D`，见
  `ml_drift/webgpu/environment.cc:271-276`）。

## 根因

1. `GetTensorDescForValue`（`ml_drift/common/gpu_model_util.cc:187-240`）
   产出的每个张量默认使用 `Layout::HWC`（`b>1` 时为 `BHWC`）+
   `create_info.storage_type`（ WebGPU 上为 TEXTURE_2D）。对空间张量没问题，
   但对形状 `{1,1,1,C}` 的 bias 张量是错的。
2. LiteRT `convert_fully_connected.cc` 的 weights-runtime 分支
   （`src/ml_drift_delegate/tflite/convert/convert_fully_connected.cc:275-281`）
   无条件把 bias 作为图消费节点传下去，即使 TFLite 里 bias 是常量。
   结果 bias 以默认 HWC + TEXTURE_2D 描述符进入 ml-drift。
3. ml-drift 里的 `FULLY_CONNECTED` handler
   （`ml_drift/common/selectors/operation_selector.cc:840-870` 及
   `1425-1460` 的重复实现）用 `GetTensor(inputs[2]->id)` 拿 bias，
   直接把 `TensorHandle` 塞给 `Convolution(src, weights, bias, conv_attr)`。
4. `Convolution` 把描述符透传给
   `SelectConvolutionWithExternalWeights` →
   `CreateConvGenericExternalWeights`，其 shader 生成
   `args.biases.Read(DST_S + sind)`（`conv_generic.cc:902`）。
5. `PerformReadSelector`（`tensor_desc.cc:699-745`）在 `args.size() == 1`
   时走 `731-738` 分支，因描述符 layout 为 HWC 且 storage 为 TEXTURE_2D
   被拒。

常量 bias / 常量 weights 路径不会命中：它经过
`gpu_model_builder.cc:247-252` 的 LINEAR `AddConstantTensor(const Tensor<Linear,
FLOAT32>&, …)` 重载，通过 `CreateConstantLinearTensorDescriptor` 生成
LINEAR + BUFFER 描述符。

### 首次修复方案为什么在 Write 时崩

第一版 `MakeBiasLinear` 通过 `AddLinearTensor(c, data_type)` 分配目标张量。
这个 helper 从默认 storage（WebGPU 上是 TEXTURE_2D）开始，指望
`UpdateToSupportedStorageType`（`tensor_desc.cc:2083`）在组合无效时回退。
在这套配置下 `CanCreateTensorWithShape` 接受 LINEAR + TEXTURE_2D，不会回退。
接着 elementwise Copy 对 LINEAR 描述符调用 `PerformWriteSelector`；LINEAR
布局下通用的 `GetPhysicalCoords` 返回 `{""}`，WebGPU TEXTURE_2D 写分支
（`tensor_desc.cc:1311`）访问 `coords[1]` → `std::vector::operator[]` 越界。

Elementwise 算子不能写入 `Layout::LINEAR` 描述符：LINEAR 写必须用专用的
`PerformWriteLinearSelector`。

## 修复

新增 `GpuModelBuilder::MakeBiasLinear(TensorHandle)` helper：

- 如果描述符已满足单参 Read 选择器（LINEAR 或 HW 布局，或
  BUFFER / IMAGE_BUFFER 存储），直接返回 `src`。
- 否则构造一个 **保留原布局和形状** 但 storage 为
  `TensorStorageType::BUFFER` 的新 `TensorDescriptor`，加入图，
  并插入一个 elementwise `Copy(src, dst)`。

Copy 两端布局保持一致后坐标生成方式一致（TEXTURE_2D 上做 HWC 读，
BUFFER 上做 HWC 写，都用 `GetPhysicalCoordsWHS`）。目标 bias 描述符
因 storage 已变为 BUFFER，能通过 `tensor_desc.cc:731-738` 的
`PerformReadSelector` 检查。

调用点：`operation_selector.cc` 两处 `FULLY_CONNECTED` handler 在
`GetTensor(inputs[2]->id)` 之后立即调用 `MakeBiasLinear`。

### 为什么不修 `CONVOLUTION_2D`

`operation_selector.cc` 中的 `CONVOLUTION_2D` handler 从来不通过
`inputs[2]` 取运行时 bias — 它的 runtime-weights 分支只读
`attr.bias.data`（本身就是 LINEAR 路径）。如果 LiteRT 转换层送来
runtime bias 会被静默丢弃，那是一个独立的、已存在的 bug，Segment Anything
不需要修它。

### 备选方案

- **改 `GetTensorDescForValue`**：全局把 `{1,1,1,C}` 张量改为
  LINEAR + BUFFER。侵入太大；其他算子（如 broadcast、reshape）依赖默认
  HWC 描述符。
- **改 `ConvertOperations` 里的 CONSTANT 节点**：只能覆盖常量 bias，
  运行时 bias 依然崩。
- **改 LiteRT `convert_fully_connected.cc`**：当 TFLite bias 是常量时
  保留 `attr.bias`。可行且对常量 bias 场景风险更低，但覆盖不了纯运行时
  bias 的 FC，且只能修 LiteRT delegate — 产生同样图结构的其他 ml-drift
  消费者仍会崩。
- **delegate 选项设 `use_buffer_storage_type = true`**：强制处处用
  BUFFER 可绕过问题，但在  iGPU 上带来性能回退，不是正解。

最终选定的 ml-drift 侧 helper 同时覆盖常量与运行时 bias，改动限定在一个
函数加两个调用点。

## 修改文件

均在 `C:\Users\junweifu\workspace\chromium\src\third_party\ml-drift\` 内：

- `ml_drift/common/gpu_model_builder.h` — 声明
  `TensorHandle MakeBiasLinear(const TensorHandle&)`。
- `ml_drift/common/gpu_model_builder.cc` — 实现。
- `ml_drift/common/selectors/operation_selector.cc` — 两处
  `FULLY_CONNECTED` handler 中调用 `MakeBiasLinear`
  （v1 使用 `node.operation.attributes`，v2 使用 `node.attr`）。

见 [`ml-drift-webgpu-bias-fix.patch`](ml-drift-webgpu-bias-fix.patch)。

## 验证计划

1. 重建 `libLiteRtWebGpuAccelerator.dll`。
2. 在  iGPU 上通过 WebGPU delegate 加载 Segment Anything。
3. 确认 `InitInferenceContextFromGraph` 不再中止。
4. 可选：与 CPU 参考对比输出正确性。因为 `Copy` 在 FP16 下是纯逐元素恒等，
   预期结果按位一致；唯一开销是每个带运行时 bias 的 FC 层多一次
   buffer 拷贝。

---

# SAM encoder 上 `SimpleMemoryArena::Commit` 的 OOM

## 现象

修完 bias 描述符之后，通过 ml-drift WebGPU delegate 加载
`segment_anything_encoder.tflite` 仍然会中止，这次在 PartitionAlloc 里：

```
KernelBase!RaiseException
partition_alloc::OnNoMemoryInternal
partition_alloc::TerminateBecauseOutOfMemory
partition_alloc::PartitionExcessiveAllocationSize      ← 单次分配 > ~2 GiB
partition_alloc::PartitionBucket::SlowPathAlloc
partition_alloc::PartitionRoot::AlignedAlloc<16>
tflite::SimpleMemoryArena::Commit
tflite::ArenaPlanner::ExecuteAllocations
tflite::Subgraph::AllocateTensors
tflite::Subgraph::ModifyGraphWithDelegateImpl
LiteRtCompiledModelT::Create
webnn::litert::GraphImplLiteRt::CreateAndBuild
```

栈里的 `IsFullyDelegated` 只是与 `Subgraph::AllocateTensors` 相邻的符号，
并不是 `Commit` 的实际调用者。

同一模型在 LiteRT CPU 路径上跑得起来。

## 背景：TFLite 张量内存怎么分配

TFLite 按 `allocation_type` 给每个张量分类：

| 类型 | 归属 |
|---|---|
| `kTfLiteMmapRo` | mmap 的 flatbuffer 权重 |
| `kTfLiteArenaRw` | 打包进 `ArenaPlanner::arena_`（可复用） |
| `kTfLiteArenaRwPersistent` | 打包进 `persistent_arena_` |
| `kTfLiteDynamic` | 每次 kernel 调用 `malloc/free` |
| delegate 自管 | TFLite 不可见（如 ml-drift 的 GPU buffer） |

**每个 subgraph 只有 2 个 `SimpleMemoryArena` 实例**（`arena_`、
`persistent_arena_`，见
[arena_planner.h](../../chromium/src/third_party/tflite/src/tensorflow/lite/arena_planner.h#L140-L146)），
不是每个张量一个。

`ArenaPlanner::PlanAllocations` 遍历 execution plan，为每个 arena 张量
记录区间 `[first_node, last_node]`，然后用按 offset 的 first-fit 贪心
把它们打包到一整块连续 buffer 里，让区间不相交的张量共享字节。
`Commit()` 执行 **一次** `AlignedAlloc<16>(high_water_mark)`。
就是这一次单次分配撞上了 PartitionAlloc 的 ~2 GiB 上限。

### delegate 对 arena 的影响

`ModifyGraphWithDelegate` 把被支持的算子融合成一个 kernel 节点。
对 CPU 侧 arena 有两个后果：

1. 完全在 delegate 内部的张量在 `ArenaPlanner` 眼里消失（它们变成 ml-drift
   拥有的 GPU buffer）。
2. 边界张量（delegate 输入/输出、图输入/输出）仍是 `kTfLiteArenaRw`。
   它们的生存区间现在都塌缩到融合出的那 **一个节点** 上，因此很多张量
   **同时存活**，彼此之间无法共享字节。对激活很大的图，`arena_` 的
   high-water 会 **升高** 而不是降低。

### `OptimizeMemoryForLargeTensors(threshold)` 的作用

上游 LiteRT 的 `InitializeRuntime` 会调用
`interpreter_options.OptimizeMemoryForLargeTensors(1 << 20)` —
所有大于 1 MiB 的张量会被移出 arena 并重新标为 `kTfLiteDynamic`
（每次调用按需分配）。这可以把 SAM / SD-turbo 这类模型的单次 Commit
分配压到远低于 2 GiB。

## 根因

[route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch](../route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch)
把这一调用注释掉了：

```cpp
// interpreter_options.OptimizeMemoryForLargeTensors(1 << 20);
```

写这个 patch 的动机：`kTfLiteDynamic` 张量会置
`Subgraph::has_dynamic_tensors_ = true`，`Subgraph::ModifyGraphWithDelegateImpl`
对只支持静态 shape 的 delegate（ml-drift 就是）会拒绝：

```
Attempting to use a delegate that only supports static-sized tensors
with a graph that has dynamic-sized tensors (tensor#%d is a
dynamic-sized tensor).
→ kTfLiteApplicationError
```

所以关掉 `OptimizeMemoryForLargeTensors` 是让 delegate 能生效的必要条件。
代价是所有大激活都留在 `arena_`，SAM encoder 上打包出的 high-water
超过 2 GiB，触发 `PartitionExcessiveAllocationSize`。

CPU 路径不会撞这个坑：CPU 用户从不 apply 静态 shape delegate，
LiteRT 保留了 `OptimizeMemoryForLargeTensors(1 << 20)`。

## 举例 — `Add → Mul → Sub`

张量大小都为 `S` 字节；节点索引 `0/1/2`。

图：

```
in0 ─┐
     Add ── A ─┐
in1 ─┘        Mul ── M ─┐
in2 ──────────┘        Sub ── out
in3 ────────────────────┘
```

生存期（图输入/输出固定为 `[-1, ∞]`）：

| 张量 | first | last |
|---|---|---|
| in0,in1 | -1 | 0 |
| in2 | -1 | 1 |
| in3 | -1 | 2 |
| A | 0 | 1 |
| M | 1 | 2 |
| out | 2 | ∞ |

### 全 CPU（无 delegate）

arena 打包可以让 `in0` 复用 `A` 的位置、`in1` 复用 `M` 的位置等。
`high_water_mark ≈ 5·S`，一次 `AlignedAlloc<16>(5·S)`。

### `Mul` 不支持 → CPU-Add | Delegate{Mul} | CPU-Sub

execution plan 仍是 3 个节点；每个张量都跨 CPU/GPU 边界，都留在 `arena_`。
布局大致仍是 5·S。

### `Add`+`Mul` 被 delegate，`Sub` 留在 CPU

execution plan 塌缩到 2 个节点：`Delegate{Add,Mul}`（idx 0）、
`Sub`（idx 1）。

- `A` 完全在 delegate 内部 → 离开 CPU arena（ml-drift 用 GPU buffer 持有）。
- arena 里的边界集合：`in0, in1, in2, in3, M, out`。
- 区间：`in0/in1/in2 [-1,0]`，`in3 [-1,1]`，`M [0,1]`，`out [1,∞]`。
  三个 delegate 输入在节点 0 时都还活着，同时 `M` 也已开始分配 —
  它们之间无法复用。
- `high_water_mark` 通常 ≥ CPU 情形；少打包一个张量抵不过区间重叠增加。

### 完全 delegated

只有一个节点 `Delegate{Add,Mul,Sub}`。`A` 与 `M` 离开 arena；
只剩 `in0..in3` 与 `out`，但所有边界输入都在这一节点期间存活
⇒ 仍然同时存在。SAM encoder 上的崩溃就是这个形态：
少量非常大的边界/激活张量在同一时刻挤在 arena 的 high-water 上。

`IsFullyDelegated` 只是 `AllocateTensors` 之后的一个查询；只要图里存在
任何 `kTfLiteArenaRw` 张量（图 IO 必然如此），arena Commit 就会执行。

## 修复方案（按工作量排序）

1. **调高阈值而非删掉调用。**
   在
   [patches/06](../route-a-webgpu-windows/patches/06-compiled-model-disable-optimize-memory.patch)
   里把 `OptimizeMemoryForLargeTensors(N)` 恢复，`N` 选到既能把 ml-drift
   需要保持静态的张量留在 arena，又能把 SAM 里几个几百 MB 的激活转成动态
   分配。用 `--enable-logging=stderr --vmodule=*subgraph*=1` 验证：
   既没有 `"only supports static-sized tensors"` 警告，**又** 能加载
   encoder。
2. **让 delegate/subgraph 检查区分 `allocation_type == kTfLiteDynamic`
   （shape 已知）与真正的动态 shape 张量。** 修改 `Subgraph::has_dynamic_tensors_`
   的记账，或改 ml-drift 的 `IsNodeSupported`，让"动态分配但静态 shape"
   的张量不再阻止 delegation。中风险，一劳永逸。
3. **在 WebNN 侧把 SAM encoder 切成子图**，让任何单个 arena 都不越 2 GiB。
   工作量最大。

建议先做方案 1：现有 patch 栈里的一行改动即可，直接解决失败分配，
不改动 delegate 语义。

## 验证钩子

- 在 `Commit()` 前后打日志输出
  `ArenaPlanner::arena_.underlying_buffer_.data_size_`（或
  `high_water_mark_`）；失败情形下 ≥ `2 * 1024 * 1024 * 1024`。
- 打开 `OptimizeMemoryForLargeTensors(64 << 20)`，观察 arena 大小骤降、
  崩溃消失。
- 如果 delegated subgraph 里重新出现动态张量，TFLite 会打印
  `"only supports static-sized tensors (tensor#N …)"` — 由此定位哪个
  大张量还需要留在 arena（或落进受支持的算子）。

---

# SAM encoder 上被 GPU delegate 拒绝的算子分布

## 数据来源

来自 [route-a-webgpu-windows/patches/10-ml-drift-log-unsupported-op-counts.patch](../route-a-webgpu-windows/patches/10-ml-drift-log-unsupported-op-counts.patch)
（在 ml-drift `GetOpsToReplaceWithOptions` 里把 lambda 拒绝的节点按 opcode
计数并追加到 `TF_LITE_KERNEL_LOG`）。跑
`segment_anything_encoder.tflite` 一次，得到：

```
Following operations are not supported by GPU delegate:
ADD: Can't parse inputs with const tensors.
ADD: Tensor "" has bad input dims size: 0.
ADD: Tensor dimensions must be less than 5.
CAST: Tensor type(INT64) is not supported.
DEQUANTIZE:
DIV: Op can only handle 1 or 2 operand(s).
GATHER: Only support 1D indices
LESS: Can't parse inputs with const tensors.
MAXIMUM: Can't parse inputs with const tensors.
MINIMUM: Can't parse inputs with const tensors.
RESHAPE: Tensor "" has bad input dims size: 6.
RESHAPE: Tensor dimensions must be less than 5.
SELECT_V2: Tensor type(INT64) is not supported.
TRANSPOSE: Expected 1 runtime input tensor(s), but node has 0 runtime input(s).
TRANSPOSE: Permutation for transpose is invalid.
TRANSPOSE: Tensor "" has bad input dims size: 6.
102 operations will run on the GPU, and the remaining 1594 operations will run on the CPU.
Unsupported node counts by opcode (nodes rejected before partitioning):
  RESHAPE: 128
  TRANSPOSE: 88
  ADD: 72
  CAST: 24
  DIV: 24
  GATHER: 24
  LESS: 24
  MAXIMUM: 24
  MINIMUM: 24
  SELECT_V2: 24
Created TensorFlow Lite XNNPACK delegate for CPU.
```

上面 `Following operations are not supported` 段是 tflite
`GraphPartitionHelper::PrepareSupportedNodes` 用 `std::set<std::string>` 去重后
的**每种"op: 原因"字符串**；下面 `Unsupported node counts by opcode` 段是
patch 10 追加的**按 opcode 累加的节点数**。同一 opcode 的多个不同原因合并计数。

## 汇总（按数量降序，共 456 个节点被拒）

| # | Opcode | 数量 | 已知原因（来自 log） |
|---|---|---:|---|
| 1 | `RESHAPE` | 128 | rank 5、rank 6 |
| 2 | `TRANSPOSE` | 88 | 无 runtime 输入 / 非法 permutation / rank 6 |
| 3 | `ADD` | 72 | 常量+常量 / rank 0 标量 / rank ≥ 5 |
| 4 | `CAST` | 24 | INT64 |
| 5 | `DIV` | 24 | 操作数个数 > 2 |
| 6 | `GATHER` | 24 | indices 非 1-D |
| 7 | `LESS` | 24 | 常量+常量 |
| 8 | `MAXIMUM` | 24 | 常量+常量 |
| 9 | `MINIMUM` | 24 | 常量+常量 |
| 10 | `SELECT_V2` | 24 | INT64 |

`102 / 1596`（GPU / 总节点）= **6.4%** 命中率。把上表最大的三项拿下（`RESHAPE + TRANSPOSE + ADD = 288`）就能翻近 3 倍。

## 按"改起来性价比"分类

按修复的 **单位 opcode 收益 / 工作量比** 分三档。

### A 档：一次修复一整片

**A1. `RESHAPE` / `TRANSPOSE` 的 rank≥5、rank=6**（120 + 40 ≈ 160 个）

SAM encoder 里的 patch embedding 与相对位置编码经常引入 5D、6D 张量。ml-drift
kernel 内部按 BHWC + Batch/Depth 处理，rank>4 直接拒。

三种可能路径：

1. **在 ml-drift `ReshapeOperationParser::IsSupported` 里放宽 rank**：如果
   reshape 的输入和输出在一个 batch 维度上就能压回 4D（例如 `[B,H,W,g,C/g]` →
   `[B,H,W,C]`），插一层"合并/展开的 view 张量"支持它。风险中等（改动核心
   layout 推断）。
2. **在 WebNN 侧或建图前把 5D/6D 拍平**：`GraphBuilderTflite` 在遇到
   `mojom::Reshape` / `mojom::Transpose` 时，如果目标 shape 有连续可折叠维度，
   等价地合并它们再发给 tflite。风险最低，且这些 reshape 本来就是 view，
   语义等价。**推荐先试这条。**
3. **在 LiteRT 的 `optimize` pass 里加个 `MergeConsecutiveViewOps`**：也可行，
   但影响面比方案 2 大。

**A2. `RESHAPE` 的 rank 5**：属于 A1 的子集，同一改法。

**A3. `TRANSPOSE: Expected 1 runtime input tensor(s), but node has 0 runtime input(s).`**

意思是 `TRANSPOSE` 的 permutation 张量是常量（builder 侧把两个输入都塞进
constant），运行时 `inputs->size` 为 1（只算 dtype 非 kTfLiteInt32
的运行时输入），ml-drift 的 `PreCheckReadValue` 断言是 0，拒。
`TransposeOperationParser::Parse` 期望 permutation 出现在 `inputs[1]`。

修复：在 WebNN 建图或 LiteRT convert 时，让 permutation 走 tflite operator
的**第二个输入槽**（依然是 constant），不要直接常量折叠掉。改动应该在
`services/webnn/tflite/graph_builder_tflite.cc` 的 `SerializeTranspose*`
路径。**低风险，收益立竿见影**。

**A4. `TRANSPOSE: Permutation for transpose is invalid`**

ml-drift 只支持 4D 的 `TransposeAttributes`（`Perm4D` 结构）。permutation
里含非 0..3 的分量（比如 5D permutation 里的 `4`）→ 拒。跟着 A1
一起处理：如果我们能保证 rank ≤ 4，这条自然就消失。

### B 档：单类型问题，改起来集中

**B1. `CAST: INT64` + `SELECT_V2: INT64`**（24 + 24 = 48 个）

SAM 里的 int64 常来自 top-k / index 计算。ml-drift 只允许
{f32, f16, i32, i8, u8, bool}。

三种可能：

- **上游 tflite converter 阶段就把 int64 常量降到 int32**（比如
  `services/webnn/tflite/graph_builder_tflite.cc` 里 constant 张量下发时如果原
  op 是 index 用途，dtype 用 int32）。这是最省事的。
- **ml-drift 里给 CAST/SELECT_V2 支持 int64**：需要新增一条 dtype 分支到
  两个 kernel + tensor descriptor。工作量中等。
- **在 tflite `optimize` pass 里替换 int64 张量为 int32**（前提是值范围能
  塞下）。

**B2. `GATHER: Only support 1D indices`**（24 个）

`GatherOperationParser::IsSupported` 要求 `indices.dims->size == 1`。SAM 里
attention 有 2-D indices。修复思路：
- 在建图侧 reshape indices 到 1-D，输出再 reshape 回去。
- 或者在 ml-drift 里加一条 2-D indices 分支（把 outer dim 当 batch）。

**B3. `DIV: Op can only handle 1 or 2 operand(s)`**（24 个）

看着像 tflite `DIV_N` 或 fused div-with-broadcast 的变体。需要 dump 出这
24 个 DIV 节点的 `inputs->size` 与 fused 属性看具体是哪种。若都是同一种
fused div，加一个 handler；若是 broadcast div，通常和 ADD 那批一起处理。

### C 档：小片修复，但影响不大

**C1. `ADD / LESS / MAXIMUM / MINIMUM: Can't parse inputs with const tensors`**
（72 + 24 + 24 + 24 = 144 个，其中 ADD 里"常量+常量"分量未知）

`ml_drift_delegate` 的元素级 handler 要求至少 1 个 runtime 输入。所有输入都是
constant 时（图上通常是 constant-folding 的漏网之鱼），parse 失败。修复：
- **在 tflite 侧 constant-fold**：这类节点跑一次也无害，让 tflite 的
  `optimize` pass 提前算掉它们即可。ml-drift 就看不到了。
- **或者在 ml-drift 里加 "全常量 → materialize constant 输出" 分支**：改动
  更集中，但对每个 elementwise handler 都要动。

**C2. `ADD: rank 0`（标量常量 + 张量）**

标量常量在 tflite 里 shape 为 `[]`，ml-drift 期望的 `BHWC` 至少 rank 1。
建图侧对标量 constant reshape 成 `[1]` 即可。

**C3. `ADD: rank ≥ 5`**：跟 A1 一起处理。

**C4. `DEQUANTIZE:`**（数量未在 counts 段出现，说明它落在 partition 内被吸收 —
或空 details 未 hash，去重成了 1 条）：SAM encoder 无量化，可暂时忽略；如果之
后跑量化模型再回头看。

## 建议的推进顺序

优先按"节点数 × 单个 CL 覆盖多类"排序：

1. **[A3] Transpose permutation 常量下发问题** — 期望：`TRANSPOSE` 88 中大约
   40-60 个消失，无风险。
2. **[A1] rank≥5 的 reshape/transpose 合并** — 期望：`RESHAPE` 128 + 剩余
   `TRANSPOSE` 一起下降，可能一次 CL 拿掉 ~180 个。
3. **[B1] int64 → int32 常量降级** — 期望：`CAST` 24 + `SELECT_V2` 24 = 48
   全消。
4. **[C1] constant-only elementwise 折叠** — 期望：144 个消失或大幅缩水。
5. 剩下 GATHER / DIV 依样处理。

每一步跑完看 counts 是否符合预期，再决定下一步。

## 记账模板

| 迭代 | 修改 | GPU 节点 | CPU 节点 | 备注 |
|---|---|---:|---:|---|
| 基线 | — | 102 | 1594 | 本节数据 |
| 1 | [A3+A1] transpose 常量输入 + rank-5 拆解（v1） | 102 | 1618 | patch 11 v1；A1 净负 (+24 RESHAPE)，A3 未命中 |
| 2 | [A3] transpose 常量输入 + DEQ(const) 链折叠（v2） | 待测 | 待测 | patch 11 v2；回滚 A1，扩 A3 |
| 3 | [B1] int64 常量降级 | | | |
| 4 | [C1] const-only elementwise | | | |
| 5 | [B2] gather 2-D indices | | | |
| 6 | [B3] div fused | | | |

## 迭代 1 复盘（patch 11 v1）

原始 patch 11 同时打了 A3 与 A1（TRANSPOSE 拆解）。实测：

```
102 GPU, 1618 CPU
RESHAPE: 152 (+24)
TRANSPOSE: 76 (-12)
```

Δ CPU = +24 = Δ RESHAPE 完全吃掉 Δ TRANSPOSE 的减量：

- **A1 的 -12 TRANSPOSE**：来源于 12 个 rank-5 transpose 被 A1 拆
  成 `RESHAPE + TRANSPOSE + TRANSPOSE`。中间那次 rank-4 transpose 被
  接受，但两侧的 RESHAPE 都是 rank-5 → 被 ml-drift 的
  `RESHAPE: Tensor dimensions must be less than 5.` 拒。净得
  `-12 TRANSPOSE + 24 RESHAPE`。**A1 是净负**，需要回滚。
- **A3 的 0 命中**：说明 SAM encoder 的 `mojom::Transpose.input_operand_id`
  在 mojom 层不是 `Kind::kConstant`，而是走了 `DequantizeLinear` 的中间
  operand。真正会导致 "0 runtime inputs" 的机制是 tflite 的
  `FP16GraphPartitionHelper::IsNodeSupported`：它对
  `DEQUANTIZE(fp16 mmap const)` 建立 `constant_dequant_map_`，然后为
  后续节点做 `RemapFp16InputTensors`——把 transpose 的 `input[0]` 临时
  改指回 fp16 mmap 常量，让 `gpu_compatibility.cc` 的
  `GetNumberOfRuntimeInputs` 数出 0 个 runtime 输入。

## 迭代 2（patch 11 v2 - 当前）

`route-a-webgpu-windows/patches/11-webnn-transpose-fold-const-and-deq-chain.patch`：

### 回滚 A1

删掉 `TryReduceTransposeRank` 及 `FoldedTransposePlan`；`SerializeTranspose`
不再对 rank-5 transpose 做 reshape 包装。保留 `SerializeTransposeOperation`
的 permutation `uint32_t → int32_t` 的小修正（对齐 ml-drift
`Tensor<Linear, INT32>` 的读法）以及 `SerializeOperation` 分发循环里的
`if (!operator_offset.IsNull())` 判断（handler 可以返回空 offset 表示
"整段被生前折叠"）。

### 扩 A3：直连常量 + DEQ(const) 链折叠

`GraphBuilderTflite::TryConstantFoldTranspose` 现在处理两种模式：

- **Case A — `Transpose(kConstant)`**：读原始字节，按 permutation 复制
  出新的 mmap constant tensor，注册到 `output_operand_id` 的
  `operand_to_tensor_info_map_` 上，返回空 offset。dtype 不变（fp16 保
  持 fp16，int8 保持 int8，等等），子字节类型（`int4`/`uint4`）不折。

- **Case B — `Transpose(DequantizeLinear(kConstant))`**：把 DEQ 数学与
  permutation 一起在建图期算掉，落成 fp32 mmap constant。触发条件：

  1. DEQ 的输入是 `kConstant`；
  2. DEQ 输出只有这一个 transpose 消费者（`operand_to_dependent_operations_`
     里 size==1）；
  3. scale / zero_point 都是标量常量（`NumberOfElements() == 1`），暂
     不支持 per-channel（需要跟踪 channel 轴经 permutation 到哪一维，
     交后续 CL）；
  4. 源 dtype 属于 {fp16, fp32, int8, uint8, int32}。

  计算路径：`fp32[i] = (source[i] - zp) * scale`，再按 permutation 落到
  fp32 mmap buffer。同时把 `lazy_serialized_dequantize_operations_` 里
  对应的 `serialized` 标记为 true，防止原 DEQ 被再次 emit。

  产物是一个 fp32 mmap constant，`operand_to_tensor_info_map_` 直接指
  向它。下游 op（通常是 MATMUL）通过 `SerializeInputTensorInfo` 拿到的
  就是 fp32 TensorInfo，不会再触发 `constant_input_tensor` 的 CAST
  折半分支，因此 `FP16GraphPartitionHelper` 也不会重映射到 mmap 常量。
  MATMUL 之类的 op 只要求 1 个 runtime 输入（activation），第 2 输入
  是常量本身就允许。

### 副作用

- **空间**：Case B 会把 fp16 常量翻倍写成 fp32；SAM encoder 里的常量总
  大小几百 MB，翻倍到 GB 级需要留意 flatbuffer 上限。若命中面过大，可
  以在后续 CL 里改成 "生成 fp16 permuted constant + 保留一次 CAST /
  DEQUANTIZE"，代价是需要给 `SerializeCastOperation` 加一条 "输入 tensor
  已是 mmap const" 的旁路，工作量更大。
- **精度**：`(v - zp) * scale` 在 fp32 里算，量化和 fp16 场景语义等价，
  没有额外的舍入误差引入。

---

# 迭代 3：全部算子 GPU 支持 + OOM 根因（当前）

## 3.1 算子支持修复

针对迭代 2 之后仍被拒的算子，在 ml-drift / LiteRT delegate 里逐一放行：

### 3.1.1 RESHAPE 0 runtime 输入

`gpu_compatibility.cc` 的 `kTfLiteBuiltinReshape` 分支原来要求恰好 1 个
runtime 输入（`CheckInputsOutputs(op_sig, /*runtime=*/1, /*outputs=*/1)`）。
常量输入（如被 `FP16GraphPartitionHelper::RemapFp16InputTensors` 重映射成
fp16 mmap 常量）时 runtime 输入为 0 → 拒。

修复（两处树都改了，注意 `use_litert_tflite=false` 时编译的是
`third_party/tflite/src/tensorflow/lite/tools/versioning/gpu_compatibility.cc`；
`litert/BUILD.gn:1346` 编译的也是 litert 树的拷贝——两处都改了）：

```cpp
case kTfLiteBuiltinReshape: {
  const int runtime_inputs = GetNumberOfRuntimeInputs(op_sig);
  if (runtime_inputs < 0 || runtime_inputs > 1) { ...error... }
  if (op_sig.outputs.size() != 1) { ...error... }
  return absl::OkStatus();
}
```

配套 `ReshapeOperationParser::Parse`：
- `runtime_inputs == 0`：用 `reader->ReadTensor` 读常量，把输出 shape 写到
  `TensorFloat32`，emit `CONSTANT` 节点（reshape 只是逻辑维度变化，数据不动）。
- `runtime_inputs >= 1`：正常 `RESHAPE` 节点。

### 3.1.2 RESHAPE/TRANSPOSE 5D 支持

`IsAllAllowedTensors` 原来一刀切拒 `dims->size >= 5`（"Tensor dimensions
must be less than 5"）。放宽到 `> 5`（允许 5D，仍拒 6D）。

配套改动：
- `ReshapeOperationParser::Parse`：输出 rank==5 时用
  `Reshape3DAttributes`（BHWDC）。
- `TransposeOperationParser::IsSupported`：perm size 上限 `4 → 5`。
- `TransposeOperationParser::Parse`：perm size==5 用 `Transpose3DAttributes`。
- ml-drift `operation_selector.cc` 的 legacy `GPUOperationFromNode`：
  RESHAPE/TRANSPOSE 都补了 `Reshape3DAttributes`/`Transpose3DAttributes`
  分支（原来 `std::any_cast<ReshapeAttributes>` 在 5D 时抛
  `std::bad_any_cast`）。
- ml-drift `transformations/remove_noop.cc` 的 `RemoveIdentityReshape`：
  遇到 `Reshape3DAttributes` 直接 SKIPPED（不再 bad_any_cast）。

### 3.1.3 TRANSPOSE 0 runtime 输入

同 RESHAPE：`gpu_compatibility.cc` 的 TRANSPOSE 分支接受 0-1 个 runtime
输入。`Parse` 里 `runtime_inputs == 0` 时读常量数据 + permutation，在 CPU
上做转置（新增 `TransposeConstantData` 辅助函数，只处理 4D BHWC），emit
`CONSTANT` 节点。

### 3.1.4 结果

全部 1260 个算子通过 GPU delegate 支持检查（full delegation）。
DEQUANTIZE 的拒绝来自 `FP16GraphPartitionHelper::IsNodeSupported`
（fp16 常量 DEQUANTIZE 故意返回 false，full delegation 场景会重新纳入），
不是 parser 问题。

## 3.2 Delegation 日志

`GetOpsToReplaceWithOptions` 的委托摘要原本走 `TF_LITE_KERNEL_LOG` → 
`context->ReportError`。LiteRT runtime 默认 `error_reporter_mode =
kLiteRtErrorReporterModeNone` → 输出全丢。

修复链（`services/webnn/tflite/graph_impl_litert.cc`）：
1. `GetCompilationOptions` 里 `SetErrorReporterMode(kLiteRtErrorReporterModeBuffer)`
2. `CompiledModel::Create` 后用 `model_->GetErrorMessages()` 把 buffer 内容
   `LOG(ERROR)` 出来 → `[WebNN] LiteRT compilation diagnostics: ...`
3. Run 失败时同样 dump（`[WebNN] LiteRT run diagnostics`）

另有 `delegate_webgpu.cc` 的 per-node GPU/CPU 委托清单（`TF_LITE_KERNEL_LOG`）、
`model_builder.cc` 的 "All N operations are supported" 分支。

## 3.3 OOM 根因链

### 3.3.1 静态分析

脚本 `tflite-dump-model/analyze_arena.py` 模拟 `ArenaPlanner` 的
first-fit-by-offset + lifetime 复用，对 `new_segment_anything_encoder.tflite`：

| 项目 | 数值 |
|---|---|
| Ops | 1260 |
| 常量权重（mmap，不进 arena） | 175.2 MB |
| Arena（非常量）张量 | 1233 个 |
| arena 张量大小总和 | 46.8 GB |
| **arena high-water mark** | **≈1.63 GB**（与实测 1673527296 B ≈ 1596 MB 吻合） |
| 最大单张量 | 768 MB × 多组（BATCH_MATMUL 中间张量，lifetime 各 1 个 node） |

### 3.3.2 为什么 full delegation 下 arena 还是大

- `AllocateTensors` 在 delegate 替换 **之前** 按原始执行计划规划 arena：
  全部 1233 个张量按 lifetime 打包 → 峰值 1.6 GB。
- delegate 替换后中间张量离开 arena，但 arena 底层 buffer **不收缩**。
- 每次 `ModifyGraphWithDelegate` 都触发一次 `EnsureMemoryAllocations`
  （STEP 3）→ 重新规划 + 重新 Commit。

### 3.3.3 为什么有两次 AllocateTensors（第二次 2 GB → OOM）

WebNN 的 `GetCompilationOptions` 注册了 **GPU + CPU(XNNPACK) 两个
accelerator**。`compiled_model.cc` 的 delegate 循环为每个 accelerator 调用
一次 `ModifyGraphWithDelegate`：

1. GPU delegate：STEP 3 CASE 1 → `EnsureMemoryAllocations` →
   分配 1.6 GB（成功），state → `kStateInvokableAndImmutable`
2. CPU delegate：STEP 3 CASE 2（`pre_delegation_state ==
   kStateInvokableAndImmutable`）→ 再次 `EnsureMemoryAllocations` →
   2 GB → PartitionAlloc `PartitionExcessiveAllocationSize` → 崩溃

### 3.3.4 `OptimizeMemoryForLargeTensors` 修复

目标：`AllocateTensors` 之前把大张量从 `kTfLiteArenaRw` 改成
`kTfLiteDynamic`（独立 malloc，不走 arena）。

问题 1：`interpreter_options.OptimizeMemoryForLargeTensors(1<<20)` 只设置了
选项，`Subgraph::OptimizeMemoryForLargeTensors()` **没有人调用**。修复：
`compiled_model.cc` 在 `builder(&interp_)` 之后对每个 subgraph 显式调用。

问题 2：该函数用 `tensor->bytes` 判断大小，但此时 `bytes` 还是 0（第一次
`AllocateTensors` 之前）。修复：`bytes==0` 时从 `dims` × 元素大小推算。

问题 3：改为 `kTfLiteDynamic` 后，TFLite 的 dynamic-tensor 检查会拒
static-shape-only delegate。修复：delegate 加
`kTfLiteDelegateFlagsAllowDynamicTensors`（delegate_webgpu.cc）。该 flag
使 `ModifyGraphWithDelegateImpl` 跳过 dynamic 检查，且 STEP 3 对
`pre_state==kStateUninvokable` 走 "no allocation needed" 分支——**编译期零
arena 分配**，第一次 Run 时才 `AllocateTensors`。

实测日志：
```
[webnn-oom] OptimizeMemoryForLargeTensors: converted=1105 skipped_input=1 skipped_bytes=734 total=1840
```

### 3.3.5 GPU-only 模式（disable CPU fallback）

新增 switch `--webnn-tflite-gpu-only`：
- `webnn_switches.h/.cc` 注册 switch 并加入
  `GetWebNNSwitchesCopiedFromGpuProcessHost()` 转发白名单
  （**不加白名单 switch 根本到不了 WebNN 进程**——这是曾经 "开关无效" 的原因）。
- `graph_impl_litert.cc` `GetCompilationOptions`：switch 存在时跳过
  `accelerators |= kCpu` 及全部 CPU options。
- 效果：只有一个 GPU delegate → 只一次 `ModifyGraphWithDelegate`。
- 注意 `compiled_model.cc:1029`：无 kCpu 且存在 non-delegated ops 时
  编译报错（全 GPU 支持时不会触发）。

"Created TensorFlow Lite XNNPACK delegate for CPU" 不再出现；
"CpuAccelerator registered" 仍会出现（注册 ≠ 创建 delegate，
`auto_registration.cc` 默认注册全部 accelerator，选择靠
`SetHardwareAccelerators` 位掩码过滤，`compiled_model.cc:982-984`）。

## 3.4 当前状态：Invoke 失败（待查）

```
[webnn-delegate] ModifyGraphWithDelegateImpl: flags=1 supports_dynamic=1 hint_full=0 pre_state=0
ERROR: [litert_compiled_model.cc:164] Failed to invoke
```

- 编译成功（OOM 已解决）。第一次 Run 的 `AllocateTensors` 也通过
  （否则报 "Failed to allocate tensors"）。
- `runner->Invoke()` 返回错误 → `compiled_model.cc:2015` "Failed to invoke"。
- 已加诊断：invoke 失败时 dump `BufferErrorReporter` 内容
  （`[webnn-run] invoke error: ...`），WebNN 侧 Run 失败也 dump。
- 待查方向：
  1. 本机 GPU 受限（EGL 仅 GLES 3.0、无 OpenCL）→ WebGPU execute 失败；
  2. delegate kernel Init 推迟到第一次 Run（Create 时 STEP 3 零分配）→
     GPU graph 构建问题在这里才暴露；
  3. constant-input RESHAPE/TRANSPOSE 生成的 CONSTANT 节点导致 GPU graph
     异常。

## 3.5 修改文件清单

| 树 | 文件 | 修改 |
|---|---|---|
| tflite | `src/tensorflow/lite/tools/versioning/gpu_compatibility.cc` | RESHAPE/TRANSPOSE 接受 0-1 runtime 输入 |
| tflite | `src/tensorflow/lite/core/subgraph.cc` | `OptimizeMemoryForLargeTensors` bytes 推算 + 计数日志；`ResizeTensorImpl` 大张量日志；`ModifyGraphWithDelegateImpl` STEP3 分支日志 |
| litert | `src/tflite/tools/versioning/gpu_compatibility.cc` | 同上（litert 树拷贝） |
| litert | `src/ml_drift_delegate/tflite/model_builder.cc` | IsAllAllowedTensors 放行 5D；Reshape/Transpose Parse 常量路径 + 5D；delegation 日志 |
| litert | `src/ml_drift_delegate/delegate/delegate_webgpu.cc` | `kTfLiteDelegateFlagsAllowDynamicTensors`；per-node 委托日志 |
| litert | `src/ml_drift_delegate/delegate/delegate_kernel.cc` | 阶段日志（BuildFinalModel/GraphToGpuModel） |
| litert | `src/litert/runtime/compiled_model.cc` | 显式调用 `OptimizeMemoryForLargeTensors`；invoke 失败 dump errors |
| ml-drift | `ml_drift/common/selectors/operation_selector.cc` | legacy GPUOperationFromNode 支持 3D(5D) attr |
| ml-drift | `ml_drift/common/transformations/remove_noop.cc` | RemoveIdentityReshape 跳过 5D attr |
| chromium | `services/webnn/tflite/graph_impl_litert.cc` | BufferErrorReporter、OOM 诊断、Run 诊断、gpu-only switch |
| chromium | `services/webnn/webnn_switches.h/.cc` | `--webnn-tflite-gpu-only` switch + 转发白名单 |
