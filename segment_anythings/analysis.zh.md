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
