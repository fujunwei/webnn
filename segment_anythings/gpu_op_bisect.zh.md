# 用 `LITERT_GPU_DEBUG_*` 定位 SAM 在 WebGPU delegate 上算错的算子

> 背景：`segment_anything` 现在能编译、能跑完（1260 个算子全部下沉 WebGPU），
> 但推理结果不对。目标是把「哪个算子算错」精确到 **node index + opcode**。
>
> 相关文档：[analysis.zh.md](analysis.zh.md)（OOM + 不支持算子分布）、
> [sam_native_runner/DESIGN.md](sam_native_runner/DESIGN.md)（standalone runner 与
> `--verify` 双跑对比）、[layer_norm_fused_impl.md](layer_norm_fused_impl.md)。

---

## 0. 当前状态（2026-08-27）

### 0.1 已定位并修复的两个 bug

用 `sam_encoder_runner`（standalone，非浏览器）走完了一轮，找到两个真 bug，
都有独立的最小复现模型（`tools/make_rank5_repro_tflite.py`，单个 <8 KB，
一轮 `--verify` 不到 1 秒）：

| bug | 位置 | 症状 | 补丁 |
|---|---|---|---|
| `pow()` 负底数 | `ml_drift/common/kernels/elementwise.cc` 非-OpenCL 分支发裸 `pow($1,$2)`；WGSL 对 `x<0` 未定义，Dawn 降成 `exp2(y*log2(x))` → NaN | 分解 LayerNorm 里 `pow(x-mean, 2)` → 后面的 `MEAN` 把 NaN 摊开 → **输出 1048576 个元素全 NaN** | `25` |
| rank-2 常量 TRANSPOSE 不搬数据 | `model_builder.cc` 的 `TransposeConstantData()`：`perm_data.size() != 4` 就 `dst_data = src_data` | 48 个权重矩阵（每 block 的 qkv/proj/fc1/fc2）只换形状标签不转置 → 误差**逐 block 累积** | `29` |

另外补了 rank-5 形状直通（`26`/`28`，消掉 100 条 `Shape mismatch` 警告）和可配
读回超时（`27`）。

数值进展（老模型 GPU vs 自己的 CPU 参考，cosine）：

```
0.401  →  0.859   修掉 rank-2 常量转置之后
```

**还没到 1.0，剩余误差未定位。** 见 §0.3。

### 0.2 方法论：这份文档里有两条建议是错的

- **§7.2「`END_NODE` 扫描」在 SAM 上不可靠。** 实测非单调：`END_NODE=512` 和
  `994` 都是 cosine 0.53179，而**全部上 GPU 反而是 0.859**（更好）。在 rank-5
  张量处切分还会自己制造 NaN（`END_NODE=96` 全 NaN，但同样这些节点全 GPU 跑
  不产生 NaN）。切点本身在制造误差，测到的不是逐节点误差。
- **§7.1「按 opcode 整类排除」在部分下沉的模型上会给出假阴性。** 老模型
  1960 个算子里只有 **207** 个真正下沉，排除 `POW`/`MEAN`/`FULLY_CONNECTED`/
  `BATCH_MATMUL` 完全没变化——因为它们**本来就不在 GPU 上**。
  先用 `LITERT_GPU_DEBUG_DUMP_NODES=1` 看 `[partition] N nodes delegated:`
  （补丁 `30` 新加）拿到真正下沉的节点表，再排除。

**任何嫌疑都要用独立最小模型确认**，这是本轮唯一没出过错的判据。

### 0.3 剩余问题与下一步

老模型 GPU 上的 207 个算子几乎全是**权重预处理**（48× rank-2 权重转置、
48× `(768,)→(1,1,1,768)` 的 γ/β reshape、48× rank-2 `(1,14)` MUL、24× 标量
MUL），主计算（matmul/softmax/attention）都在 CPU。所以**一个权重预处理算子
算错，错权重就喂进 CPU 主计算** —— 这解释了为什么 207 个"平凡"算子能造成
0.86 的 cosine 误差。剩余嫌疑就在上面那几类里，`SUB` 已排除（真阴性）。

三种定位手段目前都失效（`END_NODE` 有伪影；按类排除会因 170 个碎片分区
编译失败或挂死；只留 custom LayerNorm 会被 delegate 拒绝）。建议改用
**截断模型对拍**：把 `.tflite` 在 node K 截断、让该处张量成为图输出，再对这个
自洽小模型跑 `--verify`（同模型 CPU vs GPU）。没有分区伪影，可精确到单算子。

### 0.4 两个基准事实

- **两个 encoder 模型权重逐字节相同**（10 类大常量全同），差别只是
  `new_segment_anything_encoder.tflite` 把 LayerNorm 融成了 24 个
  `custom_call.LayerNorm`。所以老模型的 CPU 输出
  （`ref_old_cpu.bin`）对**两个模型都是有效基准**。
- `custom_call.LayerNorm` **没有 CPU kernel**，所以：纯 CPU 编译新模型会挂在
  "Node number 12 failed to prepare"；`END_NODE < 1174` 的实验会把它推到 CPU，
  结果全是垃圾（≈ -4.3e8）。**新模型上只有 `END_NODE ≥ 1174` 或不含这 24 个
  节点的 `EXCLUDE_NODES` 才是有效实验。**

### 0.5 §3 补丁状态（原文保留）

**§3 的补丁已经打进 `third_party/litert/src`，DLL 已重编并部署，工具已就绪。**
可以直接从 §5 开始做实验。

| 项 | 状态 |
|---|---|
| `delegate_webgpu.cc` 去掉 `__linux__` 守卫 + log 提到 `WARNING` | ✅ 已改（未 commit，在 litert 工作区） |
| 新增 `LITERT_GPU_DEBUG_DUMP_NODES`（节点表 dump，§3.3） | ✅ |
| 新增 `LITERT_GPU_DEBUG_ONLY_NODE_COUNT`（**encoder / decoder 作用域隔离，§13**） | ✅ |
| `libLiteRtWebGpuAccelerator.dll` 重编（bazel `-c opt`） | ✅ 首次 46 s、增量 10 s，只重编 1 个 `.cc` + 重链，**未触碰 `chrome.dll`** |
| 部署到 `chromium\src\out\Release` | ✅ DLL 8.7 MB + PDB + `webgpu_dawn.dll` |
| `tools/node_table.py`（log → 算子表 → EXCLUDE 列表） | ✅ 已验证 |

四个环境变量：

| 变量 | 作用 |
|---|---|
| `LITERT_GPU_DEBUG_DUMP_NODES=1` | 打印 `[node-table]`：`idx / builtin_code / custom` |
| `LITERT_GPU_DEBUG_ONLY_NODE_COUNT=<N>` | 下面两个只对节点数为 N 的图生效（**两模型必用**） |
| `LITERT_GPU_DEBUG_END_NODE=<N>` | 只把 node `[0, N]` 作为 GPU 候选，其余回落 CPU |
| `LITERT_GPU_DEBUG_EXCLUDE_NODES=a,b,c` | 把这些 node 从已选中的集合里剔除，回落 CPU |

> 注：`delegate_webgpu.cc` 的 `git diff` 里还有一段 **本次改动之前就存在的**
> 未提交修改（`CreateWebGpuEnvironment` 里的 `SetFlushCallback`，`:230-250`），
> 与本文档无关，提交时注意分开。

---

## 1. 结论先行

| | 内容 |
|---|---|
| **能不能用** | 能。`delegate_webgpu.cc` 里这两个变量只被 `#if defined(__linux__)` 挡住，**删掉 4 行**即可在 Windows 生效，无任何平台相关代码 |
| **重编代价** | `delegate_webgpu.cc` **只被 bazel 编进 `libLiteRtWebGpuAccelerator.dll`**，GN 完全不编它（`grep -r delegate_webgpu --include=*.gn` 无命中）→ 改它**不用重链 `chrome.dll`**，一轮迭代分钟级 |
| **推荐路径** | 先「**按 opcode 整类排除**」定位到算子类型（≤10 次实验），再在类内用 END_NODE 扫描定位到具体 node。**不要一上来就对 1260 个节点做二分** |
| **最大的坑** | node 编号是 **LiteRT 运行时执行计划**的编号。浏览器路径下模型是 `GraphBuilderTflite` 现生成的，**和硬盘上那个 `.tflite` 的编号不是一回事** —— 必须用 `--webnn-tflite-dump-model` 导出真正喂进去的模型 |
| **第一嫌疑**（2026-08-20 的猜测） | 自定义 `LAYER_NORM`（`b6a623e` 刚改）> `TRANSPOSE`（patch 24 / rank-2 常量折叠）> `FULLY_CONNECTED`/`CONV_2D` bias（patch 23）> 5D `RESHAPE`（patch 24） |
| **实际结果**（2026-08-27） | 全 NaN 的元凶是 **`POW` 负底数**（不在上面的名单里）；第二个是 **rank-2 常量 `TRANSPOSE` 不搬数据**（名单里排第二，猜对了）。**`custom_call.LayerNorm` 目前没有证据**：它散布在图里，排除其余节点会产生 24 个碎片分区被 delegate 拒绝，隔离实验做不了。详见 §0 |

---

## 2. 这两个环境变量到底做了什么

源码：`third_party/litert/src/ml_drift_delegate/delegate/delegate_webgpu.cc`
（常量声明在 `:93-94`，两处使用在 `:503-523` 和 `:539-563`）。

### 2.1 `LITERT_GPU_DEBUG_END_NODE=N`

```cpp
// delegate_webgpu.cc:503-523
int start_node_index = 0;
int end_node_index = std::numeric_limits<int>::max();
if (delegate_options->debug_delegate_partition) {          // ← 见 §2.3
  ...
#if defined(__linux__)
} else if (auto* env_debug_end_node = std::getenv(kEnvDebugEndNode)) {
  ...
  if (absl::SimpleAtoi(env_debug_end_node, &end_node_index_from_env) &&
      context->GetNodeAndRegistration(context, end_node_index_from_env,
                                      &node, &reg) == kTfLiteOk &&
      reg != nullptr) {
    end_node_index = end_node_index_from_env;
    ABSL_LOG(INFO) << ... << ": code=" << reg->builtin_code;
  }
#endif  // defined(__linux__)
}
```

`start/end` 传给 `GetOpsToReplace(...)`，最终落到
`tflite::delegates::GraphPartitionHelper::PartitionImpl`
（`third_party/tflite/src/tensorflow/lite/delegates/utils.cc:183-187`）：

```cpp
if (IsNodeSupported(...)) {
  if (node_id < start_node_index) {
    continue;
  } else if (node_id > end_node_index) {
    break;
  }
  supported_nodes_->data[supported_nodes_->size++] = node_id;
}
```

**精确语义**：

- 区间**左闭右闭** `[start, end]`。`END_NODE=0` ⇒ 只有 node 0 是候选。
- 它定义的是**候选窗口**，不是「强制 0..N 全部上 GPU」。窗口内仍要过
  `IsNodeSupported`，且窗口外的节点**根本不进入分区**。
- 落在窗口外 / 不被支持的节点 → 回落 CPU（XNNPACK）。**这就是我们的对照组**。

### 2.2 `LITERT_GPU_DEBUG_EXCLUDE_NODES=a,b,c`

```cpp
// delegate_webgpu.cc:539-563 —— 在 GetOpsToReplace 之后，直接改 ops_to_replace
for (int i = 0; i < ops_to_replace->size; ++i) {
  int node_idx = ops_to_replace->data[i];
  if (excluded_nodes.contains(node_idx)) {
    ABSL_LOG(INFO) << "Excluding node " << node_idx << " (" << i
                   << " in ops_to_replace) from WebGPU delegation.";
  } else {
    ops_to_replace->data[new_size++] = node_idx;
  }
}
ops_to_replace->size = new_size;
```

**精确语义**：

- 在分区**之后**动手，所以它能在一片已选中的连续区间里「挖洞」。
- 挖洞后 `ops_to_replace` 可能不连续 —— 没关系，
  `ReplaceNodeSubsetsWithDelegateKernels` 会自己再切成多个 delegate kernel。
- 被挖掉的节点回落 CPU。**这是做单点确认和 delta debugging 的工具**。

### 2.3 一个必须注意的互斥关系

`END_NODE` 在 `else if` 分支里 —— 如果 `delegate_options->debug_delegate_partition`
为 true，**环境变量直接被忽略**。

好消息：在 `third_party/litert/src/litert/` 全树 grep `debug_delegate_partition`
**零命中**，即 LiteRT 的公开 Options API 没有暴露这个字段，浏览器路径下它恒为
`false`（`MlDriftDelegateOptions` 由
`MlDriftWebGpuDelegateDefaultOptionsPtr()` 零初始化）。所以环境变量分支**总是可达**。

`EXCLUDE_NODES` 不在 `else if` 里，**和 `END_NODE` 可以叠加使用**。

### 2.4 对照：OpenCL 版是 `#ifndef NDEBUG`

`delegate_opencl.cc:89-93 / 467-481 / 498-517` 是同一套逻辑，但守卫是
`#ifndef NDEBUG` 而不是 `__linux__`。两个 delegate 的守卫不一致，说明
`__linux__` 只是**当初随手加的、没考虑其他平台**，不是有意的平台限制 ——
删掉它没有语义风险（`std::getenv` / `absl::SimpleAtoi` / `absl::StrSplit`
在 Windows 上都正常）。

---

## 3. Windows 启用补丁（**已应用**，此节留作记录）

改一个文件：`third_party/litert/src/ml_drift_delegate/delegate/delegate_webgpu.cc`。
实际改动见 `git -C third_party/litert/src diff ml_drift_delegate/delegate/delegate_webgpu.cc`。

### 3.1 第一步：删掉平台守卫（4 行）

```diff
@@ -505,7 +505,6 @@ TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
   if (delegate_options->debug_delegate_partition) {
     start_node_index = delegate_options->debug_first_delegate_node_index;
     end_node_index = delegate_options->debug_last_delegate_node_index;
-#if defined(__linux__)
   } else if (auto* env_debug_end_node = std::getenv(kEnvDebugEndNode)) {
     TfLiteNode* node = nullptr;
     TfLiteRegistration* reg = nullptr;
@@ -518,7 +517,6 @@ TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
                      << ". Restricting WebGPU delegation from node 0 to node "
                      << end_node_index << ": code=" << reg->builtin_code;
     }
-#endif  // defined(__linux__)
   }
   TfLiteIntArray* ops_to_replace = nullptr;
@@ -536,7 +533,6 @@ TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
   }
 
-#if defined(__linux__)
   if (auto* env_debug_exclude_nodes = std::getenv(kEnvDebugExcludeNodes)) {
@@ -560,7 +556,6 @@ TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
     ops_to_replace->size = new_size;
   }
-#endif  // defined(__linux__)
 
   // Replace the ops with delegate kernel.
```

> 常量 `kEnvDebugEndNode` / `kEnvDebugExcludeNodes`（`:93-94`）本来就在守卫外面，
> 不用动。删掉守卫后它们不再是「Windows 下未使用的常量」，顺带消掉一个潜在
> `-Wunused-const-variable`。

### 3.2 第二步：把这三条 log 提到 `WARNING`

`ABSL_LOG(INFO)` 在 `-c opt` + GPU 进程里能不能出来取决于 absl 的日志初始化和
`--enable-logging` 组合。调试开关的确认信息必须 100% 可见，否则你会分不清
「变量没生效」和「生效了但没影响结果」—— 这是最浪费时间的一种歧义。

把 `:518` / `:548` / `:555` 三处 `ABSL_LOG(INFO)` 改成 `ABSL_LOG(WARNING)` 即可。

### 3.3 第三步（强烈建议）：加一个 node 表 dump

**这是整套方法能不能落地的关键。** 你需要一张
`node index → opcode` 的表，而且它必须和 `END_NODE`/`EXCLUDE_NODES` 用的是
**同一个编号空间**。自己解析 `.tflite` flatbuffer 容易错位（见 §6），
让运行时自己打出来最可靠。

在 `:94` 后加一个常量：

```cpp
constexpr char kEnvDebugDumpNodes[] = "LITERT_GPU_DEBUG_DUMP_NODES";
```

在 `DelegatePrepare` 开头（`:503` 那两行 `int start_node_index` **之前**）插入：

```cpp
  // Dump the full node table in exactly the index space used by
  // LITERT_GPU_DEBUG_END_NODE / _EXCLUDE_NODES. Debug-only; enabled by env var.
  if (std::getenv(kEnvDebugDumpNodes)) {
    TfLiteIntArray* plan = nullptr;
    if (context->GetExecutionPlan(context, &plan) == kTfLiteOk && plan) {
      ABSL_LOG(WARNING) << "[node-table] total=" << plan->size;
      for (int i = 0; i < plan->size; ++i) {
        const int node_index = plan->data[i];
        TfLiteNode* n = nullptr;
        TfLiteRegistration* r = nullptr;
        if (context->GetNodeAndRegistration(context, node_index, &n, &r) ==
                kTfLiteOk &&
            r != nullptr) {
          ABSL_LOG(WARNING) << "[node-table] idx=" << node_index
                            << " builtin_code=" << r->builtin_code
                            << " custom=" << (r->custom_name ? r->custom_name : "-");
        }
      }
    }
  }
```

> **为什么打 `builtin_code` 而不是 op 名**：`tflite::GetOpNameByRegistration()`
> 在 `tflite/util.h`，而 `delegate_webgpu` 的 bazel target 现在只依赖
> `//tflite:builtin_ops`（`BUILD:800` 附近）。打整数 code 不用动 `BUILD`，
> code → 名字用 §6.3 的脚本在外面映射即可。
>
> **索引取 `plan->data[i]` 而不是 `i`**：`PartitionImpl` 存进
> `supported_nodes_` 的、以及 `EXCLUDE_NODES` 比对的，都是执行计划里的
> **node id**，不是它在计划中的位置。新图上两者相等，但别赌这一点。

---

## 4. 重编与部署

```powershell
$CR    = "C:\Users\fujun\workspace\chromium\src"
$WEBNN = "C:\Users\fujun\workspace\webnn"

# 1. 重编加速器 DLL（只编 bazel 侧，不碰 chrome.dll）
& "$WEBNN\route-a-webgpu-windows\scripts\build_accelerator_dll.ps1" -Mode opt

# 2. 部署到 chrome out 目录
& "$WEBNN\route-a-webgpu-windows\scripts\deploy_to_chrome.ps1" `
    -Mode opt -ChromeOutDir "$CR\out\Release"
```

依据：`build_accelerator_dll.ps1` 从 `$CR\third_party\litert\src` 里 bazel 构建
`//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dll`；
`deploy_to_chrome.ps1` 把 `libLiteRtWebGpuAccelerator.dll` + `webgpu_dawn.dll`
拷进 out 目录。全程不触碰 GN 构建图。

---

## 5. 环境变量怎么进到真正执行的进程

WebNN / LiteRT 跑在 **GPU 进程**（依据：`route-a-webgpu-windows/patches/`
的 `08-webnn-sandbox-init-full-dll-path.patch` —— GPU 进程 pre-sandbox 阶段
`LoadLibrary` 加速器 DLL）。

Chrome 的子进程**继承 browser 进程的环境块**（环境是 `CreateProcess` 时拷贝的，
sandbox 只影响之后的系统调用，不影响已经拷进来的 `environ`），所以：

```powershell
$env:LITERT_GPU_DEBUG_DUMP_NODES = "1"
$env:LITERT_GPU_DEBUG_END_NODE   = "640"

C:\Users\fujun\workspace\chromium\src\out\Release\chrome.exe `
  --no-sandbox `
  --enable-logging=stderr --v=1 `
  --enable-features=WebMachineLearningNeuralNetwork `
  --user-data-dir=C:\Users\fujun\workspace\webnn\_chrome_test_profile `
  http://localhost:8080 2>&1 | Tee-Object -FilePath bisect_640.log
```

```powershell
PS C:\WINDOWS\system32> C:\Users\junweifu\AppData\Local\Chromium\Application\chrome.exe  --no-sandbox --enable-logging=stderr --v=1 -enable-features=WebMachineLearningNeuralNetwork --disable-features=WebNNDirectML,WebNNOnnxRuntime --webnn-tflite-dump-model=C:\Users\junweifu\workpace\tflite_models\empty_result --user-data-dir=C:\Users\junweifu\workpace\tflite_models\empty_result https://10.239.115.25:8080/demos/segment-anything/ 2>&1 | Tee-Object -FilePath C:\Users\junweifu\workpace\tflite_models\empty_result\run0.log

C:\Users\junweifu\workpace\tflite_models\empty_result>py node_table.py parse C:\Users\junweifu\workpace\tflite_models\empty_result\run0.log -o encoder_nodes.tsv
```

要点：

- **每次改变量都要完全退出 chrome 再起**（环境在进程创建时定格）。
  `--user-data-dir` 用独立 profile，避免复用已有 browser 进程。
- 清掉变量：`Remove-Item Env:\LITERT_GPU_DEBUG_END_NODE`。设成空字符串**没用** ——
  `std::getenv` 返回非空指针，`SimpleAtoi("")` 失败 → 变量被忽略，行为等同未设置；
  但 `DUMP_NODES` 只判断指针非空，空串仍会触发 dump。
- `--enable-logging=stderr` 是看到 `[node-table]` / `Excluding node` 的前提。

---

## 6. node 编号在哪个空间（最容易踩的坑）

### 6.1 编号来自 LiteRT 运行时，不是硬盘上的 `.tflite`

浏览器路径是：

```
litert.js → WebNN JS API → mojo → services/webnn
          → GraphBuilderTflite 现场序列化出一个新的 .tflite
          → LiteRT CompiledModel::Create → WebGPU delegate
```

`graph_builder_tflite.cc` 是按 **WebNN mojo 图**的算子顺序重新发射的，
和你喂给 litert.js 的那个 `SAMEncoder.tflite` 的节点顺序**没有保证的对应关系**
（算子会被拆分、融合、补 transpose）。

**解法**：用现成的上游开关把真正喂进 LiteRT 的模型 dump 出来
（`services/webnn/webnn_switches.h:22-28`，当前分支 `custom_lay_norm` 已有）：

```powershell
--no-sandbox --webnn-tflite-dump-model=C:\Users\fujun\workspace\webnn\_dump_models
```

之后所有编号**一律以 dump 出来的那个文件为准**。

### 6.2 delegate 的应用顺序保证了编号是「原始」的

`litert/runtime/compiled_model.cc:990-992`：

```cpp
// Apply accelerators matching the requested hardware support to the
// model in the order they were registered.
for (auto& accelerator : env->GetAcceleratorRegistry()) {
```

注册顺序（runner 日志实证，见 DESIGN.md §9.3）：

```
RegisterAccelerator: name=GPU WebGPU
RegisterAccelerator: name=CpuAccelerator
```

**GPU 先于 XNNPACK 应用** ⇒ WebGPU delegate 的 `DelegatePrepare` 看到的是
未被任何 delegate 改写过的图，node index == dump 出来的模型里的算子序号。

⚠️ 反过来说：这个性质**只对第一个 delegate 成立**。如果哪天注册顺序变了、
或者你想在 XNNPACK 里做同样的事，编号会整体错位。每轮实验都用
`[node-table] total=N` 核对总数（现在应该是 **1260**）。

### 6.3 `builtin_code` → op 名：`tools/node_table.py`

日志里打的是整数 `builtin_code`（原因见 §3.3）。
[`tools/node_table.py`](tools/node_table.py) 负责把它和
`third_party/tflite/src/tensorflow/lite/builtin_ops.h`（210 个 opcode）join 起来：

```powershell
cd C:\Users\fujun\workspace\webnn\segment_anythings\tools

# 1. chrome 日志 → idx/code/name/custom 四列表
py node_table.py parse ..\..\bisect_dump.log -o node_table.tsv

# 2. 看算子分布，决定先排哪一类
py node_table.py hist node_table.tsv

# 3. 生成某一类的 EXCLUDE_NODES 列表（支持 builtin 名和 custom 名）
py node_table.py exclude node_table.tsv Transpose
py node_table.py exclude node_table.tsv odml.fused_layer_norm
```

`hist` 输出形如：

```
FullyConnected                     2
Conv2d                             1
Transpose                          1
Custom[odml.fused_layer_norm]      1
Reshape                            1
TOTAL                              6
```

`exclude` 把 node 列表打到 stdout、把计数打到 stderr，所以可以直接接给环境变量：

```powershell
$env:LITERT_GPU_DEBUG_EXCLUDE_NODES = (py node_table.py exclude node_table.tsv Transpose)
```

两个细节：

- 自定义算子（例如 fused LayerNorm）`builtin_code` 是
  `kTfLiteBuiltinCustom`(32)，只能靠 `custom=` 字段区分 —— `hist` 会把它显示成
  `Custom[<名字>]`，`exclude` 两种名字都认。
- `DelegatePrepare` 每个模型跑一次，**encoder 和 decoder 会各打一张表**。
  `parse` 只保留第一张（按 idx 回绕检测），并在发现多张时提示。
  所以：**一次只跑一个模型**，或者确认第一张表就是你要的那个。

---

## 7. 三种定位策略

### 7.1 策略 A：按 opcode 整类排除 —— **首选**

有了 §6.3 的全表，把某一类算子的所有 node index 一次性排除：

```powershell
# 例：排除全部 TRANSPOSE
$env:LITERT_GPU_DEBUG_EXCLUDE_NODES = (py node_table.py exclude node_table.tsv Transpose)
Remove-Item Env:\LITERT_GPU_DEBUG_DUMP_NODES -ErrorAction SilentlyContinue  # 表已经有了，别再刷屏
# 然后完全退出 chrome 再重启（§5）
```

| 结果 | 结论 |
|---|---|
| 输出变正确 | 这一类算子里有 bug，进入 §7.2 在类内定位 |
| 输出仍错但**明显变好** | 这一类是**之一**，记下来，继续排下一类 |
| 输出无变化 | 这一类无辜，划掉 |

**为什么首选**：本项目历史上的每一个 delegate bug 都是「某类算子的某个 corner
case」——rank-2 常量 transpose 折叠形状写反（DESIGN.md §10.3）、bias 描述符
（patch 23）、5D reshape/transpose（patch 24）、fused LayerNorm（`b6a623e`）。
按类排除 ≤10 次实验就能锁定算子类型，比 1260 节点二分的 11 次**信息量高得多**
（二分给你一个数字，按类排除直接给你算子名）。

### 7.2 策略 B：`END_NODE` 扫描 —— **扫描，不要二分**

```
END_NODE = 0, 64, 128, 192, ... 1259
```

每个 N 记录 `max_abs(gpu_output, reference)`，画一条曲线。

**为什么是扫描不是二分**：

1. **误差不单调**。一个算错的算子后面接 softmax / clamp / 归一化，误差可能被
   压回去，二分的「单调假设」就破了，会直接找错点。
2. **fp16 基线本身随 N 增长**。GPU fp16 vs CPU fp32 的累积误差是一条缓慢上升的
   曲线；真正的 bug 是叠在上面的**一级台阶**。只有看到整条曲线才分得清
   「缓慢上升」和「台阶」。
3. 一次实验 10 秒级（`CompiledModel::Create` 约 10 s，见 DESIGN.md §12.2），
   20 个采样点也就几分钟，没必要省。

粗扫看到台阶落在 `(N₁, N₂]` 之后，在这个区间里逐节点细扫。

**必须核对**：每轮从日志里读
`X operations will run on the GPU, and the remaining Y ...`，
确认实际下沉的算子数 ≈ N+1。见 §9.1。

### 7.3 策略 C：`EXCLUDE_NODES` 单点确认 + delta debugging

拿到嫌疑 node `K`：

```powershell
$env:LITERT_GPU_DEBUG_EXCLUDE_NODES = "K"   # 其余 1259 个仍在 GPU
```

- 输出正确 ⇒ **K 确认**。这是最强的证据：只有它回落 CPU，别的都没动。
- 输出仍错 ⇒ 有多个 culprit。对嫌疑集合做 delta debugging：
  每次排除一半，二分**集合**而不是二分**位置**。

---

## 8. 参考值（ground truth）从哪来

没有可信参考值，上面全部白搭。按可信度排序：

1. **同一模型走 CPU（XNNPACK fp32）**。最干净。浏览器里用
   `navigator.ml.createContext({deviceType: 'cpu'})` 跑一遍，把 encoder
   输出（`1×256×64×64` = 1 048 576 个 float）存成 `.bin`，之后每轮和它比。
2. **`LITERT_GPU_DEBUG_END_NODE=0`**。只有 node 0 在 GPU，其余全 CPU ——
   近似全 CPU 基线，同时验证「变量确实生效了」。
   ⚠️ **不要用 `-1` 或超大值当「全 CPU」**：
   `GetNodeAndRegistration` 对越界索引返回 `kTfLiteError`
   （`tflite/core/subgraph.cc:1912-1922` 有 `node_index >= 0` 和
   `< nodes_size` 两道 `TF_LITE_ENSURE`），条件不成立 ⇒
   `end_node_index` 保持 `INT_MAX` ⇒ **反而是全 GPU**，和你想要的正好相反。
3. **原始 `.tflite` 在 onnxruntime / tfjs / Python TFLite 里跑**。作为
   端到端 sanity check，但注意它和 WebNN 重新发射的图不是同一个图。

比较指标（在 JS 里算就行）：

```js
let maxAbs = 0, sum = 0;
for (let i = 0; i < gpu.length; i++) {
  const d = Math.abs(gpu[i] - ref[i]);
  if (d > maxAbs) maxAbs = d;
  sum += d;
}
console.log(`max_abs=${maxAbs} mean_abs=${sum / gpu.length}`);
```

SAM encoder 输出的量级大致在 ±1，所以：`max_abs < 0.05` ≈ fp16 正常噪声；
`max_abs > 0.5` ≈ 真的算错了。跑一次全 CPU vs 全 CPU 得到 0，再跑一次
fp16 全 GPU，就知道当前的「错」有多离谱。

---

## 9. 已知陷阱清单

### 9.1 `max_delegated_partitions=1` —— 窗口内只取最大连通块

`delegate_webgpu.cc:533-536` 传的是 `max_delegated_partitions=1`，
而 `model_builder.cc:7454-7458`：

```cpp
// By default, we simply get 1st largest partition as 'max_delegate_partions'
// is set to 1 by default.
std::vector<int> ops_to_replace =
    partition_helper.GetNodesOfFirstNLargestPartitions(max_delegated_partitions);
```

⇒ 如果窗口 `[0, N]` 内有不被支持的算子把它切成几段，**只有最大的那段上 GPU**。
SAM 现在 1260 个算子全支持，理论上是一整块；但一旦你在调试中改了别的东西
导致某个算子重新变成不支持，实际下沉的集合就会和你以为的不一样。

**每轮都从日志核对下沉算子数。**

### 9.2 切在中间可能把 OOM 又切出来

这正是 [analysis.zh.md](analysis.zh.md) 那个问题的反面：算子留在 CPU ⇒ tflite
arena 要为中间大张量分配空间 ⇒ 可能再次触发 PartitionAlloc
`PartitionExcessiveAllocationSize`（进程退出码 `0xE0000008` / `-536870904`）。

**某个 N 崩溃 ≠ 那个 node 算错**，那是 OOM。日志里会有
`webnn debug litert [PartitionAlloc][OOM-diag]`（patch `0001` 加的诊断）。
遇到就跳过这个 N。

> PowerShell 会把 `-536870904` 显示成 `Exit Code: 1`，用 `$LASTEXITCODE` 或
> `[Convert]::ToString($LASTEXITCODE, 16)` 确认。

### 9.3 先排除「这只是 fp16 精度问题」

`graph_impl_litert.cc:416-428`：

```cpp
gpu_options->SetPrecision(graph_requires_fp32_precision
                              ? ::litert::GpuOptions::Precision::kFp32
                              : ::litert::GpuOptions::Precision::kFp16);
```

`graph_requires_fp32_precision` 由图内容推导
（`graph_builder_tflite.cc:735-748`，逐算子 `RequiresFloat32Precision`）。

**做任何 bisect 之前，先把它硬写成 `true` 跑一轮全 GPU**
（改 `graph_builder_tflite.cc:735` 的初值，或直接改 `graph_impl_litert.cc:425`
那个三元表达式）：

- fp32 下结果**正确** ⇒ 是精度/数值范围问题，去查有没有算子在 fp16 下溢出
  （LayerNorm 的方差、attention 的 `exp`），不用 bisect 算子。
- fp32 下**仍然错** ⇒ 结构性 bug，bisect 才有意义。而且 fp32 会把基线噪声
  压到接近 0，§7.2 的台阶会非常干净。

> 先例：DESIGN.md §12.2 里 `ml_drfit_add.tflite` 的常量被错误广播，
> `--precision=fp32` 结果完全一样 —— 一次实验就把「精度问题」排除了。

### 9.4 SAM 是**两个**模型，环境变量是**全局**的

encoder + decoder 都会经过同一个 delegate，`END_NODE=640` 会**同时**截断两个图。

已经用 `LITERT_GPU_DEBUG_ONLY_NODE_COUNT` 解决 —— **见 §13**。

### 9.5 pipeline cache

`MlDriftWebGpuDelegateDefaultOptionsPtr()`（`delegate_webgpu.cc:617`）里
`serialize_program_cache = true`。改了 delegate 行为之后如果结果诡异地没变化，
先清掉序列化缓存目录再试（`serialization_dir` + `model_token` 都设了才会真正落盘，
默认没设，但值得作为「结果不随代码变化」时的第一检查项）。

### 9.6 standalone runner 在本机拿不到真 GPU

DESIGN.md §9.4 / §12.2：runner 进程选到 `Microsoft Basic Render Driver (WARP)`，
且 1M 元素输出的 `Lock(kRead)` 回读失败。而且
`services/webnn/tflite/sam_runner/` **在当前 chromium 树（分支
`custom_lay_norm`）里并不存在** —— 它是另一台机器（`junweifu`）上的产物。

⇒ **本次调试走浏览器路径**。浏览器的 GPU 进程能拿到真 adapter，
JS 侧读回也正常。runner 只有在你愿意先把它重新加回来、并按 DESIGN.md §8
注入真实 Dawn device 之后才更快。

---

## 10. 针对 SAM 的建议起手式

```
□ 0. 分离 encoder / decoder，确定是哪个模型错。
     只跑 encoder，输出 embedding 与 CPU context 的结果比 max_abs。

□ 1. LITERT_GPU_DEBUG_DUMP_NODES=1 跑一次，拿到 [node-table]，核对 total=1260。
     py tools\node_table.py parse <log> -o node_table.tsv
     py tools\node_table.py hist node_table.tsv
     （需要看算子属性/shape 时再加 --webnn-tflite-dump-model 导出模型本体）

□ 2. 强制 fp32 跑一轮（§9.3）。
     正确 → 转精度问题，本文档剩余部分不用做。
     错误 → 继续，并且后续所有实验都在 fp32 下做（基线更干净）。

□ 3. 按 opcode 整类排除，优先级：
     ① custom LAYER_NORM  ← b6a623e 刚接入 ml-drift 自定义实现，改动最新
     ② TRANSPOSE          ← rank-2 常量折叠 BHWC 写反刚修过（DESIGN.md §10.3）
                             + patch 24 的 5D transpose
     ③ FULLY_CONNECTED / CONV_2D  ← patch 23 bias descriptor 刚修过
     ④ RESHAPE            ← patch 24 的 5D reshape，规划器仍有 bhwdc 告警
                             （DESIGN.md §10.6 遗留项）
     ⑤ SOFTMAX / MUL / ADD / GATHER

□ 4. 命中某一类后，只在该类的 node 上做 §7.2 扫描 / §7.3 集合二分，
     定位到具体 node index。

□ 5. 用 EXCLUDE_NODES=<单个 K> 做最终确认：只排它一个，结果应恢复正确。

□ 6. 拿 K 的 node index → 在 dump 出来的模型里看它的输入/输出 shape 和
     属性，对照 ml-drift 里对应 parser/kernel 的实现。
```

---

## 11. 实验记账模板

| # | 变量设置 | precision | 下沉算子数（日志） | max_abs | 退出/异常 | 结论 |
|---|---|---|---|---|---|---|
| 0 | 无（全 GPU 基线） | fp16 | 1260 | | | 现状 |
| 1 | 无 | **fp32** | 1260 | | | 是否精度问题 |
| 2 | `END_NODE=0` | fp32 | 1 | | | 近似全 CPU 参考 |
| 3 | `EXCLUDE=<所有 LAYER_NORM>` | fp32 | | | | |
| 4 | `EXCLUDE=<所有 TRANSPOSE>` | fp32 | | | | |
| … | | | | | | |

> 每行都必须填「下沉算子数」——它是唯一能证明「变量真的生效了」的字段。

---

## 12. 附：`delegate_webgpu.cc` 相关行号速查

行号为**打完补丁之后**的（`git diff` 见 §0）。

| 位置 | 内容 |
|---|---|
| `:93-94` | `kEnvDebugEndNode` / `kEnvDebugExcludeNodes` 常量 |
| `:95-97` | `kEnvDebugDumpNodes` 常量（新增） |
| `:510-529` | `[node-table]` dump（新增） |
| `:531-532` | `start_node_index` / `end_node_index` 初值 |
| `:533-535` | `debug_delegate_partition` 分支（LiteRT 公开 API 无法触发） |
| `:536-548` | `END_NODE` 解析（`__linux__` 守卫已删） |
| `:550-563` | `GetOpsToReplace`（`max_delegated_partitions=1`） |
| `:565-587` | `EXCLUDE_NODES` 解析与剔除（`__linux__` 守卫已删） |
| `:634-644` | `MlDriftWebGpuDelegateDefaultOptionsPtr()` 默认值 |

其他文件：

| 文件:行 | 内容 |
|---|---|
| `ml_drift_delegate/delegate/delegate_options.h:63-73` | `debug_delegate_partition` 等字段定义 |
| `ml_drift_delegate/delegate/delegate_opencl.cc:89-93` | OpenCL 版用 `#ifndef NDEBUG`（对照） |
| `ml_drift_delegate/tflite/model_builder.cc:7446-7458` | 分区 + 只取最大块 |
| `third_party/tflite/src/tensorflow/lite/delegates/utils.cc:183-187` | `[start, end]` 左闭右闭 |
| `third_party/tflite/src/tensorflow/lite/core/subgraph.cc:1912-1922` | 越界索引返回 `kTfLiteError` |
| `litert/runtime/compiled_model.cc:990-992` | 按注册顺序应用 delegate |
| `services/webnn/tflite/graph_impl_litert.cc:416-428` | GPU 精度选择 |
| `services/webnn/tflite/graph_builder_tflite.cc:735-748` | `graph_requires_fp32_precision` 推导 |
| `services/webnn/webnn_switches.h:22-28` | `--webnn-tflite-dump-model` |

---

## 13. encoder / decoder 两个模型怎么分开调

> 场景：`https://10.239.115.25:8080/demos/segment-anything/`，
> `--webnn-tflite-dump-model=D:\tflite-dump-model` 会 dump 出两个模型
> （encoder 先、decoder 后）。环境变量是**进程级**的，`DelegatePrepare`
> 对两个图各调一次，所以裸用 `END_NODE` 会把两个模型一起截断。

### 13.1 新增的作用域开关

```
LITERT_GPU_DEBUG_ONLY_NODE_COUNT=<N>
```

设了它之后，`END_NODE` / `EXCLUDE_NODES` **只对节点数恰好等于 N 的那个图生效**，
另一个图完全不受影响（走正常的全 GPU 路径）。每个图都会打一行确认：

```
LITERT_GPU_DEBUG_ONLY_NODE_COUNT=1260, this graph has 1260 nodes -> debug knobs APPLY
LITERT_GPU_DEBUG_ONLY_NODE_COUNT=1260, this graph has  318 nodes -> debug knobs SKIPPED
```

用节点数而不是「第几个模型」做判据，是因为编译顺序会受缓存 / 页面改动影响，
节点数不会。

### 13.2 Step 0：一次跑完，拿到两个模型的节点数

```powershell
$env:LITERT_GPU_DEBUG_DUMP_NODES = "1"
Remove-Item Env:\LITERT_GPU_DEBUG_END_NODE, Env:\LITERT_GPU_DEBUG_EXCLUDE_NODES, `
            Env:\LITERT_GPU_DEBUG_ONLY_NODE_COUNT -ErrorAction SilentlyContinue

C:\Users\fujun\workspace\chromium\src\out\Release\chrome.exe `
  --no-sandbox --enable-logging=stderr --v=1 `
  --ignore-certificate-errors `
  --enable-features=WebMachineLearningNeuralNetwork `
  --webnn-tflite-dump-model=D:\tflite-dump-model\run0 `
  --user-data-dir=C:\Users\fujun\workspace\webnn\_chrome_test_profile `
  https://10.239.115.25:8080/demos/segment-anything/ 2>&1 |
  Tee-Object -FilePath D:\tflite-dump-model\run0.log
```

- `--ignore-certificate-errors`：`https://<IP>` 的证书必然不匹配，不加进不去。
  这个 flag 是存在的，定义在
  `components/network_session_configurator/common/network_switch_list.h:34`
  （用 X-macro 生成，所以直接 grep 字符串 `"ignore-certificate-errors"`
  在 `net/`、`services/network/` 里都搜不到，别被误导）。
  `content_shell` 也认它（`content/shell/browser/shell_browser_context.cc:72`）。

  WebNN 要 secure context，所以**不能**把远端 `https://<IP>` 降级成 http。
  但 `http://localhost:<port>` 属于 potentially trustworthy origin，**本身就是
  secure context**（`services/network/public/cpp/is_potentially_trustworthy_unittest.h:216`），
  所以把 demo 放到本机用 http 提供是完全可行的，而且能绕开证书和代理两个坑。

### 13.4.1 `content_shell` 上不了外网（2026-08-28）

用 `content_shell.exe` 打开外网页面（如 `www.baidu.com`）会失败，**这不是证书
配置问题，是 content_shell 根本不走代理**：

- `ShellContentBrowserClient::ConfigureNetworkContextParamsForShell()` 只设了
  user_agent / accept_language / zstd 等，**没有设 `initial_proxy_config`，
  也没有设 `proxy_config_client_receiver`**；
- 于是 `services/network/network_context.cc:3098` 兜底成
  `net::ProxyConfigWithAnnotation::CreateDirect()` —— 直连；
- 本机直连是被挡的（`Test-NetConnection www.baidu.com -Port 443` →
  `TcpTestSucceeded: False`），系统代理是 `http://proxy-ir.intel.com:911`。

**`--proxy-server` 救不了**：`services/network/` 和
`components/network_session_configurator/` 里都没有读这个 flag，它是在
`chrome/browser` 里被消费的，而 content_shell 没有那一层。

所以：**要联网就用 `chrome.exe`**（它有完整代理支持，实测能通过代理访问外网）；
要用 `content_shell` 就把页面放到 `http://localhost:<port>`（localhost 在
`NO_PROXY` 里，直连可达，且是 secure context）。
- **dump 到一个新的空目录**（`run0\`）。`D:\tflite-dump-model` 根目录里现在躺着
  8 月 4 日和 8 月 12 日的两份旧 encoder dump，混在一起会认错模型。

日志里会出现**两段** `[node-table] total=`：**第一段是 encoder，第二段是 decoder**。
记下两个 N。两段分别拆出来：

```powershell
cd C:\Users\fujun\workspace\webnn\segment_anythings\tools
# parse 只取第一张表（encoder）
py node_table.py parse D:\tflite-dump-model\run0.log -o encoder_nodes.tsv
py node_table.py hist encoder_nodes.tsv
```

> decoder 的表要单独拿：先把日志里第二段 `[node-table]` 切出来存成一个文件，
> 再对它跑 `parse`。`parse` 是按 idx 回绕检测切分的，一次只吐第一张。

### 13.3 Step 1：判定错的是哪个模型（2 次实验）

`END_NODE=0` ⇒ 该模型只有 node 0 上 GPU，其余全回落 CPU ≈ 整个模型走 CPU。

```powershell
# 实验 A —— 只让 encoder 回落 CPU，decoder 保持全 GPU
$env:LITERT_GPU_DEBUG_ONLY_NODE_COUNT = "<encoder N>"
$env:LITERT_GPU_DEBUG_END_NODE        = "0"
# 重启 chrome，看 mask

# 实验 B —— 只让 decoder 回落 CPU，encoder 保持全 GPU
$env:LITERT_GPU_DEBUG_ONLY_NODE_COUNT = "<decoder N>"
$env:LITERT_GPU_DEBUG_END_NODE        = "0"
```

| A（encoder→CPU） | B（decoder→CPU） | 结论 |
|---|---|---|
| mask 正确 | mask 仍错 | **encoder 有 bug**，去 §7 对 encoder 做 bisect |
| mask 仍错 | mask 正确 | **decoder 有 bug** |
| 都仍错 | 都仍错 | **两个都有 bug**，先修 encoder（decoder 的输入本来就是错的，先修上游） |
| 都正确 | 都正确 | 两个各自的误差都在阈值内，是**叠加**后越界 —— 大概率是 fp16 精度问题，回 §9.3 |

⚠️ **每次改环境变量都必须完全退出 chrome 再启动**（环境块在进程创建时定格），
并且每轮都要在日志里确认 `-> debug knobs APPLY` 落在你想要的那个图上。

### 13.4 Step 2：给 encoder 一个数值信号

mask 是阈值化之后的结果 —— 小误差被吃掉、大误差直接饱和，**做细粒度 bisect
的信号极差**。要拿 encoder 输出的原始 embedding（float）。

页面不是你写的也没关系，在 DevTools Console 里挂 `MLContext.readTensor`
（`ml_context.idl:330,337` 两个重载都要处理），**在跑推理之前**执行：

```js
(() => {
  const orig = MLContext.prototype.readTensor;
  window.__caps = [];
  MLContext.prototype.readTensor = async function (tensor, outputData) {
    if (outputData !== undefined) {                 // readTensor(t, buf) -> undefined
      await orig.call(this, tensor, outputData);
      const v = ArrayBuffer.isView(outputData)
        ? new Float32Array(outputData.buffer, outputData.byteOffset,
                           outputData.byteLength / 4)
        : new Float32Array(outputData);
      window.__caps.push(v.slice());
      return;
    }
    const buf = await orig.call(this, tensor);      // readTensor(t) -> ArrayBuffer
    window.__caps.push(new Float32Array(buf).slice());
    return buf;
  };
  console.log('readTensor hooked');
})();
```

跑一次推理后 `window.__caps` 里就是各个输出张量。encoder 的 embedding 是
最大的那个（MobileSAM `[1,64,64,256]` = 1 048 576 个 float）。

因为改环境变量要重启 chrome、JS 状态会丢，**基线要存成文件**：

```js
// 全 CPU 那一轮：存基线
const a = window.__caps.find(x => x.length === 1048576);
Object.assign(document.createElement('a'), {
  href: URL.createObjectURL(new Blob([a.buffer])), download: 'ref_emb.bin',
}).click();
```

之后每一轮，把 `ref_emb.bin` 拖进页面的一个 `<input type=file>`，或者直接放到
demo 的静态目录里 `fetch()` 回来，然后：

```js
const ref = new Float32Array(await (await fetch('/ref_emb.bin')).arrayBuffer());
const cur = window.__caps.find(x => x.length === ref.length);
let m = 0, s = 0;
for (let i = 0; i < ref.length; i++) {
  const d = Math.abs(cur[i] - ref[i]); if (d > m) m = d; s += d;
}
console.log(`max_abs=${m}  mean_abs=${s / ref.length}`);
```

这条 `max_abs` 就是 §7.2 扫描曲线的纵轴。

> 如果 `10.239.115.25` 那个页面是你自己的，直接在页面里加这段比较逻辑要省事得多，
> 顺便把 `max_abs` 打到 DOM 上，就不用每轮开 DevTools 了。

### 13.5 Step 3：锁定模型之后

后面就是 §7 的常规流程，唯一的区别是 **`ONLY_NODE_COUNT` 全程保持不变**，
指向那个有问题的模型：

```powershell
$env:LITERT_GPU_DEBUG_ONLY_NODE_COUNT = "<有问题的那个 N>"   # 从此不再改动

$env:LITERT_GPU_DEBUG_EXCLUDE_NODES = (py node_table.py exclude encoder_nodes.tsv Transpose)
# ... §7.1 按 opcode 整类排除
```

### 13.6 decoder 特有的注意点

- decoder 的输入之一是 encoder 的 embedding。**只要 encoder 还在 GPU 上跑且是错的，
  decoder 的一切对比都不可信**。调 decoder 之前先用
  `ONLY_NODE_COUNT=<encoder N>` + `END_NODE=0` 把 encoder 钉在 CPU 上，
  给 decoder 一个干净的输入。两个变量可以叠加，但 `ONLY_NODE_COUNT` 只能指一个图 ——
  所以这一步是：encoder 钉 CPU（`ONLY_NODE_COUNT=<encoder N>`）跑一轮存基线，
  然后切到 `ONLY_NODE_COUNT=<decoder N>` 做 decoder 的 bisect，
  此时 encoder 回到 GPU，其输出会变 —— **对比基线必须同轮重取**，别跨轮复用。
- decoder 每点一个 point 就重跑一次，同一次页面加载里能反复取样，比 encoder 好调。
- decoder 小得多（几百个节点），`hist` 之后往往一眼就能看出可疑算子，
  未必需要扫描。
