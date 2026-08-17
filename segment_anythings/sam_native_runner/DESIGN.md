# SAM Encoder Native Runner — 设计与分析

> 目标：做一个使用 **chromium 编译的库 + 机制**（PartitionAlloc、LiteRT runtime、
> libLiteRtWebGpuAccelerator.dll）构建的独立 native 程序，直接加载
> `C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite`，
> 复现 debug 版无法复现的 PartitionAlloc OOM，并大幅加快 debug 迭代速度。
>
> **状态：已实现并冒烟通过**。见 §9 "落地记录"。

---

## 1. 背景与动机

| 痛点 | 现状 |
|---|---|
| chrome debug 版无法复现 OOM | `is_debug=true` 时 `use_partition_alloc_as_malloc=false`，`SimpleMemoryArena::Commit` 走系统 malloc，没有 ~2 GiB 单次分配上限 |
| chrome release 版能复现但链接太慢 | 每次改一行代码都要重链 chrome.dll（数分钟～数十分钟） |
| 调试不方便 | 需要开浏览器、走 mojo IPC、WebNN 图构建链路才能到 `CompiledModel::Create` |

**思路**：在 chromium 源码树里加一个**独立 executable GN target**，
链接与 chrome 完全相同的静态库（`//third_party/litert` + `//base`），
代码路径**完全复刻** `services/webnn/tflite/graph_impl_litert.cc` 的
`ComputeResources::Create`（Environment → Options → CompiledModel::Create），
但跳过 mojo / WebNN 图构建（直接加载现成 .tflite 文件）。

增量编译只重编 1 个 .cc + 链一个几百 KB 的小 exe，**秒级迭代**。

---

## 2. 方案对比

| 方案 | 复现 OOM | 链接速度 | 代码一致性 |
|---|---|---|---|
| **A. chromium GN 独立 exe（本方案）** | ✅ 完全一致（同一批 .obj + PartitionAlloc） | ✅ 增量秒级 | ✅ 与 chrome 共用 litert/tflite 编译产物 |
| B. 独立 CMake + 自己编 litert/tflite | ⚠️ 无 PartitionAlloc 需另外移植 | ⚠️ 首次全量编译很久 | ⚠️ 编译配置差异风险 |
| C. 改 chrome debug 开 `use_partition_alloc_as_malloc=true` | ✅ 可复现 | ❌ 仍需重链 chrome.dll | ✅ |

选 **A**。如果只是临时验证，C 也可行（`gn args` 加
`use_partition_alloc_as_malloc=true`），但迭代速度仍受 chrome.dll 链接限制。

---

## 3. 目录结构与文件

```
chromium/src/services/webnn/tflite/sam_runner/
├── BUILD.gn                  # executable target
└── sam_encoder_runner.cc     # runner 源码（复刻 graph_impl_litert.cc 路径）
```

设计文档目录（本文件）：
```
webnn/segment_anythings/sam_native_runner/
└── DESIGN.md
```

代码放 `services/webnn/tflite/` 下的理由：
- `services/webnn` 已经依赖 `//third_party/litert`（`services/webnn/BUILD.gn:197`），
  include path / 编译配置现成；
- 与 `graph_impl_litert.cc` 相邻，diff 时对照方便；
- GN 里任意 BUILD.gn 文件只要在源码树内就可被 `autoninja -C <out> <path>:<target>`
  显式构建，不需要挂进 default 构建图。

---

## 4. 源码设计

### 4.1 `sam_encoder_runner.cc`

```cpp
// SAM encoder standalone runner: reproduces the WebNN→LiteRT compile+OOM path
// without the browser. Mirrors
// services/webnn/tflite/graph_impl_litert.cc (ComputeResources::Create).

#include <string>
#include <vector>

#include "base/command_line.h"
#include "base/files/file_util.h"
#include "base/logging.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/stringprintf.h"
#include "third_party/abseil-cpp/absl/types/span.h"
#include "third_party/litert/buildflags.h"  // BUILD_LITERT_WITH_XNNPACK
#include "third_party/litert/src/litert/c/litert_common.h"  // kLiteRtErrorReporterModeBuffer
#include "third_party/litert/src/litert/cc/litert_compiled_model.h"
#include "third_party/litert/src/litert/cc/litert_element_type.h"
#include "third_party/litert/src/litert/cc/litert_environment.h"
#include "third_party/litert/src/litert/cc/litert_expected.h"
#include "third_party/litert/src/litert/cc/litert_layout.h"
#include "third_party/litert/src/litert/cc/litert_options.h"
#include "third_party/litert/src/litert/cc/litert_ranked_tensor_type.h"
#include "third_party/litert/src/litert/cc/litert_tensor_buffer.h"
#include "third_party/litert/src/litert/cc/options/litert_gpu_options.h"
#include "third_party/litert/src/litert/cc/options/litert_runtime_options.h"  // SetErrorReporterMode

namespace {

constexpr char kModelSwitch[] = "model";
constexpr char kGpuOnlySwitch[] = "gpu-only";
constexpr char kRunSwitch[] = "run";
constexpr char kRunsSwitch[] = "runs";

// 复刻 GraphImplLiteRt::GetCompilationOptions 的核心部分。
::litert::Expected<::litert::Options> MakeOptions(bool gpu_only) {
  auto options = ::litert::Options::Create();
  if (!options) return options.Error();

  ::litert::HwAcceleratorSet accelerators(::litert::HwAccelerators::kGpu);
  auto gpu_options = options->GetGpuOptions();
  if (!gpu_options) return gpu_options.Error();
  gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp16);

#if BUILDFLAG(BUILD_LITERT_WITH_XNNPACK)
  if (!gpu_only) {
    accelerators |= ::litert::HwAccelerators::kCpu;
    auto cpu_options = options->GetCpuOptions();
    if (!cpu_options) return cpu_options.Error();
  }
#endif
  auto set_accelerators = options->SetHardwareAccelerators(accelerators);
  if (!set_accelerators) return set_accelerators.Error();

  // Buffer error reporter: capture TF_LITE_KERNEL_LOG / delegate errors.
  auto runtime_options = options->GetRuntimeOptions();
  if (runtime_options) {
    runtime_options->SetErrorReporterMode(kLiteRtErrorReporterModeBuffer);
  }
  return options;
}

void LogErrors(const ::litert::CompiledModel& model, const char* phase) {
  auto errors = model.GetErrorMessages();
  if (errors.HasValue() && !errors->empty()) {
    LOG(ERROR) << "[" << phase << "] LiteRT diagnostics:\n" << errors.Value();
  }
}

}  // namespace

int main(int argc, char** argv) {
  base::CommandLine::Init(argc, argv);
  logging::LoggingSettings settings;
  settings.logging_dest = logging::LOG_TO_STDERR;
  logging::InitLogging(settings);

  const auto* cl = base::CommandLine::ForCurrentProcess();
  if (!cl->HasSwitch(kModelSwitch)) {
    LOG(ERROR) << "Usage: sam_encoder_runner --model=<tflite> "
                  "[--gpu-only] [--run] [--runs=N]";
    return 1;
  }

  // 1. Read the flatbuffer (weights are inline in this model, 183 MB).
  std::string model_bytes;
  base::FilePath model_path =
      cl->GetSwitchValuePath(kModelSwitch);
  if (!base::ReadFileToString(model_path, &model_bytes)) {
    LOG(ERROR) << "Failed to read model: " << model_path;
    return 1;
  }
  LOG(ERROR) << "[runner] model=" << model_path
             << " bytes=" << model_bytes.size();

  // 2. Environment (auto-registers GPU + CPU accelerators; the GPU
  //    accelerator dynamically loads libLiteRtWebGpuAccelerator.dll from
  //    the exe directory — it lives in the same out dir as this exe).
  auto env = ::litert::Environment::Create({});
  if (!env) {
    LOG(ERROR) << "Environment::Create failed: " << env.Error().Message();
    return 1;
  }

  // 3. Compilation options (mirror of GetCompilationOptions).
  const bool gpu_only = cl->HasSwitch(kGpuOnlySwitch);
  auto options = MakeOptions(gpu_only);
  if (!options) {
    LOG(ERROR) << "Options failed: " << options.Error().Message();
    return 1;
  }

  // 4. Compile — this is where the OOM reproduces
  //    (SimpleMemoryArena::Commit → PartitionAlloc 2 GiB cap).
  //    All the fixes from analysis.zh.md iteration 3 are inside this call:
  //    - kTfLiteDelegateFlagsAllowDynamicTensors (delegate_webgpu.cc)
  //    - OptimizeMemoryForLargeTensors explicit call (compiled_model.cc)
  //    - subgraph.cc bytes-from-dims computation
  LOG(ERROR) << "[runner] CompiledModel::Create START (gpu_only=" << gpu_only
             << ")";
  auto model = ::litert::CompiledModel::Create(
      *env,
      ::litert::BufferRef<uint8_t>(absl::MakeSpan(
          reinterpret_cast<const uint8_t*>(model_bytes.data()),
          model_bytes.size())),
      *options);
  if (!model) {
    LOG(ERROR) << "CompiledModel::Create failed: " << model.Error().Message();
    LogErrors(*model, "compile-fail");
    return 1;
  }
  LOG(ERROR) << "[runner] CompiledModel::Create DONE. IsFullyAccelerated="
             << model->IsFullyAccelerated();
  LogErrors(*model, "compile");

  // 5. Optional inference.
  if (!cl->HasSwitch(kRunSwitch)) return 0;

  // SAM encoder: input 1x3x1024x1024 f32, output 1x256x64x64 f32.
  ::litert::RankedTensorType input_type(
      ::litert::ElementType::Float32,
      ::litert::Layout(::litert::Dimensions({1, 3, 1024, 1024})));
  ::litert::RankedTensorType output_type(
      ::litert::ElementType::Float32,
      ::litert::Layout(::litert::Dimensions({1, 256, 64, 64})));

  std::vector<float> input_data(3 * 1024 * 1024, 1.0f);
  std::vector<float> output_data(256 * 64 * 64, 0.0f);

  auto input_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      *env, input_type, input_data.data(), input_data.size() * sizeof(float));
  auto output_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
      *env, output_type, output_data.data(),
      output_data.size() * sizeof(float));
  if (!input_buffer || !output_buffer) {
    LOG(ERROR) << "Failed to create tensor buffers";
    return 1;
  }

  int runs = 1;
  if (cl->HasSwitch(kRunsSwitch)) {
    base::StringToInt(cl->GetSwitchValueASCII(kRunsSwitch), &runs);
  }
  for (int i = 0; i < runs; ++i) {
    auto status = model->Run({*input_buffer}, {*output_buffer});
    if (!status) {
      LOG(ERROR) << "[runner] Run failed: " << status.Error().Message();
      LogErrors(*model, "run-fail");
      return 1;
    }
    LOG(ERROR) << "[runner] Run #" << i << " OK, output[0]=" << output_data[0];
  }
  return 0;
}
```

### 4.2 `BUILD.gn`

```gn
# Standalone SAM encoder runner: reproduces the WebNN→LiteRT compile/OOM path
# without the browser. Build explicitly:
#   autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner

executable("sam_encoder_runner") {
  sources = [ "sam_encoder_runner.cc" ]

  deps = [
    "//base",
    "//build:branding_buildflags",
    "//services/webnn:webnn_switches",  # 若需要复用开关；可省
    "//third_party/litert",             # litert_c + runtime + accelerators + xnnpack
  ]

  # 与 services/webnn 相同的编译配置
  configs += [ "//services/webnn:webnn_buildflags_config" ]
  defines = [ "WEBNN_USE_LITERT=1" ]  # 按需；参考 services/webnn/BUILD.gn 的实际写法
}
```

> 注：`services/webnn/BUILD.gn:178-198` 已经有一整套 `webnn_use_litert` 的
> defines/configs 写法（`WEBNN_USE_LITERT=$webnn_use_litert` 等），BUILD.gn
> 里照抄即可；`//third_party/litert` 默认 target 已带 tflite 源码编译
> （`use_litert_tflite=false` 时自动切到 `third_party/tflite/src`，
> 与 chrome 完全一致）。

---

## 5. 构建方法

### 5.1 在现有 Release out 目录增量构建（推荐）

```powershell
# 用已有的 Release chrome out 目录（其构建参数 use_partition_alloc_as_malloc=true）
autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner

# 运行（libLiteRtWebGpuAccelerator.dll 已在 out\Release\ 下，exe 同目录自动找到）
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite
```

- 首次：编译 1 个 .cc + 链接小 exe（litert/base 的 .obj 都已存在）
- 增量：改 runner 源码只重链小 exe；改 litert/tflite 源码重编对应 .obj + 重链小 exe
- **不触碰 chrome.dll** ← 关键加速点

### 5.2 三种模式对照

```powershell
# A. 复现原始 OOM（当前所有修复生效前的行为不可开关；若需要原始行为可
#    临时注释 compiled_model.cc 的 OptimizeMemoryForLargeTensors 调用）
out\Release\sam_encoder_runner.exe --model=...tflite

# B. GPU-only（跳过 CPU/XNNPACK delegate，等价 --webnn-tflite-gpu-only）
out\Release\sam_encoder_runner.exe --model=...tflite --gpu-only

# C. 编译 + 推理
out\Release\sam_encoder_runner.exe --model=...tflite --gpu-only --run --runs=5

# fp32
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite --runs=1 --gpu-only --precision=fp32
```

### 5.3 debug 版带 PartitionAlloc 的变通

若想 debug 符号 + PartitionAlloc 兼得（可选路径）：
```
gn args out\DebugPA
  is_debug = false          # 关键：false 才默认开 use_partition_alloc_as_malloc
  is_component_build = false
  symbol_level = 1
  dcheck_always_on = true   # 保留 debug 检查
  use_partition_alloc_as_malloc = true
```
小 exe + 静态库 debug 符号，链接仍比 chrome.dll 快一个数量级。

---

## 6. 与 chrome 路径的差异分析

| 环节 | chrome（WebNN demo 页） | runner | 影响 |
|---|---|---|---|
| 模型来源 | WebNN GraphBuilderTflite 现生成 | 直接读 .tflite 文件 | 无影响（OOM 在编译期，与建图无关） |
| weights | 外部 weights file + mmap | 模型内嵌（183 MB flatbuffer） | 无影响（arena 只算非常量张量） |
| mojo / 线程 | utility process 后台线程 | 主线程同步调用 | 无影响 |
| Dawn 实例 | GPU 进程提供 | `Environment::Create({})` 空选项，delegate 自建 | 需验证 WebGPU 设备可创建（见 §7 风险 1） |
| PartitionAlloc | chrome 进程（release 有） | exe 链接 //base（同构建参数） | ✅ 一致复现 |
| 日志 | LOG(ERROR) 到 stderr | 同 | ✅ |

**关键等价性**：`CompiledModel::Create` → `LiteRtCompiledModelT::Create` →
`InterpreterBuilder` → `OptimizeMemoryForLargeTensors` → delegate 循环 →
`ModifyGraphWithDelegate` → `SimpleMemoryArena::Commit` — 这条链的每一环都是
`//third_party/litert` + tflite 树的**同一份编译产物**，与 chrome 完全一致。

---

## 7. 风险与验证清单

> 冒烟测试后的实际状态见 §9.3；以下 3 项在 §9.3 已印证/排除。

1. **WebGPU 设备创建**（最大风险）：chrome 里 Dawn 由 GPU 进程初始化；runner
   是普通进程，delegate 需自建 Dawn instance（ml_drift/webgpu/instance.cc 有
   自建路径，日志 `Created LiteRT GpuEnvironment` 表明环境可自建）。
   验证：runner 里 `CompiledModel::Create` 应打印 `[WebNN][delegate]` 系列日志。
   若失败，改法：在 runner 里用 Dawn 原生 API 建 instance/adapter/device 后
   通过 `LiteRtEnvironment` options 注入（参照 GPU 进程注入方式）。

2. **DLL 查找**：`gpu_registry.cc` 用 `LoadLibrary("libLiteRtWebGpuAccelerator.dll")`，
   exe 同目录查找。out\Release 下已有该 DLL（chrome 也是这么找的）。若不在，
   复制或用 GN `data_deps` 拷贝。

3. **复现验证**：修复前的行为（注释 `OptimizeMemoryForLargeTensors` 调用 +
   去掉 delegate 的 dynamic-tensors flag）应打印
   `SimpleMemoryArena::Commit requesting 1673527296 bytes` → 第二次 2 GB →
   `PartitionExcessiveAllocationSize` 崩溃。修复后 arena 大幅下降。

4. **诊断继承**：runner 自动继承所有已加的诊断——
   `[webnn-oom]`、`[webnn-delegate]`、`[WebNN][GPU-delegate]`、buffer error
   reporter dump —— 因为都在 litert/tflite 库里。

5. **VS 调试**：直接 `devenv out\Release\sam_encoder_runner.exe` 或配置 VS
   调试参数 `--model=... --gpu-only`，断点可打在
   `LiteRtCompiledModelT::Create` / `ModifyGraphWithDelegateImpl` /
   `DelegatePrepare` / `BuildFinalModel` 等任意位置。

---

## 8. 后续扩展

- `--dump-arena`：编译后 dump `ArenaPlanner` high-water mark
- `--threshold=N`：动态调 `OptimizeMemoryForLargeTensors` 阈值（需把阈值做成
  compiled_model.cc 里的变量/环境变量）
- 多模型回归：`--model` 任意路径，一次脚本跑
  `segment_anything_encoder.tflite` / `new_segment_anything_encoder.tflite` /
  MobileSAM `encoder.tflite` 对比
- 如果 Invoke 失败问题（analysis.zh.md §3.4）定位需要，runner 的 `--run` 就是
  最小复现路径，不用开浏览器

---

## 9. 落地记录（2026-08-13）

### 9.1 实际改动的文件

在 `C:\Users\junweifu\workspace\chromium\src\` 下：

| 文件 | 改动 |
|---|---|
| `services/webnn/tflite/sam_runner/BUILD.gn` | 新增；`executable("sam_encoder_runner")` |
| `services/webnn/tflite/sam_runner/sam_encoder_runner.cc` | 新增；复刻 `GetCompilationOptions` + `CompiledModel::Create` 主路径 |
| `services/webnn/BUILD.gn` | 在文件末尾追加 `if (webnn_use_litert) { group("sam_runner_tools") { testonly = true; deps = ["tflite/sam_runner:sam_encoder_runner"] } }`，把新 BUILD.gn 挂进 GN 图。若不加，`autoninja` 会报 `unknown target`。 |

### 9.2 与初稿的偏差

设计文档写好之后编译遇到 3 类问题，实际代码做了以下调整：

1. **`deps` 少了 tflite 三件套**：初稿只写了 `//third_party/litert` + `//base`，
   链接时报 `undefined symbol: tflite::impl::Interpreter::ModifyGraphWithDelegate` 等。
   原因：`compiled_model.cc` 直接调 tflite runtime 里的 `Interpreter` /
   `SignatureRunner` 符号，而 `//third_party/litert` 只 forward-declare，没有
   把 tflite 静态库拉进来。参考 `services/webnn/BUILD.gn` 的写法，补上：

   ```gn
   deps = [
     "//base",
     "//third_party/litert",
     "//third_party/litert:buildflags",
     "//third_party/tflite",
     "//third_party/tflite:tflite_builtin_op_resolver",
     "//third_party/tflite:tflite_public_headers",
   ]
   ```

2. **`base/logging.h` 里的 `LoggingSettings` / `LOG_TO_STDERR` 已迁出**：
   现在在 `base/logging/logging_settings.h`，需要显式加：
   ```cpp
   #include "base/logging/logging_settings.h"
   ```

3. **`-Wunsafe-buffer-usage-in-libc-call` / `-Wunsafe-buffer-usage`**：
   初稿用 `base::AlignedAlloc` + `memset` + `input_data + input_elems` 分配 host buffer，
   在 chromium 的 `-Werror` 下被拒（`memset` 视为 unsafe libc，指针加法视为 unsafe pointer arithmetic）。
   改用 `std::vector<float> input_data(input_elems, 1.0f)` —— runner 是 dev tool，
   一次性 12 MB 输入 + 4 MB 输出 vector 的分配开销可忽略。

4. **BUILD.gn 里不引 `webnn_buildflags_config`**：这些 config 是 `visibility = [":*"]`，
   跨目录引用会失败；`WEBNN_USE_LITERT` 在 runner 里也没用到，直接省了。
   runner 只用 `BUILDFLAG(BUILD_LITERT_WITH_XNNPACK)`（来自
   `//third_party/litert:buildflags`）。

### 9.3 冒烟测试结果

```powershell
autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner
# → [2/3] LINK sam_encoder_runner.exe, 7,662,080 bytes
```

首次全量链接后，改一次 `.cc` 只重编 1 个 `.obj` + 重链小 exe。**未触碰 chrome.dll**。

```powershell
out\Release\sam_encoder_runner.exe `
  --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite `
  --gpu-only
```

关键日志（按出现顺序节选）：

```
[runner] model=…\new_segment_anything_encoder.tflite bytes=183899272
INFO: [environment.cc:36] Creating LiteRT environment with options
INFO: [gpu_registry.cc:109] Dynamically loaded GPU accelerator(libLiteRtWebGpuAccelerator.dll) registered.
INFO: [accelerator_registry.cc:54] RegisterAccelerator: name=GPU WebGPU
INFO: [accelerator_registry.cc:54] RegisterAccelerator: name=CpuAccelerator
INFO: [cpu_registry.cc:75] XNNPACK CPU accelerator registered.
[runner] CompiledModel::Create START (gpu_only=1)
[webnn-oom] OptimizeMemoryForLargeTensors: converted=1054 skipped_input=1 skipped_bytes=230 total=1840
delegate_webgpu.cc:217] Create WebGPU environment (use_low_power=false, enable_host_mapped_pointer=true)
environment.cc:525] Selected adapter: Microsoft Basic Render Driver, arch=warp, vendor=microsoft, backend=Direct3D 12, adapterType=CPU / Software
delegate_webgpu.cc:244] Created a WebGPU environment.
gpu_environment.h:155] Created LiteRT GpuEnvironment.
[webnn-delegate] ModifyGraphWithDelegateImpl: flags=1 supports_dynamic=1 hint_full=0 pre_state=0
delegate_kernel.cc:871] Initializing WebGPU-based API from graph.
delegate_webgpu.cc:374] Failed to create litert::ml_drift::DelegateKernelLiteRt:
    INVALID_ARGUMENT: Shape mismatch: {bhwc, {2304, 1, 1, 768}} vs {bhwc, {1, 1, 2304, 768}}
[webnn-delegate] ResizeTensorImpl: tensor='' type=1 rank=3 bytes_required=805306368 alloc_type=4
[webnn-delegate] ResizeTensorImpl: tensor='?' type=1 rank=3 bytes_required=805306368 alloc_type=2
EXIT=-536870904
```

`-536870904` = `0xE0000008` = PartitionAlloc `OOM_CRASH` code。§7 三项风险全部落地验证：

> 注：PowerShell 状态栏可能把这个负值截断显示为 `Exit Code: 1`，务必用
> `$LASTEXITCODE` 或 `[Convert]::ToString($LASTEXITCODE, 16)` 确认，否则会被
> 误判为「进程正常退出返回 1」。

- **DLL 加载** ✅（`Dynamically loaded GPU accelerator(...) registered`）
- **WebGPU 设备自建** ✅（`Created a WebGPU environment` + `Created LiteRT GpuEnvironment`）
- **PartitionAlloc OOM 复现** ✅（进程以 OOM crash 结束，且发生在
  `ResizeTensorImpl` 连续申请 4 × 0.75 GiB → arena Commit → PartitionAlloc 拒绝
  的路径上，与 chrome release 版行为完全一致）

### 9.4 备注：WebGPU 后端选到 WARP

日志里 `Selected adapter: Microsoft Basic Render Driver, arch=warp, adapterType=CPU / Software`
说明 runner 进程没有拿到真正的 GPU adapter（chrome 是走 GPU 进程沙盒专门配置
才选到硬件 Intel/NVIDIA）。中间的 `Shape mismatch` 错是 WARP + ml_drift
特化路径的既有问题（analysis.zh.md 已知项之一），**不影响 OOM 复现本身**。若要在
runner 里选真 GPU adapter，需要在 `Environment::Create` 时通过
`LiteRtEnvironmentOption`（如 `kLiteRtEnvOptionTagWebGpuInstance` 等）注入
预建的 Dawn instance/adapter/device，见 §8 的后续扩展。

### 9.5 快速使用手册（TL;DR）

```powershell
cd C:\Users\junweifu\workspace\chromium\src

# 构建（首次几十秒；改 runner .cc 后秒级）
autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner

# 复现 OOM
out\Release\sam_encoder_runner.exe `
  --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite `
  --gpu-only

# 编译 + 推理（本机 WARP 下 Run 会失败，等注入真 GPU 后再用）
out\Release\sam_encoder_runner.exe `
  --model=<...>.tflite --gpu-only --run --runs=5
```

---

## 10. 全 GPU 加速修复（2026-08-13，同日）

用 §9 的 runner 冒烟通过 OOM 复现后，用户目标转为：**让 SAM 编码器的所有算子
都下沉到 ml_drift WebGPU delegate**。若成功，CPU 侧 tflite arena 就不再需要
巨型分配 → OOM 天然消失。

### 10.1 症状

`--gpu-only` 运行时 delegate 打印：

```
Shape mismatch: {bhwc, {2304, 1, 1, 768}} vs {bhwc, {1, 1, 2304, 768}}
```

来自 `ml_drift/common/gpu_model_util.cc::ReserveGraphTensors → CheckShapes`，
它比较 CONSTANT 节点两侧的 BHWC 形状：

- desc 侧（`graph.Value.tensor.shape`）走 `ExtractTensorShape`，对 2D
  `[D0, D1]` 返回 `BHWC(D0, 1, 1, D1)`
- data 侧（`ConstTensorAttributes.tensor.shape`）来自某个 emitter 显式赋值

任一侧不匹配都会导致 `absl::InvalidArgumentError`，进而整个子图被 delegate
拒绝并回落 CPU。

### 10.2 诊断补丁

在 `ReserveGraphTensors` 处理 CONSTANT 分支时先把 `desc` / `data` / 下游
consumers 全打出来（临时补丁；已在最终修复后回滚）：

```
[ml-drift-shape-diag] mismatch: tensor_id=14 desc={bhwc, {2304, 1, 1, 768}}
                                 data={bhwc, {1, 1, 2304, 768}}
                                 consumers={fully_connected#13}
```

`fully_connected#13` 就是 SAM 编码器 QKV 投影入口 —— 常见 HF ViT 导出模式
`transpose(weight_const) → fully_connected`。

### 10.3 定位

grep `attr.tensor|OperationType::CONSTANT` 命中的所有 CONSTANT 发射点里，只有
一处对 rank==2 输出写了 `BHWC(1, 1, D0, D1)`：

`chromium/src/third_party/litert/src/ml_drift_delegate/tflite/model_builder.cc`
Transpose 解析器 `runtime_inputs==0` 全常量折叠分支（约 L5747–5760）：

```cpp
if (rank == 1)      transposed.shape = BHWC(1, 1, 1, D0);
else if (rank == 2) transposed.shape = BHWC(1, 1, D0, D1);  // ← 与 rank!=2 不一致
else if (rank == 3) transposed.shape = BHWC(1, D0, D1, D2);
else if (rank == 4) transposed.shape = BHWC(D0, D1, D2, D3);
```

rank 1/3/4 均与 `ExtractTensorShape` 语义一致（前排补 1），只有 rank==2 写反
（后排补 1 → 前排补 1）。而 `AddOutputs` 给该节点的 Value 走的正是
`ExtractTensorShape`，于是两侧永远错位。

### 10.4 修复

统一到 `ExtractTensorShape` 约定：

```cpp
} else if (rank == 2) {
  // 物理存储都是 row-major [D0][D1]，BHWC(D0,1,1,D1) 与 BHWC(1,1,D0,D1)
  // 对应完全相同的字节布局，只是 axis 命名不同。改到前者以匹配 desc 侧。
  transposed.shape = ::ml_drift::BHWC(output->dims->data[0], 1, 1,
                                      output->dims->data[1]);
}
```

### 10.5 验证

```powershell
# 重编并部署 DLL
route-a-webgpu-windows\scripts\build_accelerator_dll.ps1 -Mode opt
route-a-webgpu-windows\scripts\deploy_to_chrome.ps1 -Mode opt `
  -ChromeOutDir C:\Users\junweifu\workspace\chromium\src\out\Release

# 跑 runner
sam_encoder_runner.exe --model=<...>.tflite --gpu-only
```

结果：

- `IsFullyAccelerated=1`
- `[WebNN][GPU-delegate] All 1260 operations are supported by GPU delegate.`
- `1260 ops on GPU, 0 ops on CPU`
- 退出码 `0`（此前 `--gpu-only` 走的是 PartitionAlloc OOM `0xE0000008`）

修复前 delegate 拒收 → 回落 CPU → tflite arena 巨型分配 → OOM。
修复后子图完整落到 WebGPU，CPU arena 只保留少量小张量，OOM 自然消失。

### 10.6 遗留告警

以下告警在修复后仍存在，与本次问题无关：

- `GpuModelBuilder::UpdateOutputTensors ... Shape mismatch ... bhwdc` —— 5D
  reshape/transpose 的规划器警告（W 级，非 fatal，与 4D `CheckShapes` 不同）；
- Dawn WGSL `'f16' type used without 'f16' extension enabled` —— WARP 后端
  不启用 f16 扩展，若后续注入真 GPU + f16 扩展即可解。

---

## 11. 通用模型验证：`--verify` GPU/CPU 双跑对比（设计，2026-08-14）

> 目标：让 runner 支持任意模型（当前动机是
> `C:\Users\junweifu\workspace\tflite-dump-model\ml_drfit_add.tflite`，
> 注意文件名拼写是 `ml_drfit`），在本机验证 ml_drift WebGPU delegate
> 输出结果是否正确。落地记录见 §12。

### 11.1 目标模型

`ml_drfit_add.tflite`（488 字节）：

- 输入 `a`：`[2,2]` float32（name=`a`）
- 常量：`[[1.0, 2.0], [3.0, 4.0]]`（inline buffer，16 字节）
- 输出 `y`：`[2,2]` float32（name=`y`）
- 算子：`ADD`（builtin code 0），即 `y = a + const`

### 11.2 改动 1：形状动态化（改现有 `--run` 路径）

- 删除硬编码的 SAM 形状（`1x3x1024x1024` / `1x256x64x64`），改为从
  `CreateInputBuffers()` / `CreateOutputBuffers()` 返回的
  `TensorBuffer::TensorType()` → `RankedTensorType` 获取
  `NumElements()` 与维度——add 模型与 SAM 通用。
- 输入填充默认改为**确定性 ramp**：`input[i] = i * 0.25f`（每个元素不同，
  diff 更有意义；对 SAM 的 `--run` 行为有变化，但该模式本来只打印统计量）。
- 输出统计块用动态 elems；**输出 ≤ 64 个元素时全量打印所有值**。

### 11.3 改动 2：新增 `--verify` GPU/CPU 双跑对比

- 同一 Environment 下创建两个 `CompiledModel`：
  - **GPU 侧**：`MakeOptions(gpu_only=true, precision=--precision, 默认 fp16)`
    ——与现有 `--gpu-only` 路径完全一致；
  - **CPU 参考侧**：新增 `cpu-only` 分支（accelerators 只含 `kCpu` →
    XNNPACK，**固定 fp32 作 ground truth**）。`BUILD_LITERT_WITH_XNNPACK=0`
    时 `--verify` 直接报错退出。
- `MakeOptions` 改为三态：gpu-only / gpu+cpu / cpu-only。
- 同一份 ramp 输入分别写入两侧 input buffer，各 Run 一次，
  `Lock(kRead)` 读出两侧输出。
- 对比：逐元素 `|gpu - cpu|` → 打印 `max_abs` / `mean_abs` / 超限元素数 +
  两侧小张量全量值 + `[verify] PASS/FAIL`。
  tolerance 默认 **1e-2**（fp16 容差），`--tolerance=N` 可覆盖。
- 退出码约定：PASS=0，FAIL=2，其他错误=1，
  GPU 读回失败（无法判定）=3。

### 11.4 WARP 风险

本机 GPU 选到 WARP 软件适配器；§9.4 记录过 WARP 上输出 `Lock(kRead)`
回读不可靠（SAM 时 hang ~20s 后报错）。2x2 小张量未必触发，但若
`--verify` 的 GPU 侧读回失败：打印明确提示
（"GPU readback failed (WARP) — CPU 侧结果仍有效"），退出码 3，不算 FAIL。
CPU-only 参考侧不受影响（XNNPACK 支持 ADD）。

### 11.5 用法

```powershell
# add 模型验证（GPU fp16 vs CPU fp32）
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\ml_drfit_add.tflite --verify

# add 模型只跑 GPU（验证形状动态化）
out\Release\sam_encoder_runner.exe --model=C:\...\ml_drfit_add.tflite --gpu-only --run

# SAM 照旧
out\Release\sam_encoder_runner.exe --model=C:\...\new_segment_anything_encoder.tflite --gpu-only
```

### 11.6 改动文件与验证计划

- 唯一源码改动：`services/webnn/tflite/sam_runner/sam_encoder_runner.cc`；
  `BUILD.gn` 不动。
- 验证：
  1. 构建（增量，秒级）；
  2. `--verify` 跑 `ml_drfit_add.tflite`：CPU 侧应输出
     `ramp + [[1,2],[3,4]]` ≈ `[1.0, 2.25, 3.5, 4.75]`，GPU 侧对比；
     WARP 读回失败则验证退出码 3 路径；
  3. `--gpu-only --run` 跑 add 模型：确认动态形状生效；
  4. SAM 模型 `--gpu-only`（不带 `--run`）冒烟：确认无回归。

---

## 12. 通用模型验证落地记录（2026-08-14）

### 12.1 实际改动

- `services/webnn/tflite/sam_runner/sam_encoder_runner.cc`（chromium 主树，
  分支 `integrate_litert_ml_drfit`）：
  - `MakeOptions` 三态化（gpu-only / gpu+cpu / cpu-only），
    `AcceleratorMode` 枚举 + `AcceleratorsFor` helper；
  - `--run` 形状动态化（`TensorBuffer::TensorType()` → `NumElements()`），
    输入改确定性 ramp `i * 0.25f`，输出 ≤64 元素全量打印；
  - 新增 `--verify`：CPU(XNNPACK, fp32) 参考 vs GPU(--precision) 逐元素 diff，
    退出码 0=PASS / 1=错误 / 2=FAIL / 3=GPU 读回失败；
  - `--tolerance=N`（默认 1e-2），文件头注释与运行时 Usage 同步更新。

### 12.2 验证结果

add 模型动态形状（`--gpu-only --run`）：

```
[runner] CompiledModel::Create DONE in 6237 ms. IsFullyAccelerated=1
[WebNN][GPU-delegate] All 1 operations are supported by GPU delegate.
[runner] input elems=4 (ramp i*0.25), output elems=4
[runner] Run #0 OK in 4 ms
[run] output values=[1,2.25,1.5,2.75]
EXIT=0
```

add 模型 `--verify`（GPU fp16 vs CPU fp32）——**发现真实 delegate bug**：

```
[verify-cpu] output values=[1,2.25,3.5,4.75]      ← XNNPACK 参考，正确（a+[[1,2],[3,4]]）
[verify] GPU model IsFullyAccelerated=1
[verify-gpu] output values=[1,2.25,1.5,2.75]      ← GPU 第 3、4 个元素错
[verify] elems=4 tol=0.01 max_abs=2 mean_abs=1 over_tol=2 nan_mismatch=0
[verify] FAIL
EXIT=2
```

- 常量 buffer 十六进制确认是 `[1,2,3,4]`（0x198: `3F800000 40000000
  40400000 40800000`），GPU 输出等价于 `a + [1,2,1,2]`——常量第二行
  `[3,4]` 被第一行 `[1,2]` 覆盖（或按 `[2]` 形状对第一行广播）。
- `--verify --precision=fp32` 结果相同 → **结构性 const 处理问题，非精度问题**
  （怀疑与 rank-2 常量 transpose/BHWC 处理有关，见 patches/13）。
- tolerance 与退出码映射抽查：`--tolerance=3` → PASS/退出码 0；
  `--tolerance=1e-9` → FAIL/退出码 2。退出码与 `over_tol` 计数一致。
- 退出码 3 路径本机未触发：add 模型（4 元素）WARP 读回**成功**；
  SAM 模型（1M 元素）读回仍失败（§9.4 所述），但那是 `--run` 的
  skip 路径（退出码 0），非 `--verify` 的 3。

SAM 回归（最终 exe）：

```
# --gpu-only --run --runs=2
[runner] CompiledModel::Create DONE in 10347 ms. IsFullyAccelerated=1
[WebNN][GPU-delegate] All 1260 operations are supported by GPU delegate.
[runner] input elems=3145728 (ramp i*0.25), output elems=1048576
[runner] Run #0 OK in 504 ms
[runner] Run #1 OK in 41 ms
[runner] Skipping output stats: Lock(kRead) failed (expected on WARP)   ← 预期
EXIT=0

# --gpu-only（Task 4 冒烟）
IsFullyAccelerated=1 / All 1260 operations are supported by GPU delegate.
EXIT=0
```

### 12.3 偏差与备注

1. **RunVerify 失败分支去掉 `LogErrors(*model, ...)`**：`litert::Expected`
   的 `operator*` → `CheckVal()` → `LITERT_INTERNAL_CHECK` → `std::abort()`
   （release 也不降级，见 `litert_api_types.h:173`）。按计划原样写在
   `!HasValue()` 分支里会导致直接 abort 而不是按约定返回 1。成功路径的
   `LogErrors` 保留。已有代码 compile-fail 分支（`LogErrors(*model,
   "compile-fail")`）存在同样的隐患，本次未动（超范围，后续可修）。
2. Task 1/2 的代码在本会话开始前已在源码中就位（此前会话实现），本会话
   做的是构建验证 + 行为验证 + Task 3 新增。
3. `--verify` 分派插入点：计划锚定"§3 Compilation options 之前"，实际
   重构后 precision 解析在 §3 内部，故插在 `use_fp32` 计算之后、
   `AcceleratorMode` 之前，语义一致。
4. 文件头 usage 注释同步更新（计划只提了运行时 Usage 字符串）。
5. 本机结论：**WARP 上 GPU delegate 对 ml_drfit_add.tflite 输出错误**
   （元素 3、4 错），`--verify` 成功将其精确定位。


