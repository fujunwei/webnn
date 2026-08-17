# sam_encoder_runner 通用模型验证（--verify）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `sam_encoder_runner` 支持任意形状模型并新增 `--verify` 模式：同一模型在 GPU（ml_drift WebGPU delegate）与 CPU（XNNPACK fp32 参考）各跑一次，逐元素对比验证 GPU delegate 正确性（首个用例：`ml_drfit_add.tflite`，`y = a + [[1,2],[3,4]]`）。

**Architecture:** 单文件改造 `sam_encoder_runner.cc`。`MakeOptions` 重构为三态加速器模式；`--run` 路径改为从 `CreateInputBuffers/CreateOutputBuffers` 返回的 `TensorBuffer::TensorType()` 动态获取形状；新增 `--verify` 走"CPU 参考先编译定形状 → 生成 ramp 输入 → 双跑 → 逐元素 diff"流程。退出码：PASS=0、FAIL=2、错误=1、GPU 读回失败（WARP）=3。

**Tech Stack:** C++20（chromium 风格）、LiteRT C++ API（`third_party/litert`）、GN/autoninja。

**执行环境：** 直接在 chromium 主 checkout（`C:\Users\junweifu\workspace\chromium\src`，分支 `integrate_litert_ml_drfit`）上做，**不用 worktree**——runner 依赖主树里未提交的 litert/ml_drift 修改，worktree 里无法构建。构建目录 `out\Release`。

**规范来源：** `sam_native_runner/DESIGN.md` §11（2026-08-14）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `services/webnn/tflite/sam_runner/sam_encoder_runner.cc` | 唯一源码改动：三态 options、动态形状 IO helpers、`--verify` 流程 |
| `services/webnn/tflite/sam_runner/BUILD.gn` | 不动 |
| `webnn/segment_anythings/sam_native_runner/DESIGN.md` | Task 4 追加 §12 落地记录 |

---

## Task 1: MakeOptions 三态重构（行为等价）

**Files:**
- Modify: `C:\Users\junweifu\workspace\chromium\src\services\webnn\tflite\sam_runner\sam_encoder_runner.cc`

- [ ] **Step 1: 添加 AcceleratorMode 枚举与 AcceleratorsFor helper**

在 `namespace {` 内、`kRunsSwitch` 常量之后（第 53 行附近）插入：

```cpp
// Which accelerators the compiled model may use. Mirrors the combinations
// WebNN offers: GPU-only, GPU with CPU fallback, and CPU-only (the reference
// used by --verify).
enum class AcceleratorMode { kGpuOnly, kGpuAndCpu, kCpuOnly };

// Maps a runner mode to the accelerator bit set passed to LiteRT.
::litert::HwAcceleratorSet AcceleratorsFor(AcceleratorMode mode) {
  switch (mode) {
    case AcceleratorMode::kGpuOnly:
      return ::litert::HwAcceleratorSet(::litert::HwAccelerators::kGpu);
    case AcceleratorMode::kGpuAndCpu: {
      ::litert::HwAcceleratorSet set(::litert::HwAccelerators::kGpu);
#if BUILDFLAG(BUILD_LITERT_WITH_XNNPACK)
      set |= ::litert::HwAccelerators::kCpu;
#endif
      return set;
    }
    case AcceleratorMode::kCpuOnly:
      return ::litert::HwAcceleratorSet(::litert::HwAccelerators::kCpu);
  }
  NOTREACHED();
}
```

需要新增 include（文件顶部 include 区，`base/logging/logging_settings.h` 之后）：

```cpp
#include "base/notreached.h"
```

- [ ] **Step 2: 替换 MakeOptions 为三态版本**

把现有 `MakeOptions(bool gpu_only, bool use_fp32)` 整体替换为：

```cpp
// Mirrors GraphImplLiteRt::GetCompilationOptions (graph_impl_litert.cc).
std::optional<::litert::Options> MakeOptions(AcceleratorMode mode,
                                             bool use_fp32) {
#if !BUILDFLAG(BUILD_LITERT_WITH_XNNPACK)
  if (mode == AcceleratorMode::kCpuOnly) {
    LOG(ERROR) << "CPU-only mode requires BUILD_LITERT_WITH_XNNPACK";
    return std::nullopt;
  }
#endif
  auto options = ::litert::Options::Create();
  if (!options.HasValue()) {
    LOG(ERROR) << "Options::Create failed: " << options.Error().Message();
    return std::nullopt;
  }

  if (mode != AcceleratorMode::kCpuOnly) {
    auto gpu_options = options->GetGpuOptions();
    if (!gpu_options.HasValue()) {
      LOG(ERROR) << "GetGpuOptions failed: " << gpu_options.Error().Message();
      return std::nullopt;
    }
    // Default to fp16 to match chrome release behavior. Override to fp32 for
    // devices without the WebGPU ShaderF16 feature (e.g. the WARP adapter that
    // gets selected on GPU-less CI/RDP machines) where any f16 usage in the
    // generated WGSL (Winograd Bt/At transform buffers, etc.) fails with
    //   "'f16' type used without 'f16' extension enabled".
    gpu_options->SetPrecision(use_fp32 ? ::litert::GpuOptions::Precision::kFp32
                                       : ::litert::GpuOptions::Precision::kFp16);
  }

  auto set_accelerators =
      options->SetHardwareAccelerators(AcceleratorsFor(mode));
  if (!set_accelerators.HasValue()) {
    LOG(ERROR) << "SetHardwareAccelerators failed: "
               << set_accelerators.Error().Message();
    return std::nullopt;
  }

  // Buffer error reporter: captures TF_LITE_KERNEL_LOG and delegate errors
  // so they can be dumped after compilation/inference.
  auto runtime_options = options->GetRuntimeOptions();
  if (runtime_options.HasValue()) {
    runtime_options->SetErrorReporterMode(kLiteRtErrorReporterModeBuffer);
  }

  return std::move(options.Value());
}
```

- [ ] **Step 3: 更新 main 里的调用点**

把 main 中：

```cpp
  // 3. Compilation options.
  const bool gpu_only = cl->HasSwitch(kGpuOnlySwitch);
  const std::string precision = ...（原样保留）;
  const bool use_fp32 = (precision == "fp32");
  std::optional<::litert::Options> options = MakeOptions(gpu_only, use_fp32);
  if (!options) {
    return 1;
  }
```

改为：

```cpp
  // 3. Compilation options.
  const bool gpu_only = cl->HasSwitch(kGpuOnlySwitch);
  const std::string precision = ...（原样保留）;
  const bool use_fp32 = (precision == "fp32");
  const AcceleratorMode mode =
      gpu_only ? AcceleratorMode::kGpuOnly : AcceleratorMode::kGpuAndCpu;
  std::optional<::litert::Options> options = MakeOptions(mode, use_fp32);
  if (!options) {
    return 1;
  }
```

并把第 4 步的日志行改为带 mode 名：

```cpp
  LOG(ERROR) << "[runner] CompiledModel::Create START (mode="
             << (mode == AcceleratorMode::kGpuOnly ? "gpu-only" : "gpu+cpu")
             << ")";
```

（原代码此处的 `gpu_only=` 日志同步替换。）

- [ ] **Step 4: 构建**

Run: `autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner`
Expected: 编译 1 个 .obj + 链接 exe 成功，无警告（chromium -Werror 会拦 unsafe 用法）。

- [ ] **Step 5: SAM 冒烟（行为等价验证）**

Run:
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite --gpu-only
```
Expected: 日志与重构前基线一致（`[runner] CompiledModel::Create START (mode=gpu-only)`、`IsFullyAccelerated=1`、`All 1260 operations are supported by GPU delegate.`、退出码 0）。`$LASTEXITCODE` 应为 0。

- [ ] **Step 6: Commit（可选，默认跳过）**

本树有大量未提交工作，默认**不 commit**；用户要求时：
```bash
git add services/webnn/tflite/sam_runner/sam_encoder_runner.cc
git commit -m "webnn: make sam_encoder_runner accelerator mode a three-way enum"
```

---

## Task 2: 形状动态化 + IO helpers + --run 路径改造

**Files:**
- Modify: `C:\Users\junweifu\workspace\chromium\src\services\webnn\tflite\sam_runner\sam_encoder_runner.cc`

- [ ] **Step 1: 新增 include**

include 区（`<optional>` 之后）补：

```cpp
#include <sstream>
```

（`<algorithm>`、`<cmath>`、`<cstring>`、`<vector>` 已有。）

- [ ] **Step 2: 添加 IO helpers（放在 `LogErrors` 函数之后、`}  // namespace` 之前）**

```cpp
// Shapes of the model's first input/output tensor, queried from the buffers
// LiteRT allocates for the default signature.
struct ModelIOElems {
  bool ok = false;
  std::string error;
  size_t input_elems = 0;
  size_t output_elems = 0;
};

ModelIOElems GetModelIOElems(::litert::CompiledModel& model) {
  ModelIOElems io;
  auto inputs_res = model.CreateInputBuffers();
  auto outputs_res = model.CreateOutputBuffers();
  if (!inputs_res.HasValue() || !outputs_res.HasValue()) {
    io.error = "Failed to create IO buffers: " +
               (inputs_res.HasValue() ? std::string()
                                      : inputs_res.Error().Message()) +
               " / " +
               (outputs_res.HasValue() ? std::string()
                                       : outputs_res.Error().Message());
    return io;
  }
  std::vector<::litert::TensorBuffer> inputs = std::move(inputs_res.Value());
  std::vector<::litert::TensorBuffer> outputs = std::move(outputs_res.Value());

  if (!inputs.empty()) {
    auto type = inputs[0].TensorType();
    if (!type.HasValue()) {
      io.error = "TensorType() failed for input: " + type.Error().Message();
      return io;
    }
    auto elems = type->Layout().NumElements();
    if (!elems.HasValue()) {
      io.error = "NumElements() failed for input: " + elems.Error().Message();
      return io;
    }
    io.input_elems = elems.Value();
  }
  if (!outputs.empty()) {
    auto type = outputs[0].TensorType();
    if (!type.HasValue()) {
      io.error = "TensorType() failed for output: " + type.Error().Message();
      return io;
    }
    auto elems = type->Layout().NumElements();
    if (!elems.HasValue()) {
      io.error = "NumElements() failed for output: " + elems.Error().Message();
      return io;
    }
    io.output_elems = elems.Value();
  }
  io.ok = true;
  return io;
}

// Deterministic test input: every element distinct so a diff test can
// pinpoint which elements a delegate gets wrong.
std::vector<float> GenerateRamp(size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<float>(i) * 0.25f;
  }
  return v;
}

// Input/output buffers for one model, with the input already written.
struct PreparedIO {
  bool ok = false;
  std::string error;
  size_t input_elems = 0;
  size_t output_elems = 0;
  std::vector<::litert::TensorBuffer> inputs;
  std::vector<::litert::TensorBuffer> outputs;
};

PreparedIO PrepareIO(::litert::CompiledModel& model,
                     const std::vector<float>& input_data) {
  PreparedIO io;
  auto inputs_res = model.CreateInputBuffers();
  auto outputs_res = model.CreateOutputBuffers();
  if (!inputs_res.HasValue() || !outputs_res.HasValue()) {
    io.error = "Failed to create IO buffers: " +
               (inputs_res.HasValue() ? std::string()
                                      : inputs_res.Error().Message()) +
               " / " +
               (outputs_res.HasValue() ? std::string()
                                       : outputs_res.Error().Message());
    return io;
  }
  io.inputs = std::move(inputs_res.Value());
  io.outputs = std::move(outputs_res.Value());

  if (!io.inputs.empty()) {
    auto type = io.inputs[0].TensorType();
    if (!type.HasValue()) {
      io.error = "TensorType() failed for input: " + type.Error().Message();
      return io;
    }
    auto elems = type->Layout().NumElements();
    if (!elems.HasValue()) {
      io.error = "NumElements() failed for input: " + elems.Error().Message();
      return io;
    }
    io.input_elems = elems.Value();
    if (io.input_elems != input_data.size()) {
      io.error = "input data size " + std::to_string(input_data.size()) +
                 " != model input elems " + std::to_string(io.input_elems);
      return io;
    }
    auto w = io.inputs[0].Write<float>(
        absl::MakeSpan(input_data.data(), input_data.size()));
    if (!w.HasValue()) {
      io.error = "Failed to write input: " + w.Error().Message();
      return io;
    }
  }
  if (!io.outputs.empty()) {
    auto type = io.outputs[0].TensorType();
    if (!type.HasValue()) {
      io.error = "TensorType() failed for output: " + type.Error().Message();
      return io;
    }
    auto elems = type->Layout().NumElements();
    if (!elems.HasValue()) {
      io.error = "NumElements() failed for output: " + elems.Error().Message();
      return io;
    }
    io.output_elems = elems.Value();
  }
  io.ok = true;
  return io;
}

// Runs the model once. Logs the error (including buffered LiteRT
// diagnostics) and returns false on failure.
bool RunOnce(::litert::CompiledModel& model, PreparedIO& io,
             const char* label) {
  auto status = model.Run(io.inputs, io.outputs);
  if (!status.HasValue()) {
    LOG(ERROR) << "[" << label << "] Run failed: " << status.Error().Message();
    LogErrors(model, label);
    return false;
  }
  return true;
}

// Reads the first output tensor back to host memory. Fails (without crashing)
// when the backend cannot map the buffer — e.g. WARP software WebGPU.
std::optional<std::vector<float>> ReadOutput(PreparedIO& io,
                                             std::string* error) {
  if (io.outputs.empty() || io.output_elems == 0) {
    *error = "model has no outputs";
    return std::nullopt;
  }
  std::vector<float> output(io.output_elems, 0.0f);
  auto locked = io.outputs[0].Lock(::litert::TensorBuffer::LockMode::kRead);
  if (!locked.HasValue()) {
    *error = locked.Error().Message();
    return std::nullopt;
  }
  UNSAFE_BUFFERS(std::memcpy(output.data(), locked.Value(),
                             io.output_elems * sizeof(float)));
  io.outputs[0].Unlock();
  return output;
}

// Prints min/max/mean/std plus (for small tensors) every element value.
constexpr size_t kMaxDumpElems = 64;

void PrintOutputStats(const char* label, const std::vector<float>& data,
                      size_t elems) {
  if (elems == 0) {
    LOG(ERROR) << "[" << label << "] output: elems=0";
    return;
  }
  double sum = 0.0, sumsq = 0.0;
  float mn = data[0], mx = data[0];
  size_t nan_count = 0;
  for (size_t i = 0; i < elems; ++i) {
    float v = data[i];
    if (std::isnan(v)) {
      ++nan_count;
      continue;
    }
    sum += v;
    sumsq += static_cast<double>(v) * v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  double mean = sum / elems;
  double var = sumsq / elems - mean * mean;
  LOG(ERROR) << "[" << label << "] output: elems=" << elems
             << " nan=" << nan_count << " min=" << mn << " max=" << mx
             << " mean=" << mean << " std=" << std::sqrt(std::max(0.0, var));
  if (elems <= kMaxDumpElems) {
    std::ostringstream oss;
    for (size_t i = 0; i < elems; ++i) {
      if (i) oss << ",";
      oss << data[i];
    }
    LOG(ERROR) << "[" << label << "] output values=[" << oss.str() << "]";
  }
}
```

- [ ] **Step 3: 替换 main 的推理段（原第 5 段，从 `// 5. Inference.` 到 return 0 之前）**

删除原硬编码 SAM 形状的推理段（`CreateInputBuffers` 之后的全部旧代码，含 `input_elems = 3u * 1024u * 1024u`、循环、旧 stats 块），替换为：

```cpp
  // 5. Inference with dynamic shapes (any model: SAM encoder, tiny add, ...).
  ModelIOElems io_elems = GetModelIOElems(*model);
  if (!io_elems.ok) {
    LOG(ERROR) << "[runner] " << io_elems.error;
    return 1;
  }
  std::vector<float> input_data = GenerateRamp(io_elems.input_elems);
  LOG(ERROR) << "[runner] input elems=" << io_elems.input_elems
             << " (ramp i*0.25), output elems=" << io_elems.output_elems;
  PreparedIO io = PrepareIO(*model, input_data);
  if (!io.ok) {
    LOG(ERROR) << "[runner] " << io.error;
    return 1;
  }

  int runs = 1;
  if (cl->HasSwitch(kRunsSwitch)) {
    base::StringToInt(cl->GetSwitchValueASCII(kRunsSwitch), &runs);
  }
  for (int i = 0; i < runs; ++i) {
    base::TimeTicks run_start = base::TimeTicks::Now();
    if (!RunOnce(*model, io, "run")) {
      break;  // RunOnce already logged the failure.
    }
    LOG(ERROR) << "[runner] Run #" << i << " OK in "
               << (base::TimeTicks::Now() - run_start).InMilliseconds()
               << " ms";
  }

  // Attempt readback. Non-fatal: on WARP this hangs ~20 s and then returns an
  // error (Dawn mapAsync on software adapter). Skip the failure so users see
  // it isn't a correctness signal.
  std::string read_error;
  std::optional<std::vector<float>> output = ReadOutput(io, &read_error);
  if (!output) {
    LOG(ERROR) << "[runner] Skipping output stats: Lock(kRead) failed "
                  "(expected on WARP): "
               << read_error;
    return 0;
  }
  PrintOutputStats("run", *output, io.output_elems);
  return 0;
```

- [ ] **Step 4: 构建**

Run: `autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner`
Expected: 编译链接成功。

- [ ] **Step 5: add 模型验证动态形状**

Run:
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\ml_drfit_add.tflite --gpu-only --run
```
Expected（关键行）：
```
[runner] input elems=4 (ramp i*0.25), output elems=4
```
以及后续 `[run] Run #0 OK ...`。输出读回两态之一：
- WARP 读回失败 → `[runner] Skipping output stats: Lock(kRead) failed (expected on WARP)`，退出码 0；
- 读回成功 → `[run] output values=[...]` 4 个值。

- [ ] **Step 6: SAM 回归（--run 不破坏 benchmark）**

Run:
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite --gpu-only --run --runs=2
```
Expected: `[runner] input elems=3145728 ...`、`Run #0 OK`、`Run #1 OK`（读回在 WARP 上失败属正常，退出码 0）。

- [ ] **Step 7: Commit（可选，默认跳过）**

```bash
git add services/webnn/tflite/sam_runner/sam_encoder_runner.cc
git commit -m "webnn: generalize sam_encoder_runner inference to dynamic shapes"
```

---

## Task 3: --verify GPU/CPU 双跑对比

**Files:**
- Modify: `C:\Users\junweifu\workspace\chromium\src\services\webnn\tflite\sam_runner\sam_encoder_runner.cc`

- [ ] **Step 1: 新增开关常量**

`kPrecisionSwitch` 之后加：

```cpp
constexpr char kVerifySwitch[] = "verify";
constexpr char kToleranceSwitch[] = "tolerance";  // max |gpu-cpu| per element
```

- [ ] **Step 2: 添加 RunVerify 函数（放在 `}  // namespace` 之前、PrintOutputStats 之后）**

```cpp
// Runs the model twice: once on GPU (--precision, default fp16) and once on
// CPU/XNNPACK in fp32 as ground truth, then compares element-wise.
// Returns the process exit code:
//   0 = PASS, 1 = setup/run error, 2 = FAIL, 3 = GPU readback impossible.
int RunVerify(::litert::Environment& env, const std::string& model_bytes,
              bool use_fp32, double tolerance) {
  ::litert::BufferRef<uint8_t> model_ref(absl::MakeSpan(
      reinterpret_cast<const uint8_t*>(model_bytes.data()),
      model_bytes.size()));

  // CPU reference first: its input shape defines the ramp.
  LOG(ERROR) << "[verify] compiling CPU reference (fp32)...";
  std::optional<::litert::Options> cpu_options =
      MakeOptions(AcceleratorMode::kCpuOnly, /*use_fp32=*/true);
  if (!cpu_options) {
    return 1;
  }
  auto cpu_model =
      ::litert::CompiledModel::Create(env, model_ref, *cpu_options);
  if (!cpu_model.HasValue()) {
    LOG(ERROR) << "[verify] CPU compile failed: "
               << cpu_model.Error().Message();
    LogErrors(*cpu_model, "verify-cpu-compile");
    return 1;
  }
  LogErrors(*cpu_model, "verify-cpu-compile");

  ModelIOElems io_elems = GetModelIOElems(*cpu_model);
  if (!io_elems.ok) {
    LOG(ERROR) << "[verify] " << io_elems.error;
    return 1;
  }
  std::vector<float> input_data = GenerateRamp(io_elems.input_elems);
  LOG(ERROR) << "[verify] input elems=" << io_elems.input_elems
             << " (ramp i*0.25)";

  PreparedIO cpu_io = PrepareIO(*cpu_model, input_data);
  if (!cpu_io.ok) {
    LOG(ERROR) << "[verify] CPU prepare failed: " << cpu_io.error;
    return 1;
  }
  if (!RunOnce(*cpu_model, cpu_io, "verify-cpu")) {
    return 1;
  }
  std::string cpu_read_error;
  std::optional<std::vector<float>> cpu_out =
      ReadOutput(cpu_io, &cpu_read_error);
  if (!cpu_out) {
    LOG(ERROR) << "[verify] CPU readback failed: " << cpu_read_error;
    return 1;
  }
  PrintOutputStats("verify-cpu", *cpu_out, io_elems.output_elems);

  // GPU side.
  LOG(ERROR) << "[verify] compiling GPU model ("
             << (use_fp32 ? "fp32" : "fp16") << ")...";
  std::optional<::litert::Options> gpu_options =
      MakeOptions(AcceleratorMode::kGpuOnly, use_fp32);
  if (!gpu_options) {
    return 1;
  }
  auto gpu_model =
      ::litert::CompiledModel::Create(env, model_ref, *gpu_options);
  if (!gpu_model.HasValue()) {
    LOG(ERROR) << "[verify] GPU compile failed: "
               << gpu_model.Error().Message();
    LogErrors(*gpu_model, "verify-gpu-compile");
    return 1;
  }
  auto fully_accelerated = gpu_model->IsFullyAccelerated();
  LOG(ERROR) << "[verify] GPU model IsFullyAccelerated="
             << (fully_accelerated.HasValue() && fully_accelerated.Value());
  LogErrors(*gpu_model, "verify-gpu-compile");

  ModelIOElems gpu_elems = GetModelIOElems(*gpu_model);
  if (!gpu_elems.ok) {
    LOG(ERROR) << "[verify] " << gpu_elems.error;
    return 1;
  }
  if (gpu_elems.input_elems != io_elems.input_elems ||
      gpu_elems.output_elems != io_elems.output_elems) {
    LOG(ERROR) << "[verify] GPU/CPU IO shapes differ: in "
               << gpu_elems.input_elems << "/" << io_elems.input_elems
               << " out " << gpu_elems.output_elems << "/"
               << io_elems.output_elems;
    return 1;
  }

  PreparedIO gpu_io = PrepareIO(*gpu_model, input_data);
  if (!gpu_io.ok) {
    LOG(ERROR) << "[verify] GPU prepare failed: " << gpu_io.error;
    return 1;
  }
  if (!RunOnce(*gpu_model, gpu_io, "verify-gpu")) {
    return 1;
  }
  std::string gpu_read_error;
  std::optional<std::vector<float>> gpu_out =
      ReadOutput(gpu_io, &gpu_read_error);
  if (!gpu_out) {
    LOG(ERROR) << "[verify] GPU readback failed (expected on WARP): "
               << gpu_read_error;
    LOG(ERROR) << "[verify] cannot compare; CPU reference result above is "
                  "valid. Exit code 3.";
    return 3;
  }
  PrintOutputStats("verify-gpu", *gpu_out, io_elems.output_elems);

  // Element-wise diff.
  size_t n = io_elems.output_elems;
  double max_abs = 0.0, sum_abs = 0.0;
  size_t over_tol = 0, nan_mismatch = 0;
  for (size_t i = 0; i < n; ++i) {
    float g = (*gpu_out)[i], c = (*cpu_out)[i];
    if (std::isnan(g) || std::isnan(c)) {
      ++nan_mismatch;
      continue;
    }
    double d = std::fabs(static_cast<double>(g) - c);
    sum_abs += d;
    if (d > max_abs) max_abs = d;
    if (d > tolerance) ++over_tol;
  }
  bool pass = (over_tol == 0 && nan_mismatch == 0);
  LOG(ERROR) << "[verify] elems=" << n << " tol=" << tolerance
             << " max_abs=" << max_abs
             << " mean_abs=" << (n ? sum_abs / n : 0.0)
             << " over_tol=" << over_tol << " nan_mismatch=" << nan_mismatch;
  LOG(ERROR) << "[verify] " << (pass ? "PASS" : "FAIL");
  return pass ? 0 : 2;
}
```

- [ ] **Step 3: main 分派 --verify**

在 main 中 precision 解析（`const bool use_fp32 = ...`）之后、`// 3. Compilation options.` 之前插入：

```cpp
  if (cl->HasSwitch(kVerifySwitch)) {
    double tolerance = 1e-2;
    if (cl->HasSwitch(kToleranceSwitch)) {
      if (!base::StringToDouble(cl->GetSwitchValueASCII(kToleranceSwitch),
                                &tolerance)) {
        LOG(ERROR) << "--tolerance must be a number";
        return 1;
      }
    }
    return RunVerify(*env, model_bytes, use_fp32, tolerance);
  }
```

同时更新 Usage 字符串（`!cl->HasSwitch(kModelSwitch)` 分支）：

```cpp
    LOG(ERROR) << "Usage: sam_encoder_runner --model=<tflite> [--gpu-only] "
                  "[--run] [--runs=N] [--precision=fp16|fp32] | --verify "
                  "[--precision=fp16|fp32] [--tolerance=N]";
```

- [ ] **Step 4: 构建**

Run: `autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner`
Expected: 编译链接成功。

- [ ] **Step 5: add 模型 --verify**

Run:
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\ml_drfit_add.tflite --verify
```
Expected（关键行）：
```
[verify] input elems=4 (ramp i*0.25)
[verify-cpu] output values=[1,2.25,3.5,4.75]     ← 输入 [0,0.25,0.5,0.75] + [[1,2],[3,4]]
[verify] GPU model IsFullyAccelerated=1
[verify-gpu] output values=[...]                  ← 与 CPU 一致（WARP 读回成功时）
[verify] elems=4 tol=0.01 max_abs=... over_tol=0 nan_mismatch=0
[verify] PASS
```
退出码 0（`$LASTEXITCODE` 检查）。若本机 WARP 读回失败：`[verify] GPU readback failed (expected on WARP)` + 退出码 3 —— 属预期路径，记录即可（真 GPU 注入后完整验证，见 DESIGN.md §8/§11.4）。

- [ ] **Step 6: fp32 变体与 FAIL 路径抽查**

Run:
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\ml_drfit_add.tflite --verify --precision=fp32
```
Expected: 若读回可用 → `[verify] PASS`（fp32 vs fp32 应精确）。

再抽查 FAIL 路径（故意放极小 tolerance）：
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\ml_drfit_add.tflite --verify --tolerance=1e-9
```
Expected: 若 GPU fp16 有任何量化误差 → `[verify] FAIL`、退出码 2；若两侧 fp16/fp32 恰好精确相等 → PASS 也算正常（说明 diff 逻辑没把 NaN 当 pass）。两者都可接受，重点是确认退出码与 `over_tol` 计数一致。

- [ ] **Step 7: Commit（可选，默认跳过）**

```bash
git add services/webnn/tflite/sam_runner/sam_encoder_runner.cc
git commit -m "webnn: add --verify GPU-vs-CPU output comparison to sam_encoder_runner"
```

---

## Task 4: 回归冒烟 + DESIGN.md §12 落地记录

**Files:**
- Modify: `C:\Users\junweifu\workspace\webnn\segment_anythings\sam_native_runner\DESIGN.md`

- [ ] **Step 1: SAM 全链路回归**

Run:
```powershell
out\Release\sam_encoder_runner.exe --model=C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite --gpu-only
```
Expected: `IsFullyAccelerated=1`、`All 1260 operations are supported by GPU delegate.`、退出码 0（与 §10.5 基线一致）。

- [ ] **Step 2: 追加 DESIGN.md §12 落地记录**

在 DESIGN.md 末尾追加一节，内容包含（用实际运行日志填充）：

```markdown
---

## 12. 通用模型验证落地记录（2026-08-14）

### 12.1 实际改动

- `services/webnn/tflite/sam_runner/sam_encoder_runner.cc`：
  - `MakeOptions` 三态化（gpu-only / gpu+cpu / cpu-only）；
  - `--run` 形状动态化（`TensorBuffer::TensorType()` → `NumElements()`），
    输入改确定性 ramp `i * 0.25f`，输出 ≤64 元素全量打印；
  - 新增 `--verify`：CPU(XNNPACK, fp32) 参考 vs GPU(--precision) 逐元素 diff，
    退出码 0=PASS / 1=错误 / 2=FAIL / 3=GPU 读回失败。

### 12.2 验证结果

[粘贴 add 模型 --verify 的关键日志与退出码；WARP 读回失败则记录退出码 3 路径]

### 12.3 偏差与备注

[构建/运行中与 §11 设计的偏差，如有]
```

- [ ] **Step 3: （可选）提交 DESIGN.md**

`webnn` 仓库中 `sam_native_runner/` 从未被跟踪，提交与否由用户决定：
```bash
cd C:\Users\junweifu\workspace\webnn
git add segment_anythings/sam_native_runner/DESIGN.md
git commit -m "sam_native_runner: add generic model verification design and record"
```

---

## 自审记录

- **Spec 覆盖**：§11.2 形状动态化 → Task 2；§11.3 `--verify`（三态 MakeOptions、fp32 CPU 参考、tolerance、退出码 0/1/2/3）→ Task 1 + Task 3；§11.4 WARP 读回失败提示与退出码 3 → Task 3 Step 5 的 RunVerify 读回分支；§11.5 用法 → Task 3 Step 3 的 Usage 字符串；§11.6 验证计划 4 条 → Task 2 Step 5/6、Task 3 Step 5/6、Task 4 Step 1。
- **占位符**：无 TBD/TODO；所有代码步骤含完整代码。
- **类型一致性**：`AcceleratorMode`/`ModelIOElems`/`PreparedIO`/`RunOnce`/`ReadOutput`/`PrintOutputStats`/`RunVerify` 各任务间签名一致；`MakeOptions(AcceleratorMode, bool)` 在 Task 1 定义后、Task 3 按此调用。
