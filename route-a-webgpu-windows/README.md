# Route A · WebGPU (Windows)：从 litert 编出 `LiteRtWebGpuAccelerator.dll` 并让 Chromium WebNN 用上 GPU delegate

> 本文是**使用步骤**（装依赖 → 打 patch → 编 DLL → 部署 → 跑测试）。原理 / 架构 / 各 patch 意图 / 踩坑 / Chromium 升级清单见 **[DESIGN.md](DESIGN.md)**。
>
> 本 bundle 覆盖 Linux 方案 `../route-a-webgpu/` 的 Windows 移植。

---

## 环境要求

| 组件 | 版本 | 用途 |
|---|---|---|
| Chromium 检出 + 编好一次 Release | 147+ | `chrome.exe`、libc++ `.obj`、clang-cl、lld-link |
| VS 2022（或 18） | 含 C++ 桌面工作负载 | MSVC 链接库（msvcprt.lib 等）+ `link.exe` |
| Bazelisk / Bazel | 7.7+ | litert 构建 |
| CMake | 3.20+ | Dawn 构建 |
| Python | 3.9+ | Bazel、Dawn 需 |
| Git | 任何近期版本 | 打 patch |

- 磁盘：Bazel output_base 约 20 GB，Dawn 构建目录约 5 GB。
- CPU：默认 `-march=sierraforest`（Sierra Forest 无 AVX-512）。换机按 CPU 改 `.bazelrc.user`，映射见 [DESIGN.md §2](DESIGN.md)。

---

## 1. 一次性 Setup

换机必改路径。示例（本机）：

```powershell
$CR      = "C:\Users\junweifu\workspace\chromium\src"
$WEBNN   = "C:\Users\junweifu\workspace\webnn"                       # 本 bundle 的父目录
$LITERT  = "$CR\third_party\litert\src"
$MLDRIFT = "$CR\third_party\ml-drift"
$BUNDLE  = "$WEBNN\route-a-webgpu-windows"
```

### 1.1 打包 Chromium libc++ 为静态库

Chromium `.obj` 是 LLVM bitcode，只能用 `lld-link.exe /lib` 打包。

```powershell
& "$BUNDLE\scripts\build_libcxx_lib.ps1" -ChromiumSrc $CR -Dest "$WEBNN\_cr_libcxx_link_win"
# 期望：约 3-4 MB 的 libc++.lib，看到 "__Cr@std string occurrences: >0"
```

### 1.2 编译 Dawn（webgpu_dawn.dll）

必须匹配本机 Chromium 版本号。

```powershell
& "$BUNDLE\scripts\build_dawn.ps1" `
    -SrcDir           "$WEBNN\_webgpu_dawn_src" `
    -Dest             "$WEBNN\_dawn_prebuilt_win" `
    -ChromiumVersion  "154.0.8017.0"
# 期望：_dawn_prebuilt_win\{include,lib}\，其中 lib\webgpu_dawn.dll ~10 MB。耗时 30-60 分钟。
```

### 1.3 打 patch

**litert 仓**（`$LITERT`）：

```powershell
Push-Location $LITERT
git apply "$BUNDLE\patches\00-bazelrc-user-crcxx-win.patch"                    # 生成 .bazelrc.user
git apply "$BUNDLE\patches\01-dawn-workspace-prebuilt-win.patch"               # @dawn -> 本地预编译
git apply "$BUNDLE\patches\02-workspace-local-ml-drift.patch"                  # ml_drift -> local_repository
git apply "$BUNDLE\patches\03-delegate-BUILD-weight_loader.patch"              # copybara 未替换的 label
git apply "$BUNDLE\patches\04-shared_memory_manager-BUILD-weight_loader.patch"
git apply "$BUNDLE\patches\05-delegate_webgpu-farmhash-and-pipeline-cache.patch"
git apply "$BUNDLE\patches\06-compiled-model-disable-optimize-memory.patch"
Pop-Location
```

> 若某个 patch `git apply` 失败（上游代码变了），按 [DESIGN.md §6](DESIGN.md) 的意图手动改。

**ml-drift 仓**（`$MLDRIFT`）：

```powershell
Push-Location $MLDRIFT
git apply "$BUNDLE\patches\07-ml-drift-gpu-info-uint32max.patch"               # narrowing conversion 修复
Pop-Location
```

**Chromium 仓**（`$CR`）：

```powershell
Push-Location $CR
git apply "$BUNDLE\patches\08-webnn-sandbox-init-full-dll-path.patch"        # GPU 进程 pre-sandbox 用绝对路径 LoadLibrary（安装版必需）
git apply "$BUNDLE\patches\09-chrome-release-webnn-dlls.patch"                # 让 mini_installer 打包 webgpu_dawn.dll
Pop-Location
```

### 1.4 编辑 `.bazelrc.user` 的绝对路径

patch 00 里的绝对路径是本机的，换机必改：

```powershell
# 编辑 $LITERT\.bazelrc.user，替换：
#   C:/Users/junweifu/workspace/chromium/src  -> $CR
#   C:/Users/junweifu/workspace/webnn         -> $WEBNN
#   -march=sierraforest                       -> 本机 CPU 的合适值（映射见 DESIGN.md）
```

---

## 2. 编译加速器 DLL

**首次编译**（只为了生成工具链配置），预期会因 hermetic include 检查失败：

```powershell
& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode opt -ChromiumSrc $CR -WebnnDir $WEBNN -MlDrift $MLDRIFT
```

**打 Bazel 生成的 toolchain BUILD**（一次性；`bazel clean --expunge` 后要重跑）：

```powershell
python "$BUNDLE\scripts\patch_bazel_toolchain.py"
```

**再编一次**：

```powershell
& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode opt -ChromiumSrc $CR -WebnnDir $WEBNN -MlDrift $MLDRIFT
# 期望：$LITERT\bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.dll
```

**验证 ABI**（必须用 Chromium libc++，不能用 MSVC STL）：

```powershell
$dll = "$LITERT\bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.dll"
$text = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($dll))
"__Cr@std matches: {0}" -f ($text -split "`0" | Where-Object { $_ -match "__Cr@std" }).Count   # 期望 > 100
"__1@std  matches: {0}" -f ($text -split "`0" | Where-Object { $_ -match "__1@std"  }).Count   # 期望 = 0
```

---

## 3. 部署 + 运行测试

> ⚠️ **必须加 `--no-sandbox`**。`PreSandboxWebNNInitialization()` 只在 GPU 进程 sandbox lockdown **之前** 预加载 accelerator DLL；之后建 device / 开 shader cache / 分配 GPU 资源仍需要 sandbox 外权限。不加时的典型症状：GPU 进程崩溃 / `chrome://gpu` 里 WebNN 显示 CPU / stderr 看不到 `... registered.`。

### 3.1 dev 目录直接跑（推荐迭代方式）

```powershell
& "$BUNDLE\scripts\deploy_to_chrome.ps1" -ChromeOutDir "$CR\out\Release" -Mode opt -ChromiumSrc $CR -WebnnDir $WEBNN

& "$BUNDLE\scripts\deploy_to_chrome.ps1" -ChromeOutDir "$CR\out\upstream_bots_debug" -Mode dbg -ChromiumSrc $CR -WebnnDir $WEBNN

& "$CR\out\Release\chrome.exe" `
    --no-sandbox --enable-features=WebMachineLearningNeuralNetwork `
    "file:///$WEBNN/route-a-webgpu/webnn_gpu_dispatch_test.html"
```

### 3.2 mini_installer 部署到测试机

```powershell
# 0) 前置：patch 09 已 apply；out\Release 是 is_official_build=true；加速器/Dawn DLL 已就绪。
# 1) 若之前跑过 mini_installer，先清老 staging（否则会打进老 DLL，详见 DESIGN.md §4 第 9 条）：
$Out = "$CR\out\Release"
Remove-Item -Force -ErrorAction SilentlyContinue `
    "$Out\chrome.7z", "$Out\chrome.packed.7z", "$Out\mini_installer.exe", `
    "$Out\gen\chrome\installer\mini_installer\archive.d"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
    "$Out\gen\chrome\installer\mini_installer\mini_installer\Chrome-bin"

# 2) stage DLL + 打包（脚本内部做 Rename-Item; Copy-Item 覆盖 + ninja mini_installer）
& "$BUNDLE\scripts\build_mini_installer.ps1" -ChromeOutDir "$Out" -ChromiumSrc $CR -WebnnDir $WEBNN

# 3) 自检 chrome.7z 里有两个自制 DLL
tar -tvf "$Out\chrome.7z" | Select-String "libLiteRt|webgpu_dawn"

# 4) 传 $Out\mini_installer.exe 到目标机，双击安装（默认装到 %LOCALAPPDATA%\Chromium）
# 5) 目标机上运行
& "$env:LOCALAPPDATA\Chromium\Application\chrome.exe" `
    --no-sandbox --enable-features=WebMachineLearningNeuralNetwork `
    "file:///<路径>/webnn_gpu_dispatch_test.html"
```

### 3.3 判读

- Chrome 日志（`--enable-logging=stderr --v=1` 时）：`Attempting to load GPU accelerator(LiteRtWebGpuAccelerator.dll).` + `... registered.`
- 页面上 `[WEBNN-DISP]` 输出 `readTensor result=[11,22,33,44]` → **RESULT: INFERENCE OK**
- 关键：**不再有** `std::length_error` 在 `delegate_kernel.cc:226`

---

## 4. 参考

- 原理 / 架构 / 核心难点与解决方案：**[DESIGN.md §1-§2](DESIGN.md)**
- 目录结构与各 patch 说明：**[DESIGN.md §3](DESIGN.md)**
- 已知坑与调试线索：**[DESIGN.md §4](DESIGN.md)**
- 复用 Linux bundle：**[DESIGN.md §5](DESIGN.md)**
- Chromium 版本升级 checklist：**[DESIGN.md §6](DESIGN.md)**
