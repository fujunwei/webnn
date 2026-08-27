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
$CR      = "C:\Users\fujun\workspace\chromium\src"
$WEBNN   = "C:\Users\fujun\workspace\webnn"                       # 本 bundle 的父目录
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
#   C:/Users/fujun/workspace/chromium/src  -> $CR
#   C:/Users/fujun/workspace/webnn         -> $WEBNN
#   -march=sierraforest                       -> 本机 CPU 的合适值
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
& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode dbg

& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode opt
# 期望产出：$LITERT\bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.dll (26 MB dbg / 4 MB opt)
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

> ⚠️ **必须在物理机本地控制台（`console` 会话）跑，不能走 RDP 远程桌面** — WebNN GPU 底层链是 ml_drift WebGPU delegate → Dawn → D3D12。
> - RDP 会话里渲染走的是 `Microsoft Remote Display Adapter`（软件间接显示适配器），真实 GPU（如 Intel UHD 630）**不参与**，Dawn/D3D12 无法建 device；
> - 典型症状：stderr 出现 `D3D12CreateDevice failed with DXGI_ERROR_DRIVER_INTERNAL_ERROR (0x887A0020)`，以及 `eglCreateContext: Requested GLES version (3.1) is greater than max supported (3.0)`；页面/`chrome://gpu` 里 WebNN 退化为 CPU；
> - 自查：`query session` 当前会话应为 `console`（Active），而不是 `rdp-tcp#N`；
> - 本地登录后建议先更新 GPU 驱动（Intel 老驱动 27.20.x 对 Dawn/D3D12 兼容较差）。

### 5.1 dev 目录直接跑（推荐迭代方式）

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

```
route-a-webgpu-windows/
├── README.md                                          # 本文
├── patches/
│   ├── 00-bazelrc-user-crcxx-win.patch                # 应用到 litert 仓：.bazelrc.user (crcxx_win 配置)
│   ├── 01-dawn-workspace-prebuilt-win.patch           # 应用到 litert 仓：@dawn -> 本地 webgpu_dawn.dll
│   ├── 02-workspace-local-ml-drift.patch              # 应用到 litert 仓：@ml_drift -> local_repository
│   ├── 03-delegate-BUILD-weight_loader.patch          # 应用到 litert 仓：copybara label 修复
│   ├── 04-shared_memory_manager-BUILD-weight_loader.patch  # 应用到 litert 仓
│   ├── 05-delegate_webgpu-farmhash-and-pipeline-cache.patch # 应用到 litert 仓
│   ├── 06-compiled-model-disable-optimize-memory.patch # 应用到 litert 仓：关掉 GPU 不支持的动态张量优化
│   ├── 07-ml-drift-gpu-info-uint32max.patch           # 应用到 ml-drift 仓：narrowing conversion 修复
│   ├── 08-webnn-sandbox-init-full-dll-path.patch      # 应用到 Chromium 仓：GPU 进程 pre-sandbox 用绝对路径 LoadLibrary
│   └── 09-chrome-release-webnn-dlls.patch             # 应用到 Chromium 仓：mini_installer 打包 webgpu_dawn.dll
└── scripts/
    ├── build_libcxx_lib.ps1        # 3.1: 打包 libc++.lib
    ├── build_dawn.ps1              # 3.2: 编 webgpu_dawn.dll
    ├── patch_bazel_toolchain.py    # 4:   打 Bazel 自动生成的 toolchain BUILD
    ├── build_accelerator_dll.ps1   # 4:   bazel build ...
    ├── deploy_to_chrome.ps1        # 5.1: 拷 DLL 到 chrome.exe 目录（dev 迭代用）
    └── build_mini_installer.ps1    # 5.2: stage DLL + 编 mini_installer.exe（部署到测试机用）
```

---

## 7. 已知坑与调试线索

1. **`bazel clean --expunge` 后必须重跑 `patch_bazel_toolchain.py`** — 因为 `local_config_cc/BUILD` 是 Bazel 自动生成的，`--expunge` 会重生成，之前的注入丢失。
2. **XNNPACK `.S` 文件报错 `A2044` (MASM invalid character)** — 说明 `--compiler=clang-cl` 被误设了，导致 XNNPACK 走 clang 分支引入 GAS-语法 `.S`，被 `ml64.exe` 拒绝。删掉这个 flag（保留 `USE_CLANG_CL=1` 让工具是 clang-cl，但 select 走 msvc）。
3. **运行 `flatc.exe` 报 `STATUS_ILLEGAL_INSTRUCTION`（`0xC000001D`）** — 说明 `-march=<x>` 高于本机 CPU 支持的指令集。降级 `-march`（例如从 `sapphirerapids` 改到 `sierraforest` 或 `haswell`）。
4. **XNNPACK 某个 `avx256skx` / `avx256vnni` 文件报 `avx512vl` 未启用** — patch 00 中的 `--per_file_copt` 正则覆盖不到该文件。扩展正则或新增一行。
5. **链接期 `__ExceptionPtr*` 未定义** — `msvcprt.lib` 没链上；检查 `.bazelrc.user` 里 `--linkopt=msvcprt.lib`。
6. **`std::cout / std::cerr` 报 `__declspec(dllimport)` 未定义** — `-D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS` 没生效；查 `--cxxopt`。
7. **Chrome 加载 DLL 时挂在 UI 卡顿** — 部署时旧 DLL 被 Chrome 进程持有。用 `deploy_to_chrome.ps1` 的做法：先 `Rename-Item` 再 `Copy-Item`（Windows 允许重命名被 mmap 的文件）。
8. **litert `.bazelrc.user` 已存在冲突** — patch 00 是 `new file mode`，如果目标已存在需先 `Remove-Item $LITERT\.bazelrc.user`。
9. **`mini_installer` 打包缺 `libLiteRtWebGpuAccelerator.dll` / `webgpu_dawn.dll`** —
   已在 `chrome/installer/mini_installer/chrome.release` 里加了这两条：
   ```ini
   libLiteRtWebGpuAccelerator.dll: %(VersionDir)s\
   webgpu_dawn.dll: %(VersionDir)s\
   ```
   但 clean 一次之后再改 DLL 会静默漏包／打进旧版本，根因是 Chromium `create_installer_archive.py` 的两个"缓存陷阱"叠加：

   1. **staging 幂等复制** — `chrome/tools/build/win/create_installer_archive.py` 第 213 行 `CopySectionFilesToStagingDir`：
      ```python
      for src_path in src_paths:
          dst_path = os.path.join(dst_dir, os.path.basename(src_path))
          if not os.path.exists(dst_path):
              g_archive_inputs.append(os.path.relpath(src_path, src_dir))
              shutil.copy(src_path, dst_dir)
      ```
      只在 staging 目录 `gen/chrome/installer/mini_installer/mini_installer/Chrome-bin/<ver>/` 里 **不存在** 该文件时才 copy。DLL 更新（Debug ↔ Release / rebuild bazel）后 staging 里的老 copy 不会被覆盖。且没走 copy 分支就不会把该 src 加进 `g_archive_inputs` → 也就不会写进 `archive.d` depfile。
   2. **ninja `inputs` 列表不含我们的 DLL** — `chrome/installer/mini_installer/BUILD.gn` 里 `action("mini_installer_archive")` 的 `inputs` 只列了 `chrome.dll / chrome_elf.dll / chrome.exe / locales/en-US.pak / setup.exe / chrome.release`（外加 setup 的 runtime_deps）。我们两个 DLL 既不在 `inputs`、也没能通过 depfile 声明，所以 ninja 单独更新这两个 DLL 时 **压根不会认为 action 需要 rerun**。

   两条合起来：只要 DLL 有过一次成功入 staging，后续无论怎么替换 `out\Release\*.dll`，`chrome.7z` 里都还是老版本；而某些 case（例如先跑过一次未修改 `chrome.release` 的 `mini_installer`，之后才补上 chrome.release 两条）就会永远遗漏——直到手动清干净。

   **手工每次 rebuild 前的清理清单**（推荐做成脚本）：
   ```powershell
   $Out = "C:\Users\fujun\workspace\chromium\src\out\Release"
   Remove-Item -Force -ErrorAction SilentlyContinue `
       "$Out\chrome.7z", "$Out\chrome.packed.7z", "$Out\mini_installer.exe", `
       "$Out\gen\chrome\installer\mini_installer\archive.d"
   Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
       "$Out\gen\chrome\installer\mini_installer\mini_installer\Chrome-bin"
   & "C:\Users\fujun\workspace\chromium\src\third_party\ninja\ninja.exe" -C $Out mini_installer
   # 验证
   tar -tvf "$Out\chrome.7z" | Select-String "libLiteRt|webgpu_dawn"
   ```

   **根治（可选）**：给 `chrome/installer/mini_installer/BUILD.gn` 的 `action("mini_installer_archive")` 补 `inputs`：
   ```gn
   inputs = [
     # ... existing entries ...
     "$root_out_dir/libLiteRtWebGpuAccelerator.dll",
     "$root_out_dir/webgpu_dawn.dll",
   ]
   ```
   这样 ninja 会随 DLL 变化重跑 action；但仍需保留 staging 清理逻辑，因为 py 脚本本身的幂等 copy 不会覆盖老文件。可以把这个补丁沉淀成 `patches/10-mini-installer-BUILD-inputs.patch`（未做，先靠 §5.2 的清理清单绕过）。

10. **RDP 远程桌面下 GPU 不可用（最常见的前置坑）** — 症状：stderr 出现 `D3D12CreateDevice failed with DXGI_ERROR_DRIVER_INTERNAL_ERROR (0x887A0020)` 和 `eglCreateContext: Requested GLES version (3.1) is greater than max supported (3.0)`。根因是 RDP 会话里活动显示适配器是 `Microsoft Remote Display Adapter`（软件间接显示），真实 GPU 不参与渲染，Dawn/D3D12 建不出 device。**不是机器 GPU 坏了，是会话类型不对**。解决：到物理机前本地登录 `console` 会话再跑（`query session` 应显示 `console` Active、`SESSIONNAME=Console`）；本地登录后建议更新 GPU 驱动。

---

## 8. 复用 Linux bundle

`../route-a-webgpu/webnn_gpu_dispatch_test.html` 直接可用，不必复制。

Linux patches 02/03/04（`route-a-webgpu/patches/`）与本 bundle 的 03/04/05 内容基本一致（都是 label 修复 + farmhash + pipeline cache 禁用），仅有轻微差异。如果 Linux 侧有更新，可直接同步过来。
