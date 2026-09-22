# Route A · WebGPU (Windows)：从 litert 编出 `LiteRtWebGpuAccelerator.dll` 并让 Chromium WebNN 用上 GPU delegate

> 本文是**使用步骤**（装依赖 → 打 patch → 编 DLL → 部署 → 跑测试）。各 patch 的意图内联写在 patch 文件的注释里；踩坑记录见 **§7**。
>
> 本 bundle 覆盖 Linux 方案 `../route-a-webgpu/` 的 Windows 移植。
>
> 最近一次全流程验证：Chromium `155.0.8054.0`（2026-09-17），litert `dc32e93`，ml-drift `b29199f`，含 Dawn Proc 直通改造（见 §1.4b）。

---

## 环境要求

| 组件 | 版本 | 用途 |
|---|---|---|
| Chromium 检出 + 编好一次 Release | 主线最新（本文验证于 155.x） | `chrome.exe`、libc++ `.obj`、clang-cl、lld-link |
| Visual Studio（含 C++ 桌面工作负载） | 2022 或更新（本文验证于 VS 18） | MSVC 链接库（msvcprt.lib 等）+ `link.exe`；Dawn 用 VS 生成器编译 |
| Bazelisk / Bazel | 7.7+ | litert 构建 |
| CMake | 3.20+ | Dawn 构建 |
| Python 3.x | 需能在 PATH 里解析出真实的 `python.exe`（见下方"Python 陷阱"） | Bazel genrule、Dawn CMake 都要用 |
| Git for Windows（含 `bash.exe`） | 任何近期版本 | 打 patch；Bazel genrule 的 `--shell_executable` |
| Windows 开发者模式 | 已启用 | 允许非管理员创建符号链接（Bazel `--enable_runfiles` 需要） |

- 磁盘：Bazel output_base 约 20 GB，Dawn 构建目录约 5 GB。
- CPU：默认 `-march=sierraforest`（Sierra Forest 无 AVX-512）。换机按 CPU 改 `.bazelrc.user`，参考值见下表。
- 若在企业网络后面（走代理才能访问 github.com / dawn.googlesource.com），`git`/浏览器能用不代表 Bazel 能用——Bazel 的下载器是独立的 Java HTTP 客户端，只认 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量，不认 WinINET 代理设置。`scripts/build_accelerator_dll.ps1` 会在这两个变量为空时自动从 WinINET 设置读取并注入；手动跑 `bazel` 命令时需要自己 `$env:HTTPS_PROXY = "http://<proxy>:<port>"`。

### Python 陷阱

Windows 自带的 `python.exe` 是仅会弹出 Microsoft Store 页面的"应用执行别名"，不是真解释器；depot_tools 自带的 CPython 又只有 `python3.exe`，没有 `python.exe`。Dawn 的 CMake 脚本、Bazel 的部分生成器都会调用裸 `python`。`scripts/build_dawn.ps1` 会自动探测 depot_tools 下的 bootstrap Python 目录，缺 `python.exe` 时复制一份别名并把该目录加到 PATH 最前面。

---

## 1. 一次性 Setup

**脚本不再硬编码任何机器路径**——全部按「参数 > 环境变量 > 默认值」解析（默认值从 `%WORKSPACE_ROOT%`、`%USERPROFILE%` 和 bundle 自身位置推导，规则见 `scripts/common.ps1` 头注释）。同一份 bundle 拷到任何机器/账号都能直接用；只有目录布局不标准时才需要设置环境变量（可写进 `$PROFILE` 永久生效）：

```powershell
# 可选覆盖（默认值见括号）。多 checkout / 非标准目录布局只需设 WORKSPACE_ROOT，
# webnn、chromium、depot_tools 都是它的子目录：
#   $env:WORKSPACE_ROOT      workspace 根目录（默认 %USERPROFILE%\workspace）
#   $env:CHROMIUM_SRC        Chromium 检出（默认 %WORKSPACE_ROOT%\chromium\src）
#   $env:WEB_NN              webnn 工作目录（默认 bundle 所在位置向上两级）
#   $env:ML_DRIFT_DIR        ml-drift 检出（默认 %CHROMIUM_SRC%\third_party\ml-drift）
#   $env:DEPOT_TOOLS         depot_tools（默认 %WORKSPACE_ROOT%\depot_tools）
#   $env:DAWN_PREBUILT_DIR / DAWN_SRC_DIR / CR_LIBCXX_DEST（默认 webnn 下对应目录）
# 例：$env:WORKSPACE_ROOT = "C:\Users\awx_localadmin\workspace"

# 下面的变量只供手工命令（打 patch 等）使用，按实际 checkout 位置填写：
$env:WORKSPACE_ROOT = "C:\Users\awx_localadmin\workspace"
$BUNDLE  = "${env:WORKSPACE_ROOT}\webnn\route-a-webgpu-windows" # 本 bundle 所在目录（未设 WORKSPACE_ROOT 时为 $env:USERPROFILE\workspace\webnn\route-a-webgpu-windows）
$CR      = "${env:WORKSPACE_ROOT}\chromium\src"                # 与脚本默认值一致（未设 WORKSPACE_ROOT 时为 $env:USERPROFILE\workspace\chromium\src）
$WEBNN   = Split-Path $BUNDLE -Parent
$LITERT  = "$CR\third_party\litert\src"
$MLDRIFT = "$CR\third_party\ml-drift"
```

### 1.1 打包 Chromium libc++ 为静态库

Chromium `.obj` 是 LLVM bitcode，只能用 `lld-link.exe /lib` 打包。

```powershell
& "$BUNDLE\scripts\build_libcxx_lib.ps1" -ChromiumSrc $CR -Dest "$WEBNN\_cr_libcxx_link_win"
# 期望：约 3-4 MB 的 libc++.lib，看到 "__Cr@std string occurrences: >0"
```

### 1.2 编译 Dawn（webgpu_dawn.dll）

Dawn 必须与 Chromium 检出里的 `third_party/dawn` **严格同 commit**——不是"大版本相近就行"：加速器把 prebuilt 里 `dawn_version.h` 的 20 字节 SHA1 编进自己，运行时 `dawnProcSetProcs()` 会拿它和 Chromium 传进来的 proc table 的 version 字段逐字节比对，不一致直接 abort GPU 进程（详见 §1.4b 末尾）。版本号现在自动从 `$CR\chrome\VERSION` 读取，无需手填：

```powershell
& "$BUNDLE\scripts\build_dawn.ps1"     # -ChromiumVersion 可省略（自动读 chrome\VERSION）
# 期望：_dawn_prebuilt_win\{include,lib,src}\，其中 lib\webgpu_dawn.dll ~10 MB，
# src\dawn_proc.cpp 是脚本自动从 $BUNDLE\vendor\dawn_proc.cpp 拷过来的（见 §1.4b）。
# 耗时 30-60 分钟（几乎都花在 lib\webgpu_dawn.dll 上；-define=ml_drift_use_dawn_proc=true
# 下这个 DLL 实际不会被加速器链接，见 §1.4b 后的说明，但目前仍需要跑这一步来拿 include\ 头文件）。
```

> **升级 Chromium 后要不要重跑这一步？先别急着跑**——`scripts\check_dawn_version.ps1` 会比对 prebuilt 与 Chromium 的 Dawn hash，一致就跳过这 30-60 分钟（§1.4b 末尾有完整升级清单）。

> **Dawn 分支滞后提示**：Dawn 只有在对应 Chromium milestone 切分支后才会出现 `chromium/<BUILD>` 分支，追主线 Chromium 时常常比 Dawn 新 1-2 个 build。脚本会自动查询 `dawn.googlesource.com` 上已发布的分支列表，找不到精确匹配时自动回退到「不超过所需版本的最新分支」，并打印提示，不需要手工改版本号。

### 1.3 打 patch

**litert 仓**（`$LITERT`）：

```powershell
Push-Location $LITERT
git apply "$BUNDLE\patches\00-bazelrc-user-crcxx-win.patch"                          # 生成 .bazelrc.user
git apply "$BUNDLE\patches\01-dawn-workspace-prebuilt-win.patch"                     # @dawn -> 本地预编译
git apply "$BUNDLE\patches\02-workspace-local-ml-drift.patch"                        # ml_drift -> local_repository
git apply "$BUNDLE\patches\03-composite-drop-internal-moe-experts-kernel.patch"      # 跳过 Google 内部专属的 MoE 融合 kernel
git apply "$BUNDLE\patches\07-serialization-weight-cache-python3-genrule.patch"      # 绕开 Windows MAX_PATH 限制
Pop-Location
```

> 若某个 patch `git apply` 失败（上游代码变了），看该 patch 文件顶部的注释了解意图，按同样思路手动改。

**ml-drift 仓**（`$MLDRIFT`）：

```powershell
Push-Location $MLDRIFT
git apply "$BUNDLE\patches\04-ml-drift-gpu-info-uint32max.patch"                     # narrowing conversion 修复
Pop-Location
```

**Chromium 仓**（`$CR`）：

```powershell
Push-Location $CR
git apply "$BUNDLE\patches\05-webnn-sandbox-init-full-dll-path.patch"        # GPU 进程 pre-sandbox 用绝对路径 LoadLibrary（安装版必需）
git apply "$BUNDLE\patches\06-chrome-release-webnn-dlls.patch"               # 让 mini_installer 打包 webgpu_dawn.dll
Pop-Location
```

> **patch 05 从 155.0.8054.0 起已不需要**：该版本的 `webnn_sandbox_init.cc` 已经用绝对路径 `LoadLibrary`，patch 05 会 `git apply` 失败（或 no-op）。升级到这个版本或更新时，先 `git apply --check` 探测，失败就跳过 05，只打 06。见 §5 第 14 条。

**可选**（不影响 GPU delegate 是否工作，纯调试辅助）：

```powershell
Push-Location $LITERT
git apply "$BUNDLE\patches\90-ml-drift-log-unsupported-op-counts.patch"      # 按 opcode 汇总"哪些算子被拒绝、多少个" 的日志
Pop-Location
```

### 1.4 编辑 `.bazelrc.user` 的绝对路径

patch 00 里的绝对路径是本机的，换机必改：

```powershell
# 编辑 $LITERT\.bazelrc.user，替换：
#   C:/Users/junwei/workspace/chromium/src  -> $CR 的正斜杠形式
#   C:/Users/junwei/workspace/webnn         -> $WEBNN 的正斜杠形式
#   -march=sierraforest                        -> 本机 CPU 的合适值
# （$CR / $WEBNN 按 §1 解析；设了 WORKSPACE_ROOT 时它们就是
#   %WORKSPACE_ROOT%\chromium\src 和 %WORKSPACE_ROOT%\webnn，注意
#   bazelrc 里必须用正斜杠）
```

> **patch 02（`WORKSPACE` 的 `local_repository(name = "ml_drift", ...)`）也硬编码了本机绝对路径**，同样换机必改——否则 Bazel 会安安静静地从旧机器/旧账号的 ml-drift 检出读取 `@ml_drift`，本地对 `ml_drift/webgpu/BUILD` 的任何编辑都不会生效（也不会报错）。踩过这个坑的完整排查过程见 §5 第 16 条。

### 1.4b Dawn Proc 直通（代码评审意见，2026-09-17）

评审者 Reilly 指出：加速器 DLL 不应静态链接一份独立的 `webgpu_dawn.dll`（Dawn Native 风格，直接导出 `wgpu*` 符号），而应该改走 Dawn Proc 风格——只编译 Dawn 自己生成的 `dawn_proc.cpp`（`wgpu*` 变成经由运行时可重绑定的全局 `DawnProcTable` 转发的 trampoline），这样加速器才能在运行时通过 `dawnProcSetProcs()` 绑定到 **Chromium 自己的** GPU 进程 Dawn 实例，而不是静默地跑一份自己的、独立的 Dawn/WebGPU 实例。原先 `ML_DRIFT_USE_DAWN_PROC` 这个开关一直没被设置，`graph_impl_litert.cc` 传进来的 `kWebGpuProcs`（Chromium 的 `DawnProcTable*`）在 `delegate_webgpu.cc` 里就是死代码。

修复内容：

- `third_party/dawn/workspace.bzl` 的 Windows `_WIN_ROOT_BUILD` 模板新增 `dawn_proc_lib`（编译 vendor 进来的 `src/dawn_proc.cpp`，见下）和 `libdawn_proc` alias；`webgpu_dawn`/`dawn_headers`/`webgpu_headers` 改为纯 headers-only + `cc_import`，不再默认拉近 `.dll`。
- vendor 了一份 Dawn 自己生成的 `dawn_proc.cpp` 到 `$BUNDLE/vendor/dawn_proc.cpp`，替换了它引用的 3 个 Dawn 内部头（`Compiler.h`/`assert.h`/`log.h`）为本地等价实现（`DAWN_NO_SANITIZE` 宏、`DAWN_CHECK` 宏、`dawn::ErrorLog()`）。`scripts/build_dawn.ps1` 每次运行都会把它拷进 `_dawn_prebuilt_win/src/dawn_proc.cpp`（`prebuilt_dawn` repo rule 的 Windows 分支会把这个 `src/` 目录一起拷进 `@dawn` 外部仓库），换机重新跑一遍 setup 也不会丢这份手工 patch。
- `ml_drift/webgpu/BUILD` 的 `:environment`、`:webgpu_headers` 等目标，以及 `ml_drift_delegate/delegate/BUILD` 的 `delegate_webgpu` 目标，把原本无条件的 `@dawn//:webgpu_dawn` 依赖改成 `select()`：
  ```python
  select({
      ":ml_drift_use_dawn_proc": ["@dawn//:libdawn_proc"],
      "//conditions:default": ["@dawn//:webgpu_dawn"],
  })
  ```
- `.bazelrc.user` 加一行 `build:crcxx_win --define=ml_drift_use_dawn_proc=true` 触发 `config_setting`（定义在 `ml_drift/webgpu/BUILD`）。
- `delegate_webgpu.cc` 里 `#if defined(ML_DRIFT_USE_DAWN_PROC)` 分支会从 LiteRT 的 `kLiteRtEnvOptionTagWebGpuProcs` 读出 Chromium 传入的 `DawnProcTable*` 并调用 `dawnProcSetProcs()`。

**验证**：修好后 `libLiteRtWebGpuAccelerator.dll` 的 PE import table 里彻底没有 `webgpu_dawn.dll`（`dumpbin /DEPENDENTS` 确认），linker `.params` 里只剩 `/WHOLEARCHIVE:.../dawn_proc_lib.lib`（不再同时有 `webgpu_dawn.lib`），且 §3 的 GPU 推理测试正常拿到 `readTensor result=[11,22,33,44]`。

> ⚠️ **踩过的坑**：`WORKSPACE` 里 `local_repository(name = "ml_drift", path = "...")` 是按机器硬编码的绝对路径（同 §1.4 的 `.bazelrc.user`）。如果这个 `path` 还指着别的机器/账号的 ml-drift 检出（例如换账号复制 bundle 时忘记改这一行），Bazel 会安安静静地从那个旧目录读 `@ml_drift`——对本地这份 `ml_drift/webgpu/BUILD` 的任何编辑都会被完全忽略且不报错，只会在链接期表现为奇怪的重复符号错误。换机 / 换检出目录时，`WORKSPACE` 里所有 `local_repository`/硬编码路径都要连带检查，不能只改 `.bazelrc.user`。

> **`scripts/build_dawn.ps1` 在 §1.4b 之后还需要吗？需要，不能删**——`prebuilt_dawn` repo rule（§1.4b 改过的 `workspace.bzl`）不管选没选 `ml_drift_use_dawn_proc`，都无条件要求 `$DAWN_PREBUILT_DIR` 下有 `include/`、`lib/`、`src/` 三个目录都存在，脚本仍是唯一产出 `include/`（Dawn 头文件）和 `lib/webgpu_dawn.dll` 的地方。已知的低效之处：`ml_drift_use_dawn_proc=true` 下加速器根本不链接 `lib/webgpu_dawn.dll`（只用 `include/` 里的头文件 + `src/dawn_proc.cpp`），所以脚本里 30-60 分钟的 CMake 编译有一部分是在编一个现在用不上的 DLL。评估过把这一步换成直接拷贝 Chromium 自己 `out\<dir>\gen\third_party\dawn\include\` 的产物，但 Chromium 自己的 GN 构建只生成一个精简子集（缺 `dawn/dawn_proc.h`、`webgpu/webgpu.h`、`dawn/native/*.h`、`dawn/wire/WireClient.h` 等完整 Dawn "SDK" 里的大量头文件），逐一核对替换的风险大于收益，故保留现状；只把 `dawn_proc.cpp` 这一个必须手工 patch 的文件迁移成了脚本自动从 `vendor/` 拷贝（见上）。

> **为什么升级 Chromium 后 Dawn 仍需严格同 commit（proc 直通下也成立）**：`vendor/dawn_proc.cpp` 的 `dawnProcSetProcs()` 里有硬校验——把 Chromium 传进来的 `DawnProcTable.version`（20 字节 SHA1，Chromium 构建时从自己的 `gen/third_party/dawn/include/dawn/dawn_version.h` 填入）与加速器编译时烘焙的 `dawn::kDawnVersion`（来自 `_dawn_prebuilt_win\include\dawn\dawn_version.h`，即 build_dawn.ps1 实际 fetch 到的 Dawn commit）逐字节比对，`DAWN_CHECK` 失败 → `abort()`，GPU 进程崩溃。所以"加速器不再链接 webgpu_dawn.dll"只意味着 **lib\ 里的 DLL 无关紧要**，include\ 里的这个 hash 仍是运行时硬约束（2026-09-17 验证通过，正是因为 build_dawn.ps1 fetch 的 `chromium/8054` tip 恰好等于 Chromium DEPS 钉住的 Dawn commit，两机实测 hash 相等）。
>
> **升级 Chromium 后的最小步骤**：
> 1. `gclient sync` 后**先重编 chrome**（`autoninja -C out\<dir> chrome`）——旧 chrome 二进制里的 proc table 携带旧 hash，不重编会直接触发上面的 abort（先跑加速器再重编 chrome 也会崩）。
> 2. 跑 `& "$BUNDLE\scripts\check_dawn_version.ps1"`（可用 `-ChromeOutDir` 指定别的 out 目录）：
>    - 绿灯（prebuilt hash == Chromium 构建的 hash）→ **不必重跑 build_dawn.ps1**，直接进 §2 重编加速器；
>    - 红灯 → 重跑 `build_dawn.ps1`（版本号自动读 `chrome\VERSION`），再进 §2。
> 3. 按 §2 重编 `libLiteRtWebGpuAccelerator.dll`，按 §3 重新部署。

---

## 2. 编译加速器 DLL

> **升级 Chromium 后**：先按 §1.4b 末尾的清单重编 chrome + 跑 `check_dawn_version.ps1`，再执行下面的编译。

**首次编译**（只为了生成工具链配置），预期会因 hermetic include 检查失败：

```powershell
& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode opt -ChromiumSrc $CR -MlDrift $MLDRIFT
```

**打 Bazel 生成的 toolchain BUILD**（一次性；`bazel clean --expunge` 后要重跑）：

```powershell
python "$BUNDLE\scripts\patch_bazel_toolchain.py"
```

**再编一次**：

```powershell
& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode dbg

& "$BUNDLE\scripts\build_accelerator_dll.ps1" -Mode opt
# 期望产出：$LITERT\bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.dll
# （opt ~15 MB，含静态链入的 clang_rt.builtins，见 §5 第 15 条；早期不含该 linkopt 的构建约 8.5 MB）
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

> ⚠️ **开了 `WebNNLiteRTGpuInRenderer` 时必须同时加 `--enable-unsafe-webgpu`**。这条 feature 会让 `navigator.ml.createContext({deviceType:'gpu'})` 走渲染进程内的 `ContextImplLiteRt::CreateForRenderer()`，走之前必须先在渲染进程里成功拿到一个 `wgpu::Device`（`third_party/blink/renderer/modules/ml/ml.cc` 的 `RequestWebGpuAdapter`/`OnWebGpuDeviceRequested`）——不加 `--enable-unsafe-webgpu` 时这个 WebGPU adapter/device 请求会失败，`createContext(gpu)` 直接抛 `Failed to obtain WebGPU context for LiteRT in renderer.`，加速器 DLL 根本不会被加载。完整可用的启动参数组合：
> ```
> --no-sandbox --enable-features=WebMachineLearningNeuralNetwork,WebNNLiteRTGpuInRenderer --enable-unsafe-webgpu
> ```

> **关于 RDP 远程桌面**：早期在部分机器 / 驱动组合上观察到 RDP 会话（`Microsoft Remote Display Adapter` 软件间接显示适配器接管，真实 GPU 不参与渲染）导致 `D3D12CreateDevice failed with DXGI_ERROR_DRIVER_INTERNAL_ERROR (0x887A0020)`。但本次在 Windows Server 2025 + 较新 Intel 驱动下，RDP 会话（`rdp-tcp#N`）内也完整跑通了 GPU 推理（`gpu-process` 正常常驻，`readTensor` 返回正确值）。是否受 RDP 影响似乎和具体驱动/GPU 型号相关——遇到下列症状时，优先怀疑 RDP：
> - stderr 出现 `D3D12CreateDevice failed with DXGI_ERROR_DRIVER_INTERNAL_ERROR (0x887A0020)`
> - stderr 出现 `eglCreateContext: Requested GLES version (3.1) is greater than max supported (3.0)`
> - 页面/`chrome://gpu` 里 WebNN 退化为 CPU
>
> 自查：`query session` 当前会话应为 `console`（Active）；确认无效时换回本地控制台登录，并更新 GPU 驱动（老驱动如 Intel 27.20.x 对 Dawn/D3D12 兼容较差）。

### 3.1 dev 目录直接跑（推荐迭代方式）

```powershell
& "$BUNDLE\scripts\deploy_to_chrome.ps1" -ChromeOutDir "$CR\out\Release" -Mode opt -ChromiumSrc $CR

& "$BUNDLE\scripts\deploy_to_chrome.ps1" -ChromeOutDir "$CR\out\upstream_bots_debug" -Mode dbg -ChromiumSrc $CR

& "$CR\out\Release\chrome.exe" `
    --no-sandbox --enable-features=WebMachineLearningNeuralNetwork,WebNNLiteRTGpuInRenderer `
    --enable-unsafe-webgpu `
    "file:///$WEBNN/route-a-webgpu/webnn_gpu_dispatch_test.html"
```

### 3.2 mini_installer 部署到测试机

```powershell
# 0) 前置：patch 06 已 apply；out\Release 是 is_official_build=true；加速器/Dawn DLL 已就绪。
# 1) 若之前跑过 mini_installer，先清老 staging（否则会打进老 DLL，详见 §5 第 9 条）：
$Out = "$CR\out\Release"
Remove-Item -Force -ErrorAction SilentlyContinue `
    "$Out\chrome.7z", "$Out\chrome.packed.7z", "$Out\mini_installer.exe", `
    "$Out\gen\chrome\installer\mini_installer\archive.d"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
    "$Out\gen\chrome\installer\mini_installer\mini_installer\Chrome-bin"

# 2) stage DLL + 打包（脚本内部做 Rename-Item; Copy-Item 覆盖 + ninja mini_installer）
& "$BUNDLE\scripts\build_mini_installer.ps1" -ChromeOutDir "$Out" -ChromiumSrc $CR

# 3) 自检 chrome.7z 里有两个自制 DLL
tar -tvf "$Out\chrome.7z" | Select-String "libLiteRt|webgpu_dawn"

# 4) 传 $Out\mini_installer.exe 到目标机，双击安装（默认装到 %LOCALAPPDATA%\Chromium）
# 5) 目标机上运行
& "$env:LOCALAPPDATA\Chromium\Application\chrome.exe" `
    --no-sandbox --enable-features=WebMachineLearningNeuralNetwork,WebNNLiteRTGpuInRenderer `
    --enable-unsafe-webgpu `
    "file:///<路径>/webnn_gpu_dispatch_test.html"
```

### 3.3 判读

- Chrome 日志（`--enable-logging --v=1` 时）：`Attempting to load GPU accelerator(LiteRtWebGpuAccelerator.dll).` + `... registered.`（这两条走的是 GPU 子进程自己的日志，不一定出现在 `--enable-logging=stderr` 重定向到父进程 stdio 的那份输出里；若看不到，改用 `--enable-logging`，写到 `<user-data-dir>\chrome_debug.log`，或直接用 `Get-CimInstance Win32_Process` 确认存在 `--type=gpu` 的常驻子进程）
- 页面上 `[WEBNN-DISP]` 输出 `readTensor result=[11,22,33,44]` → **RESULT: INFERENCE OK**
- 关键：**不再有** `std::length_error` 在 `delegate_kernel.cc:226`

---

## 4. 参考

```
route-a-webgpu-windows/
├── README.md                                                    # 本文
├── patches/
│   ├── 00-bazelrc-user-crcxx-win.patch                          # 应用到 litert 仓：.bazelrc.user（crcxx_win 配置，new file）
│   ├── 01-dawn-workspace-prebuilt-win.patch                     # 应用到 litert 仓：@dawn -> 本地预编译 webgpu_dawn.dll
│   ├── 02-workspace-local-ml-drift.patch                        # 应用到 litert 仓：@ml_drift -> local_repository
│   ├── 03-composite-drop-internal-moe-experts-kernel.patch      # 应用到 litert 仓：跳过 Google 内部专属的 MoE 融合 kernel（缺失的 @ml_drift//.../google/custom 包会导致 Bazel 分析失败）
│   ├── 04-ml-drift-gpu-info-uint32max.patch                     # 应用到 ml-drift 仓：narrowing conversion 修复
│   ├── 05-webnn-sandbox-init-full-dll-path.patch                # 应用到 Chromium 仓：GPU 进程 pre-sandbox 用绝对路径 LoadLibrary
│   ├── 06-chrome-release-webnn-dlls.patch                       # 应用到 Chromium 仓：mini_installer 打包 webgpu_dawn.dll
│   ├── 07-serialization-weight-cache-python3-genrule.patch      # 应用到 litert 仓：genrule 直接调 $(PYTHON3)，绕开 py_binary 深层 runfiles 触发的 Windows MAX_PATH 限制
│   └── 90-ml-drift-log-unsupported-op-counts.patch              # 可选：应用到 litert 仓，按 opcode 统计被拒绝算子数量的调试日志
├── vendor/
│   └── dawn_proc.cpp               # 1.4b: 手工 patch 过的 Dawn dawn_proc.cpp（去掉 3 个 Dawn 内部头依赖），
│                                    #       build_dawn.ps1 每次运行都会拷进 _dawn_prebuilt_win/src/
└── scripts/
    ├── common.ps1                  # 路径统一解析：参数 > 环境变量 > 默认值（WORKSPACE_ROOT / %USERPROFILE% + bundle 自身位置推导）
    ├── check_dawn_version.ps1      # 1.4b: 比对 prebuilt 与 Chromium 的 Dawn 20 字节 SHA1，判断是否要重跑 build_dawn.ps1
    ├── build_libcxx_lib.ps1        # 1.1: 打包 libc++.lib
    ├── build_dawn.ps1              # 1.2: 编 webgpu_dawn.dll + 拷贝 vendor/dawn_proc.cpp（含 python.exe 别名探测 + Dawn 分支自动回退 + 版本号自动读 chrome\VERSION）
    ├── patch_bazel_toolchain.py    # 2:   打 Bazel 自动生成的 toolchain BUILD
    ├── build_accelerator_dll.ps1   # 2:   bazel build ...（含代理自动注入 + bash.exe 自动探测）
    ├── deploy_to_chrome.ps1        # 3.1: 拷 DLL 到 chrome.exe 目录（dev 迭代用）
    └── build_mini_installer.ps1    # 3.2: stage DLL + 编 mini_installer.exe（部署到测试机用）
```

---

## 5. 已知坑与调试线索

1. **`bazel clean --expunge` 后必须重跑 `patch_bazel_toolchain.py`** — 因为 `local_config_cc/BUILD` 是 Bazel 自动生成的，`--expunge` 会重生成，之前的注入丢失。
2. **XNNPACK `.S` 文件报错 `A2044` (MASM invalid character)** — 说明 `--compiler=clang-cl` 被误设了，导致 XNNPACK 走 clang 分支引入 GAS-语法 `.S`，被 `ml64.exe` 拒绝。删掉这个 flag（保留 `USE_CLANG_CL=1` 让工具是 clang-cl，但 select 走 msvc）。
3. **运行 `flatc.exe` 报 `STATUS_ILLEGAL_INSTRUCTION`（`0xC000001D`）** — 说明 `-march=<x>` 高于本机 CPU 支持的指令集。降级 `-march`（例如从 `sapphirerapids` 改到 `sierraforest` 或 `haswell`）。
4. **XNNPACK 某个 `avx256skx` / `avx256vnni` 文件报 `avx512vl` 未启用** — patch 00 中的 `--per_file_copt` 正则覆盖不到该文件。扩展正则或新增一行。
5. **链接期 `__ExceptionPtr*` 未定义** — `msvcprt.lib` 没链上；检查 `.bazelrc.user` 里 `--linkopt=msvcprt.lib`。
6. **`std::cout / std::cerr` 报 `__declspec(dllimport)` 未定义** — `-D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS` 没生效；查 `--cxxopt`。
7. **Chrome 加载 DLL 时挂在 UI 卡顿** — 部署时旧 DLL 被 Chrome 进程持有。用 `deploy_to_chrome.ps1` 的做法：先 `Rename-Item` 再 `Copy-Item`（Windows 允许重命名被 mmap 的文件）。
8. **litert `.bazelrc.user` 已存在冲突** — patch 00 是 `new file mode`，如果目标已存在需先 `Remove-Item $LITERT\.bazelrc.user`。
9. **`mini_installer` 打包缺 `libLiteRtWebGpuAccelerator.dll` / `webgpu_dawn.dll`** —
   已在 `chrome/installer/mini_installer/chrome.release` 里加了这两条（patch 06）：
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

   **手工每次 rebuild 前的清理清单**（`build_mini_installer.ps1` 已自动做，仅供手动排障参考）：
   ```powershell
   $Out = "<chromium>\src\out\Release"
   Remove-Item -Force -ErrorAction SilentlyContinue `
       "$Out\chrome.7z", "$Out\chrome.packed.7z", "$Out\mini_installer.exe", `
       "$Out\gen\chrome\installer\mini_installer\archive.d"
   Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
       "$Out\gen\chrome\installer\mini_installer\mini_installer\Chrome-bin"
   & "<chromium>\src\third_party\ninja\ninja.exe" -C $Out mini_installer
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
   这样 ninja 会随 DLL 变化重跑 action；但仍需保留 staging 清理逻辑，因为 py 脚本本身的幂等 copy 不会覆盖老文件。未做，先靠上面的清理清单绕过。

10. **RDP 远程桌面下 GPU 可能不可用** — 见 §3 开头的说明；症状是 `D3D12CreateDevice failed with DXGI_ERROR_DRIVER_INTERNAL_ERROR (0x887A0020)` / `eglCreateContext` 报错。根因是 RDP 会话里活动显示适配器可能是 `Microsoft Remote Display Adapter`（软件间接显示），真实 GPU 不参与渲染，Dawn/D3D12 建不出 device。但这不是绝对的——本次验证环境（Windows Server 2025 + 较新 Intel 驱动）下 RDP 会话（`rdp-tcp#N`）内 GPU 推理是正常工作的。遇到上述症状时再切到本地 `console` 会话排查，并优先确认/更新 GPU 驱动。

11. **Bazel 报 `Connect timed out` 抓不到 `@FP16` / `@dawn` 等外部依赖** — 企业代理环境下，Bazel 的 Java 下载器不认 WinINET 代理，只认 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量。`build_accelerator_dll.ps1` 会在这两个变量为空时自动从注册表 `HKCU:\...\Internet Settings` 读取并注入；手动跑 `bazel build` 需要自己设置这两个变量。

12. **`py_binary` genrule 报 `AssertionError: Cannot exec() '...\_stage2_bootstrap.py': file not found`，但用 `\\?\` 前缀能证实文件确实存在** — 这是 Windows `MAX_PATH`（260 字符）限制：`py_binary` 的 runfiles 树会把当前 package 路径在 `.runfiles\litert\<package>\` 下再嵌套一层，路径长度轻松超过 260；且大多数机器（非管理员）没有开启组策略 `LongPathsEnabled`，Python 自身按短路径 API 打开文件就会"假装"文件不存在。仅靠缩短 Bazel `output_base`（如 `C:\b`）不够——`.runfiles\...` 这段固定后缀本身就已经超限。根治：像 patch 07 那样，让 genrule 通过 `@rules_python//python:current_py_toolchain` 暴露的 `$(PYTHON3)` make 变量直接调用 `.py` 源文件，完全跳过 `py_binary` 的 runfiles 树（前提是该脚本本身不依赖 Bazel runfiles API，纯 stdlib 脚本都适用）。
13. **`bash.exe failed: ... CreateProcessW(...) The system cannot find the file specified`** — `--shell_executable` 硬编码的 `C:/Program Files/Git/bin/bash.exe` 在 Git 装到用户目录（`%LOCALAPPDATA%\Programs\Git`）时不存在。`build_accelerator_dll.ps1` 已改为自动探测常见安装位置，找不到才报错；手动跑 `bazel build` 时对应加 `--shell_executable=<你的 bash.exe 绝对路径，正斜杠>`。
14. **litert 上游已合并的功能不需要再打 patch** — 随 Chromium/litert 升级，早期 bundle 里的以下 patch 已被上游吸收，本 bundle 已移除：
    - `weight_loader` label 修复（`//third_party/odml/litert/weight_loader` → `//weight_loader`）：上游 BUILD 文件已直接使用新 label。
    - `compiled_model.cc` 里 `OptimizeMemoryForLargeTensors` 相关改动：该调用点已从上游代码里整体移除。
    - `graph_builder_tflite.cc` 的 `custom_call.LayerNorm` 融合算子：已完整合入 Chromium 上游（`SerializeLayerNormalizationAsCustomCall` 等）。
    - `delegate_webgpu.cc` 的 farmhash include 路径 + pipeline-cache 回调禁用：farmhash include 已用回上游路径；`webgpu-dawn-binaries` 现在编出的 Dawn 自带 `SetDawnLoad/StoreCacheDataCallback`，无需再禁用 pipeline cache。
    - **（2026-09-12 新增）`webnn_sandbox_init.cc` 的绝对路径 `LoadLibrary`（原 patch 05）**：Chromium `155.0.8054.0` 检出里这段逻辑已经是上游代码的一部分，`git apply patches\05-webnn-sandbox-init-full-dll-path.patch` 会失败；跳过即可，不影响 GPU delegate 是否工作。
    升级后先按 §1.3 只打"确实还需要"的 patch，`git apply --check` 失败再对照本条判断是否已被上游吸收。
15. **`lld-link: error: undefined symbol: __mulsc3`（链接期，不是 hermetic-include 报错）** — 出现在链接 `libLiteRtWebGpuAccelerator.dll` 的最后一步，被引用方是 tflite 内置的 `mul` kernel 对 `std::complex<float>` 做乘法（`EvalMul` / `BroadcastMul6DSlow` 等）。`__mulsc3`（以及同族的 `__muldc3`/`__divsc3`/`__divdc3`，复数乘除法的软件实现）是 clang/compiler-rt 的 builtin helper，既不在 Chromium 的 `libc++.lib` 里，也不在 `msvcprt.lib` 里，MSVC 的 STL/CRT 从不需要它们（MSVC 用不同的复数乘法展开方式）。根治：把 Chromium 自带的 `third_party\llvm-build\Release+Asserts\lib\clang\24\lib\windows\clang_rt.builtins-x86_64.lib` 作为 `--linkopt`/`--host_linkopt` 加进 `.bazelrc.user`（放在 `libc++.lib`/`msvcprt.lib` 之后即可，lld-link 对 `.lib` 顺序不敏感）。若升级 clang 后路径里的 `24`（clang 大版本号）变了，按新版本号调整该路径。
16. **`lld-link: error: duplicate symbol: wgpuAdapterAddRef`（及一堆同类 `wgpu*` 符号）** — 打开 §1.4b 的 Dawn Proc 直通后才会出现。表面看像是 `select()` 语义有问题（`ml_drift/webgpu/BUILD` 的 `:environment` 和 `ml_drift_delegate/delegate/BUILD` 的 `delegate_webgpu` 在同一次 build 里对同一个 `config_setting` 解析出了不同分支），但真正原因是 **`WORKSPACE` 里 `local_repository(name = "ml_drift", path = "...")` 这个路径没跟着换机器/换账号一起改**——Bazel 实际读的是 `path` 指向的那份（旧的、没打 §1.4b 补丁的）`ml_drift/webgpu/BUILD`，而你以为自己在编辑的那份文件其实在另一个物理目录，编辑完全没生效。排查方法：`bazel cquery <target> --output=build --check_visibility=false` 看 `generator_location` 和 resolved deps 里的 execroot 路径，或者直接去 `<output_base>/external/ml_drift/...` 读 Bazel 实际用的那份文件，跟本地编辑的版本比对。根治：`WORKSPACE` 里所有 `local_repository`/硬编码绝对路径都要按 §1.4 的方式换成本机路径，不能假设只有 `.bazelrc.user` 需要改。
17. **运行时 GPU 进程 abort，stderr 出现 `dawn_proc: CHECK failed: version_matches (external/dawn/src/dawn_proc.cpp:71)`** — 就是 §1.4b 里 `dawnProcSetProcs()` 的 20 字节 Dawn SHA1 逐字节比对失败：加速器编译时烘焙的 Dawn commit（`_dawn_prebuilt_win\include\dawn\dawn_version.h`）和 Chromium 当前这份 `chrome.exe` 实际编出来的 Dawn commit（`out\Release\gen\third_party\dawn\include\dawn\dawn_version.h`）不一致，多半是升级 Chromium 检出后没跟着重新走 §1.4b 的升级清单。排查/修复步骤：
    1. **先重编 chrome**（`autoninja -C out\<dir> chrome`）——旧 `chrome.exe` 里的 proc table 携带旧 hash，不重编光重编 Dawn/加速器也没用。
    2. 跑 `& "$BUNDLE\scripts\check_dawn_version.ps1"`（可用 `-ChromeOutDir` 指定别的 out 目录）比对两边 hash：
       - 绿灯（一致）→ 不用重跑 `build_dawn.ps1`，直接查是不是加速器本身没跟着重编；
       - 红灯（不一致，即本条场景）→ 重跑 `& "$BUNDLE\scripts\build_dawn.ps1"`（版本号自动读 `chrome\VERSION`，会 fetch 匹配的新 Dawn commit）。
    3. 按 §2 重编 `libLiteRtWebGpuAccelerator.dll`，按 §3 重新部署。

---

## 6. 复用 Linux bundle

`../route-a-webgpu/webnn_gpu_dispatch_test.html` 直接可用，不必复制。
