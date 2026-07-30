# Route A · WebGPU:从 litert 仓编出 `libLiteRtWebGpuAccelerator.so` 并让 Chromium WebNN 用上 ml_drift GPU delegate

> 复现指南(面向同事,含全部改动 patch)。
> 背景与取舍见主设计文档 `../webnn-mldrift-gpu-integration-design.md` 的 §6c-3 / §6d。
> 快照日期:2026-07-29。本文所有 patch 已用 `git apply --reverse --check` 对当前工作树验证通过。

---

## 0. 这个方案做什么 / 不做什么

- **做**:用**预编译 Dawn**给 litert 仓那个"桩" `@dawn` 补上真身,复用 §6c 已跑通的 `crcxx`(Chromium clang+libc++)工具链,从 **litert 仓**编出 chrome 非 Android 分支会自动 `dlopen` 的 `libLiteRtWebGpuAccelerator.so`。这是**路线 A(dlopen,`--no-sandbox`)**的 WebGPU 落地。
- **为什么是 WebGPU 而不是 OpenCL**:OpenCL 路线已证实是死路——chrome 侧 litert 编译时 `LITERT_HAS_OPENCL_SUPPORT=0` 且**没有编入任何 OpenCL tensor-buffer 源**,`dispatch` 时在 `delegate_kernel_litert.cc:269` 分配共享 GPU tensor buffer 失败(推理结果全 0)。WebGPU 不同:chrome 侧 litert **默认 `LITERT_HAS_WEBGPU_SUPPORT=1` 且已编入 WebGPU tensor-buffer 层**(`tensor_buffer.cc` 的 `CreateFromWebGpuBuffer` + `kLiteRtTensorBufferTypeWebGpu*`),把 WGPU buffer 当**不透明 handle**经 registry 回调交给 `.so` 里的 `buffer_handler_webgpu`,因此**共享 tensor-buffer 层不缺**,理应得到数值正确的推理。详见主文档 §6c-2 / §6c-3。
- **不做**:正式集成(那是路线 B:静态 GN、宿主与 delegate 同一次编译、复用 Chromium `//third_party/dawn`)。本方案是关沙箱的验证/原型。

**产物(已验证编译成功,5.3MB stripped)**:导出 `LiteRtAcceleratorImpl@@VERS_1.0`;对 chrome 边界 ABI 干净(无 `libstdc++`/`libc++` DT_NEEDED,`GLIBCXX` 未定义符号 0);对 Dawn 边界 `DT_NEEDED libdawn.so` + 84 个未定义 `wgpu*` C 符号(纯 C-ABI 边界)。**运行期数值验证尚未跑通,是下一步(见 §6）。**

---

## 1. 前置条件（环境）

| 项 | 要求 |
|---|---|
| Chromium 检出 | 已 `gn gen` 并**编好 `out/Release/chrome`**(内含 WebNN + litert GPU 注册路径)。本文记 `$CR=/home/junwei/workspace/chromium/src`。 |
| 两个源码树 | `$CR/third_party/litert/src`(构建 `.so` 的根 repo)、`$CR/third_party/ml-drift`(`@ml_drift`,upstream `google-ai-edge/ml-drift`,含 `third_party/dawn/build_libdawn.sh`)。 |
| Bazel | 7.x(litert 自带 `.bazelrc`;本文用 `bazel`)。 |
| Chromium clang + libc++ | `$CR/third_party/llvm-build/Release+Asserts/bin/clang`、`$CR/third_party/libc++/src/include`、`$CR/buildtools/third_party/libc++/__config_site`。 |
| GPU | NVIDIA(本文 GTX 1060);Dawn 后端走 Vulkan/GL,需 Vulkan ICD 可用。 |
| **真实终端** | `build_libdawn.sh` 需 `sudo apt` + 网络 clone，**必须在有 TTY + 网络的终端手跑**(不能经无 TTY/无网的自动化环境)。 |

> ⚠️ **可移植性头号注意**:`patches/00-bazelrc-user-crcxx.patch` 里全是**绝对路径**(写死了 `/home/junwei/workspace/chromium/src` 与 `/home/junwei/workspace/webnn/_cr_libcxx_link`)。换环境**必须**把这些路径改成你自己的 `$CR` 和 fat-archive 目录(见 §3-步骤 B 与 §4）。

---

## 2. 改动清单（全部在 `patches/`，路径相对 litert src 根）

| patch | 文件 | 改什么 / 为什么 |
|---|---|---|
| `00-bazelrc-user-crcxx.patch` | `.bazelrc.user`(**新增**) | §6c 的 `crcxx` 工具链:用 Chromium clang23 + libc++、force-include `__config_site` 锁 `_LIBCPP_ABI_VERSION 2/__Cr`,并链接 Chromium 的 libc++ 厚归档。**含绝对路径,换环境必改。** |
| `01-dawn-workspace-prebuilt.patch` | `third_party/dawn/workspace.bzl` | 把 litert 的 `@dawn` 桩(`http_archive`+dummy BUILD)换成 `prebuilt_dawn` `repository_rule`:拷预编译 Dawn 的 `include/`+`lib/libdawn.so`,并**同时**产出根包 `@dawn//:` 与子包 `@dawn//dawn:` 两套标签(`webgpu_dawn`/`webgpu_headers`/`dawn_headers`),指向同一 `libdawn.so`。 |
| `02-delegate-BUILD-weight_loader.patch` | `ml_drift_delegate/delegate/BUILD` | 上游 copybara 漏 swap:内部 label `//third_party/odml/litert/weight_loader:external_weight_loader` → 外部 `//weight_loader:external_weight_loader`。 |
| `03-shared_memory_manager-BUILD-weight_loader.patch` | `ml_drift_delegate/delegate/shared_memory_manager/BUILD` | 同上,两处(webgpu + metal 目标)。 |
| `04-delegate_webgpu-farmhash-and-pipeline-cache.patch` | `ml_drift_delegate/delegate/delegate_webgpu.cc` | 两处源码修正(见下)。 |

`04` 的两处修正（均**不影响数值正确性**）：
1. **farmhash copybara 漏 swap**（仅 WebGPU 路径命中）：`#include "util/hash/farmhash_fingerprint.h"`(内部路径,树里没有)+ `farmhash::Fingerprint64` → 外部 `@farmhash_archive//:farmhash` 暴露的 `#include "farmhash.h"` + `util::Fingerprint64`。
2. **禁用 perf-only 的 pipeline cache**（Dawn 版本代差 + C++20 依赖）：预编译 Dawn(Chromium 147)的 `DawnCacheDeviceDescriptor` 只有 C 字段,没有 `SetDawnLoad/StoreCacheDataCallback` 这两个 C++ setter;且回调用了 `std::span`(C++20,litert 默认 c++17)。`#if 0` 掉两个 span 回调 + 去掉 `cache_desc` wiring;正确性路径不动。

> 注:`delegate_opencl.cc` 里还有一处 OpenCL 专用的 context-fallback 补丁(§6c-2),**WebGPU 路线不需要**,故未纳入本 patch 集。

---

## 3. 复现步骤

### 步骤 A — 应用 patch

```bash
CR=/home/junwei/workspace/chromium/src            # ← 改成你的 Chromium src
LITERT=$CR/third_party/litert/src
BUNDLE=/path/to/route-a-webgpu                    # ← 本 bundle 所在目录

cd "$LITERT"
# litert 仓是 git 树,可直接 apply / 回滚
git apply "$BUNDLE"/patches/01-dawn-workspace-prebuilt.patch
git apply "$BUNDLE"/patches/02-delegate-BUILD-weight_loader.patch
git apply "$BUNDLE"/patches/03-shared_memory_manager-BUILD-weight_loader.patch
git apply "$BUNDLE"/patches/04-delegate_webgpu-farmhash-and-pipeline-cache.patch
git apply "$BUNDLE"/patches/00-bazelrc-user-crcxx.patch     # 新增 .bazelrc.user
# 回滚:git apply --reverse <patch> ;或 git checkout -- <file> / rm .bazelrc.user
```

> `.bazelrc.user` 由 litert `.bazelrc` 的 `try-import %workspace%/.bazelrc.user` 自动加载(§6c)。

### 步骤 B — 准备 crcxx 工具链依赖（§6c 的一次性前置，patch 00 依赖它）

`crcxx` 链接期需要 Chromium 自己的 libc++ **厚归档**(仓里的 `libc++.a` 是 thin archive,exec root 下失效)。一次性生成:

```bash
AR=$CR/third_party/llvm-build/Release+Asserts/bin/llvm-ar
DEST=/home/junwei/workspace/webnn/_cr_libcxx_link          # ← 必须与 patch 00 里的 -L/归档路径一致
mkdir -p "$DEST"
$AR crs "$DEST/libc++_full.a"    $CR/out/Release/obj/buildtools/third_party/libc++/libc++/*.o
$AR crs "$DEST/libc++abi_full.a" $CR/out/Release/obj/buildtools/third_party/libc++abi/libc++abi/*.o
# 校验:应看到 __Cr 命名空间符号(Chromium libc++ ABI v2)
$CR/third_party/llvm-build/Release+Asserts/bin/llvm-nm "$DEST/libc++_full.a" | grep -c __Cr
```

同时确保 litert 已 `configure`(生成 `.litert_configure.bazelrc`),非交互一次即可:

```bash
cd "$LITERT"
PYTHON_BIN_PATH=$(command -v python3) USE_DEFAULT_PYTHON_LIB_PATH=1 \
TF_NEED_ROCM=0 TF_NEED_CUDA=0 TF_NEED_CLANG=1 TF_SET_ANDROID_WORKSPACE=0 \
CLANG_COMPILER_PATH=$CR/third_party/llvm-build/Release+Asserts/bin/clang \
  yes "" | python3 configure.py
```

> **换环境务必核对 `patches/00` 内所有绝对路径**:`CC=`、`BAZEL_CXXOPTS` 三个 `-isystem`、`-include .../__config_site`、`--linkopt=.../_cr_libcxx_link/libc++_full.a`(及 `libc++abi_full.a`)以及对应的 `--host_*` 镜像行。

### 步骤 C — 产出预编译 Dawn（**真实终端**）

```bash
cd "$CR/third_party/ml-drift"
./third_party/dawn/build_libdawn.sh
# 产出:/tmp/dawn/webgpu-dawn-binaries/out/latest/{include,lib/libdawn.so}
# 校验 monolithic C 符号(应非空):
nm -D /tmp/dawn/webgpu-dawn-binaries/out/latest/lib/libdawn.so | grep wgpuCreateInstance
```

> 该脚本 clone `jspanchu/webgpu-dawn-binaries`、CMake 编出 ~400MB `libdawn.so`(钉 Chromium 147),需 `sudo apt` + 网络。若已有产物可跳过。

### 步骤 D — 编 `.so`

```bash
cd "$LITERT"
DAWN_PREBUILT_DIR=/tmp/dawn/webgpu-dawn-binaries/out/latest \
bazel build --config=clang_local --config=crcxx --define ml_drift_api=wgpu -c opt \
  --check_visibility=false \
  --override_repository=ml_drift=$CR/third_party/ml-drift \
  //litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_so
# 产物:bazel-bin/litert/runtime/accelerators/gpu/libLiteRtWebGpuAccelerator.so
```

### 步骤 E — 复核 `.so` ABI 属性

```bash
SO=$(readlink -f bazel-bin/litert/runtime/accelerators/gpu/libLiteRtWebGpuAccelerator.so)
nm -D --defined-only "$SO" | grep LiteRtAcceleratorImpl        # route-A 入口,应有
readelf -d "$SO" | grep -E 'NEEDED|libdawn'                    # 应含 libdawn.so;不应含 libstdc++/libc++
nm -D -u "$SO" | grep -c GLIBCXX                               # 期望 0(crcxx libc++ 已内含)
nm -D -u "$SO" | grep -c 'wgpu'                                # 期望 ~84(C-ABI Dawn 边界)
```

### 步骤 F — 运行 + 数值验证（chrome 侧零改动）

```bash
RUN=/home/junwei/workspace/webnn/_run_wgpu
mkdir -p "$RUN"
cp "$SO" /tmp/dawn/webgpu-dawn-binaries/out/latest/lib/libdawn.so "$RUN/"
LD_LIBRARY_PATH="$RUN" $CR/out/Release/chrome \
  --no-sandbox --enable-features=WebMachineLearningNeuralNetwork \
  --enable-logging=stderr --v=1 \
  "file://$BUNDLE/webnn_gpu_dispatch_test.html" 2>&1 | tee /tmp/wgpu_dispatch.log
```

判读:
- `gpu_registry.cc` 应打 `Attempting to load GPU accelerator(libLiteRtWebGpuAccelerator.so).` + `... registered.`。
- 页面 `[WEBNN-DISP]` 日志:`build() ok; devices=["gpu",...]`,并 `readTensor result=[11,22,33,44]` → **RESULT: INFERENCE OK**。
- 对照 §6c-2 的 OpenCL 失败点(结果 `[0,0,0,0]`、`delegate_kernel_litert.cc:269`)看是否翻盘。

> `webnn_gpu_dispatch_test.html` 已随 bundle 提供:gpu context → `add` 图 → `createTensor`/`writeTensor`/`dispatch`/`readTensor`,期望 `[10,20,30,40]+[1,2,3,4]=[11,22,33,44]`。

---

## 4. 换环境时必须调整的东西（清单）

1. `patches/00` 内所有绝对路径 → 你的 `$CR`;`--linkopt` 的 fat-archive 路径 → 你的 `$DEST`(§3-B）。
2. `$DEST` 里的 `libc++_full.a` / `libc++abi_full.a` 必须重新从**你自己**的 `out/Release/obj/.../libc++*/*.o` 生成(ABI 要与你的 chrome 一致)。
3. `DAWN_PREBUILT_DIR` 若不用默认 `/tmp/dawn/...` 需相应改(`prebuilt_dawn` 读该环境变量,缺省即默认路径)。
4. `--override_repository=ml_drift=` → 你的 upstream `ml-drift` 检出路径(不再依赖 `ml-drift-main`)。

---

## 5. 已知风险 / 待验证（摘自主文档 §6d-7，按可能性排序）

1. **Dawn 版本代差**:ml_drift 用较新 `wgpu::` API。已知 pipeline-cache 的 setter/`std::span` 需 patch 04 绕过;若编译再报 `wgpu::` 成员缺失 → 换更新的 webgpu-dawn-binaries commit / 自编更新 Dawn。
2. **`libdawn.so` 必须导出 `wgpuXxx` C 符号(monolithic)**:`nm -D` 校验(步骤 C/E)。若只提供 proc-table 入口,须改开 `--define ml_drift_use_dawn_proc=true` 并补 `libdawn_proc` 目标(本方案默认走 monolithic,不需要)。
3. **`.so` 内 Dawn 在本机建 device**:GTX 1060 走 Vulkan;缺 Vulkan ICD 时落 GL。日志看 `Create WebGPU environment` 是否成功。
4. **运行期数值**:`.so` 已编译成功,但 `writeTensor/dispatch/readTensor` 端到端**尚未跑通验证**——这是本方案的最后一步(步骤 F)。

---

## 6. 状态

- ✅ `.so` 编译成功,ABI 属性符合设计(§6d-9)。
- ⏳ 运行期数值验证(步骤 F)—— **待跑**。理论上 WebGPU 有共享 tensor-buffer 层,应翻盘 OpenCL 的 `[0,0,0,0]`。
- 正式集成仍推荐路线 B(主文档 §5 P1+)。

## 7. Bundle 内容

```
route-a-webgpu/
├── README.md                          # 本文
├── webnn_gpu_dispatch_test.html       # 数值验证用例(步骤 F)
└── patches/
    ├── 00-bazelrc-user-crcxx.patch                       # 新增 .bazelrc.user(crcxx 工具链;含绝对路径,必改)
    ├── 01-dawn-workspace-prebuilt.patch                  # @dawn 桩 → prebuilt_dawn
    ├── 02-delegate-BUILD-weight_loader.patch             # weight_loader label
    ├── 03-shared_memory_manager-BUILD-weight_loader.patch# weight_loader label ×2
    └── 04-delegate_webgpu-farmhash-and-pipeline-cache.patch # farmhash + 禁用 pipeline cache
```
