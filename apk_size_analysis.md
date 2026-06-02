# Android APK Size 分析：为什么把 LiteRT CPU 推理移到 renderer 进程导致 size trybot 失败

> 关联 CL：
> - `8ff7343475ab8573e943016312958726ada7948a` — LiteRT CPU inference 移到 renderer（**arm32 size trybot 失败**）
> - `chromium-review.googlesource.com/c/chromium/src/+/7785089` — TFLite CPU/NPU 路由到 in-renderer backend（**0 size 影响，已 merged**）
>
> 文档参考：
> - `docs/speed/binary_size/metrics.md`
> - `docs/speed/binary_size/binary_size_explainer.md`
> - `docs/speed/binary_size/android_binary_size_trybot.md`

---

## 1. trybot 判定规则（划红线）

来自 `android_binary_size_trybot.md`：

- **arm32 normalized APK size 增量上限 16 KB**（arm64 是 64 KB）。
- 触发失败需在 commit footer 用 `Binary-Size: <理由>` 显式覆盖才能放行。

来自 `metrics.md` 对 normalized size 的算法：

- Native code 按 ELF section 求和、**按未压缩计**（去掉 zipalign 噪声）。
- 测量目标是 `TrichromeChrome.aab`；**App Bundle 的 normalized = 所有 `onDemand="false"` split 之和**。

---

## 2. 关键事实：renderer 只加载 base split

`binary_size_explainer.md`：

> **base split**: Loaded by every process including renderers. Keeping its dex size minimal is crucial, since it has both RAM and start-up overhead per-renderer.
> **chrome feature split**: Loaded only by the browser process at startup.

Trichrome 的 split 结构里：
- **base split** 的 `libchrome.so` 被**每个进程**加载（含 renderer）。
- **chrome split** 的 `libchrome.so` 只被 browser/特权进程加载。

所以"native 代码归到哪个 split"由它**是否被 renderer 入口可达**决定。
一旦 renderer 直接调用某段代码，它就必须出现在 base split 的 `libchrome.so` 里。

---

## 3. CL 7785089（TFLite，0 增量）做了什么

```
Subject: webnn: Route CPU/NPU requests to in-renderer TFLite backend
Files (≈170 行)：
  services/webnn/BUILD.gn                                +6
  services/webnn/tflite/context_provider_tflite.{h,cc}   +5 / +28-2
  services/webnn/webnn_context_provider_impl.{h,cc}      +1 / +33-2
  services/webnn/webnn_test_environment.cc               +99-1
```

**纯路由变更**：新增 `ShouldUseInProcessTflite()` helper，把 webnn 在 renderer 这一侧的 IPC 终点指向**已经驻留 base.so 的 TFLite 实现**。
→ 没有引入任何新 third_party 依赖，链接闭包不变，APK size 不变。

### 它能这么做的前提：TFLite 早已在 base.so

在 `chromium/src` 中 `grep -rln 'third_party/tflite"' --include=BUILD.gn` 命中的 renderer 端用户（节选）：

```
chrome/renderer/BUILD.gn
components/safe_browsing/content/renderer/phishing_classifier
components/translate/core/language_detection
components/language_detection/core
components/omnibox/browser
components/autofill/core/browser
media/webrtc
third_party/webrtc/api/audio
third_party/webrtc/modules/audio_processing/aec3/...
third_party/mediapipe
third_party/tensorflow_models
third_party/tflite_support
```

由于这些 component 长期被 renderer 加载，base.so 中早就驻留：
TFLite runtime/interpreter、`tflite_builtin_op_resolver`、`tflite_kernels`、`tflite_kernel_internals`、ruy、gemmlowp、farmhash、fft2d、fp16、neon_2_sse、flatbuffers、absl、protobuf、eigen headers、xnnpack、pthreadpool、cpuinfo。

---

## 4. CL 8ff7343（LiteRT，arm32 失败）做了什么

把 LiteRT runtime 也移到 renderer。

### LiteRT 在 chromium 里的唯一直接消费者

```
$ grep -rln 'third_party/litert"' --include=BUILD.gn chromium/src
services/webnn/BUILD.gn         ← 唯一一处
```

`services/webnn/BUILD.gn:196-201`：

```python
if (webnn_use_litert) {
  deps += [
    "//third_party/litert",
    "//third_party/litert:buildflags",
  ]
}
```

之前 webnn 是独立 utility 进程 → LiteRT 落在 chrome split / utility 路径下，**base.so 中没有 LiteRT**。

### 改到 renderer 后被链入 base.so 的闭包

`third_party/litert/BUILD.gn` 里 `group("litert")` 拢起 5 个静态库：`litert_c / litert_compiler / litert_core / litert_runtime / litert_runtime_accelerators`，再向下扇出：

| 第一阶 | 第二阶（间接） |
|---|---|
| `litert_runtime` | `tflite`、`tflite_builtin_op_resolver`、`xnnpack`、`pthreadpool` |
| `tflite` / `tflite_kernels` / `tflite_kernel_internals` | `absl`、`flatbuffers`、`farmhash`、`fft2d`、`fp16`、`neon_2_sse`、`ruy`、`gemmlowp`、`eigen headers`、`cpuinfo`（条件） |
| `litert_headers` | `absl`、`flatbuffers`、`zlib`、`tflite_proto`（间接 protobuf） |
| `tflite_litert`（experimental/genai/resource） | tflite 内核头 |
| `mutable_tflite_schema`、`weight_cache_schema_litert` | flatbuffers schema |

**全部进 base.so**。同时还包含 LiteRT 自身的全部 `.cc`：`src/litert/c/*`、`src/litert/runtime/*`、`src/litert/compiler/*`、`src/litert/core/*`、`src/weight_loader/*` 等。

---

## 5. 为什么会"在 base.so 与 chrome.so 中各保留一份"

`tflite_builtin_op_resolver` 的 `visibility` 列表（`third_party/litert/BUILD.gn`）显式包含：

```
//components/*           ← 含 components/optimization_guide/internal
//modules/*
//services/webnn/*
//third_party/litert:*
//third_party/mediapipe/*
//third_party/tflite:*
//third_party/webrtc/modules/*
```

也就是说，`optimization_guide`、`mediapipe`、`webrtc/modules`、`services/on_device_model/ml`（`enable_ml_internal`）等用户**在 chrome.so 中仍然引用** TFLite/ruy/xnnpack/absl/protobuf/flatbuffers 等。

**Native 链接器是按 link 单元跑 `--gc-sections`**，base.so 与 chrome.so 是两次独立 link，没有 R8 那种"跨 split 公共池"。所以：

- LiteRT 拉到 base.so → 它的依赖闭包在 base.so 中**完整**链接一份。
- chrome.so 中的 optimization_guide 等用户仍然需要这些依赖 → 在 chrome.so 中**仍然**链接一份。
- **两个 .so 各自独立持有同一段 absl/ruy/xnnpack/...**。这不是"被错算两次"，是 split 模型下两份 `.so` 各自完整链接的物理结果——APK 解压后磁盘上确实是两份字节，安装到设备上也是两份不同 inode 的 mmap。

---

## 6. 量级直觉与 arm32 阈值

| 增量来源 | 数量级（arm32） |
|---|---|
| LiteRT 自身（litert/* + 新增 schema/proto） | ~数百 KB |
| `tflite_kernels`（100+ 个 kernel `.cc`） | ~数百 KB |
| `tflite_kernel_internals` | ~几十 KB |
| ruy + gemmlowp + xnnpack（含上千个 micro-kernel） | **MB 级** |
| absl 模板膨胀 + protobuf + flatbuffers | 几百 KB |

阈值 **16 KB**。即便扣除 chrome.so 端可缩减的部分，base.so 端单边新增依然远超阈值。

`android_binary_size_trybot.md` 还指出 arm32 用 `-Os + AFDO`、arm64 用 `-O2 + PGO`；arm32 阈值（16 KB）也比 arm64（64 KB）紧 4 倍——所以 trybot 的 `Binary_Size_Details__arm32_` 报告先红是必然。

---

## 7. 两个 CL 对比

| 维度 | CL 7785089（TFLite） | CL 8ff7343（LiteRT） |
|---|---|---|
| 改动体量 | 6 文件、≈170 行 | 显著更大，新增 deps |
| 是否引入新 third_party 依赖到 renderer 闭包 | 否 | 是 |
| Native 链接闭包是否变化 | 不变 | 变大 |
| 新进 base.so 的代码 | 0 | LiteRT 全部 + tflite_litert 扩展 + 新增 schema/proto + 间接闭包 |
| 是否触发 base/chrome 重复链接 | 否 | 是 |
| arm32 size trybot | 通过 | 失败 |

---

## 8. 为什么 chromium 当初没把 LiteRT 放进 base

TFLite 进 base 是历史路径：phishing classifier、translate、language detection、omnibox、autofill、safe_browsing 这些 renderer 内置 ML 功能多年使用 TFLite，base 闭包早已包含。

LiteRT 是 Google 新一代 runtime，路线规划里它的目标进程是**独立 utility / on-device-model service**。CL 8ff7343 把它强行下到 base，正面打破这条边界 → trybot 立刻拦下。

---

## 9. 可行的缓解方向

1. **把 LiteRT 留在独立 utility 进程**（保持现状）—— 与 WebNN model-compilation utility 同样的设计哲学：单一职责进程 + 紧沙箱 + 不污染 base.so。
2. 若必须从 renderer 直接访问，考虑：
   - 通过 Mojo IPC 调用驻留在 utility 进程的 LiteRT（不在 renderer 链接 LiteRT）。
   - 评估能否把 LiteRT 拆到 **on-demand DFM**——但 renderer 在沙箱内无法触发 PlayCore 拉取 DFM，路径上限多。
3. 如果坚持下沉到 base，需要：
   - 推动 chrome.so 端用户（optimization_guide / mediapipe / webrtc）一并迁移，让两边的依赖能合并到 base，避免重复链接。
   - 用 `Binary-Size: <理由>` footer 明确豁免，并接受 per-renderer 启动 + RAM 成本由 explainer 指出的"base split 多出的代码每个 renderer 都付一次"。

---

## 10. 一句话总结

> **TFLite 及其底座（absl/ruy/xnnpack/flatbuffers/protobuf）在改动前已经在 base.so 中**（chrome/renderer、phishing_classifier、translate 等老用户带进去的），CL 7785089 只是改路由、不动闭包，所以 0 增量；
> **LiteRT 改动前只被 `services/webnn` 唯一引用、base.so 中没有它**，CL 8ff7343 把它放进 renderer 后，LiteRT 自身代码 + 它在 base.so 中新引入的依赖闭包 + 与 chrome.so 现有用户形成的两份重复链接，三重叠加把 arm32 normalized size 推过 **16 KB** 阈值，trybot 失败。
