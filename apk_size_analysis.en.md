# Android APK Size Analysis: Why moving LiteRT CPU inference into the renderer process fails the size trybot

> Related changes:
> - `8ff7343475ab8573e943016312958726ada7948a` — moves LiteRT CPU inference into the renderer (**arm32 size trybot fails**)
> - `chromium-review.googlesource.com/c/chromium/src/+/7785089` — routes WebNN CPU/NPU requests to an in-renderer TFLite backend (**zero size impact, already merged**)
>
> Reference docs:
> - `docs/speed/binary_size/metrics.md`
> - `docs/speed/binary_size/binary_size_explainer.md`
> - `docs/speed/binary_size/android_binary_size_trybot.md`

---

## 1. Trybot rules (the red lines)

From `android_binary_size_trybot.md`:

- **Normalized APK size growth must stay under 16 KB on arm32**, 64 KB on arm64.
- A failure can only be overridden by adding a `Binary-Size: <rationale>` footer to the commit.

From `metrics.md`, on how the normalized size is computed:

- Native code is summed across ELF sections **as if uncompressed**, with zipalign noise stripped out.
- The measurement target is `TrichromeChrome.aab`. **For App Bundles, normalized size = sum of normalized sizes of every split with `onDemand="false"`.**

---

## 2. Key fact: the renderer only loads the base split

From `binary_size_explainer.md`:

> **base split**: Loaded by every process including renderers. Keeping its dex size minimal is crucial, since it has both RAM and start-up overhead per-renderer.
> **chrome feature split**: Loaded only by the browser process at startup.

So in the Trichrome split topology:
- The **base split**'s `libchrome.so` is loaded by **every process**, including the renderer.
- The **chrome split**'s `libchrome.so` is loaded only by the browser / privileged processes.

Which split a piece of native code lands in is therefore determined by **whether it is reachable from a renderer entry point**. The moment the renderer calls into a function directly, that function — and its entire static-link closure — must appear in the base split's `libchrome.so`.

---

## 3. What CL 7785089 (TFLite, zero growth) actually does

### Why it can do this: TFLite is already in base.so

Running `grep -rln 'third_party/tflite"' --include=BUILD.gn` in `chromium/src` lights up these renderer-side users (excerpted):

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

Because these components are loaded by the renderer in shipping Chrome today, base.so has long contained:
TFLite runtime/interpreter, `tflite_builtin_op_resolver`, `tflite_kernels`, `tflite_kernel_internals`, ruy, gemmlowp, farmhash, fft2d, fp16, neon_2_sse, flatbuffers, absl, protobuf, eigen headers, xnnpack, pthreadpool, cpuinfo.

---

## 4. What CL 8ff7343 (LiteRT, arm32 fails) actually does

It moves the LiteRT runtime into the renderer.

### LiteRT has a single direct consumer in chromium

```
$ grep -rln 'third_party/litert"' --include=BUILD.gn chromium/src
services/webnn/BUILD.gn         ← the only entry
```

`services/webnn/BUILD.gn:196-201`:

```python
if (webnn_use_litert) {
  deps += [
    "//third_party/litert",
    "//third_party/litert:buildflags",
  ]
}
```

Before the change, `services/webnn` ran in an GPU process, so LiteRT lived on the chrome split / GPU path — **base.so contained no LiteRT at all.**

### What gets pulled into base.so once LiteRT moves into the renderer

In `third_party/litert/BUILD.gn`, `group("litert")` aggregates five static libraries: `litert_c / litert_compiler / litert_core / litert_runtime / litert_runtime_accelerators`. Their fan-out:

| First level | Transitive |
|---|---|
| `litert_runtime` | `tflite`, `tflite_builtin_op_resolver`, `xnnpack`, `pthreadpool` |
| `tflite` / `tflite_kernels` / `tflite_kernel_internals` | `absl`, `flatbuffers`, `farmhash`, `fft2d`, `fp16`, `neon_2_sse`, `ruy`, `gemmlowp`, `eigen` headers, `cpuinfo` (conditional) |
| `litert_headers` | `absl`, `flatbuffers`, `zlib`, `tflite_proto` (transitively `protobuf`) |
| `tflite_litert` (experimental/genai/resource) | TFLite kernel headers |
| `mutable_tflite_schema`, `weight_cache_schema_litert` | flatbuffers schemas |

**All of that lands in base.so**, alongside LiteRT's own `.cc` files: `src/litert/c/*`, `src/litert/runtime/*`, `src/litert/compiler/*`, `src/litert/core/*`, `src/weight_loader/*`, etc.

---

## 5. Why dependencies end up duplicated in base.so AND chrome.so

The `visibility` list of `tflite_builtin_op_resolver` (`third_party/litert/BUILD.gn`) explicitly includes:

```
//components/*           ← includes components/optimization_guide/internal
//modules/*
//services/webnn/*
//third_party/litert:*
//third_party/mediapipe/*
//third_party/tflite:*
//third_party/webrtc/modules/*
```

In other words, users like `optimization_guide`, `mediapipe`, `webrtc/modules`, and `services/on_device_model/ml` (`enable_ml_internal`) **continue to reference TFLite/ruy/xnnpack/absl/protobuf/flatbuffers from chrome.so**.

**The native linker runs `--gc-sections` per link unit.** base.so and chrome.so are two independent link operations; there is no R8-style cross-split common pool. Therefore:

- LiteRT pulled into base.so → its dependency closure is **fully** linked into base.so.
- The optimization_guide / mediapipe / webrtc users in chrome.so still need the same dependencies → those dependencies are **still** linked into chrome.so.
- **The two `.so` files independently each carry the same absl/ruy/xnnpack/... bytes.** This isn't a measurement bug — it's the physical result of having two separate static-link units in the split model. After the APK is unpacked the bytes really are duplicated on disk, and at install time they map into memory as two different inodes.

---

## 6. Magnitudes vs the arm32 threshold

| Source of growth | Order of magnitude (arm32) |
|---|---|
| LiteRT itself (`litert/*` + new schemas/protos) | ~hundreds of KB |
| `tflite_kernels` (100+ kernel `.cc` files) | ~hundreds of KB |
| `tflite_kernel_internals` | ~tens of KB |
| ruy + gemmlowp + xnnpack (thousands of micro-kernels) | **MB-class** |
| absl template bloat + protobuf + flatbuffers | ~hundreds of KB |

The threshold is **16 KB**. Even subtracting whatever can be trimmed on the chrome.so side, the net growth in base.so alone overshoots the limit by orders of magnitude.

`android_binary_size_trybot.md` also notes that arm32 builds with `-Os + AFDO` while arm64 builds with `-O2 + PGO`, and the arm32 threshold (16 KB) is 4× tighter than arm64 (64 KB). That's why the trybot's `Binary_Size_Details__arm32_` report fails first.

---

## 7. Side-by-side comparison

| Aspect | CL 7785089 (TFLite) | CL 8ff7343 (LiteRT) |
|---|---|---|
| Change size | 6 files, ~170 lines | substantially larger, adds new deps |
| Introduces new third_party deps to the renderer closure? | No | Yes |
| Native link closure changes? | No | Yes, grows |
| Code newly entering base.so | 0 | All of LiteRT + tflite_litert extensions + new schemas/protos + transitive closure |
| Triggers base/chrome duplication? | No | Yes |
| arm32 size trybot | passes | fails |

---

## 8. Why chromium didn't put LiteRT in base in the first place

TFLite ended up in base for historical reasons: phishing classifier, translate, language detection, omnibox, autofill, and safe_browsing have all been using TFLite from the renderer for years, so the base closure absorbed it long ago.

LiteRT is Google's next-generation runtime CL 8ff7343 forces it down into base, breaking that boundary head-on — and the trybot enforces the boundary.

---

## 10. Bottom line

> **TFLite and its underlying stack (absl/ruy/xnnpack/flatbuffers/protobuf) were already in base.so before the change**, brought in by long-standing renderer users such as `chrome/renderer`, `phishing_classifier`, and `translate`. CL 7785089 only changes routing — the link closure is untouched — so growth is zero.
> **LiteRT, by contrast, had a single consumer (`services/webnn`) and was absent from base.so before the change.** Once CL 8ff7343 places it in the renderer, the base.so closure absorbs all of LiteRT's own code, its newly-imported dependency closure, and a duplicated copy of dependencies that chrome.so still needs for its existing users. Those three forces combine to push the arm32 normalized size past the **16 KB** threshold, and the trybot fails.
