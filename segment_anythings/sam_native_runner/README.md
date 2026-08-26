# SAM Encoder Native Runner — 使用说明

> 这是**使用文档**（构建 / 命令行 / 示例）。设计原理与迭代历史见 **[DESIGN.md](DESIGN.md)**。
>
> runner 是一个独立 native 程序：直接加载 tflite 模型，复刻 chromium `WebNN→LiteRT`
> 编译路径（无浏览器），与 chrome 共用同一批 `//third_party/litert` + tflite 编译产物。
> 代码在 chromium 主树 `services/webnn/tflite/sam_runner/`。

---

## 1. 构建

```powershell
cd C:\Users\junweifu\workspace\chromium\src
autoninja -C out\Release services/webnn/tflite/sam_runner:sam_encoder_runner
# 产物：out\Release\sam_encoder_runner.exe
```

- 首次：编 1 个 `.cc` + 链小 exe（litert/base 的 .obj 早已存在），几十秒；
- 增量：改 runner 源码秒级重链；**不触碰 chrome.dll**。
- 前提：`out\Release\` 下已有 `libLiteRtWebGpuAccelerator.dll` + `webgpu_dawn.dll`
  （由 `route-a-webgpu-windows` 流程部署，exe 同目录自动找到）。

---

## 2. 命令行参数

```
sam_encoder_runner --model=<tflite>
    [--run] [--runs=N] [--precision=fp16|fp32]
    [--input=<f32 bin>] [--buffer-storage-patterns=<p1,p2,...>]
    | --verify [--precision=fp16|fp32] [--tolerance=N]
      [--dump-outputs=<path prefix>] [--input=<f32 bin>]
      [--buffer-storage-patterns=<p1,p2,...>]
```

| 参数 | 说明 |
|---|---|
| `--model=<path>` | tflite 模型路径（必填） |
| `--run` | 编译后跑推理（默认只编译） |
| `--runs=N` | 推理次数（默认 1） |
| `--precision=fp16\|fp32` | GPU 精度（默认 fp16） |
| `--verify` | GPU vs CPU(XNNPACK, 固定 fp32) 双跑逐元素对比 |
| `--tolerance=N` | 对比容差（默认 1e-2，fp16 的合理值） |
| `--dump-outputs=<prefix>` | 完整输出写 `<prefix>_cpu.bin` / `_gpu.bin` / `_run.bin`（little-endian f32） |
| `--input=<f32 bin>` | 从文件读输入（替代默认 ramp；大小必须精确 = `input_elems*4` 字节） |
| `--buffer-storage-patterns=<p1,p2,...>` | 强制指定 buffer storage 模式（调试/规避 readback 用） |

**退出码约定**：`0`=PASS · `1`=错误 · `2`=FAIL · `3`=GPU 读回失败（WARP 无法判定，CPU 侧结果仍有效）。

> 默认模式 = GPU + CPU fallback（与 WebNN 一致）。`--verify` 的 CPU 侧固定 fp32 作 ground truth。

---

## 3. 常用用法

### 3.1 编译检查（看是否全量下沉 GPU / 复现 OOM）

```powershell
out\Release\sam_encoder_runner.exe --model=C:\...\new_segment_anything_encoder.tflite
# 关键日志：IsFullyAccelerated=1 / All N operations are supported by GPU delegate.
```

### 3.2 编译 + 推理

```powershell
out\Release\sam_encoder_runner.exe --model=C:\...\model.tflite --run --runs=5
```

### 3.3 CPU vs GPU 验证（`--verify`）

```powershell
# fp16（Chrome 默认精度）
out\Release\sam_encoder_runner.exe --model=C:\...\model.tflite --verify

# fp32（消除 fp16 舍入，定位结构性问题）
out\Release\sam_encoder_runner.exe --model=C:\...\model.tflite --verify --precision=fp32

# 调容差
out\Release\sam_encoder_runner.exe --model=C:\...\model.tflite --verify --tolerance=1e-2
```

### 3.4 外部输入 + dump 完整输出（离线 diff）

```powershell
out\Release\sam_encoder_runner.exe --model=C:\...\model.tflite --verify `
    --input=C:\...\input.bin --dump-outputs=C:\...\sam_enc
# 产出 sam_enc_cpu.bin / sam_enc_gpu.bin

out\Release\sam_encoder_runner.exe --model=C:\...\model.tflite --run `
    --input=C:\...\input.bin --dump-outputs=C:\...\sam_enc_run
# 产出 sam_enc_run.bin
```

输入 bin 生成（row-major f32，与模型输入张量一致）：

```python
import numpy as np
x = preprocessed.astype(np.float32)   # 如 SAM encoder: (1,3,1024,1024) NCHW
x.tofile(r'C:\...\input.bin')
```

离线 diff（numpy）：

```python
import numpy as np
cpu = np.fromfile(r'C:\...\sam_enc_cpu.bin', dtype=np.float32)
gpu = np.fromfile(r'C:\...\sam_enc_gpu.bin', dtype=np.float32)
d = np.abs(gpu - cpu)
print('elems', cpu.size, 'max_abs', d.max(), 'mean_abs', d.mean())
idx = np.argwhere(d > 1e-2).ravel()
print('over_tol', idx.size, 'first_bad', idx[:20])

d4 = d.reshape(256, 64, 64)          # 例：SAM encoder 输出 1x256x64x64，按通道定位
per_ch = d4.mean(axis=(1, 2))
print('worst channels', np.argsort(per_ch)[::-1][:10])
```

### 3.5 图像输入预处理（SAM / MobileNet）

`sam_native_runner/` 下的脚本先把图转成模型的预处理输入，再用 numpy 转 f32 bin：

- `prepare_sam_input.ps1`：SAM —— resize 长边 1024（bilinear）+ 右下补黑 → `sam_enc_1024_bgr.bin`，
  再 `BGR→RGB → (x-mean)/std → NCHW f32` → `sam_enc_input.bin`
- `prepare_mobilenet_input.ps1`：MobileNet —— tiger.jpg 拉伸 224×224(bicubic) → BGR，
  再 `BGR→RGB → (x/127.5-1) → NCHW f32` → `tiger_input.bin`
- `inspect_tflite.py <model.tflite>`：无 tensorflow 时读模型输入/输出形状与类型
- `view_bin.py <file.bin> 1,256,64,64 [--count=100]`：查看 f32 dump 的 NaN/Inf、min/max/mean/std、前 N 个值（默认 16）

### 3.6 SAM encoder 专项：CPU（旧模型）vs GPU（新模型）对比

> 场景：`new_segment_anything_encoder.tflite` 在 GPU 上结果不对，而 CPU 正确。
> 用**旧模型 `segment_anything_encoder.tflite` 跑 CPU 作参考**，用**新模型跑 GPU**，
> 喂同一份输入 `sam_enc_input.bin`（1×3×1024×1024 f32），`--precision=fp32` 消除 fp16 舍入干扰。

```powershell
$CR    = "C:\Users\junweifu\workspace\chromium\src"
$OUT   = "C:\Users\junweifu\workspace\webnn\segment_anythings\sam_native_runner\segment_anythings_empty_result"
$M_CPU = "C:\Users\junweifu\workspace\tflite-dump-model\segment_anything_encoder.tflite"     # 旧模型（CPU 参考）
$M_GPU = "C:\Users\junweifu\workspace\tflite-dump-model\new_segment_anything_encoder.tflite" # 新模型（GPU 待验证）
$INPUT = "$OUT\sam_enc_input.bin"
$RUN   = "$CR\out\Release\sam_encoder_runner.exe"
```

**① CPU 参考（旧模型；任意机器可跑）**

```powershell
& $RUN --model=$M_CPU --verify --precision=fp32 --input=$INPUT --dump-outputs=$OUT\sam_cpu
# 产出：$OUT\sam_cpu_cpu.bin  （旧模型 CPU fp32 参考）
# 注：WARP 上旧模型的 GPU 读回会失败 → 退出码 3，只取 _cpu.bin 即可，不影响。
# 本机实测（WARP）：nan=0, min=-1.07555, max=0.814412, mean=0.0128696, std=0.15606
```

**② GPU（新模型；必须真 GPU 机器，WARP 读不回 1M 元素）**

```powershell
& $RUN --model=$M_GPU --verify --precision=fp32 --input=$INPUT --dump-outputs=$OUT\sam_gpu
# 产出：$OUT\sam_gpu_gpu.bin  ← 要对比的 GPU 结果
#        $OUT\sam_gpu_cpu.bin  （新模型 CPU fp32，可顺带对比两个模型的 CPU 差异）
# WARP 上只有 _cpu.bin；_gpu.bin 需在有真 GPU 的机器上跑（命令完全相同）。
```

**③ 离线对比（旧模型 CPU vs 新模型 GPU）**

```python
import numpy as np
cpu = np.fromfile(r'C:\...\segment_anythings_empty_result\sam_cpu_cpu.bin', dtype=np.float32)  # 旧模型 CPU
gpu = np.fromfile(r'C:\...\segment_anythings_empty_result\sam_gpu_gpu.bin', dtype=np.float32)  # 新模型 GPU
print('elems', cpu.size, 'gpu_nan', int(np.isnan(gpu).sum()))
d = np.abs(gpu - cpu)
print('max_abs', d.max(), 'mean_abs', d.mean(), 'over_tol(1e-2)', int((d > 1e-2).sum()))
d4 = d.reshape(256, 64, 64)                     # 输出 1x256x64x64，按通道定位差异
per_ch = d4.mean(axis=(1, 2))
print('worst channels', np.argsort(per_ch)[::-1][:10])
```

**④ view_bin 查看前 100 个值**

```powershell
C:/Python314/python.exe `
  C:\Users\junweifu\workspace\webnn\segment_anythings\sam_native_runner\gpu_run_package\tools\view_bin.py `
  C:\...\segment_anythings_empty_result\sam_cpu_cpu.bin 1,256,64,64 --count=100
# 打印：elems / NaN/Inf / min/max/mean/std / per-channel mean / 前 100 个值
```

---

## 4. 已知限制 / 当前状态

- **本机 adapter = WARP**（Microsoft Basic Render Driver）。SAM encoder 的 1M 元素输出
  `Lock(kRead)` 回读失败 → `--verify` 退出码 3（CPU 侧仍有效）。
- **真 GPU adapter 注入未做**（`kLiteRtEnvOptionTagWebGpuInstance`）：WARP 下无法读回大输出；
  有真 GPU 的机器上重新部署即可跑通 `--verify` 的 GPU 侧。
- **已知 bug（本机 WARP）**：`ml_drfit_add.tflite` 的 GPU 输出第 3、4 个元素错
  （const 广播问题，`--precision=fp32` 也一样 → 结构性问题）。
- **已知问题（远程真 GPU）**：SAM encoder GPU 输出曾全 NaN —— 待用 `--precision=fp32`
  区分是 fp16 还是结构性问题。
- **已验证通过**：`mobilenet.tflite` fp32 GPU vs CPU **PASS**（max_abs≈5e-4，`over_tol=0`），
  top-1/top-5 完全一致；fp16 下 top-1/top-5 也一致（数值有 ~0.05 舍入）。

---

## 5. 参考

- 设计原理 / 与 chrome 路径差异 / 风险清单 / 迭代历史：**[DESIGN.md](DESIGN.md)**
- 构建部署整个 GPU 栈（accelerator DLL / Dawn / mini_installer）：`../route-a-webgpu-windows/README.md`
