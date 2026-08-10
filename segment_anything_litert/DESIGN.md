# Segment Anything — LiteRT.js + WebNN 设计与分析文档

## 1. 背景与问题

### 1.1 现有方案：ONNX Runtime Web + WebNN

微软的 [WebNN Developer Preview](https://microsoft.github.io/webnn-developer-preview/demos/segment-anything/) 提供了一个 Segment Anything 演示，使用 ONNX Runtime Web 加载 SAM ONNX 模型，通过 WebNN execution provider 加速推演。

```
浏览器 → ONNX Runtime Web → WebNN EP → Chromium WebNN → ml-drift GPU delegate → GPU
```

### 1.2 OOM 问题分析

在 `C:\Users\junweifu\workspace\webnn\segment_anythings\analysis.md` 中已经详细分析了 OOM 的根因：

1. **直接原因**：Chromium 中 `patches/06-compiled-model-disable-optimize-memory.patch` 注释掉了 `OptimizeMemoryForLargeTensors(1 << 20)` 调用
2. **为什么注释**：`kTfLiteDynamic` 张量会被静态形状 delegate（ml-drift）拒绝，导致 `"only supports static-sized tensors"` 错误
3. **后果**：所有大型激活张量留在 `arena_` 内存池中，SAM encoder 的 packed high-water-mark 超过 2 GiB，触发 `PartitionExcessiveAllocationSize`

```
ONNX → ORT WebNN EP → ml-drift → TFLite delegate → 静态形状约束
→ 禁用 OptimizeMemoryForLargeTensors → arena 膨胀 → OOM
```

### 1.3 为什么尝试 LiteRT.js

LiteRT.js 是 Google 的原生 TFLite Web 运行时，直接加载 `.tflite` 模型。相比 ONNX 路径：

| 层级 | ONNX Runtime | LiteRT.js |
|------|-------------|-----------|
| 模型格式 | ONNX → ORT 内部表示 | TFLite（原生） |
| 算子映射 | ONNX ops → WebNN ops | TFLite ops → WebNN ops（ml-drift 原生支持） |
| 中间层 | ORT graph optimizer + partitioning | 直接 TFLite → ml-drift |
| 内存管理 | ORT arena + TFLite arena（双层） | TFLite arena（单层） |

理论上，LiteRT.js 路径减少了算子翻译层级，且单层内存管理可能降低 OOM 风险。但这需要实测验证。

## 2. 模型选择

### 2.1 候选模型

#### Full SAM (`qualcomm/Segment-Anything-Model`)

| 文件 | 大小 | 说明 |
|------|------|------|
| `SAMEncoder.tflite` | 368 MB | 完整 encoder |
| `SAMEncoderPart1-6.tflite` | ~60 MB × 6 | 拆分 encoder |
| `SAMDecoder.tflite` | 24.8 MB | Decoder |

**问题**：拆分 encoder 的 Part 6 输出 shape `[1, 64, 768, 64]` 与 decoder 期望的输入 shape `[1, 64, 64, 256]` 不匹配：
- Part 6 输出缺少 neck（1×1 conv 768→256），且 layout 不同
- 完整 368MB encoder 太大，且 OOM 风险高

#### MobileSAM (`qualcomm/MobileSam`)

| 文件 | 大小 | 说明 |
|------|------|------|
| `encoder.tflite` | 26.6 MB | ViT-Tiny encoder |
| `decoder.tflite` | 23.7 MB | 轻量 decoder |

**优势**：
- 总计仅 ~50 MB，适合浏览器加载
- Encoder 输出 `[1, 64, 64, 256]` 与 decoder 输入完全匹配
- 单文件 encoder，无需拆分拼接
- ViT-Tiny 架构，参数量小，内存压力低

**默认使用 MobileSAM**（快速启动、低内存压力），同时支持通过 URL 参数切换到 full SAM 以测试 OOM 场景。

### 2.2 模型 Tensor 详情

通过 `tflite` Python 包解析获得。

#### MobileSAM Encoder / Decoder
```
Encoder:
  输入: image              shape=[1, 1024, 1024, 3]  float32, NHWC, RGB [0,1]
  输出: image_embeddings   shape=[1, 64, 64, 256]    float32

Decoder:
  输入: image_embeddings   shape=[1, 64, 64, 256]    float32
  输入: point_coords       shape=[1, 1, 2]            float32  (x,y) 在 1024×1024 空间
  输入: point_labels       shape=[1, 1]               float32  (1=前景, 0=背景)
  输出: masks              shape=[1, 256, 256, 1]     float32  (已过 sigmoid, [0,1])
  输出: scores             shape=[1, 1]               float32
```

#### Full SAM Encoder / Decoder
```
Encoder (SAMEncoder.tflite, 368 MB):
  输入: image              shape=[1, 1024, 1024, 3]  float32, NHWC, RGB [0,1]
  输出: image_embeddings   shape=[1, 64, 64, 256]    float32  ← 与 MobileSAM 一致

Decoder (SAMDecoder.tflite, 24.8 MB):
  输入: image_embeddings   shape=[1, 64, 64, 256]    float32
  输入: point_coords       shape=[1, 2, 2]            float32  ← 2 个点
  输入: point_labels       shape=[1, 2]               float32  ← 2 个标签
  输出: masks              shape=[1, 256, 256, 1]     float32
  输出: scores             shape=[1, 1]               float32
```

**关键差异**：full SAM decoder 的 `point_coords` / `point_labels` 第二维是 2（2 个点），MobileSAM 是 1（1 个点）。代码通过 `getInputDetails()` 自动检测 decoder 支持的点数，无需手动配置。

## 3. 架构设计

### 3.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        浏览器 (Chromium)                             │
│                                                                      │
│  ┌──────────┐    ┌──────────────────────┐    ┌──────────────────┐  │
│  │ 用户上传  │    │  LiteRT.js Runtime    │    │  WebNN API       │  │
│  │ 图片      │    │  (@litertjs/core)     │    │  (navigator.ml)  │  │
│  └────┬─────┘    └──────────┬───────────┘    └────────┬─────────┘  │
│       │                     │                         │             │
│       │  Canvas 预处理       │                         │             │
│       │  1024×1024 RGB       │                         │             │
│       │  Float32 [0,1]       │                         │             │
│       │                     │                         │             │
│       ▼                     ▼                         ▼             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    推演流程                                    │   │
│  │                                                               │   │
│  │  Image → Encoder → image_embeddings [1,64,64,256]              │   │
│  │                                       │                       │   │
│  │  用户点击 → point_coords [1,1,2]      │                       │   │
│  │           → point_labels [1,1]        ▼                       │   │
│  │           → Decoder → masks [1,256,256,1]                     │   │
│  │                    → scores [1,1]                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────┐                                                     │
│  │ Canvas    │                                                     │
│  │ 渲染      │  256×256 mask → 双线性放大 → 绿色半透明叠加          │
│  └──────────┘                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 组件划分

```
segment_anything_litert/
├── index.html          # 页面结构 + Import Map
├── index.js            # 推演逻辑
│   ├── WebNN 检测      # navigator.ml.createContext()
│   ├── LiteRT 初始化   # loadLiteRt(wasmUrl, {jspi: true})
│   ├── 模型加载        # loadAndCompile(url, {accelerator: 'webnn'})
│   │   ├── URL 参数解析  # ?models= & ?encoder= & ?decoder=
│   │   └── 自动检测点数  # getInputDetails() → maxPoints
│   ├── 图像预处理      # Canvas resize → Float32Array [0,1]
│   ├── Encoder 推演    # encoder.run({image: tensor})
│   ├── Decoder 推演    # decoder.run({image_embeddings, point_coords, point_labels})
│   │   └── 自适应点数  # 取最近 maxPoints 个点, 不足时 padding
│   └── Mask 渲染       # 256×256 → display size, 绿色叠加
├── server.js           # 开发服务器 (COOP/COEP headers)
├── models/
│   ├── encoder.tflite      # MobileSAM encoder (26.6 MB, 默认)
│   ├── decoder.tflite      # MobileSAM decoder (23.7 MB, 默认)
│   ├── SAMEncoder.tflite   # Full SAM encoder (368 MB, 需下载)
│   └── SAMDecoder.tflite   # Full SAM decoder (24.8 MB, 需下载)
├── DESIGN.md           # 本设计文档
├── README.md           # 使用说明
└── inspect_model.py    # TFLite 模型检查工具
```

### 3.3 双模型切换机制

通过 URL 查询参数在 MobileSAM 和 full SAM 之间切换：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `?models=` | `./models/` | 模型目录 URL 前缀 |
| `?encoder=` | `encoder.tflite` | Encoder 文件名 |
| `?decoder=` | `decoder.tflite` | Decoder 文件名 |

**MobileSAM（默认）**:
```
http://localhost:8080/
```

**Full SAM（OOM 测试）**:
```
http://localhost:8080/?encoder=SAMEncoder.tflite&decoder=SAMDecoder.tflite
```

代码在 decoder 加载完成后调用 `decoderModel.getInputDetails()` 读取 `point_labels` 的 shape `[1, N]` 获取 `maxPoints`：
- MobileSAM: `N=1` → 每次仅使用最后一个点击
- Full SAM: `N=2` → 每次使用最近 2 个点击，不足时 padding

这样实现了一处代码同时支持两种模型，无需手动切换逻辑。

### 3.4 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 默认模型 | MobileSAM (50MB) | 快速启动，低内存压力 |
| OOM 测试模型 | Full SAM (393MB) | 通过 `?encoder=SAMEncoder.tflite` 切换 |
| 加速后端 | `accelerator: 'webnn'` | 使用 ml-drift GPU delegate |
| 设备偏好 | `devicePreference: 'gpu'` | 目标测试 GPU 路径 |
| JSPI | `{jspi: true}` | WebNN 必需的异步桥接 |
| 模块加载 | Import Map + jsDelivr CDN | 无需构建工具，浏览器原生 ESM |
| 跨域隔离 | `COEP: credentialless` | 支持 SharedArrayBuffer + CDN 资源 |
| 图片尺寸 | 1024×1024（长边缩放 + 居中填充） | 两种模型共用此输入尺寸 |
| 坐标空间 | 1024×1024 模型空间 | 与 encoder 输入一致 |
| 自适应点数 | 从 decoder input shape 自动检测 | 一套代码同时支持 Mobile (1点) 和 Full (2点) |

## 4. 实现细节

### 4.1 LiteRT.js 初始化

```javascript
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';

// JSPI 必须启用
await loadLiteRt(
  'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/',
  { jspi: true }
);
```

### 4.2 模型加载

```javascript
encoderModel = await loadAndCompile('./models/encoder.tflite', {
  accelerator: 'webnn',
  webNNOptions: { devicePreference: 'gpu' },
});
```

### 4.3 图像预处理

```javascript
function preprocessImage(img) {
  // 1. 创建 1024×1024 离屏 canvas
  // 2. 灰色 (0.5) 填充背景
  // 3. 保持宽高比缩放，居中绘制
  // 4. 提取 RGB 像素 → Float32Array [0, 1]
  // 5. 创建 Tensor: new Tensor(floatData, [1, 1024, 1024, 3])
}
```

### 4.4 Encoder 推演

```javascript
const outputs = await encoderModel.run({ image: preprocessed.tensor });
imageEmbeddings = new Float32Array(await outputs.image_embeddings.data());
// shape: [1, 64, 64, 256], NHWC
```

### 4.5 Decoder 推演（自适应点数）

```javascript
// maxPoints 从 decoder input shape 自动检测:
//   MobileSAM: point_labels [1,1] → maxPoints=1
//   Full SAM:  point_labels [1,2] → maxPoints=2
const maxPoints = decoderModel.getInputDetails()
  .find(inp => inp.name === 'point_labels').shape[1];

// 取最近 maxPoints 个点，不足时 padding
const recent = activePoints.slice(-maxPoints);
const coordsData = new Float32Array(maxPoints * 2);
const labelsData = new Float32Array(maxPoints);
for (let i = 0; i < maxPoints; i++) {
  if (i < recent.length) {
    coordsData[i * 2] = recent[i].x;
    coordsData[i * 2 + 1] = recent[i].y;
    labelsData[i] = recent[i].label;
  } else {
    // Padding: 复制最后有效点坐标 + label=0 (忽略)
    coordsData[i * 2] = recent[recent.length - 1].x;
    coordsData[i * 2 + 1] = recent[recent.length - 1].y;
    labelsData[i] = 0;
  }
}

const outputs = await decoderModel.run({
  image_embeddings: new Tensor(imageEmbeddings, [1, 64, 64, 256]),
  point_coords: new Tensor(coordsData, [1, maxPoints, 2]),
  point_labels: new Tensor(labelsData, [1, maxPoints]),
});

const mask = new Float32Array(await outputs.masks.data());
// shape: [1, 256, 256, 1], values: [0, 1] (sigmoid 已应用)
```

### 4.6 Mask 渲染

```
256×256 mask → 离屏 canvas ImageData → drawImage 缩放到显示尺寸
→ 逐像素 green 叠加 (mask > 0.5) → putImageData 到主 canvas
```

### 4.7 坐标转换

```
Canvas 点击 (px) → 原始图片坐标 → 1024×1024 模型空间

canvasX/displayWidth × originalWidth = imgX
imgX × scale + offsetX = modelX   (scale = 1024/max(w,h))
```

### 4.8 HTTP Headers (开发服务器)

```javascript
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

`credentialless` 允许跨域 CDN 资源（jsDelivr），同时启用 SharedArrayBuffer（JSPI 所需）。Chrome 120+ 支持。

## 5. Import Map 依赖链

```
@litertjs/core@2.5.3 → @litertjs/wasm-utils@2.5.3 (唯一运行时依赖)
```

```html
<script type="importmap">
{
  "imports": {
    "@litertjs/core": "https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/dist/index.js",
    "@litertjs/wasm-utils": "https://cdn.jsdelivr.net/npm/@litertjs/wasm-utils@2.5.3/dist/index.js"
  }
}
</script>
```

## 6. 与 ONNX 方案的对比

| 维度 | ONNX Runtime + WebNN | LiteRT.js + WebNN (Mobile) | LiteRT.js + WebNN (Full) |
|------|---------------------|---------------------------|--------------------------|
| **Encoder 模型** | 171 MB (FP16) / 95.6 MB (INT8) | 26.6 MB (Float32) | 368 MB (Float32) |
| **Decoder 模型** | 15.7 MB (FP16) / 4.52 MB (INT8) | 23.7 MB (Float32) | 24.8 MB (Float32) |
| **总模型大小** | ~110-187 MB | ~50 MB | ~393 MB |
| **Encoder 输出** | `[1, 256, 64, 64]` NCHW | `[1, 64, 64, 256]` NHWC | `[1, 64, 64, 256]` NHWC |
| **Decoder 多点** | 支持（动态 freeDimension） | 单点 `[1,1,2]` | 两点 `[1,2,2]` |
| **OOM 风险** | 高（双层 arena） | 低（模型小 + 单层 arena） | **待测试** |
| **算子覆盖** | ONNX→WebNN | TFLite→WebNN（原生） | TFLite→WebNN（原生） |
| **加载方式** | 默认 | `?encoder=encoder.tflite` | `?encoder=SAMEncoder.tflite&decoder=SAMDecoder.tflite` |

## 7. 已知限制与风险

1. **WebNN 算子覆盖**：模型使用的算子可能不完全被 WebNN TFLite delegate 支持，不支持的计算会 fallback 到 WASM (CPU)，可能导致性能下降

2. **OOM 测试**：MobileSAM 模型较小（~50MB），不会触发 2 GiB arena 限制。要测试 OOM 需通过 `?encoder=SAMEncoder.tflite&decoder=SAMDecoder.tflite` 加载 full SAM (393MB)

3. **CDN 依赖**：LiteRT.js 从 jsDelivr CDN 加载，首次启动需要网络连接

4. **JSPI 实验性**：JSPI (JavaScript Promise Integration) 是 WebAssembly 实验性特性，需要手动启用浏览器 flag

5. **多点限制**：MobileSAM 仅支持 1 点，full SAM 仅支持预设的 2 点。不支持 mask prompt（无 `mask_input` / `has_mask_input` 输入）

6. **浏览器兼容性**：需要 Chromium 121+，启用 `#web-machine-learning-neural-network` 和 `#enable-experimental-webassembly-features`

7. **Full SAM encoder 体积**：368 MB 下载耗时较长（取决于网络），且运行时内存峰值可能超过 2 GiB 触发 OOM

## 8. 后续测试计划

### 8.1 功能验证
- [ ] MobileSAM + LiteRT.js + WebNN 生成正确分割蒙版

### 8.2 OOM 测试——Full SAM

下载 full SAM 模型到 `models/` 目录：

```bash
cd models
curl -L -O "https://huggingface.co/qualcomm/Segment-Anything-Model/resolve/main/SAMEncoder.tflite"   # 368 MB
curl -L -O "https://huggingface.co/qualcomm/Segment-Anything-Model/resolve/main/SAMDecoder.tflite"   # 24.8 MB
```

然后访问：
```
http://localhost:8081/?encoder=SAMEncoder.tflite&decoder=SAMDecoder.tflite
```

**期望结果**：
- **如果成功**：LiteRT.js 内存管理优于 ONNX Runtime（单层 arena 避免了双层膨胀）
- **如果 OOM**：同样的 `PartitionExcessiveAllocationSize` 错误出现，说明 ml-drift delegate 层面的静态形状约束是 OOM 的根因，与上层运行时（ORT vs LiteRT.js）无关

### 8.3 性能基准
- [ ] 对比 ONNX+WebNN vs LiteRT.js+WebNN 的 encoder/decoder 延迟
- [ ] 对比 WebNN vs WebGPU vs WASM 三种 accelerator 的性能

### 8.4 算子分析
- [ ] 通过 Chrome tracing (`chrome://tracing`) 检查哪些算子 fallback 到 CPU
- [ ] 记录 WebNN delegate 实际支持的算子列表

### 8.5 NPU 测试
- [ ] 如有 NPU 硬件，测试 `devicePreference: 'npu'` 性能
- [ ] 对比 GPU vs NPU 的功耗和延迟

## 9. 参考

- [WebNN Developer Preview - Segment Anything Demo](https://microsoft.github.io/webnn-developer-preview/demos/segment-anything/)
- [LiteRT.js 官方文档](https://developers.google.com/edge/litert/web)
- [MobileSAM on HuggingFace](https://huggingface.co/qualcomm/MobileSam)
- [Segment Anything Model on HuggingFace](https://huggingface.co/qualcomm/Segment-Anything-Model)
- [WebNN LiteRT Backend 算子兼容性](https://github.com/webmachinelearning/webnn-docs/blob/main/content/en/api-reference/browser-compatibility/litert.mdx)
- [OOM Analysis](C:\Users\junweifu\workspace\webnn\segment_anythings\analysis.md)
