# Segment Anything — LiteRT.js + WebNN Demo

浏览器端 Segment Anything 演示，使用 [LiteRT.js](https://developers.google.com/edge/litert/web) + WebNN GPU 加速。

## 快速开始

### 环境要求

1. **Chromium 内核浏览器** (Chrome 121+ 或 Edge 121+)
2. 启用 WebNN: `chrome://flags/#web-machine-learning-neural-network` → **Enabled**
3. 启用 JSPI: `chrome://flags/#enable-experimental-webassembly-features` → **Enabled**

### 运行

```bash
cd segment_anything_litert
node server.js [port]
```

默认端口 8080。浏览器打开 `http://localhost:8080`。

### 操作

- **左键点击** — 添加正样本点（要分割的区域）
- **右键点击** — 添加负样本点（要排除的区域）
- **Clear Points** — 清除所有点
- **Cut Out** — 下载分割结果 PNG

## 支持两种模型

### MobileSAM（默认，50MB）

使用 [qualcomm/MobileSam](https://huggingface.co/qualcomm/MobileSam)：

| 文件 | 大小 |
|------|------|
| `models/encoder.tflite` | 26.6 MB |
| `models/decoder.tflite` | 23.7 MB |

**默认加载此模型**。无需额外配置。

### Full SAM（393MB）— 测试 OOM

使用 [qualcomm/Segment-Anything-Model](https://huggingface.co/qualcomm/Segment-Anything-Model)：

| 文件 | 大小 |
|------|------|
| `models/SAMEncoder.tflite` | 368 MB |
| `models/SAMDecoder.tflite` | 24.8 MB |

**下载模型到 models/ 目录**，然后通过 URL 参数加载：

```bash
# 下载 full SAM 模型
cd models
curl -L -O "https://huggingface.co/qualcomm/Segment-Anything-Model/resolve/main/SAMEncoder.tflite"
curl -L -O "https://huggingface.co/qualcomm/Segment-Anything-Model/resolve/main/SAMDecoder.tflite"
```

然后访问：
```
http://localhost:8080/?encoder=SAMEncoder.tflite&decoder=SAMDecoder.tflite
```

### 其他 URL 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `?models=` | `./models/` | 模型目录 URL |
| `?encoder=` | `encoder.tflite` | Encoder 文件名 |
| `?decoder=` | `decoder.tflite` | Decoder 文件名 |

示例——从 HuggingFace 直接加载 MobileSAM：
```
http://localhost:8080/?models=https://huggingface.co/qualcomm/MobileSam/resolve/main/
&encoder=encoder.tflite&decoder=decoder.tflite
```

## 架构

```
Image (1024×1024) → Encoder (TFLite) → Embeddings → Decoder (TFLite) → Mask
                         ↑ WebNN GPU                    ↑ WebNN GPU
```

- MobileSAM: 单点交互（maxPoints=1）
- Full SAM: 两点交互（maxPoints=2），自动使用最近 N 个点

## 与 ONNX Runtime WebNN 对比

| 维度 | ONNX Runtime + WebNN | LiteRT.js + WebNN |
|------|---------------------|-------------------|
| Encoder 大小 | 171 MB (FP16) | 26.6 MB (Mobile) / 368 MB (Full) |
| Decoder 大小 | 15.7 MB (FP16) | 23.7 MB (Mobile) / 24.8 MB (Full) |
| OOM 风险 (encoder) | 高 (arena > 2 GiB) | Mobile: 低 / Full: 待测试 |
| 多点支持 | 动态 freeDimension | Mobile: 1点 / Full: 2点 |
| 算子路径 | ONNX→ORT→WebNN | TFLite→ml-drift→WebNN |
| 运行时 | ONNX Runtime Web | LiteRT.js |

## 文件结构

```
segment_anything_litert/
├── index.html          # 页面 + Import Map
├── index.js            # 推演逻辑
├── server.js           # 开发服务器
├── inspect_model.py    # TFLite 模型检查工具
├── DESIGN.md           # 设计文档
├── README.md           # 本文件
└── models/
    ├── encoder.tflite      # MobileSAM encoder (默认)
    ├── decoder.tflite      # MobileSAM decoder (默认)
    ├── SAMEncoder.tflite   # Full SAM encoder (需下载)
    └── SAMDecoder.tflite   # Full SAM decoder (需下载)
```

## 浏览器 Flags

```
chrome://flags/#web-machine-learning-neural-network       → Enabled
chrome://flags/#enable-experimental-webassembly-features  → Enabled (JSPI)
```

## 参考

- [WebNN Developer Preview - SAM Demo](https://microsoft.github.io/webnn-developer-preview/demos/segment-anything/)
- [LiteRT.js 文档](https://developers.google.com/edge/litert/web)
- [OOM 分析](../segment_anythings/analysis.md)
- [设计文档](./DESIGN.md)
