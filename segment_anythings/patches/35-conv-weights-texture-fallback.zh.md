# 35-mldrift-conv-weights-texture-fallback：conv 权重纹理超限的定位与修复

> 对应补丁：`35-mldrift-conv-weights-texture-fallback.patch`
> （`third_party/ml-drift/ml_drift/common/kernels/conv_generic.cc`）
>
> 这是 patch 33（BMM 广播修复）之后暴露的**第二个 bug**：窗口注意力的
> 平铺权重在 WARP 上超出纹理上限，导致整个 dispatch 静默失效、输出全零。
> 主根因分析见 [33-bmm-batch-broadcast.zh.md](33-bmm-batch-broadcast.zh.md)。

---

## 1. 现象

应用 patch 33（B0 广播 TILE 平铺）后，完整模型 fp16 与 fp32 的 GPU 输出
**全部变成 0**（`min=0 max=0 mean=0 std=0`），日志出现 Dawn 校验错误：

```
Validation error: Texture size ([Extent3D width:16, height:65536, ...])
exceeded maximum texture size ([Extent3D width:16384, height:16384, ...]).
```

校验失败 → ComputePipeline/CommandBuffer 无效 → 队列提交被整体丢弃 →
输出缓冲保持零初始化。**没有任何 NaN 或运行错误**，是典型的静默失效。

## 2. 定位

- 最小复现：窗口注意力形态的 B0 广播 BMM
  `[64,64,12,64] @ [1,64,64,64]`（`w_b0bcast` 复现模型）在 patch 33
  之后立即复现同样错误；小形态（`[14,14,300,64] @ [1,14,64,14]`）不触发。
- 差值：窗口注意力的 left 合并 batch = 64×64 = **4096**（全局注意力只有
  196），BMM-as-conv 的 `different_weights_for_height` 权重形状为
  `OHWDI(O, Y=H=4096, X=1, D=1, I)`。
- 权重布局 `k2DX4I4YIsSpatialIAndXIsOOGroupO4` 的纹理尺寸计算：
  **height = Y × I4 = 4096 × 16 = 65536** > WARP 上限 16384，
  width = 16。
- 尺寸检查只存在于 `GetKernelParamsAdreno`（conv_generic.cc :3097-3116）：
  超限时回退 `WeightsUploadType::kGlobalMemory`。**WARP 走的是 generic /
  WebGPU 路径（:3313-3353），无条件选择 `kTexturesX4`，没有这个检查。**

## 3. 机制

`GetKernelParams` 的 vendor 分派链里，Adreno 分支自带纹理尺寸检查并回退
全局内存；其余分支（含 WARP/WebGPU 通用路径）直接选 `kTexturesX4`。
匹配 batch 的窗口 BMM（如 `[64,1,64,64]` 常量右侧）合并后同样是 4096
batch，理论上也会触发——但在 patch 33 之前，广播右侧（b==1）**不合并**
（H=64，纹理 64×16 很小），所以旧代码从未在 4096-batch 权重上跑过
纹理路径；patch 33 让广播右侧也变成 4096 batch，才暴露了这个缺口。

## 4. 修复（patch 35 的内容）

把 Adreno 的检查逻辑原样搬到 `GetKernelParams` 尾部（所有 vendor 路径
共用）：当 `weights_upload_type == kTexturesX4` 时，按当前
`kernel_params.block_size` 构造权重描述符与形状，用 `Get2dResourceSize`
计算实际纹理尺寸，超出 `GetMaxImage2DWidth/Height` 即回退
`kGlobalMemory`。

- 对已有合法纹理的路径**零影响**（检查只在原本会超限时才触发回退）；
- 对所有真 GPU 同样是纯安全性改进（超限纹理在任何 adapter 上都是无效的，
  只是部分驱动会报错而不是静默）。

## 5. 验证

```
w_b0bcast 窗口注意力复现:  PASS  max_abs=8.9e-8（纹理错误消失）
完整模型 fp16 GPU vs CPU:  cosine=0.999993  max_abs=0.0089  over_tol=0  ✅
```

配合 patch 33 与 patch 31（Winograd f16 常量）后，SAM encoder 在 WARP
（无 shader-f16 的软件适配器）上 fp16 全量 GPU 推理与 CPU 参考一致。
