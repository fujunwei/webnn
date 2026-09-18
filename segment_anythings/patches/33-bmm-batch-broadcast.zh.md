# 33-litert-bmm-batch-broadcast：BATCH_MATMUL batch 广播的定位、机制与修复

> 对应补丁：`33-litert-bmm-batch-broadcast.patch`
> （`third_party/litert/src/ml_drift_delegate/tflite/model_builder.cc`）
>
> 这是 SAM encoder 在 WebGPU delegate 上「CPU/GPU 输出不一致」的**主根因**。
> 配套上下文：[gpu_op_bisect.zh.md](../gpu_op_bisect.zh.md)、
> [35-conv-weights-texture-fallback.zh.md](35-conv-weights-texture-fallback.zh.md)。

---

## 1. 现象

`segment_anything_encoder.tflite`（含 custom LayerNorm，全量下沉 GPU，
`IsFullyAccelerated=1`）与 `no_layer_norm_segment_anything_encoder.tflite`
（XNNPACK fp32 CPU）喂同一输入，输出不一致：

```
fp16 GPU vs CPU:  max_abs=0.7197  over_tol(1e-2)=708,618 (67.6%)  cosine=0.9239
fp32 GPU vs CPU:  max_abs=0.7161  over_tol=708,258           cosine=0.9239
```

fp16 与 fp32 的 GPU 输出几乎相同（cosine 0.999991）→ **结构性问题**，
不是精度问题。

## 2. 排除过程（结论先行）

| 嫌疑 | 手段 | 结论 |
|---|---|---|
| custom LayerNorm 实现 | `ln_repro.tflite` GPU 输出 vs numpy（含 affine，eps=1e-5） | ✅ 正确（max_abs=0.0009） |
| LN 融合 vs 分解的表示差异 | 无 LN 模型 GPU+CPU 混合执行 ≈ ln 模型纯 GPU | cosine=0.999998，两者等同 |
| pow 负底数（patch 25 的 bug 类） | `powneg_repro` | bug 存在（71/128 NaN）但 ln 模型输出无 NaN → 未命中 |
| 权重常量路径（旧 patch 29 的 bug 类） | `fc_dq_repro` | 与 ref 一致（max_abs=0.007） |
| stem（patch embed conv） | ln 模型截断 K=9 `--verify` | 干净（max_abs=0.019，fp16 噪声级） |

## 3. 定位：截断模型对拍（同模型 CPU vs GPU）

对 `no_layer_norm` 模型做节点截断（`tools/truncate_tflite.py`），
`--verify` 取 CPU 参考、`--run` 取混合执行，逐检查点对比：

| 截断点 K | 检查点 | cosine | max_abs | 判定 |
|---|---|---|---|---|
| 31 | qkv 投影 FC 之后 | 0.999995 | 0.0083 | ✅ 干净 |
| 41 | attention transpose 链之后 | 0.999995 | 0.0072 | ✅ 干净 |
| **43** | **第一个 BATCH_MATMUL (QKᵀ) 之后** | **0.122** | **3.54** | ❌ **灾难性** |

K=41 干净、K=43 崩溃 → **op 42（第一个 BATCH_MATMUL）是首个出错算子**。

## 4. 最小复现

`[14,14,300,64] @ [1,14,64,14] → [14,14,300,14]`（SAM 注意力的 QKᵀ，
B0 维广播：14 vs 1）：

```
广播 batch:  GPU vs CPU  FAIL  max_abs=0.536  86% 元素超差（GPU std 只有正确的 1/3.7）
匹配 batch:  GPU vs CPU  PASS  max_abs=8.9e-8（对照，证明问题只在广播）
```

窗口注意力形态 `[64,64,12,64] @ [1,64,64,64]` 同样复现。

## 5. 机制

`BatchedMatMulOperationParser`（model_builder.cc）假设 rank-4 BMM 两侧
batch 维匹配，把 `[model_batch, matmul_batch, M, K]` 合并成
`[1, B0*B1, M, K]` 交给 ml-drift 的 matmul-as-conv（权重 H 维 = batch 槽位）：

- left `[14,14,300,64]` → 合并成 `[1,196,300,64]`（H=m*14+k，model batch 在外）；
- right `[1,14,64,14]` 因 `b==1` **不合并**，保持 H=14；

conv 权重只有 14 个 batch 槽位，left 有 196 个 → 每个 block 的注意力
都拿错 K 矩阵 → 误差逐 block 累积（旧会话在旧 litert 上定位的
rank-2 常量 TRANSPOSE bug 是同类「权重布局错」问题，但这次在 BMM 上）。

## 6. 修复（patch 33 的内容）

三种广播组合分别处理：

1. **B0 广播**（right `[1,B1,K,N]`，left `[B0,B1,M,K]`，B0>1）：
   在 Parse 里对 right 插入 **TILE（沿 H 平铺 B0 倍）**。
   TILE 的语义是 `dst[i] = src[i mod B1]`，正好给出
   `slot(m,k) = right[k]` —— 与 left 的合并布局吻合。

2. **B1 广播**（right `[B0,1,K,N]` 常量，left `[B0,B1,M,K]`，B1>1）：
   需要 `slot(m,k) = right[m]`（交织），**TILE 表达不了**——它是
   mod 索引，只会给出 `right[k]`。因为此类 right 在 SAM 里全部是
   **常量权重**（12 处：全局注意力的 V 与窗口注意力的 V），直接在
   Parse 的常量路径里做**主机端交织**（`dst[((m*B1+k)*K+i)*N+j] =
   src[(m*K+i)*N+j]`），把 `[1,B0*B1,K,N]` 的数据烘焙进常量节点，
   不产生任何图算子。

3. **其他广播组合**（rank-3 不匹配、B1 广播但 right 是 runtime 等）：
   `IsSupported` 返回 Unavailable → 该节点回落 CPU（XNNPACK 执行正确）。

## 7. 过程中踩过的两个坑（重要教训）

- **B1 用 relabel+TILE 是错的**。曾实现 `[1,B0,K,N] → [B0,1,K,N] →
  TILE(B×B1) → [1,B0*B1,K,N]`，以为 TILE 按块重复；实测 TILE 是
  **逐元素 mod 索引**，结果等于 `right[k]` 而非 `right[m]`，
  `w_b1bcast` 复现 FAIL（max_abs=0.64）后才改为主机端交织。
- **不能靠整体拒绝 B1 来"修"**。把 B1 广播在 IsSupported 全部拒绝后，
  图上出现 12 个碎片分区，`max_delegated_partitions=1` 只保留最大一块
  （75 个算子），custom LayerNorm 落进 CPU 碎片 → 输出全为 -4.3e8
  （与 gpu_op_bisect.zh.md §0.4 记录的症状一致）。**广播必须留在 GPU
  上并算对，否则破坏图连接。**

## 8. 验证

```
bmm_bcast 复现（fp32/fp16）:   PASS  max_abs=8.9e-7 / 5.3e-4
w_b0bcast 窗口注意力复现:       PASS  max_abs=8.9e-8（配合 patch 35）
完整模型 fp16 GPU vs CPU:      cosine=0.999993  max_abs=0.0089  over_tol=0  ✅
（修复前: cosine=0.9239  max_abs=0.7197  over_tol=708,618）
```

窗口注意力 B0 广播的平铺权重（4096 batch × 64×64）会进一步触发
纹理超限，见 [35-conv-weights-texture-fallback.zh.md](35-conv-weights-texture-fallback.zh.md)。

## 9. 已被上游 WebNN fix 取代

本 patch 修的是"delegate 收到了 batch 维不匹配的 BATCH_MATMUL 才需要广播"
这件事本身。更上游的做法是在生成 TFLite 图的地方（Chromium
`services/webnn/tflite/graph_builder_tflite.cc` 的 `SerializeMatmul`）
就不允许不匹配的 batch 维流出去：WebNN 的 matmul 校验本来就允许两个
操作数 batch 维不同但可广播（NumPy 规则），现在改为在生成 `BATCH_MATMUL`
之前，先用已有的 `BROADCAST_TO` helper 把较小 batch 维的一侧广播对齐，
两个操作数进图时 batch 维必然相同。

逐行重读本 patch 的 diff 可以看到，它加的三处分支（`IsSupported`
mismatch 拒绝、B1 常量交织、B0 TILE 插入）全部以 batch 维不匹配为触发
条件——一旦上游保证两侧 batch 维相同，这些分支永远不会命中，parser
走的就是本 patch 之前的老代码路径，等价于本 patch 是空操作。

已做实测验证：在 `third_party/litert/src` 里 `git revert` 掉本 patch
对应的 commit，重新编译部署 `libLiteRtWebGpuAccelerator.dll`，再跑新增的
`services_unittests` 用例 `WebNNGraphImplBackendTest.MatmulBatchDimsBroadcast`
（覆盖本 patch 修的 B0/B1 两种广播形状，通过真实 GPU delegate 执行）——
两个 case 在没有本 patch 的情况下依然通过。之后已把 revert 撤销，
`third_party/litert/src` 恢复为本 patch 仍然应用的状态。

**结论：本 patch 目前保留应用（无副作用，一旦上游 fix 到位即为空操作），
待 `graph_builder_tflite.cc` 的修复真正合入 Chromium 上游后，可以从
patch 列表里移除。**

`graph_builder_tflite.cc` 的修复已经完成并以 commit `c6367c7848`
（`webnn: broadcast mismatched matmul batch dims before BATCH_MATMUL`）
落地在本地 `chromium/src` checkout 里，对应快照见
[`34-webnn-matmul-batch-broadcast.patch`](34-webnn-matmul-batch-broadcast.patch)。
设计细节、回归测试、WPT 一致性测试补充、以及完整 SAM encoder 端到端模型级别的
验证结果见下面的 §10——但那次端到端验证是在**本 patch（33）仍然应用**的状态下
跑的（此时 33 按上面的分析应为死代码），尚未单独验证过“去掉 patch 33、只保留
`graph_builder_tflite.cc` fix”这一组合下的完整端到端模型表现（需要重新构建
不带 patch 33 的 Chromium 并重新 dump `.tflite`）；目前这一组合的验证仍然停留在
单测级别（`WebNNGraphImplBackendTest.MatmulBatchDimsBroadcast`，见上）。

## 10. 上游修复的设计与验证（补丁 34）

### 10.1 修复位置与设计

`services/webnn/tflite/graph_builder_tflite.cc` 的
`GraphBuilderTflite::SerializeMatmul`：之前把 `a`/`b` 两个操作数的 tensor
index 不做任何检查，直接传给 `SerializeMatmulOperation` 生成 `BATCH_MATMUL`。
修复后的逻辑：

1. 取 `a_dims`/`b_dims`（来自序列化好的 `TensorInfo`），仅当两者 rank
   都 `>= 3` 且 rank 相等、且 batch 前缀（除最后两维外的所有轴）不完全相等时才介入——
   这正是 WebNN matmul 校验允许、但并非所有 `BATCH_MATMUL` 后端都能正确处理的形状。
2. 逐 batch 轴取 `max(a_dims[i], b_dims[i])` 得到目标 `batch_dims`（NumPy
   广播规则下，不匹配的轴里必有一侧是 1）。
3. 对 `a`、`b` 中批次前缀不等于 `batch_dims` 的那一侧，用
   `SerializeTemporaryTensorWithByteSizeCheck` 开一个形状为
   `batch_dims + 原始 trailing two dims` 的临时张量，再用
   `SerializeBroadcastToOperation` 插入一个 `BROADCAST_TO` 算子把该操作数
   广播进去，并把该操作数的 tensor index 改指向广播后的结果。
4. 最终用（可能已被替换的）`a_tensor_index`/`b_tensor_index` 调用
   `SerializeMatmulOperation`，此时两个操作数的 batch 维必然相等。

### 10.2 为什么在 graph builder 层修，而不是继续在每个 delegate 里修

对比本 patch（33）的方案——只在 litert 的 `model_builder.cc` 里针对
B0/B1 两种具体广播形状做 TILE/主机端交织特判，其余组合一律拒绝、回退
CPU——graph builder 层的修复：

- **一次性覆盖所有后端**：GPU delegate、XNNPACK、以及未来任何新增的
  delegate 收到的图里 `BATCH_MATMUL` 两侧 batch 维永远相等，不需要每个
  delegate 各自重新实现一遍广播语义（`BatchedMatMulOperationParser` 里
  33 加的三个分支从此永远不会命中，见上文）。
- **不引入新的正确性风险**：`BROADCAST_TO` 的语义（NumPy 广播）与 WebNN
  matmul 规范定义的广播语义完全一致，这里只是把广播从"隐式依赖
  `BATCH_MATMUL` kernel 自己处理"变成图里一个显式的算子，计算结果不变。
- **对已经处理广播正确的后端无损**：XNNPACK 的 CPU `BATCH_MATMUL` kernel
  收到两侧 batch 维已经相等的输入时，走的就是它原本处理"batch 维本来就
  匹配"这一支路径，没有任何行为变化。

### 10.3 回归测试

`WebNNGraphImplBackendTest.MatmulBatchDimsBroadcast`（新增于
`services/webnn/webnn_graph_impl_backend_test.cc`）覆盖两种情形，形状和
数值直接对应本 patch（33）里 B0/B1 两种广播：

- **外层（最外侧）batch 轴广播**：`inputB` dims `[1,2,2,2]` vs
  `inputA` dims `[2,2,2,2]`。
- **内层 batch 轴广播**（更难的一种：外层轴已经在两侧都等于 2、并非 1，
  只有内层轴广播）：`inputB` dims `[2,1,2,2]` vs `inputA` dims
  `[2,2,2,2]`。

第二种情形是关键——它排除了"只处理最外层批次轴坍缩"这类简化实现（例如把
广播理解为"合并 batch 到一个轴再 tile"）能蒙混过关的可能性。

### 10.4 WPT 一致性测试补充

复查了 `matmul.https.any.js` 中已有的每一条 `(broadcast)` 用例的 shape
descriptor，发现全部只广播 `inputB` **最外层（leading）** batch 轴，且该轴
坍缩为 1（例如 `[1,1,4,5]` vs `[2,2,3,4]`）；没有一条覆盖"外层 batch 轴
已经匹配且不为 1、内层轴广播"这一场景——这正是上面 §10.3 第二个回归测试
覆盖、也是最容易被"只处理最外层坍缩"式实现漏掉的形状。为此在 float32 和
float16 两个 section 里各补充了一条同名新用例（`matmul float32/float16 4D
and 4D (broadcast an inner batch dimension) tensors`），数据与
`MatmulBatchDimsBroadcast` 第二个 case 保持一致，便于交叉核对。

### 10.5 端到端验证

除了 §9 末尾"revert 补丁 33、通过真实 GPU delegate 重跑回归测试仍然通过"
这一验证外，还做了一次完整 SAM encoder 模型级别的对拍：

```
matmul_fix_segment_anything_encoder.tflite
  （已应用 graph_builder_tflite.cc fix 重新 dump 出的模型，GPU+CPU fallback，fp16）
vs
no_layer_norm_segment_anything_encoder.tflite
  （CPU-only fp32 参考）

n=1048576  max_abs=0.009897768497467041  mean_abs=0.0003321078098277707
over_tol(0.01)=0  nan_mismatch=0   → PASS
```

注意：这次端到端验证是在**本 patch（33）仍然应用**的状态下跑的（此时按
上面的分析 33 应为死代码，不改变结果）；"去掉 patch 33、只保留
`graph_builder_tflite.cc` fix"这一组合尚未做过完整端到端模型验证，见本节
开头的说明。

### 10.6 对应补丁文件

本次修改已经作为 commit `c6367c7848` 直接提交在 `chromium/src` checkout
里（`webnn: broadcast mismatched matmul batch dims before BATCH_MATMUL`，
覆盖 `graph_builder_tflite.cc`、`webnn_graph_impl_backend_test.cc`、
`matmul.https.any.js` 三个文件）。对应的快照见
[`34-webnn-matmul-batch-broadcast.patch`](34-webnn-matmul-batch-broadcast.patch)
（用 `tools/gen_patches.py` 从该 commit 生成）。与 `litert`/`ml-drift` 的
补丁不同，这份补丁**只是归档快照，不是给 `git apply` 用的**——
`chromium/src` 不是像另外两个 third_party checkout 那样携带"裸 working
tree 修改"的仓库，这次修改已经是它自己的一个正常 commit。
