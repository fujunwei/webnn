# WebNN→TFLite 转换器中的 SDPA 融合

记录 `services/webnn/tflite/graph_builder_tflite.cc`/`.h` 中新增的
scaled-dot-product-attention（SDPA）融合 pass：把 WebNN 图里被拆解成 6 个
算子的自注意力子图，重新序列化成一个 `STABLEHLO_COMPOSITE`
(`name="odml.scaled_dot_product_attention"`) 节点，让 ml-drift 的 GPU
delegate 走它自带的融合 SDPA kernel，同时保证 CPU/XNNPACK 路径完全不变。

对应 chromium 分支 `fuse_sdpa_for_sd`，单个提交
`d7f30f0b78 "Fuse sdpa in LiteRT backend with odml.scaled_dot_product_attention"`
（`graph_builder_tflite.cc` +662/-5，`graph_builder_tflite.h` +96）。

## 背景 / 动机

`sd/model/vae_decoder_*.tflite` 由 Chromium 的 WebNN→TFLite 直转换器产出
（这个转换器不经过 StableHLO/ONNX，只是逐算子照抄 WebNN 图）。WebNN 的
`mojom::Operation` 联合体里没有原生的 attention 算子，所以 Stable
Diffusion VAE decoder 里的自注意力块被转换器拆成六步：

```
Transpose(K) → Matmul(Q, Kᵀ) → [Dequantize(fp16 scale)] → Mul(scale) → Softmax → Matmul(·, V)
```

ml-drift 的 GPU delegate（`convert_sdpa.cc` / `model_builder.cc` /
`ir_model_builder.cc`）已经认识 `STABLEHLO_COMPOSITE`+
`odml.scaled_dot_product_attention` 节点，会把它转成一个融合的
`SCALED_DOT_PRODUCT_ATTENTION` IR 算子，但从来没人喂给它这种节点 ——
转换器只会吐出拆开的 6 个算子，所以 GPU 只能按普通算子逐个跑，吃不到融合
kernel 的优化。

**硬约束**（用户明确要求，"CPU 还是走原来的方案"）：CPU/XNNPACK
必须继续执行原始的拆解算子序列，不能因为这个融合而改变 CPU 上跑的东西。
TFLite 的默认 `stablehlo_composite.cc` kernel 天然支持这个需求：只要
composite 节点带一个 decomposition subgraph，没有 delegate 认领这个节点时
就会退化去跑 decomposition subgraph —— 于是只要这个子图里放的是原始的
拆解算子，CPU 路径就完全不受影响。

## 匹配范围

只精确匹配 SD VAE decoder 里实际出现的这一种形状，不做通用 attention
matcher：单头、无 mask、`Mul(scale)` 在 `Softmax` 之前、`scale` 是
标量常量（可能被一层 `DequantizeLinear` 包着，对应 fp16 权重）。任何一步
的输出被 >1 个算子消费（"sole dependent" 检查失败）就直接放弃匹配，不勉强
融合。

## 实现结构

### 结构体 / 声明（`graph_builder_tflite.h`）

```cpp
struct ScaledDotProductAttentionFusion {
  raw_ptr<const mojom::Transpose> transpose;
  raw_ptr<const mojom::Matmul> matmul1;
  raw_ptr<const mojom::DequantizeLinear> scale_dequantize;  // 可为空
  raw_ptr<const mojom::ElementWiseBinary> mul;
  raw_ptr<const mojom::Softmax> softmax;
  raw_ptr<const mojom::Matmul> matmul2;
  OperandId q_operand_id;
  OperandId k_operand_id;
  OperandId v_operand_id;
  float scale;
};

base::flat_map<OperationId, ScaledDotProductAttentionFusion> sdpa_fusions_;
```

`sdpa_fusions_` 以链条最后一个 Matmul（Matmul#2）的 `OperationId` 为 key ——
这也是融合最终真正被发射（emit）的位置。链条里其余的算子
（transpose、matmul1、可选的 dequantize、mul、softmax）都被记入
`fused_ops_to_skip_`，序列化主循环遇到它们时直接跳过。

字段用 `raw_ptr<const T>` 而不是裸指针 —— Chromium 的
`chromium-rawptr` clang lint 强制要求，裸指针会编译报错。

### 两阶段设计：先记录匹配，最后再发射

第一版实现把融合"发射"这一步直接放在匹配到 `Transpose` 的地方，结果暴露
出一个排序 bug：WebNN 的算子列表不保证 Q、V 的生产者算子排在
`Transpose(K)` 之前 —— 它们可能在 `graph_info.operations` 里排在
`Transpose` 之后。如果在 `Transpose` 处就发射引用 Q/V 的
composite 节点，TFLite 会出现 use-before-def（消费者在生产者之前）。

修复方案是把"匹配"和"发射"彻底解耦：

1. **`RecordScaledDotProductAttentionFusion(transpose, transpose_id)`**
   ——`CreateAndBuild` 里在真正开始序列化之前的一个预处理循环，遍历全部
   `Transpose` 算子，尝试从每个 `Transpose` 出发向前匹配整条链。匹配成功
   只登记进 `sdpa_fusions_` / `fused_ops_to_skip_`，不发射任何东西。
   ```cpp
   if (base::FeatureList::IsEnabled(kApplySdpaFusion)) {
     for (size_t i = 0; i < graph_info.operations.size(); ++i) {
       if (graph_info.operations[i]->is_transpose()) {
         builder.RecordScaledDotProductAttentionFusion(
             *graph_info.operations[i]->get_transpose(), i);
       }
     }
   }
   ```

2. 主序列化循环里，`kTranspose` case 只剩跳过检查（原来的匹配逻辑已搬到
   预处理阶段）：
   ```cpp
   case mojom::Operation::Tag::kTranspose:
     if (fused_ops_to_skip_.contains(operation_index)) return base::ok();
     ASSIGN_OR_RETURN(operator_offset, SerializeTranspose(...));
   ```

3. `kMatmul` case 新增分支：命中 `sdpa_fusions_` 就在**这里**才真正调用
   `SerializeScaledDotProductAttentionComposite`：
   ```cpp
   case mojom::Operation::Tag::kMatmul:
     if (fused_ops_to_skip_.contains(operation_index)) return base::ok();
     if (auto it = sdpa_fusions_.find(operation_index); it != sdpa_fusions_.end()) {
       ASSIGN_OR_RETURN(operator_offset,
                         SerializeScaledDotProductAttentionComposite(it->second));
       break;
     }
     ASSIGN_OR_RETURN(operator_offset, SerializeMatmul(*op.get_matmul()));
   ```
   等序列化主循环走到 Matmul#2 时，Q/K/V 的生产者（不管它们原本排在哪）
   保证已经全部序列化完毕 —— WebNN 算子列表本身仍然保证"每个 operand
   的生产者排在消费者之前"这个全局约束，只是不保证相对于 `Transpose`
   这个匹配锚点的顺序。把发射点换成链条里**最后**一个算子，就天然利用了
   这个全局保证。

### `scale` 提取

`Mul` 的标量操作数有两种来源，都在匹配阶段折算成一个编译期 `float`：
- 直接常量：复用已有的 `GetFloatScalarConstant`（同时处理 fp32/fp16）。
- 被 `DequantizeLinear` 包住（真实模型里 fp16 scale 权重就是这种）：读原始
  量化值，手工套用 `(raw - zero_point) * dequant_scale` 得到最终浮点数。

### `SerializeScaledDotProductAttentionComposite`

在 Matmul#2 位置真正发射的函数，做的事：
1. 用 `SerializeInputTensorInfo` 拿 Q/K/V（Q = matmul1 的另一路输入，K =
   Transpose 之前的原始输入，V = matmul2 的另一路输入）。
2. 用 `SerializeOutputTensorInfo` 先序列化 Matmul#2 的输出 tensor
   info（这一步必须在构建 decomposition subgraph **之前**做，见下面的
   bug 1）。
3. 调用 `BuildScaledDotProductAttentionDecompositionSubgraph`，把捕获到的
   原始 mojom 算子结构体重放一遍，生成 CPU fallback 用的子图。
4. 发射一个 `STABLEHLO_COMPOSITE` 算子：
   - `inputs = [Q, K, V]`（顺序对应 ml-drift `convert_sdpa.cc` 里
     `node.inputs->data[0..2]` 的读法）
   - `builtin_options_2 = StableHLOCompositeOptions{ name:
     "odml.scaled_dot_product_attention", decomposition_subgraph_index,
     composite_attributes: flexbuffers{"scale": <float>},
     composite_attributes_format: FLEXBUFFERS }`

### Decomposition subgraph（CPU/XNNPACK 保真通道）

`stablehlo_composite.cc` 的默认 kernel：没有 delegate 认领这个节点时，
按位置把 composite 节点的输入拷进引用的 subgraph 的声明输入，跑
`Invoke()`，再把输出拷回来。因此这个子图只需要装下**原封不动**的六步
拆解算子。

`tensors_`、`operators_`、`operand_to_tensor_info_map_`、
`lazy_serialized_dequantize_operations_`、
**`graph_output_cast_operators_`** 都是子图级别的状态，构建子图前保存并
清空，构建完再恢复；`buffers_`（常量数据）是模型级别、跨子图共享，绝不能
清。子图构建完，把生成的
`flatbuffers::Offset<::tflite::SubGraph>` 追加进新成员
`decomposition_subgraphs_`；`FinishAndTakeResult` 原来硬编码只有一个
subgraph，改成 `{main_subgraph} + decomposition_subgraphs_`，主图固定是
index 0。

### Feature flag

```cpp
BASE_FEATURE(kApplySdpaFusion, ..., base::FEATURE_ENABLED_BY_DEFAULT);
```
出问题时可以 `--disable-features=ApplySdpaFusion` 一键关掉整个融合，回退
到未融合的旧行为，作为 A/B 对照。

## 踩过的两个 bug（详见 memory）

1. **`graph_output_cast_operators_` 泄漏** —— 构建子图时保存清空了这个
   deferred-cast 列表，但没有恢复，也没考虑到 Matmul#2 输出可能是需要
   cast 的 fp16 图输出。表现为模型加载直接失败（"Failed to load model
   from buffer"），主图算子引用了越界的、属于子图的 tensor index。修复：
   提前序列化主图侧的输出 tensor info 并传给子图构建函数、子图内部把
   Matmul#2 输出预置成相同 dtype、子图构建完成后把子图内产生的 cast
   算子并入子图自己的 operator 列表、真正 restore
   `graph_output_cast_operators_`。
2. **Use-before-def**（发射点选在 `Transpose` 而不是链条末尾）——
   已在上面"两阶段设计"一节说明并解决。

用 [[webnn-chrome-tflite-repro-harness]] 里记录的 Chrome flags 组合 +
`sd/webnn_sdpa_fusion_test.html`（12 组 fp32/fp16 × raw/projected ×
{8,256,4096} 序列长度的最小 WebNN 用例）+ `sd/dump_all_subgraphs.py`
（把 tflite 子图的 tensor/operator 列表整个摊开肉眼检查）找到并验证修好了
这两个 bug——比每次都跑完整 VAE decoder 模型 dump/编译/verify 快得多。

## 端到端验证结果

真实模型：`sd/model/vae_decoder_cpu.tflite`（未融合基线，1012 个主图算子，
0 个 composite）vs `sd/model/vae_decoder_gpu.tflite`（融合后重新 dump，
681 个主图算子，恰好 1 个 `STABLEHLO_COMPOSITE`，输入输出都是
`[1,4096,1,512]` FLOAT32）。

`sam_encoder_runner.exe --verify`（见 `sd/vae_verify_fusion5.log`）：

```
[verify-cpu] elems=786432 nan=0 min=-1.74138 max=0.649898  mean=-0.833536 std=0.72685
[verify-gpu] elems=786432 nan=0 min=-1.74023 max=0.648926  mean=-0.832884 std=0.726502
[verify] elems=786432 tol=0.01 max_abs=0.00522411 mean_abs=0.00110079 over_tol=0 nan_mismatch=0
[verify] PASS
```

786432 个元素零个超差，融合后的 CPU 输出与融合前完全一致（确认 "CPU 还是
走原来的方案" 这条硬约束成立），融合后 GPU 输出与 CPU 参考在 fp16 容差
内一致。

**顺带发现**：未融合模型在 GPU 上跑（走旧的拆解算子序列）输出
`mean=-30.3043`，和 CPU 参考差了约 36 倍 —— 这是 ml-drift 在拆解版
softmax/matmul 链路上一个既有的、和本次融合无关的 bug（融合工作反而绕开
了它）。之前几个 session 怀疑"融合引入了 GPU 数值回归"，实际是因为
`sam_encoder_runner.exe` 编译得太旧、`--gpu-model` 这个 CLI 开关还没编译
进去，静默地把 GPU 侧也测成了 `--model` 指向的未融合模型；另外还踩到
ml-drift WebGPU 回读默认 10 秒超时（这个模型在集显上要 ~19.5 秒），
两者都记在 [[litert-build-gotchas]] / [[mldrift-webgpu-readback-timeout]]
里，不在此文重复。

## 相关文件

- `services/webnn/tflite/graph_builder_tflite.h` / `.cc`（chromium 仓库，
  分支 `fuse_sdpa_for_sd`，提交 `d7f30f0b78`）——融合实现本体。
- `sd/webnn_sdpa_fusion_test.html` —— 隔离的最小 WebNN 测试页，12 组用例。
- `sd/dump_all_subgraphs.py` —— tflite 子图/tensor/operator 全量 dump 工具。
- `sd/model/vae_decoder_cpu.tflite` / `vae_decoder_gpu.tflite` —— 真实
  VAE decoder 的未融合/融合后 dump。
- `sd/vae_verify_fusion3.log` ~ `fusion5.log` —— 端到端 verify 尝试记录
  （3 是发现 stale exe，4 是发现 readback 超时，5 是最终 PASS）。
