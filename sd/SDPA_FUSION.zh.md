# WebNN→TFLite 转换器中的 SDPA 融合

记录 `services/webnn/tflite/graph_builder_tflite.cc`/`.h` 中新增的
scaled-dot-product-attention(SDPA)融合 pass:把 WebNN 图里被拆解成 6 个
算子的自注意力子图,重新序列化成一个 `STABLEHLO_COMPOSITE`
(`name="odml.scaled_dot_product_attention"`) 节点,让 ml-drift 的 GPU
delegate 走它自带的融合 SDPA kernel,同时保证 CPU/XNNPACK 路径完全不变。

对应 chromium 分支 `fuse_sdpa`,提交 `5e8e860757` "Fuse SDPA in LiteRT
backend"(历史版本 `20abfaa2e8`,后 amend 进 rank-3 支持与 fp16 子图修复;
`graph_builder_tflite.cc` +556/-20,`graph_builder_tflite.h` +69,WPT 测试
`sdpa_subgraph.https.any.js` +396)。早期迭代(9/5,分支
`fuse_sdpa_for_sd`、提交 `d7f30f0b78`)的验证记录保留在"历史"一节。

## 背景 / 动机

`sd/model/vae_decoder_*.tflite` 由 Chromium 的 WebNN→TFLite 直转换器产出
(这个转换器不经过 StableHLO/ONNX,只是逐算子照抄 WebNN 图)。WebNN 的
`mojom::Operation` 联合体里没有原生的 attention 算子,所以 Stable
Diffusion VAE decoder 里的自注意力块被转换器拆成六步:

```
Transpose(K) → Matmul(Q, Kᵀ) → [Dequantize(fp16 scale)] → Mul(scale) → Softmax → Matmul(·, V)
```

ml-drift 的 GPU delegate(`convert_sdpa.cc` / `model_builder.cc` /
`ir_model_builder.cc`)已经认识 `STABLEHLO_COMPOSITE`+
`odml.scaled_dot_product_attention` 节点,会把它转成一个融合的
`SCALED_DOT_PRODUCT_ATTENTION` IR 算子,但从来没人喂给它这种节点 ——
转换器只会吐出拆开的 6 个算子,所以 GPU 只能按普通算子逐个跑,吃不到融合
kernel 的优化。

**硬约束**(用户明确要求,"CPU 还是走原来的方案"):CPU/XNNPACK
必须继续执行原始的拆解算子序列,不能因为这个融合而改变 CPU 上跑的东西。
TFLite 的默认 `stablehlo_composite.cc` kernel 天然支持这个需求:只要
composite 节点带一个 decomposition subgraph,没有 delegate 认领这个节点时
就会退化去跑 decomposition subgraph —— 于是只要这个子图里放的是原始的
拆解算子,CPU 路径就完全不受影响。

## 匹配范围

只精确匹配 SD VAE decoder 里实际出现的这一种形状,不做通用 attention
matcher:单头、无 mask、`Mul(scale)` 在 `Softmax` 之前、`scale` 是
标量常量(可能被一层 `DequantizeLinear` 包着,对应 fp16 权重)。任何一步
的输出被 >1 个算子消费("sole dependent" 检查失败)就直接放弃匹配,不勉强
融合。

**rank-3 链**:真实模型的注意力链是 rank-3 `[batch, seq, head_dim]`
(Q/K/V 都是 `[1,4096,512]`;Q 路径带 reshape→transpose→reshape 的 4D
head dance)。但 ml-drift / XNNPACK 的融合 kernel 只接受 rank-4
`[batch, seq, heads, head_dim]` —— rank-3 直接喂 composite 在 GPU 上
实测输出错误(softmax 退化为 one-hot),XNNPACK 对非 rank-4 直接 Prepare
硬失败。因此:

- matcher 接受 **rank-3 链**(记 `rank3_chain` 标志)或 **rank-4 单头链**
  (`shape[2] == 1`);rank-4 多头被拒(内核把倒数第二轴读作 head 数,
  heads 折叠到别的轴会静默算错)。
- rank-3 链在 composite 前后**合成 4D reshape**(见下文),内核只看到
  rank-4 单头输入,与 9/5 验证过的 composite I/O 形状一致。
- 注:rank-4 单头链在实践中不会出现 —— rank-4 单头(axis-2==1)的 5 算子
  分解在数学上退化为 head 轴与 seq 轴的广播混叠(softmax 轴退化为
  size-1);真实模型用 rank-3 + reshape dance 正是这个原因。rank-4 分支
  仅作安全阀保留。

## 实现结构

### 结构体 / 声明(`graph_builder_tflite.h`)

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
  bool rank3_chain;  // Q/K/V 为 rank-3 [batch, seq, head_dim]
};
```

`sdpa_fusions_` 以链条最后一个 Matmul(Matmul#2)的 `OperationId` 为 key ——
这也是融合最终真正被发射(emit)的位置。链条里其余的算子
(transpose、matmul1、可选的 dequantize、mul、softmax)都被记入
`fused_ops_to_skip_`,序列化主循环遇到它们时直接跳过。

字段用 `raw_ptr<const T>` 而不是裸指针 —— Chromium 的
`chromium-rawptr` clang lint 强制要求,裸指针会编译报错。

### 两阶段设计:先记录匹配,最后再发射

第一版实现把融合"发射"这一步直接放在匹配到 `Transpose` 的地方,结果暴露
出一个排序 bug:WebNN 的算子列表不保证 Q、V 的生产者算子排在
`Transpose(K)` 之前 —— 它们可能在 `graph_info.operations` 里排在
`Transpose` 之后。如果在 `Transpose` 处就发射引用 Q/V 的
composite 节点,TFLite 会出现 use-before-def(消费者在生产者之前)。

修复方案是把"匹配"和"发射"彻底解耦:

1. **`RecordScaledDotProductAttentionFusion(transpose, transpose_id)`**
   ——`CreateAndBuild` 里在真正开始序列化之前的一个预处理循环,遍历全部
   `Transpose` 算子,尝试从每个 `Transpose` 出发向前匹配整条链。匹配成功
   只登记进 `sdpa_fusions_` / `fused_ops_to_skip_`,不发射任何东西。
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

2. 主序列化循环里,`kTranspose` case 只剩跳过检查(原来的匹配逻辑已搬到
   预处理阶段):
   ```cpp
   case mojom::Operation::Tag::kTranspose:
     if (fused_ops_to_skip_.contains(operation_index)) return base::ok();
     ASSIGN_OR_RETURN(operator_offset, SerializeTranspose(...));
   ```

3. `kMatmul` case 新增分支:命中 `sdpa_fusions_` 就在**这里**才真正调用
   `SerializeScaledDotProductAttentionComposite`:
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
   等序列化主循环走到 Matmul#2 时,Q/K/V 的生产者(不管它们原本排在哪)
   保证已经全部序列化完毕 —— WebNN 算子列表本身仍然保证"每个 operand
   的生产者排在消费者之前"这个全局约束,只是不保证相对于 `Transpose`
   这个匹配锚点的顺序。把发射点换成链条里**最后**一个算子,就天然利用了
   这个全局保证。

### `scale` 提取

`Mul` 的标量操作数有两种来源,都在匹配阶段折算成一个编译期 `float`:
- 直接常量:复用已有的 `GetFloatScalarConstant`(同时处理 fp32/fp16)。
- 被 `DequantizeLinear` 包住(真实模型里 fp16 scale 权重就是这种):读原始
  量化值,手工套用 `(raw - zero_point) * dequant_scale` 得到最终浮点数
  (`GetEffectiveScalarScale`)。

### rank-3 链的 reshape 合成

匿名命名空间的 `ToSdpaRank4Dims()` 把 `[B,N,D]` 扩成 `[B,N,1,D]`。
`SerializeScaledDotProductAttentionComposite` 对 rank-3 链:

1. **主图**:Q/K/V 各发射一个 `RESHAPE`(3D→4D temp,`operators_.emplace_back`)
   → composite 吃 4D 输入 → composite 输出写 4D temp → 再发射一个
   `RESHAPE`(4D→3D)写回 Matmul#2 输出的 operand 张量。composite 直接
   append,函数返回输出 reshape 的 offset 让 `SerializeOperation` 最后
   append,保证算子顺序:reshape×3 → composite → reshape。
2. **子图**:声明 4D 输入/输出(与 composite 签名一致;composite kernel 的
   `Prepare` 本来就会把子图输入 resize 成 composite 输入的形状),内部
   entry reshape×3(4D→3D)+ 原 6 算子 + exit reshape(3D→4D)。

### Decomposition subgraph(CPU/XNNPACK 保真通道)

`stablehlo_composite.cc` 的默认 kernel:没有 delegate 认领这个节点时,
按位置把 composite 节点的输入拷进引用的 subgraph 的声明输入,跑
`Invoke()`,再把输出拷回来。因此这个子图只需要装下**原封不动**的六步
拆解算子。

`tensors_`、`operators_`、`operand_to_tensor_info_map_`、
`lazy_serialized_dequantize_operations_`、
**`graph_output_cast_operators_`** 都是子图级别的状态,构建子图前保存并
清空,构建完再恢复;`buffers_`(常量数据)是模型级别、跨子图共享,绝不能
清。子图构建完,把生成的
`flatbuffers::Offset<::tflite::SubGraph>` 追加进
`decomposition_subgraphs_`;`FinishAndTakeResult` 把
`{main_subgraph} + decomposition_subgraphs_` 一起放进模型,主图固定是
index 0(这套多子图机制是分支上 layer-norm composite 工作的既有设施,
本次只是复用)。

**fp32 override(bug 3 的修复)**:子图内 Q/K/V 的 operand 张量用
`SerializeOperand(..., ::tflite::TensorType_FLOAT32)` 显式 override 创建,
而不是走 `SerializeInputTensorInfo` 默认路径 —— 后者对 fp16 operand 会
插入 fp16→fp32 CAST 且不更新 operand→tensor map,在子图里造成双写入张量
和无生产者 fp16 张量(见 bug 3)。未融合图里注意力链本来就全程 fp32,
override 与之一致。

### Feature flag

```cpp
BASE_FEATURE(kApplySdpaFusion, ..., base::FEATURE_ENABLED_BY_DEFAULT);
```
出问题时可以 `--disable-features=ApplySdpaFusion` 一键关掉整个融合,回退
到未融合的旧行为,作为 A/B 对照。

## 踩过的三个 bug

1. **`graph_output_cast_operators_` 泄漏** —— 构建子图时保存清空了这个
   deferred-cast 列表,但没有恢复,也没考虑到 Matmul#2 输出可能是需要
   cast 的 fp16 图输出。表现为模型加载直接失败("Failed to load model
   from buffer"),主图算子引用了越界的、属于子图的 tensor index。修复:
   提前序列化主图侧的输出 tensor info 并传给子图构建函数、子图内部把
   Matmul#2 输出预置成相同 dtype、子图构建完成后把子图内产生的 cast
   算子并入子图自己的 operator 列表、真正 restore
   `graph_output_cast_operators_`。
2. **Use-before-def**(发射点选在 `Transpose` 而不是链条末尾)——
   已在上面"两阶段设计"一节说明并解决。
3. **fp16 子图 cast 双写**(9/21 支持 rank-3 时引入并修复)—— 子图入口
   Q/K/V 若按 operand 声明的 fp16 dtype 序列化,
   `SerializeInputTensorInfo`(默认 `operation_supports_float16=false`)
   会插入 CAST 并返回 fp32 临时张量,但 map 仍指向 fp16 原始张量(TODO
   注释即此事):(a) entry reshape 写入 cast 输出 → 与 cast 双写入同一
   张量;(b) 链算子各自重新 cast 无生产者 fp16 张量 → CPU fallback 读到
   未初始化数据。14:34 的 fused dump(21 tensors/16 ops,3 个冗余 CAST)
   暴露此问题,14:52 修复后 15 tensors/10 ops 全 FLOAT32。

用 Chrome flags 组合 +
`sd/webnn_sdpa_fusion_test.html`(12 组 fp32/fp16 × raw/projected ×
{8,256,4096} 序列长度的最小 WebNN 用例)+ `sd/dump_all_subgraphs.py`
(把 tflite 子图的 tensor/operator 列表整个摊开肉眼检查)找到并验证修好了
这些 bug——比每次都跑完整 VAE decoder 模型 dump/编译/verify 快得多。

## WPT 测试

`third_party/blink/web_tests/external/wpt/webnn/conformance_tests/
sdpa_subgraph.https.any.js`:5 组用例 —— rank-3 fp32 / rank-3 fp16 /
dequantizeLinear-scale / rank-4 双头 / **rank-3 + 4D head reshape
dance**(与真实模型 Q 路径同构:reshape `[B,N,D]`→`[B,N,1,D]` →
transpose `[0,2,1,3]` → reshape 回 `[B,N,D]`)。

backend-agnostic,只测公开 API 的数值正确性(不依赖某 backend 是否融合);
在 Chromium 上 rank-3 用例走方案 A 的融合路径,reshape-dance 用例镜像
真实模型结构。期望值已手工验算。

## 端到端验证结果(2026-09-21)

真实模型:`tflite-dump-model/stable_diffusion/vae_decoder_no_sdpa.tflite`
(未融合基线,1012 个主图算子,0 个 composite,12:57 dump)vs
`vae_decoder_fused_sdpa.tflite`(融合后,14:52 dump,1011 个主图算子,
恰好 1 个 `STABLEHLO_COMPOSITE`,输入输出都是 `[1,4096,1,512]` FLOAT32,
子图 15 tensors/10 ops 全 FLOAT32)。

`sam_encoder_runner.exe --verify --model=vae_decoder_no_sdpa.tflite
--gpu-model=vae_decoder_fused_sdpa.tflite --tolerance=0.01`:

```
[verify-cpu] elems=786432 nan=0 min=-1.74138 max=0.649898  mean=-0.833536 std=0.72685
[verify-gpu] elems=786432 nan=0 min=-1.74057 max=0.650025  mean=-0.833022 std=0.726289
[verify] elems=786432 tol=0.01 max_abs=0.00224602 mean_abs=0.000790568 over_tol=0 nan_mismatch=0
[verify] PASS
```

fused 模型自身 `--model=<fused> --gpu-model=<fused>` 同样 PASS
(max_abs=0.00224578),且 fused-CPU 输出与 unfused-CPU 仅 min 末位差 1 ——
**"CPU 还是走原来的方案"这条硬约束成立**;786432 个元素零个超差,融合后
GPU 输出与 CPU 参考在 fp16 容差内一致(max_abs 0.0022,优于 9/5 时代的
0.0052)。

**环境事实(当前 ml-drift,9/21)**:ml-drift 不支持 `DEQUANTIZE`(且
TRANSPOSE 要求 INT32 perm、常量输入 RESHAPE 被拒),带 fp16 权重的模型在
纯 GPU 模式下整体编译失败(主图仅 10/1011 算子被认领;与融合无关,
9/5 时代的旧 ml-drift 支持 dequantize)。因此 `sam_encoder_runner
--verify` 的 GPU 侧从 `kGpuOnly` 改为 `kGpuAndCpu`(与浏览器
`GraphImplLiteRt::GetCompilationOptions` 一致):composite + 合成 reshape
在 GPU 上执行,其余算子与 unfused 走相同的混合路径,对比精确反映融合的
数值影响。详见 memory `mldrift-dequantize-gpu-limit`。

**顺带发现**(9/5 时代,保留):未融合模型在 GPU 上跑(走旧的拆解算子
序列)输出 `mean=-30.3043`,和 CPU 参考差了约 36 倍 —— 这是 ml-drift 在
拆解版 softmax/matmul 链路上一个既有的、和本次融合无关的 bug(融合工作
反而绕开了它)。之前几个 session 怀疑"融合引入了 GPU 数值回归",实际是
因为 `sam_encoder_runner.exe` 编译得太旧、`--gpu-model` 这个 CLI 开关还没
编译进去,静默地把 GPU 侧也测成了 `--model` 指向的未融合模型;另外还踩到
ml-drift WebGPU 回读默认 10 秒超时(这个模型在集显上要 ~19.5 秒)。

## 历史(9/5 迭代)

- 分支 `fuse_sdpa_for_sd`,提交 `d7f30f0b78`(当时的实现要求 rank-4,
  composite I/O `[1,4096,1,512]`,端到端 PASS,log 见
  `sd/vae_verify_fusion5.log`)。
- 9/21 发现:真实模型的注意力链是 rank-3,rank-4 限制导致 0 次融合;
  选定方案 A(合成 4D reshape)后落地为当前实现。

## 相关文件

- `services/webnn/tflite/graph_builder_tflite.h` / `.cc`(chromium 仓库,
  分支 `fuse_sdpa`,提交 `5e8e860757`)——融合实现本体。
- `services/webnn/tflite/sam_runner/sam_encoder_runner.cc` —— verify
  harness(GPU 侧 kGpuAndCpu)。
- `third_party/blink/web_tests/external/wpt/webnn/conformance_tests/
  sdpa_subgraph.https.any.js` —— WPT 测试。
- `sd/webnn_sdpa_fusion_test.html` —— 隔离的最小 WebNN 测试页,12 组用例。
- `sd/dump_all_subgraphs.py` / `sd/check_sdpa_scale.py` —— tflite
  子图/tensor/operator 全量 dump 工具。注:脚本内硬编码的 fujun 路径已
  失效,用 `PYTHONPATH=segment_anythings/tools;segment_anythings/
  sam_native_runner` + depot_tools `python3` 运行。
- `tflite-dump-model/stable_diffusion/` —— vae_decoder_no_sdpa.tflite /
  vae_decoder_fused_sdpa.tflite / parse_vae.py(log)。
- `sd/SDPA_FUSION_REVIEW.zh.md` —— 实现 review 与验证全记录。
