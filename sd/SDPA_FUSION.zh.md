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

## 扩展:UNet 多头注意力融合(2026-10-14)

### 起因

用 `--webnn-tflite-dump-model-skip-weights` dump 出
`tflite-dump-model/stable_diffusion/no_weights_model/unet.tflite` 后发现:
`vae_decoder.tflite` 的那一个自注意力块已经融合成
`STABLEHLO_COMPOSITE`,但 `unet.tflite` 里 **32 个 `SOFTMAX` 全部原样保留**
(224 个 `BATCH_MATMUL`,0 处出现 `scaled_dot_product_attention`)——上面
"匹配范围"一节描述的 matcher 只认单头,UNet 的 self/cross-attention 全是
多头,完全匹配不上。

### 诊断:UNet 注意力块的真实结构

用 `flatc.exe --json --raw-binary` 把 `unet.tflite` 转 JSON 逐算子核对
(权重被 skip-weights 清空,但 shape/type/op 序列完整),第一个 softmax
(index 116)附近的算子链:

```
[103] BATCH_MATMUL  (1x4096x320, 320x960)      -> 1x4096x960   # 融合的 QKV 投影
[104] SPLIT_V       -> 3x(1x4096x320)                          # 拆出 Q,K,V
[105] RESHAPE       1x4096x320 -> 1x4096x5x64                  # Q split-heads
[106] TRANSPOSE     1x4096x5x64 -> 1x5x4096x64   perm[0,2,1,3] # heads 前移
[107] RESHAPE       1x4096x320 -> 1x4096x5x64                  # K
[108] TRANSPOSE     1x4096x5x64 -> 1x5x64x4096   perm[0,2,3,1] # heads 前移 + 转置,一步到位
[109] DEQUANTIZE    fp16 标量 scale 常量 -> fp32
[110] MUL           (1x5x64x4096, scalar) -> same              # scale 乘在 Kᵀ 上
[111] RESHAPE       1x4096x320 -> 1x4096x5x64                  # V
[112] TRANSPOSE     1x4096x5x64 -> 1x5x4096x64   perm[0,2,1,3]
[113] DEQUANTIZE    同一个 fp16 标量常量 -> fp32
[114] MUL           (1x5x4096x64, scalar) -> same               # scale 乘在 Q 上
[115] BATCH_MATMUL  Q·Kᵀ -> 1x5x4096x4096
[116] SOFTMAX       axis=最后一维
[117] BATCH_MATMUL  (·V) -> 1x5x4096x64
[118] TRANSPOSE     1x5x4096x64 -> 1x4096x5x64   perm[0,2,1,3]  # heads 转回去
[119] RESHAPE       1x4096x5x64 -> 1x4096x320                  # 合并 heads
[121] BATCH_MATMUL  out-proj
```

全部 32 处注意力块分 8 种 shape 签名(16 个 self-attention + 16 个
cross-attention,K/V 来自 77-token 文本 embedding),heads ∈ {5,10,20},
head_dim 恒为 64,结构逐一致:

| Q shape | Kᵀ / V seq | heads | 种类 | 出现次数 |
|---|---|---|---|---|
| 1x5x4096x64 | 4096 / 77 | 5 | self / cross | 各 5 |
| 1x10x1024x64 | 1024 / 77 | 10 | self / cross | 各 5 |
| 1x20x256x64 | 256 / 77 | 20 | self / cross | 各 5 |
| 1x20x64x64 | 64 / 77 | 20 | self / cross(mid-block) | 各 1 |

与原 matcher 假设的三处结构性差异,缺一都匹配不上:

1. **布局**:Q/K/V 是 `[batch, heads, seq, head_dim]`(heads 在 axis 1),
   原 matcher 要求 rank-4 时 `shape[2] == 1`(退化单头),直接拒绝。
2. **scale 位置**:scale 被拆成两次,`Mul(K, s)` 和 `Mul(Q, s)`,各自
   发生在 **matmul 之前**;原 matcher 假定 `Transpose(K)` 的唯一消费者
   直接是 `matmul1`,这里唯一消费者是 `Mul`,第一步就匹配不上。
3. **入口/出口 transpose**:head 的 reshape→transpose / transpose→reshape
   搬运就发生在 matmul 链条的外侧,原 matcher 完全没有对接这部分的逻辑。

### kernel 能力核实:heads>1 完全被支持,Chromium 注释本身是错的

动手改代码前,先核实了两件事(各起了一个独立的调研 agent):

**ml-drift**:`IsSdpaSupported`
(`third_party/litert/src/ml_drift_delegate/tflite/support/support_sdpa.cc:32`)
的维度检查完全没有限制 heads 轴;真正的 kernel `MakeScaledDotProductAttention`
(`ml-drift/ml_drift/common/selectors/operation_selector.cc:52-97`)内部就是
```cpp
auto q_transposed = model_builder->Transpose(q, BHWC(0, 2, 1, 3));
auto k_transposed = model_builder->Transpose(k, BHWC(0, 2, 3, 1));
auto v_transposed = model_builder->Transpose(v, BHWC(0, 2, 1, 3));
```
把 heads 轴顶到 `BatchedMatMul` 的 batch 维上。决定性证据:ml-drift 自带的
Stable Diffusion UNet 示例
(`samples/stable_diffusion/unet_builder.cc:454-474`)做的就是完全相同的
多头 attention,字节级同构。

**XNNPACK**:`VisitDotAttentionNode`
(`tflite/src/tensorflow/lite/delegates/xnnpack/xnnpack_delegate.cc:6411`)
显式支持 MHA/MQA/GQA(`:6465-6469` 用 `key_proj.dims->data[2]` 判断是否
MQA/GQA 并走专门的 regroup 路径),测试覆盖到 `q_heads=32`。rank 必须严格
是 4(`:6473` 硬 `TF_LITE_ENSURE_EQ`),heads 轴固定读作倒数第二维
(`permute_q = {0,2,1,3}`,`:6516-6517`)—— 这部分原注释是对的。

两个 delegate 认领 `odml.scaled_dot_product_attention` 都**纯粹从 shape
推断 heads 数**,composite 本身没有、也不需要携带 `num_heads` 属性。

结论:`graph_builder_tflite.cc` 里原先 `q_shape[2] != 1` 这道门槛是
Chromium 自己转换器的限制,不是 kernel 的限制;多头 UNet 注意力块理论上
完全可以走融合路径,而且 **UNet 图里 heads 轴搬运的 transpose 本来就
存在**(上面算子链的 106/108/112/118),融合时把它们吸收掉而不是新插入
一份,composite 直接读写 `[batch, seq, heads, head_dim]`——与 kernel 要的
布局完全一致,相当于**零额外数据搬运**的融合。

GQA/MQA(K/V heads 少于 Q)被排除在匹配范围外:XNNPACK 有专门的 regroup
路径处理,但 ml-drift 的 `IsSdpaSupported` 没有对应逻辑,两个 delegate
行为会不一致,所以干脆不匹配这种情况,留给未融合路径。

### 实现(`graph_builder_tflite.h`/`.cc`,`dump_larger_model` 分支,未提交)

`bool rank3_chain` 换成三态枚举(`graph_builder_tflite.h:888`):

```cpp
enum class SdpaOperandLayout {
  kRank3,           // [batch, seq, head_dim],原有的合成 4D reshape 路径
  kRank4SeqHeads,   // [batch, seq, heads, head_dim],kernel 要的布局,原样直喂
  kRank4HeadsSeq,   // [batch, heads, seq, head_dim],UNet 的真实布局,
                    // 吸收前后的 transpose 而不是插入 reshape
};
```

`ScaledDotProductAttentionFusion`(`:925`)按需扩成可选字段:
`q_transpose`/`v_transpose`/`output_transpose`(吸收的四个 transpose 里的
三个,K 的那个复用原有的 `transpose` 字段,因为它同时承担"转置"与"heads
前移"两个作用)、`q_scale_dequantize`/`q_mul`、`k_scale_dequantize`/`k_mul`
(拆开的两次 scale),原有的 `scale_dequantize`/`mul` 仍然保留用于
"scale 乘在 matmul 结果上"这种(rank-3/rank-4-单头)情形。`scale` 字段
现在是三处 scale 的乘积(`q_scale * k_scale * post_scale`,UNet 里
`post_scale` 恒为 1,因为它压根没有第三次 scale)。

新增一个以 **softmax 为锚点**的匹配器
`RecordMultiHeadScaledDotProductAttentionFusion()`(`:8942`),与原有的
以 `Transpose(K)` 为锚点的 `RecordScaledDotProductAttentionFusion()`
并行跑(`CreateAndBuild()` 里先跑 transpose 扫描、再跑 softmax 扫描,
`:812-841`,保证同一条链不会被两个 matcher 重复认领)。匹配步骤:

1. `softmax.axis` 必须是最后一维(原 matcher 从未校验这个,是本次顺带
   补上的缺口)。
2. 剥离 softmax 输入上可能存在的 scale `Mul`——`TryResolveSdpaScaleMul()`
   (`:8841`)：判断某个 operand 是否由一个"其中一个操作数是可解析标量
   常量(或被 `dequantizeLinear` 包住)的 `Mul`"产生,如果是,剥掉它拿到
   更上游的真实输入和这一步的浮点 scale;如果该 `Mul`(或其 scale 的
   `dequantizeLinear`)还被别的地方消费,直接放弃匹配而不是部分融合。
3. 顺着 `matmul1`→反查其 a/b 操作数各自是否也套了一层 `TryResolveSdpaScaleMul`
   (对应 Q 和 K 各自的那次 scale)。
4. `matmul2` 的左操作数必须正好是 softmax 输出(`matmul(V, weights)` 是
   不同的计算,原 matcher 没卡这点,这里补上)。
5. `GetAbsorbableSdpaTransposeId()`(`:8914`)分别检查 Q/K/V 的真实输入
   是否各自经过一个**置换完全匹配**(`kSdpaHeadsFirstPermutation = {0,2,1,3}`
   给 Q/V,`kSdpaHeadsFirstTransposedPermutation = {0,2,3,1}` 给 K——
   heads 前移与"转置"一步到位)**且输出无其他消费者**的 `Transpose`;
   同时检查 `matmul2` 输出之后是否跟着一个用同一置换转回去的
   `Transpose`。四个条件但凡一个不满足就放弃。
6. 最后校验 Q/K/V 三者 `[batch, heads, head_dim]` 必须一致、只允许 seq
   长度不同(自注意力 vs 交叉注意力的唯一差异),显式排除 GQA/MQA。

命中后把四个 transpose、`matmul1`、`softmax`、各处 scale 的 `Mul`/
`DequantizeLinear` 一并记入 `fused_ops_to_skip_`,在 `matmul2` 的位置
发射 composite——延续原有"在链条最后一个算子才发射"的两阶段设计,规避
use-before-def。

`SerializeScaledDotProductAttentionComposite()` /
`BuildScaledDotProductAttentionDecompositionSubgraph()` 按 `fusion` 里
哪些指针非空决定重放哪些算子(`kRank4HeadsSeq` 比原来多重放 4 个
transpose 和 2 组 scale mul/dequantize),`kRank4SeqHeads`/`kRank4HeadsSeq`
都不需要合成 reshape,`composite_output_tensor_index` 直接复用
`SerializeOutputTensorInfo` 返回的 tensor——这两种布局下 composite 的
输入输出形状本来就与 kernel 要的一致。

### 当前状态:**融合计数已验证全量命中,数值/性能验证尚未做**

重新 dump 出的 `model1.tflite`(用 `flatc.exe --json --raw-binary` 转
JSON 后用脚本统计算子类型)确认:主图里 `SOFTMAX` 从 32 降到 **0**,新增
**32** 个 `odml.scaled_dot_product_attention` composite,对应生成
**32** 个 `scaled_dot_product_attention_decomposition` 子图(每个恰好
1 个 `SOFTMAX` + 2 个 `BATCH_MATMUL`,即未融合时的原始算子,用于
CPU/XNNPACK 没认领 composite 时的保真回退)。UNet 的 16 个 self-attention
+ 16 个 cross-attention 块**全部**命中多头匹配器,一个没漏。

统计方法:

```
$outdir = "$env:TEMP\tflite_json"
& flatc.exe --json --raw-binary --strict-json -o $outdir schema.fbs -- model1.tflite
# 用 Python 读 JSON,按 operator_codes[opcode_index].builtin_code 分类计数,
# 并在 STABLEHLO_COMPOSITE 上读 builtin_options_2.name 区分
# odml.scaled_dot_product_attention / odml.group_norm 等不同 composite。
```

遗留的两个风险点仍未验证:

1. **精度**:原图是 `(Q·s)·(K·s)ᵀ` 两次缩放后再相乘,融合后变成对
   `matmul1` 的结果一次性乘 `s²`。数学上等价,但如果 kernel 内部实现是
   在 `QKᵀ` **之后**才应用 scale(常见做法),这完全匹配;但要留意 fp16
   累加路径上是否有任何一步因为合并而改变了中间量级。
2. **回归**:`vae_decoder.tflite` 走的 `kRank3` 路径只是把原来的
   `bool rank3_chain` 机械替换成枚举分支,语义上应该等价,但这次改动
   没有重新单独验证它。

验证计划(尚未执行):跑 SD-Turbo demo 页面肉眼比对出图;有空的话仿照
2026-09-21 那次验证,extend `sam_encoder_runner` 或类似工具做 UNet
粒度的 CPU vs GPU 数值 diff;以及下面的性能对比方法。

## 性能对比方法:有/无 SDPA 融合(2026-10-14)

不需要额外写代码,用已有的 feature flag + Chrome 内置 Perfetto tracing
就能做 A/B 对比。

### 1. 开关融合——feature flag

复用"Feature flag"一节提到的 `kApplySdpaFusion`
(`graph_builder_tflite.cc:73`,默认开启),命令行一个开关关掉:

```
# 有 SDPA 融合(默认)
chrome.exe --no-sandbox --enable-features=WebMachineLearningNeuralNetwork \
  --disable-features=WebNNDirectML,WebNNOnnxRuntime

# 无 SDPA 融合(对照组)
chrome.exe --no-sandbox --enable-features=WebMachineLearningNeuralNetwork \
  --disable-features=WebNNDirectML,WebNNOnnxRuntime,ApplySdpaFusion
```

两组其余参数(GPU device、模型、页面)保持完全一致,只有这一个 feature
的差异。

### 2. 精确计时——Chrome 内置 tracing,`webnn` category

WebNN 自带 Perfetto trace(`services/webnn/public/cpp/webnn_trace.cc:14`
category 叫 `"webnn"`),`graph_impl_litert.cc` 里每次 `Dispatch` 都会打
一个 `"Run inference"` step(`:403`),正好是 SDPA 融合影响到的那段
耗时,且已经排除了 `"Acquire resources"`(`:843`)/
`"Set up input and output buffers"`(`:373`)这些非计算开销。命令行直接
录制成文件,不用手动点 chrome://tracing:

```
chrome.exe --no-sandbox --enable-features=WebMachineLearningNeuralNetwork \
  --disable-features=WebNNDirectML,WebNNOnnxRuntime \
  --trace-startup=webnn --trace-startup-file=trace_with_sdpa.json
```

跑完若干轮推理后关闭浏览器(trace 在进程退出或 buffer 满时落盘),把
`.json` 拖进 https://ui.perfetto.dev,搜索 `"Run inference"`,看这个
slice 的 duration 分布。两组(有/无 `ApplySdpaFusion`)各跑一遍,对比
同一个 step 名字的耗时。

### 3. 跑测注意事项

- **跑够轮数**:SD-Turbo 每次生成会多次调用 UNet(每个 timestep 一次),
  第一次调用通常包含 LiteRT 编译/delegate 下发的一次性开销,建议看第
  2 轮之后的稳定值,或者直接看多次 `"Run inference"` slice 的中位数
  而不是均值。
- **确认 delegate 真的认领了 composite**:如果 ML Drift/XNNPACK 没认领
  `STABLEHLO_COMPOSITE`(比如遇到不支持的 shape),会静默跑
  decomposition 子图里的未融合算子,这种情况下两组耗时应该几乎没差别
  ——如果测出来没有任何提升,先加 `--vmodule=*litert*=1,*webnn*=1` 确认
  GPU 日志里没有 fallback 警告。
- **同一硬件、同一轮次对齐**:UNet 每个 timestep 都会触发一次
  `Dispatch`,对比时要确保两组统计的是同一个阶段(比如都是第 3 个
  timestep 之后),不要把冷启动的那一轮也算进去。

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
- `tflite-dump-model/stable_diffusion/no_weights_model/unet.tflite` ——
  UNet 多头扩展的诊断样本(skip-weights dump,结构完整、权重为空)。
