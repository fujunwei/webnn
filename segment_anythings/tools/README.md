# Tools for debugging SAM on the WebGPU delegate

Everything here is plain Python with no TensorFlow dependency — the `.tflite`
files are read (and written) through the minimal flatbuffer reader in
[`../sam_native_runner/inspect_tflite.py`](../sam_native_runner/inspect_tflite.py).

Most scripts drive `sam_encoder_runner.exe`; see
[`../gpu_op_bisect.zh.md`](../gpu_op_bisect.zh.md) for the surrounding workflow
and, importantly, §0.2 for which localisation methods are known to give wrong
answers on this model.

## Model inspection

| Script | What it does |
|---|---|
| `rank5_scan.py <model>` | Inventory of rank≥5 tensors and the ops producing/consuming them. Answers "how much of ml-drift needs 5D support". |
| `node_table.py` | `[node-table]` log lines → opcode table → `EXCLUDE_NODES` lists. |
| `view_bin.py <f32.bin>` | Dump stats of a raw float32 blob (nan count, min/max/mean/std, first values). |
| `conv_audit.py` | Convolution parameter audit. |

## Repro model generation

`make_rank5_repro_tflite.py` hand-writes tiny `.tflite` files that isolate one
suspected bug each. All are under 8 KB and verify in well under a second, which
makes them the fast inner loop — the full encoder takes ~30 s to compile.

```powershell
py make_rank5_repro_tflite.py            <out.tflite> [<in.bin>]   # 5D reshape/transpose/broadcast-add
py make_rank5_repro_tflite.py --rank4    <out.tflite>              # same ops, all rank 4 (control)
py make_rank5_repro_tflite.py --pow-neg  <out.tflite> [<in.bin>]   # pow(negative, runtime 2.0)
py make_rank5_repro_tflite.py --transpose2d <out.tflite>           # rank-2 constant TRANSPOSE
py make_rank5_repro_tflite.py --bcast5d  <out.tflite> [<in.bin>]   # 5D broadcast ADD
py make_rank5_repro_tflite.py --layernorm <out.tflite> [<in.bin>]  # one custom_call.LayerNorm (real flexbuffer bytes)
```

Run one with:

```powershell
sam_encoder_runner.exe --model=<out.tflite> --verify --precision=fp32 `
                       --input=<in.bin> --tolerance=1e-3
```

`--verify` compiles the model twice (CPU reference and GPU) and compares, so it
needs no external ground truth. It only works on models with no custom ops.

Caveat: `--bcast5d` currently reproduces a *crash*
(`object_reader.cc: Check failed: CanReadValue`) rather than the numeric bug it
was written for — a rank-5 constant operand to an elementwise op hits
`AddInput()`, whose precondition is that the input is *not* constant. It does
not model SAM's relative-position-bias ADDs, whose operands are runtime values.

## Localisation

| Script | What it does |
|---|---|
| `bisect_end_node.py` | Binary-search `LITERT_GPU_DEBUG_END_NODE` for the first node whose delegation diverges from a reference. **Unreliable on SAM** — see gpu_op_bisect.zh.md §0.2. |
| `exclude_sweep.py` | Exclude opcode classes and score against a reference; can binary-search over the set of classes. |
| `exclude_class.py` | Exclude one opcode class, restricted to the nodes *actually delegated*. Use this rather than `exclude_sweep.py` on partially-delegated models, otherwise untouched classes read as false negatives. |
| `fp16_zeros_bisect.py` | Delta-debug with a binary all-zero/nonzero signal over non-LayerNorm nodes (fp16 mode); `--fp32 --ref=` switches to a cosine-jump signal. Found the fp16 all-zeros culprit (patch 31). LayerNorm nodes never leave the GPU. |

Get the truly-delegated node list first (needs patch `30`):

```powershell
$env:LITERT_GPU_DEBUG_DUMP_NODES=1
sam_encoder_runner.exe --model=<model> --precision=fp32   # look for "[partition] N nodes delegated:"
```

## Maintenance

`gen_patches.py` regenerates `../patches/*.patch` from the two third-party
working trees, splitting by logical change rather than by file. See
[`../patches/README.md`](../patches/README.md).

## Environment variables

| Variable | Effect |
|---|---|
| `LITERT_GPU_DEBUG_DUMP_NODES=1` | print the candidate node table and the delegated node list |
| `LITERT_GPU_DEBUG_ONLY_NODE_COUNT=<N>` | scope the two below to the graph with N nodes (SAM ships two models) |
| `LITERT_GPU_DEBUG_END_NODE=<N>` | only nodes `[0, N]` are GPU candidates |
| `LITERT_GPU_DEBUG_EXCLUDE_NODES=a,b,c` | drop these nodes from the GPU partition |
| `MLD_WEBGPU_READBACK_TIMEOUT_SECONDS=<s>` | readback budget, default 10; the SAM encoder needs ~11 s (patch `27`) |
