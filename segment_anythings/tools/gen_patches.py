"""Regenerate segment_anythings/patches/*.patch from the two working trees.

Both third_party checkouts carry uncommitted work that spans several unrelated
fixes, and some files mix hunks from different fixes (operation_selector.cc and
delegate_kernel.cc in particular). This splits the working-tree diff into one
patch per logical change so they can be committed separately.

Each entry is (patch_filename, repo, subject, body, [(path, hunks_or_None)]).
`hunks` is a list of 0-based hunk indices within that file's diff; None keeps
every hunk. Entries listed in COMMITS are sourced from that commit's diff
against its parent instead of a bare working-tree/staged diff (used for the
chromium/src checkout, which commits its WebNN changes).

Usage: py gen_patches.py [--check]
"""
import argparse
import os
import subprocess
import sys

LITERT = r"C:\Users\junwei\workspace\chromium\src\third_party\litert\src"
MLDRIFT = r"C:\Users\junwei\workspace\chromium\src\third_party\ml-drift"
CHROMIUM = r"C:\Users\junwei\workspace\chromium\src"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches")


def file_diff(repo, path, commit=None):
    # The chromium/src checkout commits its WebNN changes (unlike the
    # litert/ml-drift checkouts, which carry bare working-tree edits), so
    # those patches are sourced from a specific commit instead.
    if commit:
        args = ["git", "diff", commit + "~1", commit, "--", path]
    else:
        args = ["git", "diff", "--", path]
    return subprocess.run(args, cwd=repo,
                          capture_output=True, text=True, check=True).stdout


def select_hunks(diff, hunks):
    """Keep only the listed hunks, preserving the file header."""
    if hunks is None:
        return diff
    lines = diff.splitlines(keepends=True)
    header, i = [], 0
    while i < len(lines) and not lines[i].startswith("@@"):
        header.append(lines[i])
        i += 1
    chunks, cur = [], None
    for line in lines[i:]:
        if line.startswith("@@"):
            if cur is not None:
                chunks.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur is not None:
        chunks.append(cur)
    kept = []
    for idx in hunks:
        if idx >= len(chunks):
            raise SystemExit("hunk %d missing (file has %d)" % (idx, len(chunks)))
        kept.extend(chunks[idx])
    return "".join(header) + "".join(kept)


PATCHES = [
    ("31-mldrift-winograd-f16-constants.patch", MLDRIFT,
     "ml-drift: bake Winograd transform constants as FLOAT32 in WGSL",
     """The 3x3-conv Winograd kernels bake the Bt/At transform matrices into the
WGSL source via BufferToKernelLanguage, using the input tensor's data type.
In fp16 mode that yields `const Bt_buffer = array<f16, 36>(...)`, which Dawn
rejects on devices without the shader-f16 feature ("'f16' type used without
'f16' extension enabled"). The invalid shader makes the whole command buffer
invalid, so the dispatch silently never runs and the output buffers stay
zero-initialized: the fused SAM encoder (single GPU fragment, fp16 mode)
read back exactly 1048576 zeros with nan=0 and no error.

Root-caused with a shader-dump hook (patch 32): the only two WGSL modules
containing f16 were the Winograd Bt/At kernels. The fix bakes the constants
as FLOAT32 in all four call sites; the kernels already read them into f32
arrays, so precision only improves (f16 baking also quantized values like
sqrt(0.5) to 0.70703125).

Verified on the Intel UHD 630 (no shader-f16): new fused model fp16 now
outputs cosine=0.999977 vs the same model at fp32, and no WGSL validation
errors remain. The fp32 path was already unaffected (src data type = f32).""",
     [("ml_drift/common/kernels/winograd.cc", None)]),

    ("33-litert-bmm-batch-broadcast.patch", LITERT,
     "litert: broadcast BATCH_MATMUL batch axes in the ML Drift parser",
     """BatchedMatMulOperationParser assumed every rank-4 BMM has matching
batch dims ([model_batch, matmul_batch, M, K] on both sides). The SAM
encoder attention broadcasts instead:

  - B0 broadcast (QK^T): left [14,14,300,64] @ right [1,14,64,14]. The
    left merges to [1,196,300,64] (H = model batch outer, slot m*B1+k =
    left[m][k]) but the right stayed [1,14,64,14], so the conv weights
    indexed only 14 of the 196 batch slots and every block's attention
    ran against the wrong K. Cosine vs CPU 0.924 -> fixed with a TILE of
    the right along H (block repeat: slot m*B1+k reads right[k]).
  - B1 broadcast (attn@V weights, constant): left [14,14,300,64] @ right
    [14,1,64,14]. A plain TILE cannot express slot m*B1+k = right[m]
    (TILE repeats blocks, not interleaves), so the constant data is
    interleaved on the host and baked into the const node as
    [1,B0*B1,K,N].

Any other batch broadcast combination is refused in IsSupported so the
node falls back to CPU instead of computing wrong results.""",
     [("ml_drift_delegate/tflite/model_builder.cc", None)]),

    ("34-webnn-matmul-batch-broadcast.patch", CHROMIUM,
     "webnn: broadcast mismatched matmul batch dims before BATCH_MATMUL",
     """WebNN's matmul validation allows the batch dims (every axis but the
trailing two) of the two operands to differ as long as they are
NumPy-broadcastable. GraphBuilderTflite::SerializeMatmul previously passed
such operands straight through to TFLite's BATCH_MATMUL unchanged. TFLite's
BATCH_MATMUL spec permits this, and the XNNPACK CPU kernel implements it
correctly, but not every BATCH_MATMUL backend does: some only handle the
case where one operand is unbatched (rank 2) and the other is batched, and
silently mis-compute cases where both operands are batched (rank >= 3, same
rank) but disagree on one or more batch axes -- see
33-litert-bmm-batch-broadcast.patch for the concrete GPU-delegate failure
this caused on the SAM encoder.

Fix this at the graph-builder level instead of per-backend: insert explicit
BROADCAST_TO ops ahead of BATCH_MATMUL for whichever operand has smaller
batch dims, so every backend always receives operands with matching batch
dims. This cannot change the result on backends that already broadcast
correctly (XNNPACK), it only moves which op performs the replication, and it
makes 33-litert-bmm-batch-broadcast.patch's GPU-side workaround dead code
(verified by reverting it and re-running the regression test below through
the real GPU delegate -- still passes).

Add WebNNGraphImplBackendTest.MatmulBatchDimsBroadcast, covering both an
outermost batch-axis broadcast and an inner batch-axis broadcast (the
harder case, since the outer axis already matches at a non-1 value).

Add matching WPT conformance test cases to matmul.https.any.js (float32 and
float16) for the inner-batch-axis-broadcast shape, which was not previously
covered by any existing broadcast test case there (all prior "(broadcast)"
cases only broadcast the outermost batch axis).""",
     [("services/webnn/tflite/graph_builder_tflite.cc", None),
      ("services/webnn/webnn_graph_impl_backend_test.cc", None),
      ("third_party/blink/web_tests/external/wpt/webnn/conformance_tests/matmul.https.any.js",
       None)]),

    ("35-mldrift-conv-weights-texture-fallback.patch", MLDRIFT,
     "ml-drift: fall back to global memory when conv weights exceed texture",
     """GetKernelParamsAdreno checks the kTexturesX4 weights resource size
against the adapter's image2D limits and falls back to kGlobalMemory,
but the non-Adreno paths (including WARP and the WebGPU generic path)
selected kTexturesX4 unconditionally. A different_weights_for_height
conv with H = 4096 (the SAM window attention's matmul-as-conv) then
allocates a 16 x 65536 weights texture, which Dawn rejects on adapters
with 16384^2 limits; the invalid pipeline silently kills the dispatch and
the output reads back all zeros.

Move the same check into the generic GetKernelParams tail so every path
degrades to global memory instead of emitting an invalid texture.""",
     [("ml_drift/common/kernels/conv_generic.cc", None)]),

    ("36-litert-ir-bmm-batch-broadcast.patch", LITERT,
     "litert: fix batch broadcast direction in the IR-pipeline BATCH_MATMUL converter",
     """ConvertBatchMatMul (the ml_drift_delegate/tflite/convert/* front end used
by the IrModel pipeline, a separate code path from the legacy
model_builder.cc parser fixed by 33-litert-bmm-batch-broadcast.patch)
flattens each operand's leading batch dims into a single axis before
handing left/right to ml-drift's BatchedMatMul, e.g. left [B,H,M,K] ->
[1,B*H,M,K]. ml-drift's BMM-as-conv kernel broadcasts a smaller right
batch against the *last* dim of this flattened axis (i.e. against H),
matching TFLite's BATCH_MATMUL only when the right operand's own
broadcast dim is B (right = [1,H,K,N] with r.b==1). SAM's decoder
instead broadcasts a right operand shaped [B,1,K,N] (r.b!=1, r.h==1)
against a left [B,H,M,K] -- TFLite's right-aligned NumPy broadcast
picks left's *outermost* batch dim (B), but ml-drift's flattened
conv-as-matmul broadcasts against the innermost one (H), so every group
but the first read the wrong right-hand rows.

Detect this specific mismatch (left_is_5d and right_is_5d both false,
l.b!=1, l.h!=1, r.b!=1, r.h==1) and swap the left operand's (b,h) axes
with a TRANSPOSE before flattening, then swap the output back with a
matching trailing TRANSPOSE, so ml-drift's last-dim broadcast lines up
with the batch dim TFLite actually broadcasts against.""",
     [("ml_drift_delegate/tflite/convert/convert_batch_matmul.cc", None)]),

    ("37-litert-enable-ir-model-pipeline.patch", LITERT,
     "litert: default the GPU delegate to the IrModel pipeline",
     """use_ir_model gated which of two independent front ends builds the
ml-drift graph: the legacy GraphFloat32 pipeline (model_builder.cc,
fixed for BATCH_MATMUL broadcast by 33-litert-bmm-batch-broadcast.patch)
or the newer IrModel pipeline (convert_batch_matmul.cc et al., fixed by
36-litert-ir-bmm-batch-broadcast.patch). It defaulted to false, so the
SAM decoder -- and any other model exercised through the default
delegate options -- ran the legacy parser and never took the IrModel
converter's batch-broadcast fix. Flip the default to true so the fixed
pipeline is what actually executes; both pipelines remain compiled and
selectable at runtime for callers that explicitly set the option.""",
     [("ml_drift_delegate/delegate/delegate_options.h", None)]),

    ("38-mldrift-bmm-conv-weights-dst-height.patch", MLDRIFT,
     "ml-drift: size BMM-as-conv weights layout by dst height, not right H*W",
     """GpuModelBuilder::BatchedMatMul's conv-based path always requests
different_weights_for_height=true, so the packed conv weights buffer
must have one row per destination row (0..dst_shape.h-1): the kernel
indexes it by output row and broadcasts the right operand's batch over
dst height. WeightsConversion was instead sized from the right
operand's own OHWI shape (H*W = right's batch count), which only
matches dst_shape.h when the right operand isn't broadcast. Once
36-litert-ir-bmm-batch-broadcast.patch's left/right flattening produces
a dst_shape.h larger than the right operand's H*W (e.g. dst_shape.h=6
attention rows broadcasting right's 2 real K/V batches), the packer
only wrote 2 rows and the kernel read rows 2-5 out of the allocated
buffer, so 4 of every 6 attention blocks got garbage instead of the
broadcast K/V weights.

Grow only the layout's H so H*W == dst_shape.h, keeping W (the
converter's source-row divisor, `sp % W`) equal to the right operand's
real batch count, so the broadcast still cycles through the correct
rows instead of just allocating more space.""",
     [("ml_drift/common/gpu_model_builder.cc", None)]),
]

# Patches sourced from a specific commit instead of a bare working-tree/staged
# diff. Used for the chromium/src checkout (which commits its WebNN changes)
# and for litert/ml-drift fixes that were committed on a feature branch rather
# than left as uncommitted working-tree edits.
COMMITS = {
    "34-webnn-matmul-batch-broadcast.patch": "c6367c7848",
    "36-litert-ir-bmm-batch-broadcast.patch": "f0a65ed5e29dbec29d1a0f365d38e48da227bd47",
    "37-litert-enable-ir-model-pipeline.patch": "75fc311ff27c99179984f6d931dada017a9243b4",
    "38-mldrift-bmm-conv-weights-dst-height.patch": "8f7989895f8420a2b9975f1c5c620730a87f3fdd",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print sizes instead of writing files")
    args = ap.parse_args()

    for name, repo, subject, body, files in PATCHES:
        commit = COMMITS.get(name)
        parts = [subject, "", body, ""]
        for path, hunks in files:
            d = file_diff(repo, path, commit)
            if not d.strip():
                raise SystemExit("no diff for %s in %s" % (path, repo))
            parts.append(select_hunks(d, hunks))
        text = "\n".join(parts[:4]) + "\n" + "".join(parts[4:])
        dest = os.path.join(OUT, name)
        if args.check:
            print("%-52s %6d bytes" % (name, len(text)))
        else:
            with open(dest, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print("wrote %s (%d bytes)" % (name, len(text)))


if __name__ == "__main__":
    main()
