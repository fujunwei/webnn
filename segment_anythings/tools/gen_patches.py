"""Regenerate segment_anythings/patches/*.patch from the two working trees.

Both third_party checkouts carry uncommitted work that spans several unrelated
fixes, and some files mix hunks from different fixes (operation_selector.cc and
delegate_kernel.cc in particular). This splits the working-tree diff into one
patch per logical change so they can be committed separately.

Each entry is (patch_filename, repo, subject, body, [(path, hunks_or_None)]).
`hunks` is a list of 0-based hunk indices within that file's diff; None keeps
every hunk.

Usage: py gen_patches.py [--check]
"""
import argparse
import os
import subprocess
import sys

LITERT = r"C:\Users\fujun\workspace\chromium\src\third_party\litert\src"
MLDRIFT = r"C:\Users\fujun\workspace\chromium\src\third_party\ml-drift"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches")


def file_diff(repo, path):
    return subprocess.run(["git", "diff", "--", path], cwd=repo,
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
    ("24-mldrift-5d-reshape-transpose.patch", MLDRIFT,
     "ml-drift: handle 5D reshape/transpose attributes",
     """Completes 5D (BHWDC) reshape/transpose support in ml-drift:

- add std::any_cast branches for Reshape3DAttributes and
  Transpose3DAttributes in the legacy GPUOperationFromNode selector;
- skip 5D identity-reshape in RemoveIdentityReshape to avoid
  std::bad_any_cast.

NOTE: an earlier revision of this patch also demoted the shape-mismatch
warning in GpuModelBuilder::UpdateOutputTensors to VLOG(1), on the theory
that the mismatch was a harmless axis-relabeling artifact. That was wrong
-- the truncated shape is the one installed by SetOutputDescriptor(), so
the warning marks real data loss. That hunk is deliberately gone; see
patch 26, which removes the mismatch instead of hiding it.""",
     [("ml_drift/common/selectors/operation_selector.cc", [1, 2]),
      ("ml_drift/common/transformations/remove_noop.cc", None)]),

    ("25-mldrift-pow-negative-base.patch", MLDRIFT,
     "ml-drift: guard pow() against negative bases off the OpenCL path",
     """WGSL, MSL and GLSL all leave pow(x, y) undefined for x < 0; Dawn lowers
it to exp2(y * log2(x)), which is NaN. The OpenCL path already handles
this via PowUsingNativePowr, but the shared non-OpenCL path emitted a
bare pow($1, $2).

Any decomposed LayerNorm computes pow(x - mean, 2), and x - mean is
negative for about half of its elements, so a single such node turns the
whole downstream tensor into NaN once the following MEAN spreads it. On
the SAM encoder this made all 1048576 output elements NaN; excluding just
the two POW nodes (delegate indices 1229 and 1248) restored finite output.

The existing pow(x,2) -> x*x lowering does not help here because it lives
in CreateElementwiseOneRuntimeOneScalar, and this model reaches POW with a
runtime exponent (an fp16 constant 2.0 behind DEQUANTIZE -> RESHAPE), so
the two-input kernel is selected instead.

Mirrors PowUsingNativePowr's contract, including its treatment of a
non-integer exponent with a negative base.""",
     [("ml_drift/common/kernels/elementwise.cc", None)]),

    ("26-mldrift-rank5-shape-plumbing.patch", MLDRIFT,
     "ml-drift: allocate rank-5 graph values with their real shape",
     """GraphFloat32::Value holds a TensorRef<BHWC>, so a frontend with a 5D
tensor has to drop a dimension to store it. ReserveGraphTensors() then
allocated e.g. [5,14,5,14,768] as BHWC(5,14,5,14) -- 768x too small --
and UpdateOutputTensors() overwrote the correct 5D descriptor with the
truncated one via SetOutputDescriptor(). On the SAM encoder this produced
100 shape-mismatch warnings and garbage through the whole window-attention
path.

Adds CreateGpuModelInfo::rank5_shapes, a ValueId -> BHWDC side table that
frontends able to see the original rank can populate (see the companion
litert patch). GetTensorDescForValue() then builds a BHWDC/HWDC descriptor
for those values, mirroring the layout selection already used by
GpuModelBuilder::AddTensor(b, h, w, d, c, ...).

Also switches the legacy elementwise selector to read shapes from the
tensor descriptors rather than Value::tensor.shape, which is BHWC and
therefore truncated; the IR selector already did this. Without it the
rank-5 broadcast ADDs of SAM's relative-position bias pick the wrong
broadcast.

The kernels needed no changes -- reshape.cc, transpose.cc and
elementwise.cc already handle Axis::DEPTH.""",
     [("ml_drift/common/BUILD", None),
      ("ml_drift/common/gpu_model.h", None),
      ("ml_drift/common/gpu_model_util.cc", None),
      ("ml_drift/common/selectors/operation_selector.cc", [0])]),

    ("27-mldrift-webgpu-readback-timeout.patch", MLDRIFT,
     "ml-drift: make the WebGPU readback timeout configurable",
     """MapAsync on the readback buffer only completes once every previously
queued command has executed, so its timeout is really a budget for the
whole inference rather than for the copy. The hardcoded 10s is not enough
for a large graph on an integrated GPU: the SAM encoder needs about 11s
on an Intel UHD 630 and failed with "Timed out waiting for future: 10s",
which reads like a hang or an OOM rather than a slow device.

Adds MLD_WEBGPU_READBACK_TIMEOUT_SECONDS. Default behaviour is unchanged.""",
     [("ml_drift/webgpu/webgpu_api_util.cc", None)]),

    ("28-litert-rank5-shape-collection.patch", LITERT,
     "litert: recover the real shape of rank-5 values for ml-drift",
     """BuildFinalModel() records every value's shape through
ExtractTensorShape(), which drops everything past the 4th dimension
because GraphFloat32::Value holds a TensorRef<BHWC>. ml-drift then
allocated rank-5 tensors far too small (see the companion ml-drift patch
adding CreateGpuModelInfo::rank5_shapes).

Value::tensor.ref is the TFLite tensor index, so the original dims are
still reachable after BuildFinalModel() returns. Walk the graph values
once and populate rank5_shapes, guarding against a stale ref by checking
that the truncation of those dims is what the value actually carries.

This covers every op uniformly rather than each parser separately: on the
SAM encoder 140 rank-5 tensors are produced by RESHAPE, TRANSPOSE and
broadcast ADD.""",
     [("ml_drift_delegate/delegate/BUILD", None),
      ("ml_drift_delegate/delegate/delegate_kernel.cc", [1, 3])]),

    ("29-litert-transpose-constant-permutation.patch", LITERT,
     "litert: actually permute constant data for non-4D TRANSPOSE",
     """TransposeConstantData() bailed out with

    if (perm_data.size() != 4 || elements == 0) { dst_data = src_data; }

so a rank-2 or rank-3 constant TRANSPOSE relabelled the shape but left
the data untouched. The SAM encoder has 48 rank-2 transposes -- the
qkv/proj/fc1/fc2 weight matrix of each of the 12 blocks, reached as
fp16 constant -> DEQUANTIZE -> TRANSPOSE -- so every projection ran
against a wrongly laid out weight matrix, and the error accumulated per
block. Cosine against a CPU reference went from 0.401 to 0.859 once
fixed. A 3x4 repro gives CPU [0,4,8,1,5,9,2,6,10,3,7,11] against GPU
[0,1,2,...,11].

Introduces BhwcPermFromTflitePerm(), which re-expresses a rank 2/3/4
TFLite permutation as a BHWC axis permutation following the axis
placement ExtractTensorShape() uses, and shares it between the constant
and runtime paths. The runtime path is behaviour-preserving for all three
ranks; it previously open-coded the same mapping.

Two related fixes fall out of sharing the mapping:
- the constant path built rank-3 shapes as BHWC(1,D0,D1,D2) while
  ExtractTensorShape() uses BHWC(D0,1,D1,D2);
- an unsupported rank now refuses the node so it falls back to CPU,
  rather than silently shipping untransposed weights.""",
     [("ml_drift_delegate/tflite/model_builder.cc", None)]),

    ("30-litert-partition-node-logging.patch", LITERT,
     "litert: log which nodes the GPU delegate actually takes",
     """The [node-table] dump lists every node considered for delegation. On a
partially-delegated model that is far more than what runs on the GPU --
the SAM encoder's decomposed variant offers 1960 nodes and delegates 207
of them, in 170 fragments -- which makes LITERT_GPU_DEBUG_EXCLUDE_NODES
experiments look like no-ops when the excluded class was never delegated
in the first place.

Print delegate_params->nodes_to_replace under the existing
LITERT_GPU_DEBUG_DUMP_NODES flag.""",
     [("ml_drift_delegate/delegate/delegate_kernel.cc", [0, 2])]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print sizes instead of writing files")
    args = ap.parse_args()

    for name, repo, subject, body, files in PATCHES:
        parts = [subject, "", body, ""]
        for path, hunks in files:
            d = file_diff(repo, path)
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
