"""Find which opcode class computes wrong on GPU, by excluding classes from it.

LITERT_GPU_DEBUG_EXCLUDE_NODES drops the given nodes from the GPU partition so
they run on CPU. If excluding a set of opcodes makes the output match a trusted
CPU reference, the offending op is in that set; binary-searching the set costs
log2(classes) runs instead of one run per class.

Unlike LITERT_GPU_DEBUG_END_NODE this does not cut the graph at an arbitrary
point, so it does not manufacture partition-boundary artifacts at rank-5
tensors -- which is what made END_NODE bisection unreliable on this model.

Only safe on models with no custom ops: an excluded custom_call.LayerNorm has
no CPU kernel. Use segment_anything_encoder.tflite, not the "new_" one.

Usage:
  py exclude_sweep.py --model M --input IN --ref REF --total-nodes N
                      [--codes 6,22,18]   # test one explicit set and stop
"""
import argparse
import collections
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "sam_native_runner"))
from inspect_tflite import Table  # noqa: E402

RUNNER = (r"C:\Users\fujun\workspace\chromium\src\out\upstream_bots_debug"
          r"\sam_encoder_runner.exe")
TMP_PREFIX = "D:/tflite-dump-model/_sweep"


def nodes_by_opcode(model_path):
    buf = open(model_path, "rb").read()
    model = Table(buf, struct.unpack_from("<I", buf, 0)[0])
    codes = []
    for i in range(model.vec_len(1)):
        oc = model.vec_table(1, i)
        b = (oc.scalar(0, "<b") or 0) if oc.has(0) else 0
        d = (oc.scalar(3, "<i") or 0) if oc.has(3) else 0
        codes.append(max(b, d))
    sg = model.vec_table(2, 0)
    out = collections.defaultdict(list)
    for oi in range(sg.vec_len(3)):
        out[codes[sg.vec_table(3, oi).scalar(0, "<i") or 0]].append(oi)
    return out


def run(args, exclude_nodes, ref):
    out = TMP_PREFIX + "_run.bin"
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ)
    env["MLD_WEBGPU_READBACK_TIMEOUT_SECONDS"] = "900"
    env["LITERT_GPU_DEBUG_ONLY_NODE_COUNT"] = str(args.total_nodes)
    if exclude_nodes:
        env["LITERT_GPU_DEBUG_EXCLUDE_NODES"] = ",".join(
            str(n) for n in sorted(exclude_nodes))
    subprocess.run(
        [RUNNER, "--model=" + args.model, "--run", "--runs=1",
         "--input=" + args.input, "--dump-outputs=" + TMP_PREFIX,
         "--precision=fp32"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out):
        return float("nan")
    x = np.fromfile(out, dtype=np.float32)
    if x.shape != ref.shape or not np.isfinite(x).all():
        return float("nan")
    return float(x @ ref) / (np.linalg.norm(x) * np.linalg.norm(ref))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--total-nodes", type=int, required=True)
    p.add_argument("--codes", default=None)
    p.add_argument("--threshold", type=float, default=0.999)
    args = p.parse_args()

    ref = np.fromfile(args.ref, dtype=np.float32)
    by_code = nodes_by_opcode(args.model)

    if args.codes:
        codes = [int(c) for c in args.codes.split(",")]
        nodes = [n for c in codes for n in by_code.get(c, [])]
        print("exclude codes=%s (%d nodes) cosine=%.5f"
              % (codes, len(nodes), run(args, nodes, ref)), flush=True)
        return

    # Most-frequent classes first: a wrong op repeated many times is both more
    # likely and more damaging than a rare one.
    classes = [c for c, _ in sorted(by_code.items(),
                                    key=lambda kv: -len(kv[1]))]
    print("candidate classes: %s" % classes, flush=True)

    base = run(args, [], ref)
    print("baseline (nothing excluded) cosine=%.5f" % base, flush=True)

    def good(cs):
        nodes = [n for c in cs for n in by_code[c]]
        c = run(args, nodes, ref)
        print("  exclude %-28s (%4d nodes) cosine=%.5f"
              % (",".join(map(str, cs))[:28], len(nodes), c), flush=True)
        return c >= args.threshold

    if not good(classes):
        print("excluding every class still does not match; more than one bug, "
              "or the reference is wrong")
        return

    # Invariant: excluding `cur` fixes it. Shrink until a single class remains.
    cur = classes
    while len(cur) > 1:
        half = len(cur) // 2
        if good(cur[:half]):
            cur = cur[:half]
        else:
            cur = cur[half:]
    print("culprit opcode: %d (%d nodes)" % (cur[0], len(by_code[cur[0]])))


if __name__ == "__main__":
    main()
