"""Binary-search LITERT_GPU_DEBUG_END_NODE for the first node that diverges.

Runs the model with nodes [0, N] as GPU candidates (everything after N falls
back to CPU), dumps the output, and scores it against a trusted CPU reference.
Nodes at or below the first bad index are the ones that break the result.

Only safe on models with no custom ops -- on a model containing
custom_call.LayerNorm the CPU tail cannot execute, so every low-N run returns
garbage and the search is meaningless. Use segment_anything_encoder.tflite
(1960 ops, no custom ops), not new_segment_anything_encoder.tflite.

Usage:
  py bisect_end_node.py --model <m.tflite> --input <in.bin> --ref <ref.bin>
                        --total-nodes N [--lo 0] [--hi N-1] [--threshold 0.99]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

RUNNER = (r"C:\Users\fujun\workspace\chromium\src\out\upstream_bots_debug"
          r"\sam_encoder_runner.exe")
TMP_PREFIX = "D:/tflite-dump-model/_bisect"


def cosine(path, ref):
    x = np.fromfile(path, dtype=np.float32)
    if x.shape != ref.shape or not np.isfinite(x).all():
        return float("nan")
    return float(x @ ref) / (np.linalg.norm(x) * np.linalg.norm(ref))


def run(args, end_node, ref):
    out = TMP_PREFIX + "_run.bin"
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ)
    env["MLD_WEBGPU_READBACK_TIMEOUT_SECONDS"] = "900"
    env["LITERT_GPU_DEBUG_ONLY_NODE_COUNT"] = str(args.total_nodes)
    env["LITERT_GPU_DEBUG_END_NODE"] = str(end_node)
    subprocess.run(
        [RUNNER, "--model=" + args.model, "--run", "--runs=1",
         "--input=" + args.input, "--dump-outputs=" + TMP_PREFIX,
         "--precision=" + args.precision],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out):
        return float("nan")
    return cosine(out, ref)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--total-nodes", type=int, required=True)
    p.add_argument("--lo", type=int, default=0)
    p.add_argument("--hi", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--precision", default="fp32")
    args = p.parse_args()

    ref = np.fromfile(args.ref, dtype=np.float32)
    hi = args.hi if args.hi is not None else args.total_nodes - 1
    lo = args.lo

    # lo is expected good, hi expected bad; verify both so a wrong assumption
    # shows up immediately instead of producing a bogus midpoint.
    for label, n in (("lo", lo), ("hi", hi)):
        c = run(args, n, ref)
        print("  probe %s END_NODE=%-5d cosine=%.5f" % (label, n, c), flush=True)

    while hi - lo > 1:
        mid = (lo + hi) // 2
        c = run(args, mid, ref)
        good = c >= args.threshold
        print("  END_NODE=%-5d cosine=%.5f  %s" % (mid, c, "OK" if good else "BAD"),
              flush=True)
        if good:
            lo = mid
        else:
            hi = mid
    print("first diverging node index: %d (last good END_NODE=%d)" % (hi, lo))


if __name__ == "__main__":
    main()
