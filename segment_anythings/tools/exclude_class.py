"""Exclude one opcode class -- restricted to the nodes actually delegated.

On a partially-delegated model most nodes of a class never reach the GPU, so
excluding the whole class is a no-op and reads as a false negative. Feed this
the list printed by the delegate's "[partition] N nodes delegated:" line
(LITERT_GPU_DEBUG_DUMP_NODES=1) so only real GPU nodes are excluded.

Usage:
  py exclude_class.py --model M --input IN --ref REF --total-nodes N
                      --delegated FILE --code 18
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
TMP = "D:/tflite-dump-model/_excl"


def nodes_by_opcode(model_path):
    buf = open(model_path, "rb").read()
    model = Table(buf, struct.unpack_from("<I", buf, 0)[0])
    codes = []
    for i in range(model.vec_len(1)):
        oc = model.vec_table(1, i)
        a = (oc.scalar(0, "<b") or 0) if oc.has(0) else 0
        d = (oc.scalar(3, "<i") or 0) if oc.has(3) else 0
        codes.append(max(a, d))
    sg = model.vec_table(2, 0)
    out = collections.defaultdict(list)
    for oi in range(sg.vec_len(3)):
        out[codes[sg.vec_table(3, oi).scalar(0, "<i") or 0]].append(oi)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--total-nodes", type=int, required=True)
    p.add_argument("--delegated", required=True)
    p.add_argument("--code", type=int, required=True)
    args = p.parse_args()

    ref = np.fromfile(args.ref, dtype=np.float32)
    delegated = set(int(x) for x in open(args.delegated).read().split(","))
    nodes = [n for n in nodes_by_opcode(args.model)[args.code]
             if n in delegated]

    out = TMP + "_run.bin"
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ)
    env["MLD_WEBGPU_READBACK_TIMEOUT_SECONDS"] = "900"
    env["LITERT_GPU_DEBUG_ONLY_NODE_COUNT"] = str(args.total_nodes)
    env["LITERT_GPU_DEBUG_EXCLUDE_NODES"] = ",".join(map(str, sorted(nodes)))
    subprocess.run(
        [RUNNER, "--model=" + args.model, "--run", "--runs=1",
         "--input=" + args.input, "--dump-outputs=" + TMP,
         "--precision=fp32"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(out):
        print("code %-4d (%3d delegated nodes) -> run produced no output"
              % (args.code, len(nodes)))
        return
    x = np.fromfile(out, dtype=np.float32)
    if x.shape != ref.shape or not np.isfinite(x).all():
        print("code %-4d (%3d delegated nodes) -> nan/shape mismatch"
              % (args.code, len(nodes)))
        return
    print("code %-4d (%3d delegated nodes) -> cosine=%.6f"
          % (args.code, len(nodes),
             float(x @ ref) / (np.linalg.norm(x) * np.linalg.norm(ref))))


if __name__ == "__main__":
    main()
