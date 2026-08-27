"""Delta-debug the fp16 all-zeros failure on new_segment_anything_encoder.tflite.

The new fused model runs fully on the GPU delegate; in fp16 the output is
exactly all zeros (fp32 is fine, the single-op LayerNorm repro is fine). The
signal is binary, so delta-debug over execution order works: at each round,
exclude the first half of the candidate (non-LayerNorm) nodes from the GPU
partition. The excluded nodes run on CPU/XNNPACK (fp32). If the output stops
being all zero, the culprit is inside the excluded half; shrink accordingly.

The 24 custom_call.LayerNorm nodes never leave the GPU (they have no CPU
kernel, pushing one to CPU makes compilation fail).

Usage:
  py fp16_zeros_bisect.py <nodes.tsv> <model.tflite> <input.bin> <runner.exe>
                           <workdir> [--all-but-ln | --start=<n>]
                           [--fp32 --ref=<cpu_output.bin>]

  --all-but-ln   one experiment: everything except the 24 LayerNorms on CPU
  --start=<n>    resume the binary search from a candidate list of n nodes
                 (printed by an earlier --list run); default: full bisect
  --fp32         fp32 bisect: the signal is cosine vs the --ref file instead
                 of the fp16 all-zeros test; "culprit in excluded half" when
                 the cosine rises above the baseline by a fixed margin.
"""
import os
import subprocess
import sys

import numpy as np

OUTPUT_ELEMS = 1048576


def parse_tsv(path):
    nodes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("idx"):
                continue
            idx, code, name, custom = line.split("\t")[:4]
            nodes.append((int(idx), code, name, custom))
    return nodes


def non_ln_indices(nodes):
    return [idx for idx, code, name, custom in nodes
            if custom != "custom_call.LayerNorm"]


def all_zero(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != OUTPUT_ELEMS:
        return None  # short read: treat as no output
    return not a.any()


def run_round(runner, model, input_bin, dump_prefix, exclude, precision="fp16",
              timeout_s=600):
    env = dict(os.environ)
    env["LITERT_GPU_DEBUG_ONLY_NODE_COUNT"] = "1260"
    if exclude:
        env["LITERT_GPU_DEBUG_EXCLUDE_NODES"] = ",".join(map(str, exclude))
    env["MLD_WEBGPU_READBACK_TIMEOUT_SECONDS"] = "120"
    cmd = [runner, "--model=%s" % model, "--run", "--precision=%s" % precision,
           "--input=%s" % input_bin, "--dump-outputs=%s" % dump_prefix]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout_s)
    dump = dump_prefix + "_run.bin"
    verdict = all_zero(dump) if os.path.exists(dump) else None
    tail = "\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-6:])
    return verdict, tail


def cosine(dump_path, ref):
    a = np.fromfile(dump_path, dtype=np.float32)
    if a.size != ref.size:
        return None
    return float(np.dot(a, ref) / (np.linalg.norm(a) * np.linalg.norm(ref)))


def main():
    nodes_tsv, model, input_bin, runner, workdir = sys.argv[1:6]
    flags = sys.argv[6:]
    nodes = parse_tsv(nodes_tsv)
    cands = non_ln_indices(nodes)
    print("non-LayerNorm candidates: %d" % len(cands), flush=True)

    fp32 = "--fp32" in flags
    precision = "fp32" if fp32 else "fp16"
    ref = None
    if fp32:
        ref_flag = next((f for f in flags if f.startswith("--ref=")), None)
        if not ref_flag:
            print("--fp32 requires --ref=<cpu_output.bin>")
            return
        ref = np.fromfile(ref_flag.split("=", 1)[1], dtype=np.float32)
        print("reference elems=%d" % ref.size, flush=True)

    def score(dump_path, v):
        """fp16: True=all-zero, False=nonzero, None=no output.
        fp32: cosine float, None=no output."""
        if fp32:
            return cosine(dump_path, ref)
        return v

    if "--all-but-ln" in flags:
        dump = os.path.join(workdir, "fp16_bb_allbutln")
        v, tail = run_round(runner, model, input_bin, dump, cands, precision)
        print("all-but-ln verdict: %s" % score(dump + "_run.bin", v))
        print(tail)
        return

    if "--list" in flags:
        print(",".join(map(str, cands)))
        return

    start = None
    for f in flags:
        if f.startswith("--start="):
            start = [int(x) for x in f.split("=", 1)[1].split(",") if x]
    if start is not None:
        s = set(start)
        cands = [i for i in cands if i in s]
        print("resumed with %d candidates" % len(cands), flush=True)

    # Round 0: no exclusion must reproduce the known-bad baseline.
    dump = os.path.join(workdir, "fp16_bb_00")
    v, tail = run_round(runner, model, input_bin, dump, [], precision)
    base = score(dump + "_run.bin", v)
    if fp32:
        print("round 00 (no exclude): cosine=%.6f" % (base or float("nan")))
    else:
        print("round 00 (no exclude): %s" %
              ("ALL-ZERO (baseline confirmed)" if base is True else base))
    print(tail)
    if fp32:
        if base is None:
            print("baseline produced no output; aborting", flush=True)
            return
    elif base is not True:
        print("baseline not all-zero; aborting", flush=True)
        return

    rnd = 1
    while len(cands) > 1:
        half = cands[:len(cands) // 2]
        dump = os.path.join(workdir, "fp16_bb_%02d" % rnd)
        v, tail = run_round(runner, model, input_bin, dump, half, precision)
        s = score(dump + "_run.bin", v)
        if fp32:
            # Excluding buggy nodes pulls the cosine toward 1.0; a clear jump
            # means the excluded half contains a culprit.
            improved = s is not None and s > base + 0.15
            cands = half if improved else cands[len(cands) // 2:]
            verdict = "cosine=%.4f %s" % (s or 0.0,
                                          "-> culprit in excluded half"
                                          if improved else "-> no jump")
        else:
            if s is False:
                cands = half
                verdict = "NONZERO -> culprit in excluded half"
            elif s is True:
                cands = cands[len(cands) // 2:]
                verdict = "ALL-ZERO -> culprit in GPU half"
            else:
                print("round %02d: NO OUTPUT (excluded %d) -- aborting" %
                      (rnd, len(half)), flush=True)
                print(tail)
                print("resume: --start=%s" % ",".join(map(str, cands)),
                      flush=True)
                return
            verdict = "%s" % verdict
        print("round %02d: excluded %d, %s, cands now %d" %
              (rnd, len(half), verdict, len(cands)), flush=True)
        print(tail)
        rnd += 1

    if len(cands) == 1:
        k = cands[0]
        dump = os.path.join(workdir, "fp16_bb_final")
        v, tail = run_round(runner, model, input_bin, dump, [k], precision)
        s = score(dump + "_run.bin", v)
        if fp32:
            print("final confirm node %d alone: cosine=%.4f" % (k, s or 0.0))
        else:
            print("final confirm node %d alone: %s" %
                  (k, "NONZERO (CONFIRMED)" if s is False else ("ALL-ZERO" if s
                                                                else "NO OUTPUT")))
        print(tail)
        print("suspect node index: %d" % k)
        print("candidates exhausted: %s" % ",".join(map(str, cands)))


if __name__ == "__main__":
    main()
