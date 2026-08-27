"""Throwaway: enumerate rank>=5 tensors in a .tflite and the ops around them.

Answers "how much of ml-drift needs real 5D support" by listing, for every
tensor with rank >= 5, which opcode produces it and which opcodes consume it.

Usage: py rank5_scan.py <model.tflite>

Reuses the minimal flatbuffer reader from ../sam_native_runner/inspect_tflite.py.
"""
import collections
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "sam_native_runner"))
from inspect_tflite import Table, shape_of  # noqa: E402
import struct  # noqa: E402

# builtin_code -> name, only the ones we expect to see here.
BUILTIN = {
    0: "ADD", 3: "CONV_2D", 4: "DEPTHWISE_CONV_2D", 6: "DEQUANTIZE",
    9: "FULLY_CONNECTED", 18: "MUL", 22: "RESHAPE", 25: "SOFTMAX",
    28: "RSQRT", 39: "TRANSPOSE", 40: "MEAN", 41: "SUB", 42: "DIV",
    45: "PAD", 47: "STRIDED_SLICE", 49: "SQUEEZE", 53: "CAST",
    59: "SPLIT", 61: "EXP", 75: "SQUARE", 78: "GATHER", 92: "SUM",
    101: "POW", 102: "ARG_MIN", 126: "SLICE", 158: "BATCH_MATMUL",
    18: "MUL",
}


def op_name(builtins, custom_names, opcode_index):
    bc = builtins[opcode_index]
    if bc == 32:  # CUSTOM
        return "CUSTOM(%s)" % custom_names[opcode_index]
    return BUILTIN.get(bc, "op#%d" % bc)


def main(path):
    with open(path, "rb") as f:
        buf = f.read()
    model = Table(buf, struct.unpack_from("<I", buf, 0)[0])

    n_codes = model.vec_len(1)
    builtins, custom_names = [], []
    for i in range(n_codes):
        oc = model.vec_table(1, i)
        # OperatorCode: 0=deprecated_builtin_code(byte), 3=builtin_code(int32).
        # Both default to 0 (ADD) and are then omitted from the buffer.
        bc = (oc.scalar(0, "<b") or 0) if oc.has(0) else 0
        dbc = (oc.scalar(3, "<i") or 0) if oc.has(3) else 0
        builtins.append(max(bc, dbc))
        custom_names.append(oc.string(1))

    sg = model.vec_table(2, 0)
    n_tensors = sg.vec_len(0)
    n_ops = sg.vec_len(3)

    shapes = [shape_of(sg.vec_table(0, i)) for i in range(n_tensors)]

    producer = {}
    consumers = collections.defaultdict(list)
    for oi in range(n_ops):
        op = sg.vec_table(3, oi)
        # Field defaults are omitted from the buffer, so opcode_index 0 reads
        # back as None.
        name = op_name(builtins, custom_names, op.scalar(0, "<i") or 0)
        for k in range(op.vec_len(1)):
            ti = op.vec_scalar(1, k, "<i", 4)
            if ti >= 0:
                consumers[ti].append((oi, name))
        for k in range(op.vec_len(2)):
            ti = op.vec_scalar(2, k, "<i", 4)
            if ti >= 0:
                producer[ti] = (oi, name)

    rank5 = [i for i in range(n_tensors) if len(shapes[i]) >= 5]
    print("total tensors: %d, rank>=5: %d, ops: %d"
          % (n_tensors, len(rank5), n_ops))

    by_rank = collections.Counter(len(shapes[i]) for i in rank5)
    print("rank histogram:", dict(sorted(by_rank.items())))

    prod_hist = collections.Counter(
        producer.get(i, (None, "<graph-input/const>"))[1] for i in rank5)
    cons_hist = collections.Counter(
        n for i in rank5 for _, n in consumers[i])
    print("\nproducers of rank>=5 tensors:", dict(prod_hist.most_common()))
    print("consumers of rank>=5 tensors:", dict(cons_hist.most_common()))

    print("\ndistinct (rank, producer -> consumers) chains:")
    chains = collections.Counter()
    for i in rank5:
        p = producer.get(i, (None, "<input/const>"))[1]
        c = ",".join(sorted(n for _, n in consumers[i])) or "<graph-output>"
        chains["%-8s %-14s -> %s" % (tuple(shapes[i]), p, c)] += 1
    for k, v in chains.most_common():
        print("  %3dx %s" % (v, k))


if __name__ == "__main__":
    main(sys.argv[1])
