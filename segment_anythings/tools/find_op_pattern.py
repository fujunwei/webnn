"""Find a builtin-op sequence (by name) in a .tflite and dump the operators
around each match: opcode, input/output tensor names+shapes+dtypes, and any
constant operand's actual values (decoding fp16 buffers as needed).

Reuses the minimal flatbuffer reader from inspect_tflite.py (no tensorflow
dependency). Field indices below all come directly from
tensorflow/compiler/mlir/lite/schema/schema.fbs (Model, SubGraph, Operator,
Tensor, Buffer table field order).

Usage:
    py find_op_pattern.py <model.tflite> [MUL,SOFTMAX,BATCH_MATMUL]
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sam_native_runner"))
from inspect_tflite import Table, TYPE_NAMES, shape_of, quant_of  # noqa: E402

BUILTIN_NAMES = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D", 6: "DEQUANTIZE", 9: "FULLY_CONNECTED",
    14: "LOGISTIC", 17: "MAX_POOL_2D", 18: "MUL", 22: "RESHAPE",
    23: "RESIZE_BILINEAR", 25: "SOFTMAX", 26: "SPACE_TO_DEPTH", 28: "TANH",
    32: "CUSTOM", 34: "PAD", 39: "TRANSPOSE", 40: "MEAN", 41: "SUB",
    42: "DIV", 47: "EXP", 65: "SLICE", 78: "POW", 74: "SUM", 92: "SQUARE",
    99: "SQUARED_DIFFERENCE", 117: "HARD_SWISH", 126: "BATCH_MATMUL",
    206: "STABLEHLO_COMPOSITE",
}
# Operator table field indices (tensorflow/compiler/mlir/lite/schema/schema.fbs).
_OP_FIELD_BUILTIN_OPTIONS_2_TYPE = 11
_OP_FIELD_BUILTIN_OPTIONS_2 = 12
_BUILTIN_OPTIONS_2_STABLEHLO_COMPOSITE_OPTIONS = 21
NAME_TO_BUILTIN = {v: k for k, v in BUILTIN_NAMES.items()}


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        root_off = struct.unpack_from("<I", self.buf, 0)[0]
        self.model = Table(self.buf, root_off)

        n_ops = self.model.vec_len(1)
        self.opcodes = []
        for i in range(n_ops):
            oc = self.model.vec_table(1, i)
            bc = oc.scalar(0, "<b") if oc.has(0) else 0
            # builtin_code (field 3) is int32 in the schema, unlike the
            # legacy deprecated_builtin_code (field 0, int8) -- codes >127
            # (e.g. STABLEHLO_COMPOSITE=206) only fit in the int32 field.
            dbc = oc.scalar(3, "<i") if oc.has(3) else None
            self.opcodes.append(dbc if dbc is not None else bc)

        self.sg = self.model.vec_table(2, 0)
        self.n_tensors = self.sg.vec_len(0)
        self.n_ops_sg = self.sg.vec_len(3)

        # buffers: Model field 4.
        self.n_buffers = self.model.vec_len(4)

    def tensor(self, idx):
        t = self.sg.vec_table(0, idx)
        type_code = t.scalar(1, "<b")
        if type_code is None:
            type_code = 0  # default TensorType == FLOAT32
        return {
            "index": idx,
            "name": t.string(3) or "",
            "shape": shape_of(t),
            "type": TYPE_NAMES.get(type_code, type_code),
            "buffer": t.scalar(2, "<I") or 0,
            "quant": quant_of(t),
        }

    def buffer_bytes(self, buf_idx):
        if buf_idx is None or buf_idx == 0:
            return None
        buf_t = self.model.vec_table(4, buf_idx)
        p = buf_t.obj(0)  # data:[ubyte], field 0
        if p is None:
            return None
        n = struct.unpack_from("<I", self.buf, p)[0]
        return self.buf[p + 4:p + 4 + n]

    def op(self, idx):
        o = self.sg.vec_table(3, idx)
        opcode_index = o.scalar(0, "<I") or 0
        builtin = self.opcodes[opcode_index]
        n_in = o.vec_len(1)
        n_out = o.vec_len(2)
        ins = [o.vec_scalar(1, i, "<i", 4) for i in range(n_in)]
        outs = [o.vec_scalar(2, i, "<i", 4) for i in range(n_out)]
        result = {"index": idx, "builtin": builtin,
                  "name": BUILTIN_NAMES.get(builtin, str(builtin)),
                  "inputs": ins, "outputs": outs}
        if builtin == NAME_TO_BUILTIN["STABLEHLO_COMPOSITE"]:
            options_2_type = o.scalar(_OP_FIELD_BUILTIN_OPTIONS_2_TYPE, "<B")
            if options_2_type == _BUILTIN_OPTIONS_2_STABLEHLO_COMPOSITE_OPTIONS:
                composite = o.table(_OP_FIELD_BUILTIN_OPTIONS_2)
                if composite is not None:
                    result["composite_name"] = composite.string(0)
                    result["decomposition_subgraph_index"] = composite.scalar(
                        1, "<i")
        return result

    def all_ops(self):
        return [self.op(i) for i in range(self.n_ops_sg)]


def decode_const(m, tensor_desc):
    """Returns a numpy array if the tensor has an inline constant buffer."""
    raw = m.buffer_bytes(tensor_desc["buffer"])
    if raw is None:
        return None
    dtype = {"FLOAT32": np.float32, "FLOAT16": np.float16,
             "INT32": np.int32}.get(tensor_desc["type"])
    if dtype is None:
        return None
    arr = np.frombuffer(raw, dtype=dtype)
    shape = tensor_desc["shape"] or [arr.size]
    try:
        return arr.reshape(shape)
    except ValueError:
        return arr


def describe_tensor_ref(m, idx):
    if idx < 0:
        return "(none)"
    t = m.tensor(idx)
    const = decode_const(m, t)
    s = f"#{idx} {t['name']} shape={t['shape']} type={t['type']}"
    if const is not None:
        flat = const.reshape(-1)
        preview = flat[:8].tolist()
        s += f" CONST values[:8]={preview}"
        if flat.size > 1:
            s += f" (min={flat.min()} max={flat.max()} all_equal={bool(np.all(flat == flat[0]))})"
    return s


def main(path, pattern_str):
    m = Model(path)
    pattern = [NAME_TO_BUILTIN[p] for p in pattern_str.split(",")]
    ops = m.all_ops()
    builtins = [o["builtin"] for o in ops]

    matches = []
    for i in range(len(ops) - len(pattern) + 1):
        if builtins[i:i + len(pattern)] == pattern:
            matches.append(i)

    print(f"model={path} total_ops={len(ops)} pattern={pattern_str} "
          f"matches_at_op_index={matches}")
    for start in matches:
        print(f"\n=== match starting at op index {start} ===")
        for j in range(len(pattern)):
            o = ops[start + j]
            extra = ""
            if "composite_name" in o:
                extra = (f" name={o['composite_name']!r} "
                         f"decomposition_subgraph_index="
                         f"{o['decomposition_subgraph_index']}")
            print(f"op[{o['index']}] {o['name']}{extra}")
            for k, ti in enumerate(o["inputs"]):
                print(f"    in[{k}]  {describe_tensor_ref(m, ti)}")
            for k, ti in enumerate(o["outputs"]):
                print(f"    out[{k}] {describe_tensor_ref(m, ti)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "MUL,SOFTMAX,BATCH_MATMUL")
