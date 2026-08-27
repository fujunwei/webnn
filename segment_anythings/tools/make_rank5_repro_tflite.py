"""Generate a tiny .tflite that reproduces the SAM rank-5 truncation bug.

The SAM encoder feeds three op types rank-5 tensors (see rank5_scan.py):
RESHAPE, TRANSPOSE and broadcast ADD. `ExtractTensorShape()` in the LiteRT
ml-drift delegate truncates any rank>=5 tensor to its first four dims, so the
GPU tensors for those values are allocated far too small and the whole
window-attention path produces garbage.

This model chains all three patterns behind a single 4D input/output so it can
be driven by `sam_encoder_runner --verify`, which compares the GPU result
against a CPU reference:

    x   : [1, 28, 28, 8]                        (input)
    r1  = reshape(x,  [2, 14, 2, 14, 8])        window partition  (5D out)
    t1  = transpose(r1, [0, 2, 1, 3, 4])        -> [2, 2, 14, 14, 8]
    r2  = reshape(t1, [4, 14, 14, 8])           back to 4D
    r3  = reshape(r2, [4, 14, 14, 8, 1])        rel-pos bias operands
    r4  = reshape(r2, [4, 14, 14, 1, 8])
    s   = add(r3, r4)                           -> [4, 14, 14, 8, 8] (5D bcast)
    out = reshape(s,  [4, 14, 14, 64])          (output)

A `--rank4` variant emits the same op chain with every tensor kept at rank 4.
Nothing about it is broken today; it exists to catch regressions in the shared
reshape/transpose/broadcast-add paths while the rank-5 support is changed.

Usage: py make_rank5_repro_tflite.py [--rank4|--pow-neg|--transpose2d|--bcast5d|--layernorm] <out.tflite> [<out_input.bin>]

Writing the flatbuffer by hand keeps this free of a TensorFlow dependency.
"""
import struct
import sys

import flatbuffers
import numpy as np

# schema_v3 enum values.
BUILTIN_ADD = 0
BUILTIN_DEQUANTIZE = 6
BUILTIN_RESHAPE = 22
BUILTIN_CUSTOM = 32
BUILTIN_TRANSPOSE = 39
BUILTIN_POW = 78
OPT_NONE = 0
OPT_ADD = 11         # BuiltinOptions_AddOptions
OPT_RESHAPE = 17     # BuiltinOptions_ReshapeOptions
OPT_TRANSPOSE = 26   # BuiltinOptions_TransposeOptions
TYPE_FLOAT32 = 0
TYPE_FLOAT16 = 1
TYPE_INT32 = 2


def end_vector(b, n):
    """flatbuffers >=2.0 dropped EndVector's argument."""
    try:
        return b.EndVector()
    except TypeError:
        return b.EndVector(n)


def int_vector(b, values):
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependInt32(v)
    return end_vector(b, len(values))


def byte_vector(b, data):
    b.StartVector(1, len(data), 1)
    for v in reversed(data):
        b.PrependByte(v)
    return end_vector(b, len(data))


def offset_vector(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return end_vector(b, len(offsets))


class ModelWriter:
    """Accumulates tensors/operators, then serializes a single-subgraph model."""

    def __init__(self):
        self.tensors = []    # (shape, type, name, buffer_index)
        self.buffers = [b""]  # buffer 0 is the conventional empty buffer
        # (builtin_code, inputs, outputs, opt_kind, opt_arg, custom)
        # custom is None or (custom_name, custom_options_bytes).
        self.ops = []

    def tensor(self, shape, name, dtype=TYPE_FLOAT32, data=None):
        buf = 0
        if data is not None:
            self.buffers.append(data.tobytes())
            buf = len(self.buffers) - 1
        self.tensors.append((list(shape), dtype, name, buf))
        return len(self.tensors) - 1

    def const_i32(self, values, name):
        return self.tensor([len(values)], name, TYPE_INT32,
                           np.asarray(values, dtype=np.int32))

    def reshape(self, src, new_shape, name):
        shape_t = self.const_i32(new_shape, name + "/shape")
        dst = self.tensor(new_shape, name)
        self.ops.append((BUILTIN_RESHAPE, [src, shape_t], [dst],
                         OPT_RESHAPE, new_shape, None))
        return dst

    def transpose(self, src, perm, out_shape, name):
        perm_t = self.const_i32(perm, name + "/perm")
        dst = self.tensor(out_shape, name)
        self.ops.append((BUILTIN_TRANSPOSE, [src, perm_t], [dst],
                         OPT_TRANSPOSE, None, None))
        return dst

    def add(self, a, b_, out_shape, name):
        dst = self.tensor(out_shape, name)
        self.ops.append((BUILTIN_ADD, [a, b_], [dst], OPT_ADD, None, None))
        return dst

    def dequantize(self, src, shape, name):
        dst = self.tensor(shape, name)
        self.ops.append((BUILTIN_DEQUANTIZE, [src], [dst], OPT_NONE, None,
                         None))
        return dst

    def pow(self, base, exponent, out_shape, name):
        dst = self.tensor(out_shape, name)
        self.ops.append((BUILTIN_POW, [base, exponent], [dst], OPT_NONE, None,
                         None))
        return dst

    def custom_op(self, name, ins, out_shape, out_name, custom_bytes):
        """One custom operator with FLEXBUFFERS custom options."""
        dst = self.tensor(out_shape, out_name)
        self.ops.append((BUILTIN_CUSTOM, ins, [dst], OPT_NONE, None,
                         (name, custom_bytes)))
        return dst

    def serialize(self, inputs, outputs):
        b = flatbuffers.Builder(4096)

        buffer_offsets = []
        for data in self.buffers:
            data_off = byte_vector(b, data) if data else None
            b.StartObject(1)
            if data_off is not None:
                b.PrependUOffsetTRelativeSlot(0, data_off, 0)
            buffer_offsets.append(b.EndObject())
        buffers_vec = offset_vector(b, buffer_offsets)

        # One OperatorCode per distinct (builtin, custom name), in first-use
        # order. Keys are 2-tuples so two custom ops with different names do
        # not collapse onto builtin code 32.
        codes = []
        for code, *_unused, custom in self.ops:
            key = (code, custom[0] if custom else None)
            if key not in codes:
                codes.append(key)
        code_offsets = []
        for code, custom_name in codes:
            # CreateString must run outside StartObject (flatbuffers nesting).
            custom_name_off = (b.CreateString(custom_name)
                               if custom_name is not None else None)
            b.StartObject(4)
            if custom_name_off is None:
                # deprecated_builtin_code caps at 127; builtin_code is real one.
                b.PrependInt8Slot(0, min(code, 127), 0)
                b.PrependInt32Slot(2, 1, 0)   # version
                b.PrependInt32Slot(3, code, 0)
            else:
                b.PrependInt8Slot(0, BUILTIN_CUSTOM, 0)
                b.PrependUOffsetTRelativeSlot(1, custom_name_off, 0)
                b.PrependInt32Slot(2, 1, 0)   # version
                b.PrependInt32Slot(3, BUILTIN_CUSTOM, 0)
            code_offsets.append(b.EndObject())
        codes_vec = offset_vector(b, code_offsets)

        tensor_offsets = []
        for shape, dtype, name, buf in self.tensors:
            shape_off = int_vector(b, shape)
            name_off = b.CreateString(name)
            b.StartObject(4)
            b.PrependUOffsetTRelativeSlot(0, shape_off, 0)
            b.PrependInt8Slot(1, dtype, 0)
            b.PrependUint32Slot(2, buf, 0)
            b.PrependUOffsetTRelativeSlot(3, name_off, 0)
            tensor_offsets.append(b.EndObject())
        tensors_vec = offset_vector(b, tensor_offsets)

        op_offsets = []
        for code, ins, outs, opt_kind, opt_arg, custom in self.ops:
            if opt_kind == OPT_RESHAPE:
                new_shape_off = int_vector(b, opt_arg)
                b.StartObject(1)
                b.PrependUOffsetTRelativeSlot(0, new_shape_off, 0)
                opt_off = b.EndObject()
            elif opt_kind == OPT_ADD:
                b.StartObject(2)
                b.PrependInt8Slot(0, 0, 0)  # fused_activation_function = NONE
                opt_off = b.EndObject()
            else:  # TransposeOptions is an empty table
                b.StartObject(0)
                opt_off = b.EndObject()
            custom_off = None
            if custom is not None:
                custom_off = byte_vector(b, custom[1])
            ins_off = int_vector(b, ins)
            outs_off = int_vector(b, outs)
            b.StartObject(7)
            b.PrependUint32Slot(0, codes.index((code, custom[0] if custom
                                               else None)), 0)
            b.PrependUOffsetTRelativeSlot(1, ins_off, 0)
            b.PrependUOffsetTRelativeSlot(2, outs_off, 0)
            b.PrependInt8Slot(3, opt_kind, 0)
            b.PrependUOffsetTRelativeSlot(4, opt_off, 0)
            if custom_off is not None:
                b.PrependUOffsetTRelativeSlot(5, custom_off, 0)
                b.PrependInt8Slot(6, 1, 0)  # CustomOptionsFormat_FLEXBUFFERS
            op_offsets.append(b.EndObject())
        ops_vec = offset_vector(b, op_offsets)

        sg_in = int_vector(b, inputs)
        sg_out = int_vector(b, outputs)
        sg_name = b.CreateString("main")
        b.StartObject(5)
        b.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
        b.PrependUOffsetTRelativeSlot(1, sg_in, 0)
        b.PrependUOffsetTRelativeSlot(2, sg_out, 0)
        b.PrependUOffsetTRelativeSlot(3, ops_vec, 0)
        b.PrependUOffsetTRelativeSlot(4, sg_name, 0)
        subgraph = b.EndObject()
        subgraphs_vec = offset_vector(b, [subgraph])

        desc = b.CreateString("rank5 reshape/transpose/broadcast-add repro")
        b.StartObject(5)
        b.PrependUint32Slot(0, 3, 0)  # schema version
        b.PrependUOffsetTRelativeSlot(1, codes_vec, 0)
        b.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
        b.PrependUOffsetTRelativeSlot(3, desc, 0)
        b.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
        model = b.EndObject()

        b.Finish(model, file_identifier=b"TFL3")
        return bytes(b.Output())


def build_rank5():
    m = ModelWriter()
    x = m.tensor([1, 28, 28, 8], "x")

    r1 = m.reshape(x, [2, 14, 2, 14, 8], "window_partition")
    t1 = m.transpose(r1, [0, 2, 1, 3, 4], [2, 2, 14, 14, 8], "window_perm")
    r2 = m.reshape(t1, [4, 14, 14, 8], "windows")

    r3 = m.reshape(r2, [4, 14, 14, 8, 1], "rel_h")
    r4 = m.reshape(r2, [4, 14, 14, 1, 8], "rel_w")
    s = m.add(r3, r4, [4, 14, 14, 8, 8], "rel_pos_bias")
    out = m.reshape(s, [4, 14, 14, 64], "out")

    return m.serialize([x], [out]), (1, 28, 28, 8)


def build_rank4():
    """Same op mix, nothing above rank 4 — a control for the 4D code paths."""
    m = ModelWriter()
    x = m.tensor([1, 28, 28, 8], "x")

    r1 = m.reshape(x, [2, 14, 28, 8], "split_rows")
    t1 = m.transpose(r1, [0, 2, 1, 3], [2, 28, 14, 8], "perm")
    r2 = m.reshape(t1, [4, 14, 14, 8], "windows")

    r3 = m.reshape(r2, [4, 14, 14, 8], "lhs")
    r4 = m.reshape(r2, [4, 14, 14, 8], "rhs")
    s = m.add(r3, r4, [4, 14, 14, 8], "sum")
    out = m.reshape(s, [4, 14, 112], "out")

    return m.serialize([x], [out]), (1, 28, 28, 8)


def build_pow_neg():
    """pow(x, 2) where x goes negative and the exponent is a runtime value.

    Mirrors the SAM neck's decomposed LayerNorm2d: the exponent is an fp16
    constant 2.0 reached through DEQUANTIZE, so the delegate sees a runtime
    second input and picks the two-input POW kernel instead of the scalar
    lowering at elementwise.cc:547.
    """
    m = ModelWriter()
    x = m.tensor([1, 4, 4, 8], "x")
    # Scalar (rank-0) fp16 constant, dequantized then reshaped to [1] -- this is
    # the exact chain SAM uses (nodes 1227/1228/1229). A [1]-shaped constant
    # feeding POW directly gets folded to a scalar and takes the safe path.
    c = m.tensor([], "two_f16", TYPE_FLOAT16, np.asarray([2.0], np.float16))
    d = m.dequantize(c, [], "two_f32")
    e = m.reshape(d, [1], "two_1d")
    out = m.pow(x, e, [1, 4, 4, 8], "x_squared")
    return m.serialize([x], [out]), (1, 4, 4, 8)


def build_transpose2d():
    """Rank-2 TRANSPOSE of a dequantized constant weight matrix.

    This is SAM nodes 29/30: an fp16 [K, N] weight -> DEQUANTIZE -> TRANSPOSE
    with perm [1, 0]. The encoder has 48 of these (qkv/proj/fc1/fc2 in each of
    12 blocks). Dimensions are deliberately non-square so that a permutation
    that silently passes the data through is visible in the output.

    The input is added to the transposed constant so the runner has something
    to drive; feed zeros and the output *is* the transposed constant.
    """
    k, n = 3, 4
    m = ModelWriter()
    x = m.tensor([n, k], "x")
    w = m.tensor([k, n], "w_f16", TYPE_FLOAT16,
                 np.arange(k * n, dtype=np.float16).reshape(k, n))
    d = m.dequantize(w, [k, n], "w_f32")
    t = m.transpose(d, [1, 0], [n, k], "w_t")
    out = m.add(x, t, [n, k], "out")
    return m.serialize([x], [out]), (n, k)


def build_bcast5d():
    """The two rel-pos-bias broadcast ADDs of SAM (old-model nodes 95 and 96).

    Node 95 is full (+) c==1, node 96 is full (+) d==1. build_rank5() covers the
    case where the two operands broadcast on *different* axes; this covers a
    full tensor added to a single broadcasting operand, which is what the
    encoder actually does. A rank-5 constant cannot fold into a BHWC attribute,
    so the parser adds it as a runtime input -- the same two-input broadcast
    kernel the encoder hits.
    """
    b, h, w, d, c = 4, 6, 6, 5, 7
    m = ModelWriter()
    x = m.tensor([b, h, w, d * c], "x")
    a = m.reshape(x, [b, h, w, d, c], "a")
    k1 = m.tensor([b, h, w, d, 1], "bias_c1", TYPE_FLOAT32,
                  (np.arange(b * h * w * d, dtype=np.float32) % 11.0 - 5.0)
                  .reshape(b, h, w, d, 1))
    t1 = m.add(a, k1, [b, h, w, d, c], "add_bcast_c")
    k2 = m.tensor([b, h, w, 1, c], "bias_d1", TYPE_FLOAT32,
                  (np.arange(b * h * w * c, dtype=np.float32) % 13.0 - 6.0)
                  .reshape(b, h, w, 1, c))
    t2 = m.add(t1, k2, [b, h, w, d, c], "add_bcast_d")
    out = m.reshape(t2, [b, h, w, d * c], "out")
    return m.serialize([x], [out]), (b, h, w, d * c)


def build_layernorm():
    """One custom_call.LayerNorm op, f32 input/scale/bias/output.

    The custom options bytes are copied verbatim from the real fused model
    (`new_segment_anything_encoder.tflite`), so epsilon and the flexbuffer
    layout are bit-identical to what the encoder's 24 LayerNorms carry.
    `sam_encoder_runner --run` drives it GPU-only (the op has no CPU kernel);
    compare against the numpy reference:
        (x - mean(x, axis=-1)) / sqrt(var(x, axis=-1) + eps) * scale + bias
    """
    m = ModelWriter()
    x = m.tensor([1, 4, 4, 8], "x")
    scale = m.tensor([8], "scale", TYPE_FLOAT32,
                     np.asarray([1.0, 0.9, 0.8, 1.1, 1.2, 0.7, 1.05, 0.95],
                                np.float32))
    bias = m.tensor([8], "bias", TYPE_FLOAT32,
                    np.asarray([0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15],
                               np.float32))
    out = m.custom_op(
        "custom_call.LayerNorm", [x, scale, bias], [1, 4, 4, 8], "out",
        bytes.fromhex("657073696c6f6e0001090000030000000100000001000000"
                      "bd3786350e052601"))
    return m.serialize([x], [out]), (1, 4, 4, 8)


def main(*argv):
    argv = list(argv)
    rank4 = "--rank4" in argv
    pow_neg = "--pow-neg" in argv
    transpose2d = "--transpose2d" in argv
    bcast5d = "--bcast5d" in argv
    layernorm = "--layernorm" in argv
    for flag in ("--rank4", "--pow-neg", "--transpose2d", "--bcast5d",
                 "--layernorm"):
        if flag in argv:
            argv.remove(flag)
    model_path, input_path = argv[0], (argv[1] if len(argv) > 1 else None)
    if layernorm:
        blob, in_shape = build_layernorm()
    elif bcast5d:
        blob, in_shape = build_bcast5d()
    elif transpose2d:
        blob, in_shape = build_transpose2d()
    elif pow_neg:
        blob, in_shape = build_pow_neg()
    elif rank4:
        blob, in_shape = build_rank4()
    else:
        blob, in_shape = build_rank5()
    with open(model_path, "wb") as f:
        f.write(blob)
    print("wrote %s (%d bytes)" % (model_path, len(blob)))
    if input_path:
        n = int(np.prod(in_shape))
        # Deterministic, non-trivial, and spread over a range where a truncated
        # tensor cannot accidentally match the reference.
        data = (np.arange(n, dtype=np.float32) % 37.0) - 18.0
        with open(input_path, "wb") as f:
            f.write(data.tobytes())
        print("wrote %s (%d floats)" % (input_path, n))


if __name__ == "__main__":
    main(*sys.argv[1:])
