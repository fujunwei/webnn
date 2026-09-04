"""Truncate a .tflite flatbuffer at operator K (in-place length edit).

Keeps operators [0, K-1] and rewrites the subgraph outputs to the last
operator's outputs. The operators vector is truncated by shortening its
length field in place (the trailing entries become dead bytes the parser
ignores). The outputs are rewritten into the original outputs vector when
they fit; otherwise a new vector is appended and the subgraph's field slot
is repointed (uoffset relative to the slot).

Usage: py truncate_tflite.py <in.tflite> <K> <out.tflite>
"""
import struct
import sys

sys.path.insert(0, r"C:\Users\fujun\workspace\webnn\segment_anythings\sam_native_runner")
from inspect_tflite import Table


def u32(buf, pos):
    return struct.unpack_from("<I", buf, pos)[0]


def main(in_path, k, out_path):
    buf = bytearray(open(in_path, "rb").read())
    root_off = struct.unpack_from("<I", buf, 0)[0]
    model = Table(buf, root_off)
    sg = model.vec_table(2, 0)  # first subgraph

    n_ops = sg.vec_len(3)
    if not (0 < k <= n_ops):
        sys.exit("K=%d out of range [1, %d]" % (k, n_ops))
    ops_vec = sg.obj(3)  # operators vector, at its length prefix

    # Outputs of operator K-1 become the new graph outputs.
    last_op_off = u32(buf, ops_vec + 4 + 4 * (k - 1))
    last_op = ops_vec + 4 + 4 * (k - 1) + last_op_off
    last_op_vto = last_op - struct.unpack_from("<i", buf, last_op)[0]
    fo2 = struct.unpack_from("<H", buf, last_op_vto + 4 + 2 * 2)[0]
    out_vec = last_op + fo2 + u32(buf, last_op + fo2)
    n_out = u32(buf, out_vec)
    outs = [struct.unpack_from("<i", buf, out_vec + 4 + 4 * j)[0]
            for j in range(n_out)]

    # 1. Truncate the operators vector in place: just shorten the length.
    struct.pack_into("<I", buf, ops_vec, k)

    # 2. Rewrite the subgraph outputs.
    sg_fo2 = struct.unpack_from("<H", buf, sg.vto + 4 + 2 * 2)[0]
    sg_out_slot = sg.pos + sg_fo2
    sg_out_vec = sg_out_slot + u32(buf, sg_out_slot)
    n_orig = u32(buf, sg_out_vec)
    if len(outs) <= n_orig:
        # Fits in the original vector; rewrite in place.
        struct.pack_into("<I", buf, sg_out_vec, len(outs))
        for j, o in enumerate(outs):
            struct.pack_into("<i", buf, sg_out_vec + 4 + 4 * j, o)
    else:
        # Append a new vector and repoint the subgraph's outputs field.
        end = len(buf)
        appended = struct.pack("<I", len(outs))
        for o in outs:
            appended += struct.pack("<i", o)
        while len(appended) % 4:
            appended += b"\x00"
        buf.extend(appended)
        struct.pack_into("<I", buf, sg_out_slot, end - sg_out_slot)

    with open(out_path, "wb") as f:
        f.write(buf)

    print("wrote %s: %d ops, outputs=%s" % (out_path, k, outs))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
