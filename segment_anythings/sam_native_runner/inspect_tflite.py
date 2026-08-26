"""Minimal TFLite flatbuffer inspector (no tensorflow needed).

Reads the input/output tensor name, shape, element type, and quantization
(scale / zero_point) of a .tflite model.
"""
import struct
import sys

TYPE_NAMES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8", 4: "INT64",
    5: "STRING", 6: "BOOL", 7: "INT16", 8: "COMPLEX64", 9: "INT8",
    10: "FLOAT64", 11: "COMPLEX128", 12: "UINT64", 13: "RESOURCE",
    14: "VARIANT", 15: "UINT32", 16: "UINT16", 17: "INT4", 18: "BFLOAT16",
}


class Table:
    def __init__(self, buf, pos):
        self.buf = buf
        self.pos = pos
        self.vto = pos - struct.unpack_from("<i", buf, pos)[0]
        self.vsize = struct.unpack_from("<H", buf, self.vto)[0]

    def _fo(self, fid):
        off = self.vto + 4 + fid * 2
        if off + 2 > self.vto + self.vsize:
            return 0
        return struct.unpack_from("<H", self.buf, off)[0]

    def has(self, fid):
        return self._fo(fid) != 0

    def scalar(self, fid, fmt):
        fo = self._fo(fid)
        if fo == 0:
            return None
        return struct.unpack_from(fmt, self.buf, self.pos + fo)[0]

    def obj(self, fid):
        fo = self._fo(fid)
        if fo == 0:
            return None
        off = struct.unpack_from("<I", self.buf, self.pos + fo)[0]
        return self.pos + fo + off

    def table(self, fid):
        p = self.obj(fid)
        return Table(self.buf, p) if p is not None else None

    def string(self, fid):
        p = self.obj(fid)
        if p is None:
            return None
        n = struct.unpack_from("<I", self.buf, p)[0]
        return self.buf[p + 4:p + 4 + n].decode("utf-8", "replace")

    def vec_len(self, fid):
        p = self.obj(fid)
        return struct.unpack_from("<I", self.buf, p)[0] if p is not None else 0

    def vec_table(self, fid, i):
        p = self.obj(fid)
        if p is None:
            return None
        ep = p + 4 + i * 4
        off = struct.unpack_from("<I", self.buf, ep)[0]
        return Table(self.buf, ep + off)

    def vec_scalar(self, fid, i, fmt, size):
        p = self.obj(fid)
        if p is None:
            return None
        return struct.unpack_from(fmt, self.buf, p + 4 + i * size)[0]


def shape_of(t):
    n = t.vec_len(0)
    return [t.vec_scalar(0, i, "<i", 4) for i in range(n)]


def quant_of(t):
    q = t.table(4)
    if q is None:
        return None
    ns = q.vec_len(2)
    nz = q.vec_len(3)
    scale = [q.vec_scalar(2, i, "<f", 4) for i in range(ns)]
    zp = [q.vec_scalar(3, i, "<q", 8) for i in range(nz)]
    return {"scale": scale, "zero_point": zp}


def main(path):
    with open(path, "rb") as f:
        buf = f.read()
    root_off = struct.unpack_from("<I", buf, 0)[0]
    model = Table(buf, root_off)

    # Operator codes (builtin) — just for identification.
    n_ops = model.vec_len(1)
    builtins = []
    for i in range(n_ops):
        oc = model.vec_table(1, i)
        bc = oc.scalar(0, "<b") if oc.has(0) else None
        dbc = oc.scalar(3, "<b") if oc.has(3) else None
        builtins.append(bc if dbc is None else dbc)
    print("operator_codes (builtin):", builtins)

    n_sg = model.vec_len(2)
    print("num_subgraphs:", n_sg)
    sg = model.vec_table(2, 0)
    n_tensors = sg.vec_len(0)
    inputs = [sg.vec_scalar(1, i, "<i", 4) for i in range(sg.vec_len(1))]
    outputs = [sg.vec_scalar(2, i, "<i", 4) for i in range(sg.vec_len(2))]
    print("subgraph0 inputs:", inputs)
    print("subgraph0 outputs:", outputs)
    print("num_tensors:", n_tensors)

    def describe(idx):
        t = sg.vec_table(0, idx)
        return {
            "index": idx,
            "name": t.string(3),
            "shape": shape_of(t),
            "type": TYPE_NAMES.get(t.scalar(1, "<b"), t.scalar(1, "<b")),
            "quant": quant_of(t),
        }

    for idx in inputs:
        print("INPUT:", describe(idx))
    for idx in outputs:
        print("OUTPUT:", describe(idx))


if __name__ == "__main__":
    main(sys.argv[1])
