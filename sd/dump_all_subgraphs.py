"""Dump every subgraph of a .tflite: declared inputs/outputs, all tensors
(shape+dtype+const-ness) and all operators. Written for inspecting the
decomposition subgraph emitted alongside an `odml.scaled_dot_product_attention`
STABLEHLO_COMPOSITE.

Usage: py dump_all_subgraphs.py <model.tflite>
"""
import struct
import sys

sys.path.insert(0, r"C:\Users\fujun\workspace\webnn\segment_anythings\tools")
sys.path.insert(0, r"C:\Users\fujun\workspace\webnn\segment_anythings\sam_native_runner")
from find_op_pattern import Model, BUILTIN_NAMES  # noqa: E402


def ivec(table, field):
    """Read a [int32] vector field as a python list."""
    p = table.obj(field)
    if p is None:
        return []
    n = struct.unpack_from("<I", table.buf, p)[0]
    return list(struct.unpack_from("<%di" % n, table.buf, p + 4))


def main(path):
    m = Model(path)
    n_sub = m.model.vec_len(2)
    print(f"{path}\n  subgraphs={n_sub} buffers={m.n_buffers}")
    for si in range(n_sub):
        m.sg = m.model.vec_table(2, si)
        m.n_tensors = m.sg.vec_len(0)
        m.n_ops_sg = m.sg.vec_len(3)
        # SubGraph: tensors(0), inputs(1), outputs(2), operators(3), name(4)
        sg_in = ivec(m.sg, 1)
        sg_out = ivec(m.sg, 2)
        name = m.sg.string(4) or ""
        print(f"\n  === subgraph[{si}] name={name!r} "
              f"tensors={m.n_tensors} ops={m.n_ops_sg}")
        print(f"      inputs={sg_in} outputs={sg_out}")
        for ti in range(m.n_tensors):
            t = m.tensor(ti)
            const = "const" if t["buffer"] and m.buffer_bytes(t["buffer"]) else "     "
            tag = []
            if ti in sg_in:
                tag.append(f"IN[{sg_in.index(ti)}]")
            if ti in sg_out:
                tag.append(f"OUT[{sg_out.index(ti)}]")
            print(f"      t#{ti:<3} {t['type']:<8} {str(t['shape']):<20} "
                  f"{const} {' '.join(tag):<8} {t['name']}")
        for oi in range(m.n_ops_sg):
            o = m.sg.vec_table(3, oi)
            opcode_idx = o.scalar(0, "<I") or 0
            builtin = m.opcodes[opcode_idx]
            print(f"      op[{oi}] {BUILTIN_NAMES.get(builtin, builtin):<22} "
                  f"in={ivec(o, 1)} out={ivec(o, 2)}")


if __name__ == "__main__":
    main(sys.argv[1])
