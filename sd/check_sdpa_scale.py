"""Decode the STABLEHLO_COMPOSITE op's flexbuffer attributes to check the
emitted 'scale' value against expectation.
"""
import struct
import sys

sys.path.insert(0, r"C:\Users\fujun\workspace\webnn\segment_anythings\tools")
sys.path.insert(0, r"C:\Users\fujun\workspace\webnn\segment_anythings\sam_native_runner")
from find_op_pattern import Model, NAME_TO_BUILTIN  # noqa: E402
from flatbuffers import flexbuffers  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "model/vae_decoder_gpu.tflite"
m = Model(path)
ops = m.all_ops()
target = NAME_TO_BUILTIN["STABLEHLO_COMPOSITE"]
for o in ops:
    if o["builtin"] != target:
        continue
    idx = o["index"]
    op_table = m.sg.vec_table(3, idx)
    # StableHLOCompositeOptions fields: name(0), decomposition_subgraph_index(1),
    # composite_attributes(2, [ubyte]), composite_attributes_format(3).
    options2 = op_table.table(12)  # builtin_options_2 field index
    name = options2.string(0)
    subgraph_index = options2.scalar(1, "<i")
    attr_vec_ptr = options2.obj(2)
    attr_bytes = b""
    if attr_vec_ptr is not None:
        n = struct.unpack_from("<I", m.buf, attr_vec_ptr)[0]
        attr_bytes = m.buf[attr_vec_ptr + 4: attr_vec_ptr + 4 + n]
    print(f"op[{idx}] name={name!r} decomposition_subgraph_index={subgraph_index} "
          f"attr_bytes_len={len(attr_bytes)}")
    if attr_bytes:
        root = flexbuffers.GetRoot(bytearray(attr_bytes))
        print("  flexbuffer map:", root.AsMap.Value)
