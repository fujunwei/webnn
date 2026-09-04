import struct, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("find_op_pattern.py")), "..", "sam_native_runner"))
from find_op_pattern import Model, NAME_TO_BUILTIN

path = r"C:\Users\fujun\workspace\webnn\sd\model\vae_decoder_gpu.tflite"
m = Model(path)
ops = m.all_ops()
for o in ops:
    if o["builtin"] == NAME_TO_BUILTIN["STABLEHLO_COMPOSITE"]:
        print("op", o["index"], o)
        raw_op = m.sg.vec_table(3, o["index"])
        options_2_type = raw_op.scalar(11, "<B")
        composite = raw_op.table(12)
        print("options_2_type", options_2_type)
        name = composite.string(0)
        decomp_idx = composite.scalar(1, "<i")
        print("name", name, "decomp_idx", decomp_idx)
        p = composite.obj(2)
        if p is not None:
            n = struct.unpack_from("<I", m.buf, p)[0]
            attr_bytes = m.buf[p+4:p+4+n]
            print("attr bytes len", n, list(attr_bytes))
        fmt = composite.scalar(3, "<b")
        print("attributes_format", fmt)
