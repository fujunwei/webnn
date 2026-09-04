import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sam_native_runner"))
from find_op_pattern import Model, NAME_TO_BUILTIN, describe_tensor_ref

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\fujun\workspace\webnn\sd\model\vae_decoder_gpu.tflite"
m = Model(path)
ops = m.all_ops()

# Map each tensor index to the op that produces it, so we can walk backward
# from the composite's inputs / forward from its output.
producer = {}
for o in ops:
    for out in o["outputs"]:
        producer[out] = o
consumer = {}
for o in ops:
    for inp in o["inputs"]:
        consumer.setdefault(inp, []).append(o)

for o in ops:
    if o["builtin"] != NAME_TO_BUILTIN["STABLEHLO_COMPOSITE"]:
        continue
    print(f"=== STABLEHLO_COMPOSITE op[{o['index']}] "
          f"name={o.get('composite_name')!r} "
          f"decomposition_subgraph_index={o.get('decomposition_subgraph_index')} ===")
    for label, ti in zip(["Q", "K", "V"], o["inputs"]):
        print(f"  composite in ({label}): {describe_tensor_ref(m, ti)}")
        p = producer.get(ti)
        if p:
            print(f"    <- produced by op[{p['index']}] {p['name']}")
            for k, pin in enumerate(p["inputs"]):
                print(f"         reshape-in[{k}] {describe_tensor_ref(m, pin)}")
    for ti in o["outputs"]:
        print(f"  composite out: {describe_tensor_ref(m, ti)}")
        for c in consumer.get(ti, []):
            print(f"    -> consumed by op[{c['index']}] {c['name']}")
            for k, cout in enumerate(c["outputs"]):
                print(f"         reshape-out[{k}] {describe_tensor_ref(m, cout)}")
