"""Audit CONV_2D / DEPTHWISE_CONV_2D ops in a .tflite against WebNN's conv2d
groups validation.

WebNN rejects a conv2d when (graph_validation_utils.cc:781-787):

    input_channels % groups != 0  ||
    filter_input_channels != input_channels / groups

TFLite expresses grouping implicitly: CONV_2D is a grouped conv whenever the
filter's input-channel dim is smaller than the input's, and DEPTHWISE_CONV_2D
carries a depth_multiplier. A TFLite->WebNN converter has to recover `groups`
from the shapes, so this prints the shapes and the groups value each op needs.

Usage:
  py conv_audit.py <model.tflite> [--all]
"""

import argparse
import sys

from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator

# code -> name, for reporting
_OP_NAMES = {
    v: k for k, v in vars(BuiltinOperator).items() if isinstance(v, int)
}


def shape_of(subgraph, idx):
    if idx < 0:
        return None
    t = subgraph.Tensors(idx)
    return [int(t.Shape(i)) for i in range(t.ShapeLength())]


def name_of(subgraph, idx):
    if idx < 0:
        return "-"
    raw = subgraph.Tensors(idx).Name()
    return raw.decode("utf-8", "replace") if raw else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--all", action="store_true",
                    help="print every conv, not just the failing ones")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model = Model.GetRootAs(bytearray(f.read()), 0)

    codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        # BuiltinCode() is the modern field; DeprecatedBuiltinCode() caps at 127.
        bc = oc.BuiltinCode()
        codes.append(bc)

    print(f"model    : {args.model}")
    print(f"subgraphs: {model.SubgraphsLength()}")

    total_bad = 0
    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        print(f"\n=== subgraph {s}: {sg.OperatorsLength()} ops, "
              f"{sg.TensorsLength()} tensors ===")
        for n in range(sg.OperatorsLength()):
            op = sg.Operators(n)
            bc = codes[op.OpcodeIndex()]
            is_conv = bc == BuiltinOperator.CONV_2D
            is_dw = bc == BuiltinOperator.DEPTHWISE_CONV_2D
            if not (is_conv or is_dw):
                continue

            in_shape = shape_of(sg, op.Inputs(0))
            flt_shape = shape_of(sg, op.Inputs(1))
            out_shape = shape_of(sg, op.Outputs(0))
            if not in_shape or not flt_shape or len(in_shape) != 4:
                print(f"  [{n}] {_OP_NAMES.get(bc)}  UNEXPECTED RANK "
                      f"in={in_shape} filter={flt_shape}")
                continue

            in_c = in_shape[3]      # TFLite is NHWC
            out_c = out_shape[3] if out_shape and len(out_shape) == 4 else None

            if is_conv:
                # filter is OHWI: [out_c, h, w, in_c/groups]
                flt_in_c = flt_shape[3]
                needed_groups = in_c // flt_in_c if flt_in_c else 0
                layout = "ohwi"
            else:
                # filter is 1HWO: [1, h, w, in_c * depth_multiplier]
                flt_in_c = flt_shape[0]
                needed_groups = in_c
                layout = "ihwo"

            # Replay WebNN's check with the groups the shapes imply.
            ok = (needed_groups > 0 and in_c % needed_groups == 0 and
                  flt_in_c == in_c // needed_groups)
            # And with groups=1, which is what a converter emits if it ignores
            # grouping entirely.
            ok_g1 = (flt_in_c == in_c)

            grouped = needed_groups != 1
            dw_mult = (out_c // in_c) if (is_dw and out_c and in_c) else None

            if not args.all and ok and ok_g1:
                continue

            flag = "" if ok else "  <-- FAILS EVEN WITH CORRECT GROUPS"
            if ok and not ok_g1:
                flag = "  <-- needs groups=%d; groups=1 would throw" % needed_groups
            total_bad += 0 if (ok and ok_g1) else 1

            print(f"  [{n}] {_OP_NAMES.get(bc, bc)}{flag}")
            print(f"       input  {in_shape}  (in_c={in_c})")
            print(f"       filter {flt_shape}  (filter_in_c={flt_in_c}, "
                  f"expects layout '{layout}')")
            print(f"       output {out_shape}" +
                  (f"  depth_multiplier={dw_mult}" if dw_mult is not None else ""))
            print(f"       implied groups={needed_groups}"
                  f"  grouped={grouped}"
                  f"  webnn_ok(groups={needed_groups})={ok}"
                  f"  webnn_ok(groups=1)={ok_g1}")
            print(f"       tensors: in='{name_of(sg, op.Inputs(0))}' "
                  f"filter='{name_of(sg, op.Inputs(1))}'")

    print(f"\nconvs needing non-trivial groups or outright invalid: {total_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
