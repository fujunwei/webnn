"""Turn `[node-table]` delegate log lines into an opcode table and
LITERT_GPU_DEBUG_EXCLUDE_NODES lists.

The `[node-table]` lines come from the debug dump added to
`ml_drift_delegate/delegate/delegate_webgpu.cc::DelegatePrepare`, which is
enabled by setting LITERT_GPU_DEBUG_DUMP_NODES=1. Those indices are in the
same space that LITERT_GPU_DEBUG_END_NODE / _EXCLUDE_NODES operate on, which
is why we read them back from the log instead of parsing the .tflite.

Usage:
  py node_table.py parse  <chrome.log> [-o node_table.tsv]
  py node_table.py hist   <node_table.tsv>
  py node_table.py exclude <node_table.tsv> <OpName> [<OpName> ...]

See ../gpu_op_bisect.zh.md for the surrounding workflow.
"""

import argparse
import collections
import io
import re
import sys

BUILTIN_OPS_H = (
    r"C:\Users\fujun\workspace\chromium\src\third_party\tflite\src"
    r"\tensorflow\lite\builtin_ops.h"
)

# "[node-table] idx=17 builtin_code=9 custom=-"
NODE_RE = re.compile(
    r"\[node-table\]\s+idx=(\d+)\s+builtin_code=(\d+)\s+custom=(\S+)"
)
TOTAL_RE = re.compile(r"\[node-table\]\s+total=(\d+)")


def load_opcode_names(header_path):
    """code -> CamelCase op name, from tflite's builtin_ops.h."""
    text = io.open(header_path, encoding="utf-8", errors="replace").read()
    names = {}
    for name, code in re.findall(r"kTfLiteBuiltin(\w+)\s*=\s*(\d+)", text):
        # First definition wins; later aliases map to the same code.
        names.setdefault(int(code), name)
    if not names:
        sys.exit(f"no opcodes found in {header_path}")
    return names


def iter_log_lines(path):
    """Yield lines from a chrome log, auto-detecting UTF-16 vs UTF-8.

    Windows PowerShell 5.1 writes UTF-16LE by default for `>` and
    Tee-Object, so logs captured that way don't match a plain utf-8 read.
    """
    with open(path, "rb") as f:
        head = f.read(2)
        f.seek(0)  # hand the BOM back to the codec below
        if head in (b"\xff\xfe", b"\xfe\xff"):
            enc = "utf-16"
        else:
            enc = "utf-8-sig"
        text = io.TextIOWrapper(f, encoding=enc, errors="replace")
        yield from text


def cmd_parse(args):
    names = load_opcode_names(args.builtin_ops)
    tables = []  # (declared_total, [(idx, code, name, custom), ...])
    cur_total, cur_rows, seen = None, [], set()
    pending_total = None
    for line in iter_log_lines(args.log):
        m = TOTAL_RE.search(line)
        if m:
            # A total= line always heads a new table. If rows are in flight,
            # close that table off first, using the total it was bound to.
            if cur_rows:
                tables.append((cur_total, cur_rows))
                cur_rows, seen = [], set()
            pending_total = int(m.group(1))
            continue
        m = NODE_RE.search(line)
        if m:
            idx, code, custom = int(m.group(1)), int(m.group(2)), m.group(3)
            row = (idx, code, names.get(code, f"Unknown{code}"), custom)
            # A repeated idx also starts a new table (DelegatePrepare runs
            # once per model, so SAM logs hold an encoder table then a
            # decoder), even if no total= line separates them.
            if idx in seen:
                tables.append((cur_total, cur_rows))
                cur_rows, seen = [], set()
            if not cur_rows:
                cur_total, pending_total = pending_total, None
            seen.add(idx)
            cur_rows.append(row)
    if cur_rows:
        tables.append((cur_total, cur_rows))

    if not tables:
        sys.exit(
            "no [node-table] lines found. Was LITERT_GPU_DEBUG_DUMP_NODES set, "
            "and did chrome run with --enable-logging=stderr?"
        )

    declared_total, first_table = tables[0]

    out = io.open(args.out, "w", encoding="utf-8", newline="\n")
    out.write("idx\tcode\tname\tcustom\n")
    for idx, code, name, custom in first_table:
        out.write(f"{idx}\t{code}\t{name}\t{custom}\n")
    out.close()

    print(f"wrote {len(first_table)} nodes to {args.out}")
    if declared_total is not None and declared_total != len(first_table):
        print(
            f"WARNING: log said total={declared_total} but parsed "
            f"{len(first_table)} — the log is probably truncated.",
            file=sys.stderr,
        )
    if len(tables) > 1:
        sizes = ", ".join(str(len(t)) for _, t in tables)
        print(
            f"note: log held {len(tables)} tables ({sizes} nodes); "
            f"kept the first table only.",
            file=sys.stderr,
        )


def read_table(path):
    rows = []
    with io.open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return rows


def cmd_hist(args):
    rows = read_table(args.table)
    counts = collections.Counter(
        # Custom ops are only distinguishable by their name.
        (name if custom == "-" else f"{name}[{custom}]")
        for _, _, name, custom in rows
    )
    width = max(len(k) for k in counts)
    for name, n in counts.most_common():
        print(f"{name:<{width}}  {n:>5}")
    print(f"{'TOTAL':<{width}}  {len(rows):>5}")


def cmd_exclude(args):
    rows = read_table(args.table)
    wanted = {w.lower() for w in args.opname}
    idxs = [
        idx
        for idx, _, name, custom in rows
        if name.lower() in wanted or custom.lower() in wanted
    ]
    if not idxs:
        sys.exit(f"no nodes matched {args.opname}; try `hist` to see the names")
    print(",".join(str(i) for i in idxs))
    print(f"# {len(idxs)} nodes", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("parse", help="chrome log -> node_table.tsv")
    sp.add_argument("log")
    sp.add_argument("-o", "--out", default="node_table.tsv")
    sp.add_argument("--builtin-ops", default=BUILTIN_OPS_H)
    sp.set_defaults(func=cmd_parse)

    sp = sub.add_parser("hist", help="opcode histogram")
    sp.add_argument("table")
    sp.set_defaults(func=cmd_hist)

    sp = sub.add_parser("exclude", help="print an EXCLUDE_NODES list")
    sp.add_argument("table")
    sp.add_argument("opname", nargs="+")
    sp.set_defaults(func=cmd_exclude)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
