"""Audit: does DNNKernels read the attribute keys the exporter actually writes?

`export_graphs.py` names an op's attributes two different ways depending on how
the traced call was written: positional args become `arg{j}`, keyword args keep
their own name (`approximate`, `dim`, ...). Nothing normalises the two, so a
`runop!` that reads `op.attrs["arg1"]` silently sees nothing when the graph
recorded `approximate`, and falls back to its default.

That is not hypothetical: `gelu.default` was read as `arg1 == "tanh"` and every
graph in `gen/graphs` records it as `{"approximate": "tanh"}`, so the tanh
branch had never once been taken.

This walks `DNNKernels/src/*.jl` for the attr keys each `runop!` reads, walks
every graph JSON for the keys each aten actually carries, and prints the two
one-sided differences:

  READ BUT NEVER PRESENT   a dead branch — the gelu class of bug
  PRESENT BUT NEVER READ   a semantic the runtime is ignoring

Neither is automatically a bug (a default may be the only value in the tree),
so the output is a list to check by hand, not a verdict.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path("/sim/Programmieren/VideoEdit/dev/JuliaVision/DNNKernels/src")
GRAPHS = Path("/sim/Programmieren/VideoEdit/gen/graphs")

VAL_RE = re.compile(r'::Val\{Symbol\("([^"]+)"\)\}')
VAL_BARE_RE = re.compile(r'::Val\{:([A-Za-z_][A-Za-z0-9_!]*)\}')
ATTR_RE = re.compile(r'(?:op|o)\.attrs(?:\s*,\s*|\[)\s*"([^"]+)"')
GET_RE = re.compile(r'get\((?:op|o)\.attrs,\s*"([^"]+)"')
HAS_RE = re.compile(r'haskey\((?:op|o)\.attrs,\s*"([^"]+)"')


def read_keys():
    """aten -> {attr key read in its runop!}, by scanning until the next runop!."""
    out = defaultdict(set)
    for f in sorted(SRC.rglob("*.jl")):
        cur = None
        for line in f.read_text().splitlines():
            if "runop!" in line:
                m = VAL_RE.search(line) or VAL_BARE_RE.search(line)
                cur = m.group(1) if m else None
                if cur and "." not in cur:
                    cur = cur + ".default"
            if cur is None:
                continue
            for rx in (GET_RE, HAS_RE, ATTR_RE):
                for m in rx.finditer(line):
                    out[cur].add(m.group(1))
    return out


def present_keys():
    """aten -> {attr key seen in any exported graph}, and which graphs."""
    out = defaultdict(set)
    where = defaultdict(set)
    for f in sorted(GRAPHS.glob("*/*.json")):
        if f.name == "op_histogram.json":
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "ops" not in d:
            continue
        for o in d["ops"]:
            for k in o.get("attrs", {}):
                out[o["aten"]].add(k)
                where[(o["aten"], k)].add(f.parent.name)
    return out, where


def main():
    read, (present, where) = read_keys(), present_keys()

    print("=== READ BUT NEVER PRESENT (dead branch) ===")
    for aten in sorted(read):
        if aten not in present:
            continue
        dead = {k for k in read[aten] if k.startswith("arg")} - present[aten]
        if dead:
            print(f"{aten}: reads {sorted(dead)}, graphs carry {sorted(present[aten])}")

    print()
    print("=== PRESENT BUT NEVER READ (ignored semantic) ===")
    for aten in sorted(present):
        if aten not in read:
            continue
        unread = present[aten] - read[aten] - {"dtype", "layout", "pin_memory", "device"}
        for k in sorted(unread):
            print(f"{aten}: '{k}' ignored, in {sorted(where[(aten, k)])[:4]}")

    print()
    print("=== ATENS IN GRAPHS WITH NO runop! FOUND (scanner blind spots) ===")
    missing = sorted(set(present) - set(read))
    print(f"{len(missing)}: {missing[:25]}")


if __name__ == "__main__":
    sys.exit(main())
