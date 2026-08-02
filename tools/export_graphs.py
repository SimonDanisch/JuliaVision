"""torch.export every graph in graphs.py to core ATen, and emit the JSON.

This is the only source correctness depends on (lava-dnn.md "Two sources,
separated"). The Inductor plan is merged in later by dump_plan.py and is
advisory.

Shapes stay symbolic in h, w (the stride-16 plane); full resolution is the
derived dim 16h x 16w, so resolution never becomes a variant axis and never
reaches a Julia type parameter.

    uv run tools/export_graphs.py
"""

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch
from torch.export import Dim, export

import common
import graphs as G
import patches


def dims():
    """The dim algebra. Everything derives from h, w so export knows the
    encoder's strides are exact rather than three unrelated symbols."""
    h = Dim("h", min=2, max=256)
    w = Dim("w", min=2, max=256)
    return {"h": h, "w": w, "h2": 2 * h, "w2": 2 * w, "h4": 4 * h, "w4": 4 * w,
            "h8": 8 * h, "w8": 8 * w, "H": 16 * h, "W": 16 * w}


def resolve(spec, table):
    return {k: table[v] for k, v in spec.items()}


# ATen ops that alias their input rather than compute anything. They become a
# parent + offset + strides descriptor in the emitter, never an allocation and
# never a kernel launch. sym_size/_assert_* are host-side facts, dropped here
# because every shape is already resolved symbolically in the JSON.
VIEW_OPS = {
    "view.default", "_unsafe_view.default", "permute.default", "transpose.int",
    "t.default", "unsqueeze.default", "squeeze.dims", "squeeze.dim",
    "slice.Tensor", "expand.default", "alias.default", "select.int",
    "split_with_sizes.default", "detach.default",
}
DROP_OPS = {"_assert_tensor_metadata.default", "_assert_scalar.default"}

# every dim name in graphs.py is a fixed multiple of h or w
ROOT = {"h": "h", "H": "h", "h2": "h", "h4": "h", "h8": "h",
        "w": "w", "W": "w", "w2": "w", "w4": "w", "w8": "w"}


def symbol_names(ep, specs):
    """sympy symbol -> our dim name, read off the placeholders we declared."""
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    user = [n for n in placeholders if n.name in ep.graph_signature.user_inputs]
    out = {}
    for node, spec in zip(user, specs):
        val = node.meta.get("val")
        if val is None:
            continue
        for idx, name in spec.items():
            expr = val.shape[idx].node.expr
            syms = list(expr.free_symbols)
            if len(syms) == 1:
                out[syms[0]] = ROOT[name]
    return out


def sym_str(d, names):
    """A single (possibly symbolic) extent, as an int or a string in h, w."""
    import sympy

    if isinstance(d, int):
        return d
    expr = d.node.expr if hasattr(d, "node") else d
    if getattr(expr, "is_Integer", False):
        return int(expr)
    return str(expr.subs({s: sympy.Symbol(n) for s, n in names.items()}))


def shape_str(val, names):
    """Serialize a (possibly symbolic) shape to ints and strings in h, w."""
    import sympy

    return [sym_str(d, names) for d in val.shape]


def const(x):
    """A non-tensor argument, as it will appear in attrs."""
    if isinstance(x, float) and not math.isfinite(x):
        # json.dump would emit bare Infinity/NaN, which is not JSON and no
        # strict parser accepts it; the reader turns these back into floats
        return "-inf" if x < 0 else ("inf" if x > 0 else "nan")
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [const(v) for v in x]
    if isinstance(x, torch.dtype):
        return str(x).removeprefix("torch.")
    if isinstance(x, torch.device):
        return None
    return str(x)


def convert(ep, specs, name):
    """ExportedProgram -> the buffers/ops JSON of lava-dnn.md."""
    sig = ep.graph_signature
    names = symbol_names(ep, specs)
    param_of = dict(sig.inputs_to_parameters)
    param_of.update(sig.inputs_to_buffers)
    # Lifted tensor constants are weights too, whatever torch.export calls them.
    # A tensor a module holds as a plain attribute rather than a registered
    # buffer — SAM 2's neck builds its four positional encodings that way — gets
    # hoisted to a graph *input* by torch.export and lands in `ep.constants`
    # rather than the state_dict. Left in this bucket it turns into something the
    # caller has to pass in on every single call, which for SAM 2 means four
    # tensors up to 256x256x256 handed to the encoder per frame. They are
    # constant; treat them as the weights they are. `save_constants` writes their
    # values next to the state_dict so `key` resolves.
    param_of.update(getattr(sig, "inputs_to_lifted_tensor_constants", {}))

    buffers, ops = {}, []
    produced_at, last_use = {}, {}

    def add_buffer(nid, val, kind, **extra):
        b = {"id": nid, "kind": kind}
        if isinstance(val, (tuple, list)):
            # multi-result op (layer_norm, max_pool2d_with_indices, sdpa); the
            # elements are reached through getitem views
            b["shapes"] = [shape_str(v, names) if hasattr(v, "shape") else None for v in val]
            b["dtypes"] = [str(v.dtype).removeprefix("torch.") if hasattr(v, "dtype") else None
                           for v in val]
        elif val is not None and hasattr(val, "shape"):
            b["shape"] = shape_str(val, names)
            b["dtype"] = str(val.dtype).removeprefix("torch.")
        b.update(extra)
        buffers[nid] = b
        return b

    inputs, outputs = [], []
    for n in ep.graph.nodes:
        if n.op != "placeholder":
            continue
        val = n.meta.get("val")
        if n.name in param_of:
            add_buffer(n.name, val, "weight", key=param_of[n.name])
        else:
            add_buffer(n.name, val, "external")
            inputs.append(n.name)

    # liveness is indexed in op space, not node space: views and asserts do not
    # advance it, so the intervals line up with the emitted pass list
    for n in ep.graph.nodes:
        if n.op == "call_function":
            if str(n.target).removeprefix("aten.") in DROP_OPS:
                continue
            i = len(ops)
            args, attrs = [], {}
            for j, a in enumerate(n.args):
                if isinstance(a, torch.fx.Node):
                    args.append(a.name)
                    last_use[a.name] = i
                elif isinstance(a, (list, tuple)) and any(isinstance(v, torch.fx.Node) for v in a):
                    # A list mixing buffers and constants, e.g. constant_pad_nd's
                    # [0, sub_141]. The buffers go into `in` so the dependency and
                    # liveness are tracked, but the list itself has to survive with
                    # its order and its constants intact, so it is also emitted as
                    # an attr with "$name" standing for a buffer reference.
                    for v in a:
                        if isinstance(v, torch.fx.Node):
                            args.append(v.name)
                            last_use[v.name] = i
                    attrs[f"arg{j}"] = [f"${v.name}" if isinstance(v, torch.fx.Node) else const(v)
                                        for v in a]
                else:
                    attrs[f"arg{j}"] = const(a)
            for k, v in n.kwargs.items():
                if isinstance(v, torch.fx.Node):
                    args.append(v.name)
                    last_use[v.name] = i
                else:
                    attrs[k] = const(v)

            target = str(n.target).removeprefix("aten.")
            val = n.meta.get("val")

            # Not a tensor: a size computed on the host (h*w, padding to a
            # multiple of 8, ...). Keep it as a symbolic scalar so the ops that
            # consume it still resolve, but it is never a kernel or a buffer.
            if isinstance(val, (int, bool, float, torch.SymInt, torch.SymBool, torch.SymFloat)):
                add_buffer(n.name, None, "host", expr=sym_str(val, names))
                produced_at[n.name] = i
                continue
            # Views alias their parent: parent + offset + strides, no allocation
            # and no kernel. getitem is the same thing for a tuple result.
            if "getitem" in target or target in VIEW_OPS:
                add_buffer(n.name, val, "view", of=args[0] if args else None,
                           op=target, attrs=attrs)
                produced_at[n.name] = i
                continue
            add_buffer(n.name, val, "transient")
            produced_at[n.name] = i
            ops.append({"id": n.name, "aten": target, "in": args, "out": n.name,
                        "attrs": {k: v for k, v in attrs.items() if v is not None}})
        elif n.op == "output":
            flat = n.args[0]
            flat = flat if isinstance(flat, (list, tuple)) else [flat]
            for a in flat:
                if isinstance(a, torch.fx.Node):
                    outputs.append(a.name)
                    last_use[a.name] = len(list(ep.graph.nodes))

    n_ops = len(ops)
    for bid, b in buffers.items():
        if b["kind"] == "transient":
            if bid in outputs:
                b["kind"] = "external"
            else:
                b["live"] = [produced_at.get(bid, 0), last_use.get(bid, n_ops)]

    return {"name": name, "symbols": sorted({n for n in names.values()}),
            "inputs": inputs, "outputs": outputs,
            "buffers": list(buffers.values()), "ops": ops,
            "fusion_groups": [], "barriers": []}


def save_constants(ep):
    """The tensor values behind `inputs_to_lifted_tensor_constants`.

    Keyed by the same names `convert` writes into each weight buffer's `key`, so
    merging this into the state_dict before `save_file` is all it takes for them
    to load as ordinary weights.
    """
    return {k: v for k, v in getattr(ep, "constants", {}).items()
            if isinstance(v, torch.Tensor)}


def precision_ctx(precision):
    """The dtype policy the graph is traced under.

    `autocast` is what MatAnyone2 actually ships: inference_matanyone2.py wraps
    main in @safe_autocast_decorator(), which is autocast(enabled=True), so conv
    and matmul run in fp16 while reductions, norms and cat stay fp32.

    torch.export captures that policy as explicit `_to_copy` nodes, so the dtype
    of every buffer becomes data in the JSON rather than a runtime policy we
    would have to reimplement - provided run_decompositions() runs INSIDE the
    same context. Called outside, it re-traces and silently drops every cast.
    """
    if precision == "autocast":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()


def run(out_dir, precision="autocast"):
    common.bootstrap()
    patches.apply()  # bit-exact; `uv run tools/patches.py` is the proof
    model, dev = common.load_model()
    trace = json.loads((common.GEN / "graphs" / "trace.json").read_text())
    tp = trace["typeparams"]

    table = dims()
    built = G.build(model, tp)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, histogram = {}, {}
    for name, (mod, args, specs) in built.items():
        resolved = tuple(resolve(s, table) for s in specs)
        try:
            patches.clear_pe_caches(model)
            with torch.no_grad(), precision_ctx(precision):
                ep = export(mod.eval(), args, dynamic_shapes=resolved, strict=False)
                ep = ep.run_decompositions()   # must stay inside the context
        except Exception as e:
            print(f"  {name}: EXPORT FAILED: {type(e).__name__}: {str(e)[:300]}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        g = convert(ep, specs, name)
        (out_dir / f"{name}.json").write_text(json.dumps(g, indent=1))
        results[name] = g
        for o in g["ops"]:
            histogram[o["aten"]] = histogram.get(o["aten"], 0) + 1
        nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
        dt = {}
        for b in g["buffers"]:
            if b.get("dtype"):
                dt[b["dtype"]] = dt.get(b["dtype"], 0) + 1
        print(f"  {name}: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights, "
              f"dtypes {dt}")

    (out_dir / "op_histogram.json").write_text(json.dumps(histogram, indent=1, sort_keys=True))
    print(f"\n{len(histogram)} distinct ATen ops across all graphs:")
    for op, n in sorted(histogram.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}  {op}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"],
                    help="autocast matches the shipped model; fp32 is the debug oracle")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else common.GEN / "graphs" / f"aten-{a.precision}"
    run(out, a.precision)
