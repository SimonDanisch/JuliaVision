"""Merge Inductor's fusion decisions into the exported graph JSONs.

`export_graphs.py` is the correctness source; this is the *plan*, and it is
advisory — dropping it changes speed, never results.

Why it matters: DNNKernels runs one kernel per ATen op, ~2000 dispatches a step,
and at ~13 us each that launch structure *is* the step time. PyTorch reaches
~110 steps/s on the same model because Inductor fuses the same graph into tens
of kernels. Rather than re-deriving which ops may fuse, take Inductor's answer.

    uv run tools/dump_plan.py [--precision autocast]

Writes `fusion_groups` into each `gen/graphs/aten-<precision>/<name>.json`:

    [{"kernel": "triton_poi_fused_add_relu_7", "ops": ["add_12", "relu_9"]}, ...]

## Getting the names to line up

Compiling the *original* module and matching names against our export does not
work, and the failure is not subtle. The two see different graphs: our export
runs `run_decompositions()` to core ATen and lifts every view into a buffer, so
it has 1 `sub` where Inductor's post-grad graph has 44 (it expands batch-norm)
and 0 `view` where Inductor has 176. Any name- or position-based matching across
that is guesswork.

So compile the exported program's own module — `export(...).run_decompositions()`
then `.module()`. Inductor's *pre-grad* graph is then exactly our graph, and its
`preToPost` mapping is keyed by our own node names. Composing `preToPost` with
`postToCppCode` gives our op id -> the Triton kernels it lands in, with no
heuristic in the middle.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Inductor's post-grad name for an op vs ours. Only the ones that actually
# differ need an entry.
ALIAS = {
    "convert_element_type": "_to_copy",
    "_unsafe_view": "view",
}

# Ops Inductor hands to a library rather than generating: they appear in a
# kernel's provenance because they produced or consumed its data, not because
# the kernel computes them, and fusing around them would be nonsense.
EXTERN = {"convolution", "mm", "addmm", "bmm", "_scaled_dot_product_flash_attention",
          "_scaled_dot_product_efficient_attention", "max_pool2d_with_indices",
          "convolution_backward"}


def base(name):
    """`convert_element_type_3` -> `_to_copy`; `add` -> `add`."""
    stem = re.sub(r"_\d+$", "", name)
    return ALIAS.get(stem, stem)


def prename(key):
    """Inductor's pre-grad provenance key -> our op id.

    Ops with more than one result carry the overload in the key:
    `native_layer_norm_default_2` where our id is `native_layer_norm_2`. Without
    stripping it, 150 keys match nothing — 96 layer norms, 48 attentions and 6
    max-pools — and every one of them is *reported as unfused* rather than as
    unmatched. That read as "Inductor runs layer norm on its own too", which is
    false: it fuses all 96 into the surrounding elementwise kernels.

    The old `matched == 0` check cannot catch this. Partial loss is the likely
    failure, not total loss, so `groups_from` now reports the unmatched keys.
    """
    return re.sub(r"_default(_\d+)?$", lambda m: m.group(1) or "", key)


def groups_from(mapping_path, our_ops):
    """
    Fused Triton kernels -> our op ids.

    `preToPost` is keyed by our own node names (we compiled our exported
    module), so composing it with `postToCppCode` needs no name matching.
    """
    m = json.loads(Path(mapping_path).read_text())
    pre_to_post = m.get("preToPost", {})
    post_to_cpp = m.get("postToCppCode", {})
    ours = {o["id"] for o in our_ops}
    order = {o["id"]: i for i, o in enumerate(our_ops)}

    per_kernel = defaultdict(list)
    matched, unmatched = 0, []
    for rawpre, posts in pre_to_post.items():
        pre = prename(rawpre)
        if pre not in ours:
            unmatched.append(rawpre)
            continue
        matched += 1
        if base(pre) in EXTERN:
            continue                      # cuDNN/cuBLAS, not fusible by us
        # Key on the *call site* (`name:N`), not the kernel name. Inductor emits
        # one kernel definition and calls it from several places; collapsing
        # those put six unrelated `relu`s in a single "group" and would have had
        # us fusing ops that never run together.
        calls = {c for p in posts for c in post_to_cpp.get(p, ())
                 if c.startswith("triton_")}
        for k in calls:
            per_kernel[k].append(pre)

    if matched == 0:
        return None, "no pre-grad node name matched an exported op id"
    # Inductor's pre-grad graph holds nodes ours does not (lifted placeholders,
    # `getitem` on multi-result ops), so some slack is normal — but a name
    # convention drifting is a silent, partial loss, and the count is what makes
    # it visible.
    if unmatched:
        h = defaultdict(int)
        for k in unmatched:
            h[base(k)] += 1
        top = sorted(h.items(), key=lambda kv: -kv[1])[:4]
        print(f"    ({len(unmatched)} pre-grad keys matched no op id: "
              + ", ".join(f"{k}x{v}" for k, v in top) + ")")

    out, seen = [], set()
    for call, ids in per_kernel.items():
        ids = sorted(set(ids), key=lambda i: order[i])
        if len(ids) < 2:
            continue                      # a group of one is not a fusion
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kernel": call.rsplit(":", 1)[0], "call": call, "ops": ids})
    return out, None


def capture(mod, args, ctx, dbg, dynamic_shapes=None):
    """Export under `ctx`, let Inductor compile that, return its provenance file.

    The export has to be the *same* export the graph JSON came from — same
    precision context, same `run_decompositions()`, same wrapper module — or
    `preToPost` is keyed by node names that do not exist on our side and
    `groups_from` matches nothing.
    """
    import torch
    import torch._inductor.config as ic
    from torch.export import export

    ic.trace.enabled = True
    ic.trace.provenance_tracking_level = 1
    ic.trace.debug_dir = dbg
    # Without this a second run hits the FX graph cache, skips codegen entirely
    # and emits no debug directory at all — which looks exactly like "provenance
    # is unavailable" rather than "nothing was compiled".
    ic.force_disable_caches = True
    before = set(glob.glob(dbg + "/torchinductor/*"))
    with torch.no_grad(), ctx:
        ep = export(mod.eval(), args, dynamic_shapes=dynamic_shapes, strict=False)
        ep = ep.run_decompositions()      # must stay inside the context
        torch.compile(ep.module(), backend="inductor")(*args)
    fresh = sorted(set(glob.glob(dbg + "/torchinductor/*")) - before, key=os.path.getmtime)
    for d in reversed(fresh):
        p = Path(d) / "inductor_provenance_tracking_node_mappings.json"
        if p.exists():
            return p
    return None


def compile_one(mod, args, specs, table, precision, dbg):
    """Export exactly as `export_graphs.py` does, then let Inductor compile that."""
    import contextlib

    import torch

    ctx = (torch.amp.autocast(device_type="cuda", enabled=True)
           if precision == "autocast" else contextlib.nullcontext())
    resolved = tuple({k: table[v] for k, v in s.items()} for s in specs)
    return capture(mod, args, ctx, dbg, dynamic_shapes=resolved)


def merge(jf, mapping_path, name):
    """Write `fusion_groups` into an exported graph JSON. Returns groups, ops."""
    g = json.loads(Path(jf).read_text())
    grps, why = groups_from(mapping_path, g["ops"])
    if grps is None:
        print(f"  {name}: name alignment failed ({why}); leaving fusion_groups empty")
        return 0, 0
    g["fusion_groups"] = grps
    Path(jf).write_text(json.dumps(g, indent=1))
    # Distinct ops: a node can appear in several kernels (Inductor recomputes
    # cheap producers rather than materialising them), so summing group sizes
    # overcounts and can exceed the op count outright.
    fused = len({i for x in grps for i in x["ops"]})
    print(f"  {name}: {len(grps)} groups, {fused}/{len(g['ops'])} distinct ops fused")
    return len(grps), fused


def run(precision="autocast", dbg="/tmp/lavadnn_inductor"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
    import common
    import export_graphs as EX
    import graphs as G
    import patches

    common.bootstrap()
    patches.apply()
    model, _ = common.load_model()
    trace = json.loads((common.GEN / "graphs" / "trace.json").read_text())
    built = G.build(model, trace["typeparams"])
    table = EX.dims()
    gdir = common.GEN / "graphs" / f"aten-{precision}"
    os.makedirs(dbg, exist_ok=True)

    total_groups = total_ops = 0
    for name, (mod, args, specs) in built.items():
        jf = gdir / f"{name}.json"
        if not jf.exists():
            print(f"  {name}: no exported graph, skipping")
            continue
        g = json.loads(jf.read_text())
        try:
            patches.clear_pe_caches(model)
            mp = compile_one(mod, args, specs, table, precision, dbg)
        except Exception as e:
            print(f"  {name}: COMPILE FAILED {type(e).__name__}: {str(e)[:160]}")
            continue
        if mp is None:
            print(f"  {name}: no provenance emitted")
            continue
        a, b = merge(jf, mp, name)
        total_groups += a
        total_ops += b
    print(f"total: {total_groups} groups, {total_ops} ops fused")


def run_sam2(size="large", precision="autocast", decoder_precision="fp32",
             which=("encoder", "decoder"), dbg="/tmp/lavadnn_inductor"):
    """The same, for the two graphs `export_sam2.py` writes.

    These are the graphs `DNNKernels` actually runs — 1353 ops in the encoder
    against 189 in the older `aten-autocast/encode_image` — and they had no
    plan at all: `dump_plan.py` was written for `graphs.py`'s seven-graph
    decomposition and was never pointed at this one.

    Kept separate from `run()` rather than generalised, because the two differ
    in more than a path: `export_sam2` wraps the model itself (normalisation
    inside the graph), exports no dynamic dimensions, and gives the decoder its
    own precision policy.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")
    import torch

    import common
    import export_graphs as EG
    import export_sam2 as ES

    common.bootstrap()
    gdir = common.GEN / "graphs" / f"sam2-{size}"
    os.makedirs(dbg, exist_ok=True)

    model = ES.build(size)
    dev = next(model.parameters()).device
    res = model.image_size
    image = torch.zeros(1, 3, res, res, device=dev)
    enc = ES.Encoder(model).to(dev).eval()

    total = []
    if "encoder" in which:
        mp = capture(enc, (image,), EG.precision_ctx(precision), dbg)
        total.append(merge(gdir / "sam2_encoder.json", mp, "sam2_encoder")
                     if mp else print("  sam2_encoder: no provenance emitted"))

    if "decoder" in which:
        dec = ES.Decoder(model).to(dev).eval()
        with torch.no_grad(), EG.precision_ctx(precision):
            feats = enc(image)
        nfpn = len(feats) // 2
        f0, f1, f2 = feats[0], feats[1], feats[nfpn - 1]
        point = torch.full((1, ES.MAXPOINTS, 2), res / 2.0, device=dev)
        label = torch.full((1, ES.MAXPOINTS), -1, dtype=torch.int32, device=dev)
        label[0, 0] = 1
        if decoder_precision == "fp32":
            f0, f1, f2 = (t.float() for t in (f0, f1, f2))
        del feats
        torch.cuda.empty_cache()
        mp = capture(dec, (f0, f1, f2, point, label),
                     EG.precision_ctx(decoder_precision), dbg)
        total.append(merge(gdir / "sam2_decoder.json", mp, "sam2_decoder")
                     if mp else print("  sam2_decoder: no provenance emitted"))
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"])
    ap.add_argument("--model", default="aten", choices=["aten", "sam2"],
                    help="'aten' is graphs.py's seven-graph decomposition; "
                         "'sam2' is the encoder/decoder pair DNNKernels runs")
    ap.add_argument("--only", default="encoder,decoder",
                    help="for --model sam2: which of the two to compile")
    a = ap.parse_args()
    if a.model == "sam2":
        run_sam2(precision=a.precision, which=tuple(a.only.split(",")))
    else:
        run(a.precision)
