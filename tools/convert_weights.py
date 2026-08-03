"""matanyone2.pth -> weights.safetensors, keyed exactly as the graph JSONs are.

Taken from the constructed model's state_dict rather than the raw checkpoint,
because load_weights (matanyone2.py:290) does channel surgery on the way in and
because the normalisation constants are registered non-persistent, so they are
absent from the .pth but present in every graph.

Norm folding is deliberately NOT done here. run_decompositions lowers eval-mode
BatchNorm to per-channel mul/add, which the fused-kernel dedup in emit_kernels
picks up as part of the conv+norm+relu chain anyway; folding it into the conv
weights would be a graph rewrite for no additional win.

    uv run tools/convert_weights.py
"""

import argparse
import json
from pathlib import Path

import torch

import common


def run(out_path, graph_dir, dtype):
    common.bootstrap()
    model, _ = common.load_model()
    sd = model.state_dict()

    wanted = {}
    for f in sorted(graph_dir.glob("*.json")):
        if f.stem == "op_histogram":
            continue
        g = json.loads(f.read_text())
        for b in g["buffers"]:
            if b["kind"] == "weight":
                wanted.setdefault(b["key"], []).append(g["name"])

    # pixel_mean/pixel_std are registered non-persistent (matanyone2.py:58), so
    # they never reach the state_dict even though every image graph divides by them
    for k in list(wanted):
        if k not in sd:
            obj = model
            for part in k.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, torch.Tensor):
                sd[k] = obj

    missing = [k for k in wanted if k not in sd]
    if missing:
        raise SystemExit(f"{len(missing)} weights referenced by graphs but absent from the "
                         f"state_dict: {missing[:8]}")

    cast = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    out = {}
    for k in sorted(wanted):
        t = sd[k].detach().cpu()
        # integer buffers (num_batches_tracked and friends) keep their dtype
        out[k] = t.to(cast) if t.is_floating_point() else t

    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path), metadata={"source": "matanyone2.pth", "dtype": dtype})

    unused = sorted(set(sd) - set(wanted))
    total = sum(t.numel() * t.element_size() for t in out.values())
    print(f"{len(out)} tensors, {total / 1e6:.1f} MB {dtype} -> {out_path}")
    print(f"{len(unused)} state_dict entries unused by any graph"
          + (f" (e.g. {unused[:3]})" if unused else ""))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.GEN / "weights.safetensors"))
    ap.add_argument("--graphs", default=str(common.GEN / "graphs" / "aten"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    a = ap.parse_args()
    run(Path(a.out), Path(a.graphs), a.dtype)
