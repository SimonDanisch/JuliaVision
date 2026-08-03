"""Per-layer reference activations for the Julia side to diff against.

Runs the real pipeline on a short clip, captures the actual inputs each graph
is called with, then replays each exported graph through an interpreter that
records *every* node result. That is what makes "the first mismatching layer is
the bug" (lava-dnn.md, Verification order) possible - end-to-end comparison
alone only tells you that something is wrong.

Small by default: the refs are for correctness, not for timing, so a downscaled
clip keeps refs.safetensors to a sane size while exercising identical code.

    uv run tools/dump_refs.py --max-size 128
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

import common
import export_graphs as EG
import graphs as G
import patches


# network method -> (graph name, how to flatten its args into graph inputs)
def graph_for(op, kwargs):
    if op == "encode_mask":
        return "encode_mask_deep" if kwargs.get("deep_update", True) else "encode_mask_shallow"
    return op if op in ("encode_image", "transform_key", "pixel_fusion", "readout_query",
                        "pred_uncertainty", "segment") else None


def flatten(args):
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(v for v in a if isinstance(v, torch.Tensor))
    return out


def capture(max_size, frames, warmup, precision="autocast"):
    """One representative real call per graph, with its actual input tensors."""
    common.bootstrap()
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

    model, device = common.load_model()
    captured = {}

    for name in ("encode_image", "transform_key", "encode_mask", "pixel_fusion",
                 "readout_query", "pred_uncertainty", "segment"):
        bound = getattr(model, name)

        def make(name, bound):
            def wrapper(*args, **kwargs):
                out = bound(*args, **kwargs)
                g = graph_for(name, kwargs)
                if g is not None and g not in captured:
                    captured[g] = [t.detach().clone() for t in flatten(args)]
                return out
            return wrapper

        setattr(model, name, make(name, bound))

    src = common.UPSTREAM / "inputs" / "video" / "test-sample1"
    vframes, _, length, _ = common.read_frames(src)
    vframes = torch.cat([vframes[0].unsqueeze(0).repeat(warmup, 1, 1, 1), vframes], 0).float()
    h, w = vframes.shape[-2:]
    if min(h, w) > max_size:
        nh, nw = int(h / min(h, w) * max_size), int(w / min(h, w) * max_size)
        vframes = F.interpolate(vframes, size=(nh, nw), mode="area")

    mask = np.array(Image.open(common.UPSTREAM / "inputs" / "mask" / "test-sample1.png").convert("L"))
    mask = gen_erosion(gen_dilate(mask, 10, 10), 10, 10)
    mask = torch.from_numpy(mask).float().to(device)
    if mask.shape[-2:] != vframes.shape[-2:]:
        mask = F.interpolate(mask[None, None], size=vframes.shape[-2:], mode="nearest")[0, 0]

    processor = InferenceCore(model, cfg=model.cfg)
    with torch.inference_mode(), EG.precision_ctx(precision):
        for ti in range(min(length + warmup, frames + warmup)):
            image = (vframes[ti] / 255.0).float().to(device)
            if ti == 0:
                processor.step(image, mask, objects=[1])
                processor.step(image, first_frame_pred=True)
            elif ti <= warmup:
                processor.step(image, first_frame_pred=True)
            else:
                processor.step(image)

    print(f"captured inputs at {tuple(vframes.shape[-2:])} for: {sorted(captured)}")
    return model, captured, tuple(vframes.shape[-2:])


class Recorder(torch.fx.Interpreter):
    """Runs a graph and keeps every intermediate."""

    def __init__(self, gm):
        super().__init__(gm)
        self.results = {}

    def run_node(self, n):
        out = super().run_node(n)
        if isinstance(out, torch.Tensor):
            self.results[n.name] = out.detach().contiguous().cpu()
        elif isinstance(out, (tuple, list)):
            for i, v in enumerate(out):
                if isinstance(v, torch.Tensor):
                    self.results[f"{n.name}.{i}"] = v.detach().contiguous().cpu()
        return out


def run(out_path, max_size, frames, warmup, precision="autocast"):
    common.bootstrap()
    patches.apply()

    # TF32 is on by default for conv and matmul on Ampere and later, and its
    # 10-bit mantissa shows up as ~2e-4 relative error - the same order as a
    # real bug in a fused kernel. A reference that cannot distinguish the two
    # is useless, so the refs are full fp32.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, captured, res = capture(max_size, frames, warmup, precision)

    trace = json.loads((common.GEN / "graphs" / "trace.json").read_text())
    built = G.build(model, trace["typeparams"])
    table = EG.dims()

    tensors, manifest = {}, {}
    for name, (mod, example, specs) in built.items():
        if name not in captured:
            print(f"  {name}: never called on this clip, skipped")
            continue
        args = tuple(captured[name])
        # Export from the canonical example inputs, exactly as export_graphs
        # does, then *run* the resulting program on the real ones. Re-exporting
        # from the real shapes would risk different node names, and the refs are
        # only useful if they key to the same names as the emitted graph.
        try:
            patches.clear_pe_caches(model)
            with torch.no_grad(), EG.precision_ctx(precision):
                ep = torch.export.export(mod.eval(), example,
                                         dynamic_shapes=tuple(EG.resolve(s, table) for s in specs),
                                         strict=False).run_decompositions()
        except Exception as e:
            print(f"  {name}: export failed: {type(e).__name__}: {str(e)[:160]}")
            continue

        rec = Recorder(ep.module())
        t0 = time.perf_counter()
        with torch.no_grad(), EG.precision_ctx(precision):
            rec.run(*args)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        for i, a in enumerate(args):
            tensors[f"{name}/in{i}"] = a.detach().cpu()
        for k, v in rec.results.items():
            tensors[f"{name}/node/{k}"] = v

        g = json.loads((common.GEN / "graphs" / f"aten-{precision}" / f"{name}.json").read_text())
        manifest[name] = {
            "inputs": [{"name": n, "shape": list(a.shape), "dtype": str(a.dtype).removeprefix("torch.")}
                       for n, a in zip(g["inputs"], args)],
            "outputs": g["outputs"],
            "n_nodes_recorded": len(rec.results),
            "seconds": round(dt, 4),
        }
        print(f"  {name}: {len(rec.results)} intermediates, {dt * 1e3:.1f} ms")

    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"resolution": f"{res[0]}x{res[1]}"})
    (out_path.parent / f"refs_manifest-{precision}.json").write_text(
        json.dumps({"resolution": list(res), "graphs": manifest}, indent=1))

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"\n{len(tensors)} tensors, {total / 1e6:.1f} MB -> {out_path}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-size", type=int, default=128)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out) if a.out else common.GEN / f"refs-{a.precision}.safetensors"
    run(out, a.max_size, a.frames, a.warmup, a.precision)
