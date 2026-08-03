"""Per-node reference activations for SAM 2.1's two graphs.

Same contract as `dump_refs.py` — `"<graph>/in<i>"` and `"<graph>/node/<name>"`
in one safetensors file — so `DNNKernels.verifygraph` reads it without knowing
which model produced it, and "the first mismatching layer is the bug" applies
here too.

    uv run tools/dump_sam2_refs.py --size large

The click is fixed rather than random: a reference you cannot regenerate
identically is not a reference. It lands on the bird in `media/spatz.png` if
that file exists, and on the centre of a synthetic frame otherwise, so this runs
without any of the editor's media.

The encoder is the expensive half — 1491 ops at 1024x1024, whose intermediates
run to tens of GB — so by default only its six outputs are recorded, and the
decoder is recorded node by node. `--nodes all` overrides that when the encoder
itself needs bisecting; expect it to need a smaller `--size`.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import export_graphs as EG
import export_sam2 as ES

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()


class Recorder(torch.fx.Interpreter):
    """Runs a graph and keeps intermediates, optionally only a chosen few.

    Two differences from `dump_refs.Recorder`, both forced by this model:

    * every result is **cloned**. `.contiguous()` on a view returns the same
      storage, and safetensors refuses to write aliased tensors outright — SAM 2's
      decoder is ~40% reshapes, so almost every node aliases another.
    * `keep` bounds what is held. Recording all 1491 encoder nodes at 1024x1024
      is tens of GB; passing the graph's output names records the six that matter
      and lets the rest be freed as the interpreter walks past them.
    """

    def __init__(self, gm, keep=None):
        super().__init__(gm)
        self.keep = keep
        self.results = {}

    def store(self, name, t):
        if self.keep is None or name in self.keep:
            self.results[name] = t.detach().cpu().contiguous().clone()

    def run_node(self, n):
        out = super().run_node(n)
        if isinstance(out, torch.Tensor):
            self.store(n.name, out)
        elif isinstance(out, (tuple, list)):
            for i, v in enumerate(out):
                if isinstance(v, torch.Tensor):
                    self.store(f"{n.name}.{i}", v)
        return out


def frame(res):
    """A real frame at the model's square input size, or a synthetic stand-in."""
    for cand in (ROOT / "media" / "spatz.png", ROOT / "media" / "frame.png"):
        if cand.exists():
            from PIL import Image
            im = Image.open(cand).convert("RGB").resize((res, res), Image.BILINEAR)
            a = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
            return a[None], cand.name
    # Deterministic and not flat: a flat image makes every attention row equal,
    # which hides exactly the reduction-order differences this is meant to catch.
    g = torch.linspace(0, 1, res)
    a = torch.stack([g[None, :].expand(res, res),
                     g[:, None].expand(res, res),
                     (g[None, :] * g[:, None])], 0)
    return a[None], "synthetic gradient"


def run(out_path, size, which, nodes, precision, decprec):
    # TF32's 10-bit mantissa reads as ~2e-4 relative error, the same order as a
    # real bug in a fused kernel — same reasoning as dump_refs.py.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = ES.build(size)
    dev = next(model.parameters()).device
    res = model.image_size
    image, src = frame(res)
    image = image.to(dev)
    print(f"frame: {src} at {res}x{res}")

    tensors, manifest = {}, {}

    enc, dec = ES.Encoder(model).to(dev).eval(), ES.Decoder(model).to(dev).eval()

    with torch.no_grad(), EG.precision_ctx(precision):
        feats = enc(image)
        nfpn = len(feats) // 2
        f0, f1, f2 = feats[0], feats[1], feats[nfpn - 1]
        # Model-space pixel coordinates: one real point at the centre of the
        # frame, the remaining slots padded with SAM's -1 "not a point". Same
        # shape the decoder graph was exported for, so the refs exercise the
        # padding path the editor will actually hit with one click.
        point = torch.full((1, ES.MAXPOINTS, 2), res / 2.0, device=dev)
        label = torch.full((1, ES.MAXPOINTS), -1, dtype=torch.int32, device=dev)
        label[0, 0] = 1

        # Each graph is recorded under the policy it was exported with — see
        # `export_sam2.main`. Recording the decoder under autocast while the graph
        # it is compared against is fp32 would report a dtype difference as a bug.
        df = [t.float() for t in (f0, f1, f2)] if decprec == "fp32" else [f0, f1, f2]
        todo = []
        if which in ("both", "encoder"):
            todo.append(("sam2_encoder", enc, (image,), precision))
        if which in ("both", "decoder"):
            todo.append(("sam2_decoder", dec, (df[0], df[1], df[2], point, label), decprec))

        for name, mod, args, prec in todo:
            with EG.precision_ctx(prec):
                ep = torch.export.export(mod.eval(), args, strict=False).run_decompositions()
            g = json.loads((ROOT / "gen" / "graphs" / f"sam2-{size}" / f"{name}.json").read_text())
            # Per-node for the decoder, outputs-only for the encoder: 1491 nodes
            # of up to 1x256x256x256 do not fit anywhere sensible. The encoder is
            # ordinary convolution and attention that these graphs already run for
            # MatAnyone; if its outputs match, there is nothing to bisect, and if
            # they do not, `--nodes all` at a smaller size is the next step.
            allnodes = nodes == "all" or (nodes == "auto" and name == "sam2_decoder")
            keep = None if allnodes else set(g["outputs"])
            rec = Recorder(ep.module(), keep)
            t0 = time.perf_counter()
            with EG.precision_ctx(prec):
                rec.run(*args)
            dt = time.perf_counter() - t0

            for i, a in enumerate(args):
                tensors[f"{name}/in{i}"] = a.detach().cpu().contiguous().clone()
            for k, v in rec.results.items():
                tensors[f"{name}/node/{k}"] = v

            manifest[name] = {
                "inputs": [{"name": n, "shape": list(a.shape),
                            "dtype": str(a.dtype).removeprefix("torch.")}
                           for n, a in zip(g["inputs"], args)],
                "outputs": g["outputs"],
                "n_nodes_recorded": len(rec.results),
                "seconds": round(dt, 4),
            }
            print(f"  {name}: {len(rec.results)} intermediates, {dt * 1e3:.1f} ms")

    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"resolution": f"{res}x{res}"})
    (out_path.parent / "refs_manifest.json").write_text(
        json.dumps({"resolution": [res, res], "graphs": manifest}, indent=1))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"\n{len(tensors)} tensors, {total / 1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(ES.CKPTS))
    ap.add_argument("--graphs", default="both", choices=["both", "encoder", "decoder"])
    ap.add_argument("--nodes", default="auto", choices=["auto", "all", "outputs"],
                    help="auto = every node for the decoder, outputs only for the encoder")
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"])
    ap.add_argument("--decoder-precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "gen" / "graphs" / f"sam2-{a.size}" / "refs.safetensors"
    run(out, a.size, a.graphs, a.nodes, a.precision, a.decoder_precision)
