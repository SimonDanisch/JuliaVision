"""
Export the Wan 2.2 VAE decoder to DNNKernels' graph JSON + weights.

Same front-end-only shape as `export_basicvsrpp.py`: build the module, export,
hand the `ExportedProgram` to `export_graphs.convert`.

Two things about loading this model are not discoverable from its own defaults,
and both fail with errors that point somewhere else:

  * **`z_dim=48`.** `_video_vae` defaults to 16 and the mismatch surfaces as a
    channel-size error on `conv2.weight`, which reads like a corrupt checkpoint.
  * **`temperal_downsample=[False, True, True]`.** The default does three
    temporal downsamples; `wan_ti2v_5B.py`'s `vae_stride=(4, 16, 16)` means two.
    The mismatch surfaces as *missing* `time_conv` keys — i.e. the checkpoint
    looks incomplete rather than the config looking wrong.

The file is loaded by path because importing it through the `wan` package pulls
in `decord`, `diffusers` and `transformers`; `vae2_2.py` itself needs only torch
and einops.

    uv run tools/export_wanvae.py --frames 3 --size 4
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_graphs as EG

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()
VAE_PY = ROOT / "dev" / "Wan2.2" / "wan" / "modules" / "vae2_2.py"
CKPT = ROOT / "gen" / "wan22" / "Wan2.2_VAE.pth"
ZDIM = 48


def load_vae(ckpt=CKPT):
    spec = importlib.util.spec_from_file_location("wanvae22", VAE_PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules["wanvae22"] = m
    spec.loader.exec_module(m)
    return m._video_vae(pretrained_path=str(ckpt), z_dim=ZDIM,
                        temperal_downsample=[False, True, True], device="cpu").eval()


class Decoder(torch.nn.Module):
    """`decode` takes `scale = [mean, 1/std]` and computes `z / scale[1] +
    scale[0]`, so a list of zeros divides by zero and the whole output is NaN —
    which looks like a broken model rather than a bad argument.

    Identity scaling is used here on purpose: the latent normalisation is a
    constant affine map the caller applies, and pinning it to identity keeps the
    exported graph a pure decoder that DNNKernels can be checked against. Production
    use passes the real per-channel statistics."""

    def __init__(self, vae, scale=(0.0, 1.0)):
        super().__init__()
        self.vae = vae
        self.scale = list(scale)

    def forward(self, z):
        return self.vae.decode(z, self.scale)


def main(frames: int, size: int, out: Path, precision: str):
    vae = load_vae()
    dec = Decoder(vae).eval()
    args = (torch.randn(1, ZDIM, frames, size, size),)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(dec, args, strict=False)
        ep = ep.run_decompositions()

    g = EG.convert(ep, ({},), "wanvae_decoder")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wanvae_decoder.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(dec.named_parameters()),
                            **dict(dec.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))
    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"wanvae_decoder: {len(g['ops'])} ops, {len(g['buffers'])} buffers, "
          f"{nw} weights, {len(tensors)} tensors")
    for op, n in sorted(hist.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:5}  {op}")
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out) if a.out else ROOT / "gen" / "graphs" / f"wanvae-{a.precision}"
    main(a.frames, a.size, outdir, a.precision)
