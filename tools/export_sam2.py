"""
Export SAM 2.1's single-image path to DNNKernels' graph JSON + weights.

Two graphs, because the editor calls them at different rates: the **encoder**
runs once per frame you mark (0.31 s for the large variant at 1080p on this
card), the **decoder** runs per click (0.04 s). Splitting them is not an
optimisation detail — it is what makes clicking feel immediate, and a single
fused graph would re-embed the frame on every point.

    uv run tools/export_sam2.py --size large --out gen/graphs/sam2-large

What the two graphs are:

  encoder   image (1, 3, 1024, 1024) -> the FPN's three feature maps. SAM 2
            resizes to a square 1024 itself; the editor does the same before
            handing a frame over, so the graph's extents stay static.
  decoder   those features + up to `MAXPOINTS` points -> three masks at 256x256
            and their IoU predictions. Three, because a click is ambiguous (a
            stripe, the jacket, the person) and SAM offers all three rather than
            guessing; the choice belongs to the user, not to an argmax. Unused
            point slots carry label -1, SAM's own "not a point".

The prompt encoder is folded into the decoder graph: it is a handful of
embeddings and a positional encoding, and keeping it separate would mean a third
graph whose only job is to add two tensors.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.export import export

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()
sys.path.insert(0, str(ROOT / "tools"))

import export_graphs as EG  # noqa: E402

MAXPOINTS = 16

CKPTS = {
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
}


class Encoder(torch.nn.Module):
    """Image -> backbone features, exactly what `set_image` computes and caches.

    The ImageNet normalisation SAM 2 keeps in `SAM2Transforms` is folded in here,
    so the graph takes plain 0..1 RGB and the mean/std exist once, in the export,
    rather than being restated in Julia where they could drift from the
    checkpoint they belong to. Everything else `set_image` does — the resize to a
    1024 square — stays with the caller, which has the frame and knows how it
    wants to sample it.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[:, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[:, None, None])

    def forward(self, image):
        out = self.model.forward_image((image - self.mean) / self.std)
        # `_prepare_backbone_features` picks the last `num_feature_levels` maps and
        # flattens them; keeping the raw maps instead leaves the reshaping to the
        # caller, where it is a view rather than an op in the graph.
        return tuple(out["backbone_fpn"]) + tuple(out["vision_pos_enc"])


class Decoder(torch.nn.Module):
    """Features + N points -> three masks and their IoU predictions.

    `point` is `(1, N, 2)` in the model's own 1024-square pixel coordinates and
    `label` is `(1, N)`: 1 for "this is the subject", 0 for "this is not", and
    **-1 for "no point here"**. The editor maps a preview click into that space
    with the same matrices it uses for everything else on the canvas.

    N is fixed at export rather than made a dynamic dimension, because SAM
    already has the concept a variable point count would buy: its prompt encoder
    replaces every `-1`-labelled point with a learned `not_a_point` embedding, so
    a graph exported for N points answers correctly for any k <= N by padding the
    labels. That keeps the shapes static — a decode is 251 ops on a 64x64
    embedding either way, so the unused slots cost nothing worth a dynamic shape.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat0, feat1, feat2, point, label):
        sparse, dense = self.model.sam_prompt_encoder(
            points=(point, label), boxes=None, masks=None)
        low_res_masks, iou, _, _ = self.model.sam_mask_decoder(
            image_embeddings=feat2,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[feat0, feat1],
        )
        return low_res_masks, iou


def device():
    """CUDA when there is one, and it matters for more than speed.

    `scaled_dot_product_attention` lowers differently per device:
    `run_decompositions()` keeps it as one `_scaled_dot_product_*` op on CUDA,
    but on CPU there is no fused kernel to dispatch to, so it decomposes into
    the math form — an explicit softmax over a materialised attention matrix,
    plus the `logical_not`/`where` pair that guards fully-masked rows.

    For SAM 2's encoder that is the difference between a fused attention and a
    `(1, 8, 4096, 4096)` tensor — 512 MiB — for each of its global-attention
    blocks, several of them live at once. Exported from CPU the encoder needs
    15 GB and does not fit on a 20 GB card; exported from CUDA it is the same
    network with the attention left whole.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def build(size, dev=None):
    from sam2.build_sam import build_sam2

    ckpt, cfg = CKPTS[size]
    model = build_sam2(cfg, str(ROOT / "dev" / "sam2" / "checkpoints" / ckpt),
                       device=dev or device())
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(CKPTS))
    ap.add_argument("--points", type=int, default=MAXPOINTS,
                    help="point slots the decoder graph is exported for; "
                         "unused slots carry label -1")
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"],
                    help="encoder precision; autocast is what SAM 2 itself runs under "
                         "and what puts its matmuls on fp16 tensor cores")
    ap.add_argument("--decoder-precision", default="fp32", choices=["autocast", "fp32"],
                    help="decoder precision, fp32 by default: it is 205 ops on a 64x64 "
                         "embedding, so fp16 buys no measurable time and only costs "
                         "precision in the mask logits the threshold is taken from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    npoints = args.points
    out = Path(args.out or (ROOT / "gen" / "graphs" / f"sam2-{args.size}"))
    out.mkdir(parents=True, exist_ok=True)

    model = build(args.size)
    dev = next(model.parameters()).device
    res = model.image_size
    image = torch.zeros(1, 3, res, res, device=dev)

    # `.to(dev)` on the wrappers, not just the model: `Encoder` registers the
    # normalisation buffers itself, and they are created wherever the wrapper is.
    enc, dec = Encoder(model).to(dev).eval(), Decoder(model).to(dev).eval()

    with torch.no_grad(), EG.precision_ctx(args.precision):
        feats = enc(image)
        print(f"encoder outputs: {[tuple(f.shape) for f in feats]}")

    graphs = {}
    # The state_dict is not the whole story: torch.export lifts tensors a module
    # holds as plain attributes (SAM 2's four neck positional encodings) into
    # `ep.constants`, and `EG.convert` emits those as weights. Collect them here
    # or the encoder graph asks its caller for a 256x256x256 tensor per frame.
    constants = {}
    # `run_decompositions()` must run INSIDE the same context or it re-traces and
    # silently drops every cast — see `EG.precision_ctx`.
    with torch.no_grad(), EG.precision_ctx(args.precision):
        ep = export(enc, (image,), strict=False).run_decompositions()
        graphs["sam2_encoder"] = EG.convert(ep, ({},), "sam2_encoder")
        constants.update(EG.save_constants(ep))

        nfpn = len(feats) // 2
        f0, f1, f2 = feats[0], feats[1], feats[nfpn - 1]
        point = torch.full((1, npoints, 2), res / 2.0, device=dev)
        label = torch.full((1, npoints), -1, dtype=torch.int32, device=dev)
        label[0, 0] = 1
    # The decoder is exported under its own policy, in its own context. The
    # features it consumes come out of the encoder in whatever dtype that graph
    # ends in, so they are cast to the decoder's input dtype here — `DNNKernels`'
    # `decode` does the same on the device side, from the graph's declared types.
    with torch.no_grad(), EG.precision_ctx(args.decoder_precision):
        df = [t.float() for t in (f0, f1, f2)] if args.decoder_precision == "fp32" else [f0, f1, f2]
        ep = export(dec, (df[0], df[1], df[2], point, label),
                    strict=False).run_decompositions()
        graphs["sam2_decoder"] = EG.convert(ep, ({},) * 5, "sam2_decoder")
        constants.update(EG.save_constants(ep))

    hist = {}
    for name, g in graphs.items():
        (out / f"{name}.json").write_text(json.dumps(g, indent=1))
        for o in g["ops"]:
            hist[o["aten"]] = hist.get(o["aten"], 0) + 1
        print(f"  {name}: {len(g['ops'])} ops, {len(g['buffers'])} buffers")

    # The wrapper's state_dict, not the model's. `torch.export` names weights by
    # their path inside the module it was handed, so every key here is
    # `model.image_encoder...` — saving `model.state_dict()` gives the same
    # tensors under names one level short, and every buffer lookup misses. Both
    # wrappers hold the model under the same attribute, so the encoder's dict
    # covers the decoder too, plus the normalisation constants only it declares.
    tensors = {k: v.detach().cpu().contiguous() for k, v in enc.state_dict().items()}
    tensors.update({k: v.detach().cpu().contiguous() for k, v in constants.items()})
    save_file(tensors, str(out / "weights.safetensors"))
    print(f"  weights: {len(tensors)} tensors ({len(constants)} lifted constants)")

    missing = sorted({b["key"] for g in graphs.values() for b in g["buffers"]
                      if b["kind"] == "weight"} - set(tensors))
    if missing:
        raise SystemExit(f"{len(missing)} weights the graphs ask for are not in the "
                         f"safetensors, e.g. {missing[:5]}")
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    print(f"\n{len(hist)} distinct ATen ops:")
    for op, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}  {op}")
    return graphs


if __name__ == "__main__":
    main()
