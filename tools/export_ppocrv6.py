"""
Export PP-OCRv6 medium (text detection and text recognition) to DNNKernels'
graph JSON + weights.

    uv run tools/export_ppocrv6.py --device cpu
    uv run tools/export_ppocrv6.py --device cpu --page 1280x1792

Two graphs, two directories under `gen/graphs/`:

  - `ppocrv6_det`: a page, normalised, `(1, 3, H, W)` -> the probability that a
    pixel is inside a text line, `(1, 1, H, W)`. Static: `--page` (default
    1280x1792, A4 at about 155 dpi, both sides multiples of 32 as the
    detector's five stride-2 stages need). The caller scales a page into it
    and pads the rest.
  - `ppocrv6_rec`: a batch of `b` text lines scaled to 48 pixels high,
    `(b, 3, 48, W)` with `b` and `W = 16w` symbolic -> per line and time step
    (`W / 8` of them) the most likely character and its probability. The softmax over 18710 classes and the
    argmax both stay in the graph, so what crosses back to the host is two
    short vectors rather than `W / 8 x 18710` floats. CTC decoding (drop
    repeats, drop blanks) is the caller's, with `characters.json`.

**From Hugging Face Transformers, not Paddle.** Transformers 5.x carries native
PyTorch ports (`pp_ocrv6_medium_det`, `pp_ocrv6_small_rec`, which the medium
recognizer loads as), with the official safetensors under
`PaddlePaddle/PP-OCRv6_medium_{det,rec}_safetensors`. Apache-2.0, code and
weights.

**RGB in, not BGR.** Upstream feeds OpenCV's BGR order to both networks. The
input channels of each network's first convolution are reversed here instead,
so the graph takes RGB and nothing on the Julia side has to remember to swap.
The normalisation is left outside the graph, as in every runner here:

  - detector: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225),
    which is upstream's BGR-ordered pair read in RGB order;
  - recogniser: mean 0.5, std 0.5; padding to the right is 0 after
    normalisation, which is what upstream pads with.

**CPU export is fine here.** The recogniser has two SVTR attention blocks over
`W / 8` tokens; on CPU `run_decompositions` spells them as `bmm` + `_softmax`,
which DNNKernels runs, and the sequences are a few hundred tokens at most.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.export import Dim

import export_graphs as EG
from common import find_root

ROOT = find_root()
GEN = ROOT / "gen"

DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
REC_REPO = "PaddlePaddle/PP-OCRv6_medium_rec_safetensors"
ORI_REPO = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
ORI_SIZE = 224

DET_MEAN, DET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
REC_MEAN, REC_STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
REC_HEIGHT = 48


def bgr_to_rgb_stem(model):
    """Reverse the input channels of the first convolution that reads the image."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            with torch.no_grad():
                m.weight.copy_(m.weight.flip(1))
            return m
    raise RuntimeError("no 3-channel convolution found")


class Detector(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(pixel_values=x).last_hidden_state


class Orienter(nn.Module):
    """Logits for the page being turned 0, 90, 180 or 270 degrees clockwise."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(pixel_values=x).last_hidden_state


class Recogniser(nn.Module):
    """Probabilities, then the best class per time step, inside the graph."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        probs = self.net(pixel_values=x).last_hidden_state   # (1, T, C), softmaxed
        best, index = probs.max(dim=2)
        return best, index


def load():
    from transformers import AutoModelForObjectDetection, AutoImageProcessor, PPOCRV6SmallRecForTextRecognition
    det = AutoModelForObjectDetection.from_pretrained(DET_REPO).eval()
    rec = PPOCRV6SmallRecForTextRecognition.from_pretrained(REC_REPO).eval()
    chars = AutoImageProcessor.from_pretrained(REC_REPO).character_list
    bgr_to_rgb_stem(det)
    bgr_to_rgb_stem(rec)
    return det, rec, chars


def write(out, name, model, g, refs):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(g, indent=1))
    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(model.named_parameters()), **dict(model.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))
    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))
    save_file({k: v.detach().cpu().clone().contiguous() for k, v in refs.items()}, str(out / "reference.safetensors"))
    nparam = sum(v.numel() for v in dict(model.named_parameters()).values())
    print(f"{name}: {len(g['ops'])} ops, {nparam / 1e6:.1f}M parameters")
    print("  ops:", json.dumps(hist, sort_keys=True))


def sample_page(w, h, dev):
    """A page-like input: dark lines of 'text' on light paper, normalised."""
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    ink = ((yy % 40) < 14) & ((yy // 40) % 2 == 0) & ((xx % 23) < 15) & (xx > 80) & (xx < w - 80) & (yy > 100) & (yy < h - 100)
    rgb = torch.where(ink, 0.1, 0.95).float().expand(3, h, w).unsqueeze(0)
    mean = torch.tensor(DET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(DET_STD, device=dev).view(1, 3, 1, 1)
    return ((rgb - mean) / std).contiguous()


class Recorder(torch.fx.Interpreter):
    """Runs a graph and keeps every intermediate, as `dump_refs.py` does."""

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


def verification(det, rec, dev, precision, out_root):
    """Small versions of both graphs with every node's PyTorch result, for
    `DNNKernels.verifygraph`: `gen/graphs/ppocrv6_{det,rec}_verify/`."""
    for name, model, x, specs, dyn in (
        ("ppocrv6_det_verify", det, sample_page(256, 256, dev), ({},), None),
        ("ppocrv6_rec_verify", rec, sample_page(320, REC_HEIGHT, dev), ({},), None),
    ):
        with torch.no_grad(), EG.precision_ctx(precision):
            ep = torch.export.export(model, (x,), strict=False).run_decompositions()
        g = EG.convert(ep, specs, name)
        rec_ = Recorder(ep.module())
        with torch.no_grad():
            rec_.run(x)
        refs = {f"{name}/in0": x}
        refs.update({f"{name}/node/{k}": v for k, v in rec_.results.items()})
        write(out_root / name, name, model, g, refs)
        (out_root / name / "refs.safetensors").unlink(missing_ok=True)
        (out_root / name / "reference.safetensors").rename(out_root / name / "refs.safetensors")


def export_orienter(dev, precision, out_root, reference_image=None):
    """`ppocr_docori`: PP-LCNet x1.0 doc orientation, `(1, 3, 224, 224)` ->
    four logits. Upstream scales the short side to 256 and crops the centre
    224, ImageNet-normalised; RGB here for the same reason as the others."""
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    net = AutoModelForImageClassification.from_pretrained(ORI_REPO).eval()
    bgr_to_rgb_stem(net)
    model = Orienter(net).to(dev).eval()
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, (torch.randn(1, 3, ORI_SIZE, ORI_SIZE, device=dev),), strict=False)
        ep = ep.run_decompositions()
    g = EG.convert(ep, ({},), "ppocr_docori")
    if reference_image is not None:
        # A real page through upstream's own preprocessing, so the reference is
        # a confident answer rather than four similar logits.
        import numpy as np
        from PIL import Image
        proc = AutoImageProcessor.from_pretrained(ORI_REPO)
        img = np.asarray(Image.open(reference_image).convert("RGB"))
        # The processor takes RGB and hands the network BGR; the stem here was
        # turned to take RGB, so the channels go back.
        x = proc(images=img, return_tensors="pt")["pixel_values"].flip(1).contiguous().to(dev)
    else:
        x = sample_page(ORI_SIZE, ORI_SIZE, dev)
    with torch.no_grad():
        refs = {"input": x, "logits": model(x)}
    write(out_root / "ppocr_docori", "ppocr_docori", model, g, refs)


def main(page, dev, precision, out_root, verify=False, graphs=("det", "rec", "ori"), reference_image=None):
    if "ori" in graphs:
        export_orienter(dev, precision, out_root, reference_image)
        if graphs == ("ori",):
            return
    det, rec, chars = load()
    det, rec = Detector(det).to(dev).eval(), Recogniser(rec).to(dev).eval()
    w, h = page
    if verify:
        return verification(det, rec, dev, precision, out_root)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(det, (torch.randn(1, 3, h, w, device=dev),), strict=False)
        ep = ep.run_decompositions()
    g = EG.convert(ep, ({},), "ppocrv6_det")
    with torch.no_grad():
        x = sample_page(w, h, dev)
        refs = {"input": x, "probability": det(x)}
    write(out_root / "ppocrv6_det", "ppocrv6_det", det, g, refs)

    # Lines are read in batches, so the batch is symbolic as well as the width.
    # Exported at a batch of 2: torch.export specialises a dimension it sees at 1.
    wdim, bdim = Dim("w", min=2, max=200), Dim("b", min=1, max=64)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(rec, (torch.randn(2, 3, REC_HEIGHT, 640, device=dev),),
                                 dynamic_shapes=({0: bdim, 3: 16 * wdim},), strict=False)
        ep = ep.run_decompositions()
    g = EG.convert(ep, ({0: "b", 3: "W"},), "ppocrv6_rec")
    with torch.no_grad():
        ramp = torch.linspace(-1, 1, 640, device=dev).view(1, 1, 1, 640).expand(1, 3, REC_HEIGHT, 640)
        page = sample_page(640, REC_HEIGHT * 3, dev)[:, :, 40:40 + REC_HEIGHT, :]
        # Two different lines, so a batch that mixed them up would not match.
        line = torch.cat([ramp, page]).contiguous()
        best, index = rec(line)
        refs = {"input": line, "best": best, "index": index}
    write(out_root / "ppocrv6_rec", "ppocrv6_rec", rec, g, refs)
    (out_root / "ppocrv6_rec" / "characters.json").write_text(json.dumps(list(chars), ensure_ascii=False))
    print(f"  {len(chars)} classes, index 0 is the CTC blank")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--page", default="1280x1792", help="detector input, WxH, multiples of 32")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--precision", default="fp32")
    p.add_argument("--out", type=Path, default=GEN / "graphs")
    p.add_argument("--graphs", default="det,rec,ori", help="which to export: det, rec, ori")
    p.add_argument("--reference-image", default=None, help="a page for the orienter's reference")
    p.add_argument("--verify", action="store_true",
                   help="write small graphs with every node's PyTorch result instead")
    a = p.parse_args()
    w, h = (int(v) for v in a.page.lower().split("x"))
    if w % 32 or h % 32:
        raise SystemExit("--page sides must be multiples of 32")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA; pass --device cpu (see the module docstring)")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    main((w, h), a.device, a.precision, a.out, a.verify, tuple(a.graphs.split(",")), a.reference_image)
