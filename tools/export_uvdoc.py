"""Export PaddlePaddle/UVDoc_safetensors to DNNKernels (fp32, CPU).

    uv run tools/export_uvdoc.py --reference-image curled-page.jpg

RGB [0,1], (1,3,712,488) -> normalized backward grid (1,2,45,31).
The first convolution is changed from BGR to RGB. Resize and grid sampling
remain outside the network, both bilinear with align_corners=True, matching
Transformers' UVDocImageProcessor. Code and weights: Apache-2.0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from huggingface_hub import hf_hub_download

import export_graphs as EG
from common import GEN
from export_ppocrv6 import bgr_to_rgb_stem, write

REPO = "PaddlePaddle/UVDoc_safetensors"


class UVDoc(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(pixel_values=x).last_hidden_state


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(1)
    net = AutoModel.from_pretrained(REPO, revision=args.revision).eval()
    processor = AutoImageProcessor.from_pretrained(REPO, revision=args.revision)
    if args.reference_image:
        image = Image.open(args.reference_image).convert("RGB")
    else:
        yy, xx = np.mgrid[:417, :293]
        values = np.where((yy % 25 < 4) & (xx > 25) & (xx < 265), 30, 240).astype(np.uint8)
        image = Image.fromarray(np.stack([values, values, values], -1))
    processed = processor(images=image, return_tensors="pt")
    x = processed["pixel_values"].flip(1).contiguous()
    original = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    with torch.no_grad():
        expected = net(pixel_values=processed["pixel_values"]).last_hidden_state
        bgr_to_rgb_stem(net)
        model = UVDoc(net).eval()
        grid = model(x)
        torch.testing.assert_close(grid, expected, atol=1e-5, rtol=1e-5)
        unwarped = F.grid_sample(original, F.interpolate(grid, size=original.shape[-2:],
                                 mode="bilinear", align_corners=True).permute(0, 2, 3, 1), align_corners=True)
        ep = torch.export.export(model, (x,), strict=False).run_decompositions()
        graph = EG.convert(ep, ({},), "uvdoc")
    out = args.out_root / "uvdoc"
    write(out, "uvdoc", model, graph, {"input": x, "grid": grid, "original": original, "unwarped": unwarped})
    (out / "provenance.json").write_text(json.dumps({"repo": REPO,
        "revision": Path(hf_hub_download(REPO, "config.json", revision=args.revision)).parent.name, "torch": torch.__version__,
        "input": "RGB float32 [0,1]; bilinear align_corners=True"}, indent=2))
    Image.fromarray((unwarped[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)).save(out / "reference.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--revision", default="bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4")
    parser.add_argument("--out-root", type=Path, default=GEN / "graphs")
    main(parser.parse_args())
