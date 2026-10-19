"""Export PP-DocLayoutV3 (a page's regions, their kind and reading order) to
DNNKernels' graph JSON + weights.

    uv run tools/export_ppdoclayout.py --reference-image page.png

One graph, `ppdoclayoutv3`: a page resized to 800x800, RGB in [0, 1],
`(1, 3, 800, 800)` -> for each of 300 queries the class logits (25 kinds), the
box `(cx, cy, w, h)` normalised to the page, and the pairwise reading-order
logits `(300, 300)`. The masks upstream also predicts, for polygons, are left
out: the runner reports boxes. Scores, the threshold and the reading order
(votes over the order logits, then a sort) are the runner's, on 300 values.

From Hugging Face Transformers (`pp_doclayout_v3`), weights
`PaddlePaddle/PP-DocLayoutV3_safetensors`. Apache-2.0, code and weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForObjectDetection

import export_graphs as EG
from common import GEN
from export_ppocrv6 import write

REPO = "PaddlePaddle/PP-DocLayoutV3_safetensors"


class Layout(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(pixel_values=x)
        return out.logits, out.pred_boxes, out.order_logits


def page(path):
    if path:
        return Image.open(path).convert("RGB")
    # Lines of "text" on paper, so the reference has regions to find.
    yy, xx = np.mgrid[:1100, :800]
    ink = ((yy % 40) < 14) & ((yy // 200) % 2 == 0) & (xx > 80) & (xx < 720) & (yy > 100)
    values = np.where(ink, 30, 245).astype(np.uint8)
    return Image.fromarray(np.stack([values] * 3, -1))


def main(args):
    torch.set_num_threads(8)
    net = AutoModelForObjectDetection.from_pretrained(REPO, revision=args.revision).eval()
    processor = AutoImageProcessor.from_pretrained(REPO, revision=args.revision)
    x = processor(images=[page(args.reference_image)], return_tensors="pt")["pixel_values"]
    model = Layout(net).eval()
    with torch.no_grad():
        logits, boxes, order = model(x)
        ep = torch.export.export(model, (x,), strict=False).run_decompositions()
        graph = EG.convert(ep, ({},), "ppdoclayoutv3")
    out = args.out_root / "ppdoclayoutv3"
    write(out, "ppdoclayoutv3", model, graph, {"input": x, "logits": logits, "boxes": boxes, "order": order},
          EG.save_constants(ep))
    labels = [net.config.id2label[k] for k in range(len(net.config.id2label))]
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "provenance.json").write_text(json.dumps({"repo": REPO,
        "revision": Path(hf_hub_download(REPO, "config.json", revision=args.revision)).parent.name,
        "torch": torch.__version__, "input": "RGB float32 [0,1], 800x800, bicubic without antialiasing"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out-root", type=Path, default=GEN / "graphs")
    main(parser.parse_args())
