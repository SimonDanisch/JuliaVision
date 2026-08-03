"""End-to-end reference: padded inputs and the alpha matte PyTorch produces.

Feeds the Julia driver exactly what InferenceCore sees - already padded to a
multiple of 16, already scaled to [0,1] - so a difference in the matte is a
difference in the model and not in the preprocessing.

    uv run tools/dump_e2e.py --max-size 128 --frames 6

Writes gen/e2e-<precision>.safetensors with `frames`, `mask`, `alpha`.
"""

import argparse
import json
from pathlib import Path

import torch

import common
import export_graphs as EG
import patches


def run(out_path, max_size, frames, warmup, precision):
    common.bootstrap()
    patches.apply()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion
    from matanyone2.utils.tensor_utils import pad_divide_by

    model, device = common.load_model()
    processor = InferenceCore(model, cfg=model.cfg)

    src = common.UPSTREAM / "inputs" / "video" / "test-sample1"
    vframes, _, length, _ = common.read_frames(src)
    h, w = vframes.shape[-2:]
    if min(h, w) > max_size:
        nh, nw = int(h / min(h, w) * max_size), int(w / min(h, w) * max_size)
        vframes = F.interpolate(vframes, size=(nh, nw), mode="area")

    mask = np.array(Image.open(common.UPSTREAM / "inputs" / "mask" / "test-sample1.png").convert("L"))
    mask = gen_erosion(gen_dilate(mask, 10, 10), 10, 10)
    mask = torch.from_numpy(mask).float().to(device)
    if mask.shape[-2:] != vframes.shape[-2:]:
        mask = F.interpolate(mask[None, None], size=vframes.shape[-2:], mode="nearest")[0, 0]

    n = min(length, frames)
    kept, alphas = [], []
    with torch.inference_mode(), EG.precision_ctx(precision):
        for ti in range(n):
            image = (vframes[ti] / 255.0).float().to(device)
            padded, _ = pad_divide_by(image, 16)
            kept.append(padded.detach().float().cpu())
            if ti == 0:
                # inference_matanyone2.py:97-102 repeats the first frame n_warmup
                # times *in the clip*, so frame 0 gets the mask step, then its own
                # first_frame_pred, then one more per repeated frame: warmup + 2
                # steps in total. Getting this off by one changes the converged
                # matte in ambiguous regions.
                processor.step(image, mask, objects=[1])
                for _ in range(warmup + 1):
                    out = processor.step(image, first_frame_pred=True)
            else:
                out = processor.step(image)
            # alpha at the padded resolution, matching what the driver returns
            alpha, _ = pad_divide_by(processor.output_prob_to_mask(out), 16)
            alphas.append(alpha.detach().float().cpu())

    padded_mask, _ = pad_divide_by(mask, 16)
    tensors = {
        "frames": torch.stack(kept, -1).contiguous(),     # (3, H, W, T)
        "mask": padded_mask.detach().float().cpu().contiguous(),
        "alpha": torch.stack(alphas, -1).contiguous(),    # (H, W, T)
    }

    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path),
              metadata={"precision": precision, "warmup": str(warmup), "frames": str(n)})
    print(f"{n} frames at {tuple(tensors['alpha'].shape[:2])} "
          f"(alpha range {tensors['alpha'].min():.3f}..{tensors['alpha'].max():.3f}) -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=128)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else common.GEN / f"e2e-{a.precision}.safetensors"
    run(out, a.max_size, a.frames, a.warmup, a.precision)
