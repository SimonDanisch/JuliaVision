"""Per-step driver state, so the Julia driver can be diffed step by step.

The graphs are verified individually and the memory read is verified against
dump_memread, so a remaining end-to-end difference has to be in the
orchestration: which variant runs, what gets threaded into the next step, and
in what order. This records the four values that survive a step, exactly as
InferenceCore holds them, so the first step that diverges names the bug.

    uv run tools/dump_driver.py --steps 15
"""

import argparse
from pathlib import Path

import torch

import common
import export_graphs as EG
import patches


def run(out_path, max_size, steps, warmup, precision):
    common.bootstrap()
    patches.apply()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

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

    out = {}
    n = {"i": -1}

    def snapshot():
        i = n["i"]
        if i >= steps:
            return
        out[f"s{i}/last_mask"] = processor.last_mask.detach().float().cpu()
        out[f"s{i}/last_pix_feat"] = processor.last_pix_feat.detach().float().cpu()
        out[f"s{i}/last_msk_value"] = processor.last_msk_value.detach().float().cpu()
        sens = processor.memory.get_sensory(processor.object_manager.all_obj_ids)
        out[f"s{i}/sensory"] = sens.detach().float().cpu()
        out[f"s{i}/nvalid"] = torch.tensor(
            [processor.memory.work_mem.size(next(iter(processor.memory.work_mem.buckets)))
             if processor.memory.work_mem.buckets else 0], dtype=torch.float32)

    def tick(*a, **kw):
        n["i"] += 1
        r = processor.step(*a, **kw)
        snapshot()
        return r

    with torch.inference_mode(), EG.precision_ctx(precision):
        for ti in range(length):
            if n["i"] >= steps:
                break
            image = (vframes[ti] / 255.0).float().to(device)
            if ti == 0:
                tick(image, mask, objects=[1])
                for _ in range(warmup + 1):
                    tick(image, first_frame_pred=True)
            else:
                tick(image)

    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(out_path),
              metadata={"steps": str(steps), "warmup": str(warmup)})
    print(f"{len({k.split('/')[0] for k in out})} steps -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(Path(a.out) if a.out else common.GEN / "driver.safetensors",
        a.max_size, a.steps, a.warmup, a.precision)
