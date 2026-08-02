"""Reference tensors for the memory bank and its read.

Everything else in the pipeline is an exported graph checked against
refs.safetensors. The bank and the read are hand-authored (DNNKernels memory.jl),
so they are the only part with no oracle - which makes them the first place to
look when the end-to-end matte disagrees but every graph passes.

Captures, at one ordinary (non-memory) frame:
  the bank contents PyTorch holds       -> checks add!/eviction ordering
  the query key and selection           -> the read's inputs
  similarity / affinity / visual readout -> the read's three stages

    uv run tools/dump_memread.py --at 14
"""

import argparse
from pathlib import Path

import torch

import common
import export_graphs as EG
import patches


def run(out_path, max_size, at, warmup, precision):
    common.bootstrap()
    patches.apply()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    from matanyone2.inference import memory_manager as mm
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.inference.memory_manager import MemoryManager
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

    model, device = common.load_model()
    processor = InferenceCore(model, cfg=model.cfg)

    grabbed = {}
    step = {"n": -1}

    # the read's three stages live in MemoryManager's namespace, not memory_utils'
    orig_sim, orig_soft = mm.get_similarity, mm.do_softmax
    orig_readout = MemoryManager._readout

    def wrap_sim(mk, ms, qk, qe, *a, **kw):
        out = orig_sim(mk, ms, qk, qe, *a, **kw)
        if step["n"] == at:
            # do_softmax runs in place and would overwrite this
            grabbed.update(mem_key=mk.clone(), mem_shrinkage=ms.clone(),
                           query_key=qk.clone(), selection=qe.clone(),
                           similarity=out.clone())
        return out

    def wrap_soft(sim, *a, **kw):
        out = orig_soft(sim, *a, **kw)
        if step["n"] == at:
            grabbed["affinity"] = (out[0] if isinstance(out, tuple) else out).clone()
        return out

    def wrap_readout(self, affinity, v, uncert_mask=None):
        out = orig_readout(self, affinity, v, uncert_mask)
        if step["n"] == at:
            grabbed.update(mem_value=v, visual_readout=out)
        return out

    mm.get_similarity, mm.do_softmax = wrap_sim, wrap_soft
    MemoryManager._readout = wrap_readout

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

    def tick(*a, **kw):
        step["n"] += 1
        return processor.step(*a, **kw)

    with torch.inference_mode(), EG.precision_ctx(precision):
        for ti in range(min(length, at + 4)):
            image = (vframes[ti] / 255.0).float().to(device)
            if ti == 0:
                tick(image, mask, objects=[1])
                for _ in range(warmup + 1):
                    tick(image, first_frame_pred=True)
            else:
                tick(image)
            if step["n"] >= at:
                break

    if "similarity" not in grabbed:
        raise SystemExit(f"step {at} never ran a memory read; the clip only reached "
                         f"step {step['n']}")

    # the bank as PyTorch holds it right after the captured read
    wmem = processor.memory.work_mem
    bucket = next(iter(wmem.buckets))
    grabbed["bank_key"] = wmem.key[bucket]
    grabbed["bank_shrinkage"] = wmem.shrinkage[bucket]
    grabbed["bank_value"] = torch.stack([wmem.value[o] for o in wmem.buckets[bucket]], dim=1)
    grabbed["objmem"] = torch.stack(
        [processor.memory.obj_v[o] for o in wmem.buckets[bucket]], dim=1)

    tensors = {k: v.detach().float().contiguous().cpu() for k, v in grabbed.items()}
    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path),
              metadata={"step": str(at), "perm_end": str(wmem.perm_end_pt[bucket])})
    for k, v in tensors.items():
        print(f"  {k:18} {tuple(v.shape)}")
    print(f"-> {out_path} (perm_end_pt={wmem.perm_end_pt[bucket]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=128)
    ap.add_argument("--at", type=int, default=14, help="step index to capture")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else common.GEN / "memread.safetensors"
    run(out, a.max_size, a.at, a.warmup, a.precision)
