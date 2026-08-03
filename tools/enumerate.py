"""Trace the MatAnyone2 inference path and enumerate its variants.

Stage 1 of the Python side (lava-dnn.md). Runs the real pipeline on a real clip
and records, per step:

  - the guard values that pick a branch in InferenceCore.step
  - the sequence of network-level calls, with every tensor shape in and out
  - the size of the cross-frame state (working memory, sensory, object memory)

Steps whose call sequence is identical collapse into one variant. What is left
is the variant set the Julia side has to record command buffers for, plus the
list of places where a shape varies with frame index rather than resolution -
those are what get padded to a fixed capacity.

The guards are re-derived here from the same expressions as
inference_core.py:288-301, then cross-checked against the calls that actually
happened, so drift from upstream shows up as a failed check rather than a
silently wrong graph.

    uv run tools/enumerate.py --frames 40
"""

import argparse
import json
from collections import OrderedDict

import torch

import common


def sig(x):
    """Structural signature of an argument: shapes and dtypes, no values."""
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype).removeprefix("torch.")}
    if isinstance(x, (list, tuple)):
        return [sig(v) for v in x]
    if isinstance(x, dict):
        return {str(k): sig(v) for k, v in x.items()}
    if isinstance(x, (int, float, bool, str)) or x is None:
        return x
    return type(x).__name__


# The network entry points InferenceCore drives. Everything below these is pure
# tensor code and gets exported to ATen separately; this level is only about
# which of them run, in what order, on what shapes.
TRACED = [
    "encode_image",
    "transform_key",
    "encode_mask",
    "pixel_fusion",
    "readout_query",
    "pred_uncertainty",
    "segment",
]


class Trace:
    def __init__(self):
        self.calls = []          # flat list of every traced call
        self.steps = []          # one record per InferenceCore.step
        self.current = None

    def wrap(self, network):
        for name in TRACED:
            bound = getattr(network, name)

            def make(name, bound):
                def wrapper(*args, **kwargs):
                    out = bound(*args, **kwargs)
                    rec = {
                        "op": name,
                        "in": [sig(a) for a in args],
                        "kw": {k: sig(v) for k, v in kwargs.items()},
                        "out": sig(out),
                    }
                    self.calls.append(rec)
                    if self.current is not None:
                        self.current["calls"].append(rec)
                    return out

                return wrapper

            setattr(network, name, make(name, bound))
        return network

    def record(self, name, args, kwargs, out):
        rec = {"op": name, "in": [sig(a) for a in args],
               "kw": {k: sig(v) for k, v in kwargs.items()}, "out": sig(out)}
        self.calls.append(rec)
        if self.current is not None:
            self.current["calls"].append(rec)

    def wrap_memory_ops(self):
        """Trace the affinity path, which lives in MemoryManager rather than in
        the network, and is the only place a shape depends on frame index.

        memory_manager.py does `from ...memory_utils import get_similarity`, so
        the name has to be patched in *its* namespace, not the origin module.
        """
        from matanyone2.inference import memory_manager as mm
        from matanyone2.inference.memory_manager import MemoryManager

        for name in ("get_similarity", "do_softmax"):
            fn = getattr(mm, name)

            def make(name, fn):
                def wrapper(*args, **kwargs):
                    out = fn(*args, **kwargs)
                    self.record(name, args, kwargs, out)
                    return out
                return wrapper

            setattr(mm, name, make(name, fn))

        readout = MemoryManager._readout

        def wrapped_readout(mgr, affinity, v, uncert_mask=None):
            out = readout(mgr, affinity, v, uncert_mask)
            self.record("memory_readout", (affinity, v), {}, out)
            return out

        MemoryManager._readout = wrapped_readout


def memstate(processor):
    """Sizes of everything that survives a frame."""
    mem = processor.memory
    wm = mem.work_mem
    out = {
        "engaged": mem.engaged,
        "buckets": {str(b): list(o) for b, o in wm.buckets.items()},
        "work_key": {str(b): list(k.shape) for b, k in wm.key.items()},
        "work_value": {str(o): list(v.shape) for o, v in wm.value.items()},
        "work_shrinkage": {str(b): list(s.shape) for b, s in wm.shrinkage.items()},
        "perm_end_pt": {str(b): int(p) for b, p in wm.perm_end_pt.items()},
        "sensory": {str(o): list(s.shape) for o, s in mem.sensory.items()},
        "obj_v": {str(o): list(v.shape) for o, v in mem.obj_v.items()},
        "max_work_tokens": getattr(mem, "max_work_tokens", None),
    }
    return out


def guards(processor, mask, objects, end, first_frame_pred):
    """The branch decisions of InferenceCore.step, before it mutates state.

    Mirrors inference_core.py:280-301.
    """
    curr_ti = processor.curr_ti + 1
    last_mem_ti = processor.last_mem_ti
    om = processor.object_manager

    is_mem_frame = ((curr_ti - last_mem_ti >= processor.mem_every) or (mask is not None)) and (not end)
    need_segment = (mask is None) or (om.num_obj > 0 and not om.has_all(objects))
    update_sensory = ((curr_ti - last_mem_ti) in processor.stagger_ti) and (not end)

    if first_frame_pred:
        curr_ti, last_mem_ti = 0, 0
        is_mem_frame = need_segment = update_sensory = True

    return {
        "curr_ti": curr_ti,
        "last_mem_ti": last_mem_ti,
        "first_frame_pred": bool(first_frame_pred),
        "has_mask": mask is not None,
        "is_mem_frame": bool(is_mem_frame),
        "need_segment": bool(need_segment),
        "update_sensory": bool(update_sensory),
        "first_frame_read": bool(curr_ti == 0),
    }


def check(step):
    """Cross-check derived guards against the calls that actually ran."""
    g, ops = step["guards"], [c["op"] for c in step["calls"]]
    problems = []
    if g["need_segment"] != ("segment" in ops):
        problems.append("need_segment disagrees with segment() being called")
    # read_first_frame skips affinity, so pred_uncertainty only runs on the
    # regular read path (memory_manager.py:249).
    expect_uncert = g["need_segment"] and not g["first_frame_read"]
    if expect_uncert != ("pred_uncertainty" in ops):
        problems.append("first_frame_read disagrees with pred_uncertainty() being called")
    if "encode_mask" not in ops:
        problems.append("encode_mask never ran; step always computes last_msk_value")
    return problems


def variant_key(step):
    """Two steps are the same variant if the same graph runs.

    Keyed on the call sequence *and* the keyword arguments, because some of
    them select a branch inside the callee rather than around it -
    encode_mask(deep_update=False) skips the sensory GRU entirely
    (big_modules.py MaskEncoder), so it is a different graph under the same
    op name. Constant kwargs (chunk_size, need_weights) fold out on their own.
    """
    return tuple((c["op"], json.dumps(c["kw"], sort_keys=True)) for c in step["calls"])


def run(frames, warmup, out_path, max_size, clip="test-sample1"):
    common.bootstrap()
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

    model, device = common.load_model()
    trace = Trace()
    trace.wrap(model)
    trace.wrap_memory_ops()
    processor = InferenceCore(model, cfg=model.cfg)

    vid = common.UPSTREAM / "inputs" / "video"
    src = vid / clip if (vid / clip).exists() else vid / f"{clip}.mp4"
    vframes, fps, length, name = common.read_frames(src)
    vframes = torch.cat([vframes[0].unsqueeze(0).repeat(warmup, 1, 1, 1), vframes], 0).float()
    length = min(length + warmup, frames + warmup)

    if max_size > 0:
        h, w = vframes.shape[-2:]
        if min(h, w) > max_size:
            nh, nw = int(h / min(h, w) * max_size), int(w / min(h, w) * max_size)
            vframes = F.interpolate(vframes, size=(nh, nw), mode="area")

    mask = np.array(Image.open(common.UPSTREAM / "inputs" / "mask" / f"{clip}.png").convert("L"))
    mask = gen_erosion(gen_dilate(mask, 10, 10), 10, 10)
    mask = torch.from_numpy(mask).float().to(device)
    if max_size > 0 and mask.shape[-2:] != vframes.shape[-2:]:
        mask = F.interpolate(mask[None, None], size=vframes.shape[-2:], mode="nearest")[0, 0]

    def step(image, mask=None, objects=None, first_frame_pred=False):
        rec = {
            "guards": guards(processor, mask, objects, False, first_frame_pred),
            "calls": [],
        }
        trace.current = rec
        out = processor.step(image, mask, objects=objects, first_frame_pred=first_frame_pred)
        trace.current = None
        rec["memory_after"] = memstate(processor)
        rec["problems"] = check(rec)
        trace.steps.append(rec)
        return out

    with torch.inference_mode():
        for ti in range(length):
            image = (vframes[ti] / 255.0).float().to(device)
            if ti == 0:
                step(image, mask, objects=[1])
                step(image, first_frame_pred=True)
            elif ti <= warmup:
                step(image, first_frame_pred=True)
            else:
                step(image)

    # collapse steps into variants
    variants = OrderedDict()
    for i, s in enumerate(trace.steps):
        k = variant_key(s)
        ops = [op if kw in ("{}", '{"chunk_size": -1, "need_weights": false}')
               else f"{op}({kw.strip('{}')})" for op, kw in k]
        v = variants.setdefault(k, {"ops": ops, "steps": [], "guards": [], "shapes": set()})
        v["steps"].append(i)
        v["guards"].append({g: s["guards"][g] for g in
                            ("first_frame_pred", "has_mask", "is_mem_frame",
                             "need_segment", "first_frame_read")})
        v["shapes"].add(json.dumps([c["in"] for c in s["calls"]], sort_keys=True))

    mcfg = model.cfg.model
    report = {
        "source": {"clip": str(src), "frames": length, "warmup": warmup,
                   "resolution": list(vframes.shape[-2:]), "fps": fps},
        # Everything Julia folds as a type parameter. num_objects and the two
        # object-dependent C_in values live here rather than being baked into
        # the emitted kernels, so a multi-object checkpoint would be a new
        # instantiation instead of an emitter change. The released weights are
        # single-object: mask_encoder.conv1 is (64,4,7,7), and load_weights
        # (matanyone2.py:297) random-inits the 5th channel, so multi-object is
        # not verifiable against this checkpoint.
        "typeparams": {
            "num_objects": 1,
            "single_object": model.single_object,
            "batch": 2 if processor.cfg.flip_aug else 1,
            "key_dim": mcfg.key_dim,
            "value_dim": mcfg.value_dim,
            "sensory_dim": mcfg.sensory_dim,
            "pixel_dim": mcfg.pixel_dim,
            "embed_dim": mcfg.embed_dim,
            "ms_dims": list(mcfg.pixel_encoder.ms_dims),
            "up_dims": list(mcfg.mask_decoder.up_dims),
            "num_heads": mcfg.object_transformer.num_heads,
            "num_blocks": mcfg.object_transformer.num_blocks,
            "num_queries": mcfg.object_transformer.num_queries,
            "ff_dim": mcfg.object_transformer.ff_dim,
            "mask_encoder_c_in": int(model.mask_encoder.conv1.weight.shape[1]),
            "sensory_compress_c_in": int(model.pixel_fuser.sensory_compress.weight.shape[1]),
            "top_k": processor.cfg.top_k,
            "mem_every": processor.mem_every,
            "mem_capacity_frames": processor.memory.max_mem_frames + 1,
        },
        "config": {k: str(processor.cfg[k]) for k in
                   ("mem_every", "max_mem_frames", "use_long_term", "top_k",
                    "chunk_size", "flip_aug", "stagger_updates")},
        "variants": [
            {"id": f"v{i}", "ops": v["ops"], "n_steps": len(v["steps"]),
             "steps": v["steps"], "guards": dedupe(v["guards"]),
             "distinct_input_shapes": len(v["shapes"])}
            for i, (k, v) in enumerate(variants.items())
        ],
        "steps": trace.steps,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    return report


def dedupe(dicts):
    seen, out = set(), []
    for d in dicts:
        k = json.dumps(d, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def summarize(report):
    print(f"clip {report['source']['frames']} steps at {report['source']['resolution']}")
    print(f"config: {report['config']}\n")
    for v in report["variants"]:
        print(f"{v['id']}  x{v['n_steps']:<4} steps={v['steps'][:6]}{'...' if len(v['steps']) > 6 else ''}")
        print(f"    ops: {' -> '.join(v['ops'])}")
        for g in v["guards"]:
            print(f"    guards: {', '.join(k for k, on in g.items() if on) or '(none)'}")
        if v["distinct_input_shapes"] > 1:
            print(f"    !! {v['distinct_input_shapes']} distinct input shapes -> needs padding")
        print()

    problems = [(i, p) for i, s in enumerate(report["steps"]) for p in s["problems"]]
    print(f"guard cross-check: {'OK' if not problems else problems}")

    # where does the cross-frame state change size?
    sizes = [s["memory_after"]["work_key"].get("0", [None])[-1] for s in report["steps"]]
    print(f"work_mem tokens by step: {sizes}")
    print(f"max_work_tokens: {report['steps'][-1]['memory_after']['max_work_tokens']}"
          f"  saturates at {max(n for n in sizes if n)}")

    # every op whose input shapes are not a pure function of resolution
    varying = {}
    for s in report["steps"]:
        for c in s["calls"]:
            varying.setdefault(c["op"], set()).add(json.dumps(c["in"], sort_keys=True))
    print("\nops with >1 distinct input shape (candidates for pad-to-capacity):")
    for op, shapes in varying.items():
        if len(shapes) > 1:
            dims = sorted({tuple(d["shape"]) for sh in shapes for d in json.loads(sh)
                           if isinstance(d, dict) and "shape" in d})
            print(f"  {op}: {len(shapes)} shapes  {dims}")

    print(f"\ntypeparams: {json.dumps(report['typeparams'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-size", type=int, default=-1)
    ap.add_argument("--clip", default="test-sample1")
    ap.add_argument("--out", default=str(common.GEN / "graphs" / "trace.json"))
    a = ap.parse_args()
    from pathlib import Path
    summarize(run(a.frames, a.warmup, Path(a.out), a.max_size, a.clip))
