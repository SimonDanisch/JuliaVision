"""What SAM 2.1 costs in PyTorch on this card, split the way the editor uses it.

    uv run tools/sam2_pytorch_baseline.py --size large

Two numbers, because the editor pays them at different rates: `encode` runs once
when marking starts, `decode` runs on every click. A single "SAM takes 350 ms"
would hide that the interactive half is 40x cheaper than the half it depends on.

Measured on the exported wrappers rather than through `SAM2ImagePredictor`, so
this is the same computation DNNKernels runs — the predictor also resizes,
normalises and post-processes, and timing those here would flatter or punish the
comparison depending on which side does them.

**`--compile` measures the other denominator, and it is not optional to know it.**
Everything in the README compares against eager, on the stated reasoning that
the graph we run came out of `torch.export` so a fused Inductor build would be
"a compiler we do not have against a runtime we do". That argument does not hold
up: DNNKernels fuses hard — `fuseattention` alone moved DepthAnything from
37.4 ms to 14.8 — so eager-vs-us is our fused build against their unfused one.
Both numbers are worth having and they answer different questions; what is not
defensible is reporting one of them as "PyTorch". The two write to separate
files so an eager record is never silently overwritten by a compiled one.
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import export_graphs as EG
import export_sam2 as ES

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()


def smclock():
    """Shader clock in MHz, whichever vendor's tool is here.

    Recorded with the result because it is part of it: the Ada card idles at
    495 MHz of a 3105 MHz maximum and the 8060S at 600 MHz, so a benchmark that
    does not say what clock it ran at is not comparable with one that ran at a
    different clock.

    Both vendors, because the comparison this file feeds now runs on both: the
    NVIDIA box is where the published numbers came from, and the APU is where
    Lava's own numbers come from. Returning `None` on the APU — which is what
    asking `nvidia-smi` alone did — drops the one field that makes two runs
    comparable, silently.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    try:
        out = subprocess.run(["rocm-smi", "-c"], capture_output=True, text=True,
                             check=True)
        m = re.search(r"sclk clock level: \d+: \((\d+)Mhz\)", out.stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def refspath(size):
    """The reference activations: the local dump, else the published artifact.

    Resolved through `SAM2Runner/Artifacts.toml` rather than a hardcoded hash —
    the tree hash IS the artifact's identity and that file is where it is
    declared, so this cannot drift from what the Julia side loads.
    """
    local = ROOT / "gen" / "graphs" / f"sam2-{size}" / "refs.safetensors"
    if local.is_file():
        return local
    import tomllib
    decl = ROOT / "SAM2Runner" / "Artifacts.toml"
    if not decl.is_file():
        return None
    entry = tomllib.loads(decl.read_text()).get(f"sam2-{size}-refs")
    if entry is None:
        return None
    p = Path.home() / ".julia" / "artifacts" / entry["git-tree-sha1"] / "refs.safetensors"
    return p if p.is_file() else None


def parity(enc, size, dev, precision):
    """The encoder's six outputs against the refs, on the refs' own input.

    **Before any timing is reported, not after.** A number from a model built on
    the wrong checkpoint or run under the wrong precision policy is worse than
    no number, because it looks like a result. The refs came out of this same
    PyTorch wrapper, so agreement should be near-exact; a mismatch means the
    setup here differs from the one that produced them, and that is exactly what
    this is for.
    """
    import json

    from safetensors.torch import load_file

    path = refspath(size)
    if path is None:
        return {"checked": False, "why": "no refs.safetensors locally or in the artifact"}
    manifest = json.loads((path.parent / "refs_manifest.json").read_text())
    names = manifest["graphs"]["sam2_encoder"]["outputs"]
    refs = load_file(str(path))
    image = refs["sam2_encoder/in0"].to(dev)
    with torch.no_grad(), EG.precision_ctx(precision):
        got = enc(image)
    per = {}
    for i, name in enumerate(names):
        want = refs[f"sam2_encoder/node/{name}"].to(dev).float()
        per[name] = (got[i].float() - want).abs().max().item()
    return {"checked": True, "per_output": per,
            "failed": [(n, d) for n, d in per.items() if outofband(n, d)]}


def outofband(name, d):
    """Whether this output's deviation is one to refuse a timing over."""
    if name in KNOWN_DRIFT:
        lo, hi = KNOWN_DRIFT[name]
        return not lo < d < hi
    return d > 2e-2


# `add_129` does NOT match the references, on purpose, and the same band is
# pinned on the Julia side (`SAM2Runner/test/runtests.jl`, "graphs vs PyTorch,
# layer by layer"). The dump covers the encoder at its six outputs only, so that
# node carries the accumulated fp16 difference of the 543 unchecked ops before
# it. Measured here at 0.271 through PyTorch on ROCm against references dumped
# on CUDA — inside the band the Julia gate pins for Lava, which is what says the
# drift belongs to the fp16 accumulation and not to either runtime.
#
# A flat tolerance is the wrong gate for that: 2e-2 rejects a correct setup, and
# anything loose enough to admit 0.271 stops noticing a real break. So the
# signature is pinned instead — this node in this band, every other output
# tight — and a change either way fails. Localising it is
# `dump_sam2_refs.py --nodes all`, which the ROCm venv now makes possible here.
KNOWN_DRIFT = {"add_129": (0.2, 0.6)}


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"min": ts[0], "p50": ts[len(ts) // 2], "mean": sum(ts) / len(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(ES.CKPTS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 (default off, to match the fp32 reference)")
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"],
                    help="must match what the graphs were exported under, or the "
                         "comparison is between two different computations")
    ap.add_argument("--compile", action="store_true",
                    help="measure torch.compile instead of eager — the other "
                         "denominator; writes a separate file")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--sdpa", default="fused", choices=["fused", "math"],
                    help="`math` forces the decomposition XLA is stuck with, so "
                         "the two can be compared on the same algorithm")
    ap.add_argument("--no-parity", action="store_true",
                    help="report timings without checking the refs first. Off by "
                         "default and it should stay off: see `parity`")
    a = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.set_float32_matmul_precision("high" if a.tf32 else "highest")

    if not torch.cuda.is_available():
        raise SystemExit(
            "no GPU visible to torch — there is nothing to compare against here.\n"
            "  On the 8060S this means a ROCm build of torch; a +cuXXX wheel sees "
            "no device on an AMD box and would quietly measure the CPU.")
    dev = "cuda"
    model = ES.build(a.size).to(dev)
    res = model.image_size
    enc = ES.Encoder(model).to(dev).eval()
    dec = ES.Decoder(model).to(dev).eval()

    # Parity BEFORE compiling and before timing: it validates the checkpoint and
    # the precision policy, which is what makes the number mean anything, and it
    # is cheapest on the eager module.
    par = {"checked": False, "why": "--no-parity"} if a.no_parity else \
        parity(enc, a.size, dev, a.precision)
    if par["checked"] and par["failed"]:
        detail = ", ".join(f"{n} {d:.3e}" for n, d in par["failed"])
        raise SystemExit(
            f"refs parity failed: {detail}\n"
            f"  expected: every output under 2e-2 except {sorted(KNOWN_DRIFT)}, "
            f"which is pinned to {KNOWN_DRIFT}.\n"
            "  The model here is not the model the references came from — check "
            "--size and --precision before trusting any timing.")

    # Held open across parity, warm-up and timing, so every call below runs the
    # same attention. Scoped narrower it would decompose some calls and not
    # others, which is the one thing that must not vary here.
    sdpactx = sdpa_kernel(SDPBackend.MATH) if a.sdpa == "math" else contextlib.nullcontext()
    sdpactx.__enter__()

    if a.compile:
        # Compiled before the clock ramp below, so the 200 warm-up calls pay
        # Inductor's compilation as well as bringing the clock up and neither
        # lands in the measured window. `dynamic=False` because the export fixes
        # every shape: letting Dynamo specialise avoids a recompile that would
        # otherwise show up as one slow iteration in the middle of the run.
        enc = torch.compile(enc, mode=a.compile_mode, dynamic=False)
        dec = torch.compile(dec, mode=a.compile_mode, dynamic=False)

    image = torch.rand(1, 3, res, res, device=dev)
    with torch.no_grad():
        with EG.precision_ctx(a.precision):
            feats = enc(image)
        nfpn = len(feats) // 2
        f0, f1, f2 = feats[0], feats[1], feats[nfpn - 1]
        point = torch.full((1, ES.MAXPOINTS, 2), res / 2.0, device=dev)
        label = torch.full((1, ES.MAXPOINTS), -1, dtype=torch.int32, device=dev)
        label[0, 0] = 1

        # Same precision policy the graphs carry, and a warm-up long enough to
        # bring the SM clock up: this card idles at 495 MHz of 3105 and a short
        # benchmark measures the ramp rather than the kernel.
        with torch.no_grad(), EG.precision_ctx(a.precision):
            for _ in range(200):
                enc(image)
            torch.cuda.synchronize()
            e = bench(lambda: enc(image), a.warmup, a.iters)
            d = bench(lambda: dec(f0, f1, f2, point, label), a.warmup, a.iters)

    out = {"size": a.size, "res": res, "tf32": a.tf32, "precision": a.precision,
           "compiled": a.compile, "sdpa": a.sdpa,
           "compile_mode": a.compile_mode if a.compile else None,
           "sm_clock_mhz": smclock(),
           "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__,
           "parity": par,
           "encode_ms": e, "decode_ms": d}
    print(json.dumps(out, indent=1))
    # A separate file per mode. One name for both would let a compiled run
    # overwrite the eager record that every published comparison rests on.
    name = ("pytorch_baseline" + ("_compiled" if a.compile else "")
            + ("_math" if a.sdpa == "math" else "") + ".json")
    dest = ROOT / "gen" / "graphs" / f"sam2-{a.size}" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nencode {e['p50']:.1f} ms   decode {d['p50']:.2f} ms   "
          f"({'compiled/' + a.compile_mode if a.compile else 'eager'}, p50, TF32 "
          f"{'on' if a.tf32 else 'off'})")


if __name__ == "__main__":
    main()
