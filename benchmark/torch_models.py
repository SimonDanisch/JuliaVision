"""The PyTorch half of the suite: the same models, the same parameters, eager
and `torch.compile`d.

    benchmark/torch.sh benchmark/torch_models.py --out benchmark/results/torch.json
    benchmark/torch.sh benchmark/torch_models.py sam2-encode rife

Parameters come from `models.toml` and are ASSERTED here, never re-derived — see
that file's header for why there is only one place to write a shape down.

**Both eager and compiled, because they answer different questions.** The graph
Lava runs came out of `torch.export`, which is eager PyTorch's own
decomposition, so EAGER is the like-for-like comparison: same decomposition,
different runtime. `torch.compile` is Inductor fusing that graph into kernels we
do not generate, so it is not a like-for-like anything — it is the target. A
table with only one of them either flatters us or reads as a fantasy gap, so
both are recorded and labelled.

A model whose compile fails or is not attempted records `null` with a reason
rather than being absent: a missing row reads as "we did not get to it" and a
recorded reason reads as what it is.
"""

import argparse
import contextlib
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import torch

# **After `import torch`, never before.** The ROCm interpreter has torch and
# little else; the model setups below need safetensors, transformers, kokoro and
# hydra, which live in the project's own .venv. That venv also contains a CUDA
# torch, and anything placed on `PYTHONPATH` is searched BEFORE the running
# interpreter's own site-packages — so exporting it would silently import the
# CUDA build into the ROCm process and leave `torch.cuda.is_available()` False
# with no explanation. Appending here, once the real torch is already bound, is
# what makes the layering safe.
_extra = os.environ.get("JULIAVISION_TORCH_EXTRA_SITE", "")
for _p in filter(None, _extra.split(os.pathsep)):
    if _p not in sys.path:
        sys.path.append(_p)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))     # the export_* modules live there

with open(HERE / "models.toml", "rb") as fh:
    SPEC = tomllib.load(fh)


class Unsupported(Exception):
    """This model cannot be run in this mode, with the reason as the message."""


def params(name):
    return SPEC["models"][name]["params"]


# ── telemetry ────────────────────────────────────────────────────────────────

def clock_mhz():
    """Shader clock in MHz from AMD sysfs, or None.

    Recorded with the result because it is part of it: a baseline that does not
    say what clock it ran at is not comparable with one that ran at another, and
    a baseline taken too SLOW flatters us, so it fails in the comfortable
    direction and nothing complains. `tools/measure.jl` reads the same files.
    """
    for hw in sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*")):
        try:
            return int(hw.joinpath("freq1_input").read_text()) / 1e6
        except OSError:
            continue
    return None


def bench(fn, *, warmup, iters, ramp):
    """Median wall time per call in ms, after holding the card at its clock.

    `ramp` is not a warm-up for the caches — `warmup` is that. It exists because
    the boost clock takes seconds of SUSTAINED load to come up on this card, and
    a benchmark that starts measuring before it does reports the ramp. The same
    defect cost the Julia harness an 11x error on short kernels.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ramp:
        fn()
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    p50 = ts[len(ts) // 2]
    return {"times_ms": ts, "min": ts[0], "p50": p50,
            "spread": (ts[-1] - ts[0]) / p50, "clock_mhz": clock_mhz()}


def compiled(mod):
    """`torch.compile` with the defaults, so this measures what a user gets."""
    return torch.compile(mod)


# ── the models ───────────────────────────────────────────────────────────────
#
# Each setup returns `make(compile: bool) -> fn`. Returning a factory rather
# than a function is what lets a model choose WHICH module Inductor wraps: the
# composite ones (kokoro) have host Python between their halves and only the
# halves can be compiled.


def setup_sam2(dev):
    import export_graphs as EG
    import export_sam2 as ES

    p = params("sam2-encode")
    model = ES.build(p["variant"]).to(dev)
    if model.image_size != p["resolution"]:
        raise Unsupported(f"spec says {p['resolution']}, checkpoint is {model.image_size}")
    enc = ES.Encoder(model).to(dev).eval()
    image = torch.zeros(1, 3, p["resolution"], p["resolution"], device=dev)

    # **The autocast context is entered ONCE, not per call.** `autocast.__exit__`
    # runs `torch.clear_autocast_cache()`, so a `with` inside the timed function
    # throws away the fp16 casts of every fp32 weight and re-does them on the
    # next call — a 3% overstatement of PyTorch, i.e. in the direction that
    # flatters us. The stack is deliberately left open for the closure's life.
    stack = contextlib.ExitStack()
    stack.enter_context(torch.no_grad())
    stack.enter_context(EG.precision_ctx("autocast"))

    def make(compile):
        m = compiled(enc) if compile else enc

        def fn(_keep=stack):
            m(image)
        return fn

    return make


def setup_whisper(dev):
    from transformers import WhisperForConditionalGeneration

    p = params("whisper-encode")
    model = WhisperForConditionalGeneration.from_pretrained(
        p["checkpoint"], torch_dtype=torch.float16).to(dev).eval()
    enc = model.model.encoder
    mel = torch.zeros(1, p["mel_bins"], p["mel_frames"], device=dev, dtype=torch.float16)

    def make(compile):
        m = compiled(enc) if compile else enc

        def fn():
            with torch.inference_mode():
                m(mel)
        return fn

    return make


def setup_depthanything(dev):
    from export_depthanything import Depth, load_model

    p = params("depthanything")
    model = Depth(load_model()).to(dev).eval()
    x = torch.zeros(1, 3, p["height"], p["width"], device=dev)

    def make(compile):
        m = compiled(model) if compile else model

        def fn():
            with torch.no_grad():
                m(x)
        return fn

    return make


def setup_neurallut(dev):
    from export_neurallut import LUTPredictor, load_upstream

    p = params("neurallut")
    classifier, luts = load_upstream(p["colorspace"], p["pairing"])
    model = LUTPredictor(classifier, luts).eval().to(dev)
    x = torch.zeros(1, 3, p["height"], p["width"], device=dev)

    def make(compile):
        m = compiled(model) if compile else model

        def fn():
            with torch.no_grad():
                m(x)
        return fn

    return make


def setup_rife(dev):
    from export_rife import Interp, load_flownet, padded

    p = params("rife")
    net, _unused = load_flownet()
    model = Interp(net).to(dev).eval()
    h, w = padded(p["height"]), padded(p["width"])
    if (w, h) != (p["width"], p["height"]):
        raise Unsupported(f"spec says {p['width']}x{p['height']}, padding gives {w}x{h}")
    imgs = torch.zeros(1, 6, h, w, device=dev)
    timestep = torch.full((1, 1, 1, 1), p["timestep"], device=dev)

    def make(compile):
        m = compiled(model) if compile else model

        def fn():
            with torch.no_grad():
                m(imgs, timestep)
        return fn

    return make


def setup_kokoro(dev):
    """The two model halves and the alignment between them — not phonemisation.

    `export_kokoro`'s own split, so this times the same computation the graphs
    carry. fp32 because that is what the published artifact contains, however
    much an fp16 export exists.
    """
    import export_kokoro as EK
    from kokoro import KModel

    p = params("kokoro-speak")
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(EK.WEIGHTS / "config.json"),
                   model=str(EK.WEIGHTS / "kokoro-v1_0.pth")).eval().to(dev)
    phonemes = p["phonemes"]
    ids = torch.tensor([[0] + [model.vocab[c] for c in phonemes if c in model.vocab] + [0]],
                       device=dev)
    ntokens = ids.shape[1]
    if ntokens != p["tokens"]:
        raise Unsupported(f"spec says {p['tokens']} tokens, this vocab gives {ntokens}")
    voice = torch.load(EK.WEIGHTS / f"{p['voice']}.pt", map_location="cpu",
                       weights_only=True)
    ref_s = voice[ntokens - 1].to(dev)
    text = EK.TextHalf(model).eval()
    voc = EK.VocoderHalf(model).eval()

    def make(compile):
        # The alignment between the halves is a host gather and cannot be
        # compiled; wrapping the halves is what Inductor can actually do here.
        t = compiled(text) if compile else text
        v = compiled(voc) if compile else voc

        def fn():
            with torch.no_grad():
                d, t_en, duration = t(ids, ref_s)
                idx, _ = EK.align(duration, ntokens)
                en = d.transpose(-1, -2)[:, :, idx]
                asr = t_en[:, :, idx]
                v(en, asr, ref_s)
        return fn

    return make


def setup_matanyone(dev):
    # `bootstrap` first: `matanyone2` lives in the dev checkout and is not on
    # `sys.path` until it runs, so importing InferenceCore above it fails.
    from common import bootstrap, load_model

    bootstrap()
    # SAM 2 builds through Hydra and leaves the global instance initialised, so
    # every later Hydra user in the same process dies with "GlobalHydra is
    # already initialized". Ordering the table around it would be a trap for
    # whoever adds the next model; clearing it here is local to the cause.
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    from matanyone2.inference.inference_core import InferenceCore

    p = params("matanyone-step")
    net, device = load_model(dev)
    core = InferenceCore(net, device=device)
    W, H = p["width"], p["height"]
    img = torch.full((3, H, W), 0.5, device=device)
    msk = torch.zeros(H, W, device=device)
    msk[(H // 4):(3 * H // 4), (W // 4):(3 * W // 4)] = 1.0

    with torch.no_grad():
        core.step(img, msk, objects=[1])

    def make(compile):
        if compile:
            # `step` mutates a memory bank across calls, so what Inductor would
            # trace is one state of an object that is not the same object next
            # call. Reported rather than attempted: a guard-thrashing compile
            # would produce a number, and it would be a number for a different
            # program.
            raise Unsupported("InferenceCore.step carries mutable memory-bank state")

        def fn():
            with torch.no_grad():
                core.step(img)
        return fn

    return make


SETUPS = {
    "sam2-encode": setup_sam2,
    "whisper-encode": setup_whisper,
    "kokoro-speak": setup_kokoro,
    "matanyone-step": setup_matanyone,
    "depthanything": setup_depthanything,
    "neurallut": setup_neurallut,
    "rife": setup_rife,
}
ORDER = ["sam2-encode", "whisper-encode", "kokoro-speak", "matanyone-step",
         "depthanything", "neurallut", "rife"]

_missing = set(SPEC["models"]) - set(SETUPS)
if _missing:
    raise SystemExit(f"models.toml declares {sorted(_missing)} and torch_models.py "
                     f"has no setup for them")


def main(want, iters, ramp, out, modes):
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible to torch — there is nothing to compare against "
                         "here. See benchmark/README.md for the ROCm interpreter.")

    # TF32 off everywhere: its 10-bit mantissa is a different computation, and
    # every export in this tree was produced with it off.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"PyTorch {torch.__version__}, TF32 off — {name}")
    results = {}
    for model in want:
        if model not in SETUPS:
            print(f"{model:18s} no such model")
            continue
        spec = SPEC["models"][model]
        entry = {"shape": spec["shape"], "dtype": spec["dtype"], "note": spec["note"]}
        try:
            make = SETUPS[model](dev)
        # `SystemExit` too, and deliberately: several `export_*` helpers call
        # `sys.exit()` with an instruction when an upstream checkout or a weight
        # file is missing. That is a perfectly good message and a terrible way to
        # end a seven-model suite, so it is recorded as this model's reason and
        # the rest still run.
        except (Exception, SystemExit) as e:     # noqa: BLE001 — reported, not swallowed
            print(f"{model:18s} SETUP FAILED: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:80]}")
            entry["error"] = f"{type(e).__name__}: {e}"
            results[model] = entry
            continue

        for mode in modes:
            print(f"{model:18s} {mode:9s} ", end="", flush=True)
            try:
                fn = make(mode == "compiled")
                # A compiled model spends its first calls in Inductor; that is
                # not what this measures, and `warmup` is inside `bench`.
                r = bench(fn, warmup=5 if mode == "eager" else 3,
                          iters=iters, ramp=ramp)
            except Unsupported as e:
                print(f"not supported: {e}")
                entry[mode] = {"unsupported": str(e)}
                continue
            except (Exception, SystemExit) as e:  # noqa: BLE001
                print(f"FAILED: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
                entry[mode] = {"error": f"{type(e).__name__}: {e}"}
                continue
            entry[mode] = r
            print(f"{r['p50']:9.2f} ms  ±{100 * r['spread']:4.1f}%  "
                  f"clock {r['clock_mhz'] or float('nan'):.0f} MHz")
            torch.cuda.empty_cache()
        results[model] = entry
        del make
        torch.cuda.empty_cache()

    payload = {"device": name, "torch": torch.__version__, "modes": modes,
               "models": results}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=SPEC["suite"]["samples"])
    ap.add_argument("--ramp", type=float, default=SPEC["suite"]["ramp_seconds"])
    ap.add_argument("--modes", default="eager,compiled",
                    help="comma-separated subset of eager,compiled")
    ap.add_argument("--out", default=str(HERE / "results" / "torch.json"))
    a = ap.parse_args()
    main(a.models or ORDER, a.iters, a.ramp, a.out, a.modes.split(","))
