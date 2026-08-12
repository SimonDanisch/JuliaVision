"""
Every ported model's forward pass in PyTorch, on this card, through one harness.

    uv run tools/baseline_all.py                 # all of them
    uv run tools/baseline_all.py sam2 rife       # a subset

The other half of `tools/bench_all.jl`, and neither is meaningful without the
other: "≥ PyTorch" is a target, and a target needs a denominator measured the
same way on the same card (GUARDRAILS §5). `bench_all.jl`'s docstring has
referred to this file for months; until now it did not exist, so five of the
seven models had a cost and no verdict.

Matched to the Julia side as closely as the two runtimes allow:

  * **the same dtype as the artifact we actually load**, not the one the export
    was meant to produce. Read off the loaded weights, not off the export
    script — `depthanything`, `neurallut` and `rife` all run fp32, and Kokoro
    ships an fp32 artifact although an fp16 export exists. Comparing our fp32
    against PyTorch fp16 would measure a dtype choice and call it an engine gap.
  * the same shapes `bench_all.jl` feeds;
  * TF32 **off** in fp32, so PyTorch is not handed the tensor cores for every
    matmul while Lava runs true fp32;
  * a real clock ramp before anything is recorded. This card idles at 210 MHz of
    a 3105 MHz maximum and takes seconds of sustained load to boost, so a short
    benchmark measures the ramp. `ramp` holds it up; see `measure.jl`'s
    `plateau` for what the same mistake did on the Julia side.
  * the median reported, not the min, and the spread beside it.

**No `torch.compile`.** The graph Lava runs came out of `torch.export`, which is
eager PyTorch's own decomposition. Comparing against a fused Inductor build would
be comparing a compiler we do not have against a runtime we do.

Two entries do not cover quite what the Julia row covers, and say so in the
`note` column rather than being quietly compared:

  * `kokoro` times the model halves, not phonemisation — our `speak` includes our
    own G2P and PyTorch's `KPipeline` would time misaki's, which is a different
    program in a different language.
  * `matanyone` times upstream's `InferenceCore.step`, which carries its own
    memory bank; ours is a fixed 5-frame capacity. Same work, different
    bookkeeping.
"""

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import find_root  # noqa: E402  (tools/ is symlinked; see find_root)

ROOT = find_root()
GEN = ROOT / "gen"


def smclock():
    """SM clock in MHz, or None where `nvidia-smi` is not available.

    Recorded with the result because it is part of it. A baseline that does not
    say what clock it ran at is not comparable with one that ran at another —
    and a baseline taken too SLOW flatters us, so it fails in the comfortable
    direction and nothing complains.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def bench(fn, *, warmup=5, iters=15, ramp=6.0):
    """Median wall time per call in ms, after holding the card at its clock.

    `ramp` is not a warm-up for the caches — `warmup` is that. It exists because
    the boost clock takes seconds of *sustained* load to come up on this card,
    and a benchmark that starts measuring before it does reports the ramp. The
    same defect cost the Julia harness an 11x error on short kernels.

    **6 seconds, and that is measured rather than picked.** At 2 s the SAM 2 row
    read 69.4-69.5 ms; `sam2_pytorch_baseline.py`, which ramps 200 iterations
    (~14 s), reads 67.3 on the same card minutes apart. A longer ramp giving the
    FASTER number is the signature of a card still climbing, so 2 s was still
    measuring the climb.
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
    return {"min": ts[0], "p50": p50, "spread": (ts[-1] - ts[0]) / p50}


# ── the models ───────────────────────────────────────────────────────────────
#
# Each returns `(fn, shape, dtype, note)`. `fn` takes no arguments and runs one
# forward; everything it needs is built here, outside the timed region.


def setup_sam2(dev):
    import export_graphs as EG
    import export_sam2 as ES

    model = ES.build("large").to(dev)
    res = model.image_size
    enc = ES.Encoder(model).to(dev).eval()
    image = torch.rand(1, 3, res, res, device=dev)

    # **The autocast context is entered ONCE, not per call.** `autocast.__exit__`
    # runs `torch.clear_autocast_cache()`, so a `with` inside the timed function
    # throws away the fp16 casts of every fp32 weight and re-does them on the next
    # call. That read 69.5 ms against `sam2_pytorch_baseline.py`'s 67.3 on the same
    # card — a 3% overstatement of PyTorch, i.e. in the direction that flatters us.
    # The stack is deliberately left open for the life of the closure.
    stack = contextlib.ExitStack()
    stack.enter_context(torch.no_grad())
    stack.enter_context(EG.precision_ctx("autocast"))

    def fn(_keep=stack):
        enc(image)

    return fn, f"{res}x{res}", "autocast fp16", ""


def setup_whisper(dev):
    from transformers import WhisperForConditionalGeneration

    dtype = torch.float16
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo", torch_dtype=dtype).to(dev).eval()
    enc = model.model.encoder
    mel = torch.zeros(1, 128, 3000, device=dev, dtype=dtype)

    def fn():
        with torch.inference_mode():
            enc(mel)

    return fn, "30 s window", "fp16", ""


def setup_depthanything(dev):
    from export_depthanything import Depth, load_model

    model = Depth(load_model()).to(dev).eval()
    x = torch.rand(1, 3, 518, 518, device=dev)

    def fn():
        with torch.no_grad():
            model(x)

    return fn, "518x518", "fp32", ""


def setup_neurallut(dev):
    from export_neurallut import LUTPredictor, load_upstream

    classifier, luts = load_upstream("sRGB", "paired")
    model = LUTPredictor(classifier, luts).eval().to(dev)
    x = torch.rand(1, 3, 256, 256, device=dev)

    def fn():
        with torch.no_grad():
            model(x)

    return fn, "256x256 classifier", "fp32", ""


def setup_rife(dev):
    from export_rife import Interp, load_flownet, padded

    net, _unused = load_flownet()
    model = Interp(net).to(dev).eval()
    h, w = padded(1152), padded(1920)
    imgs = torch.rand(1, 6, h, w, device=dev)
    timestep = torch.full((1, 1, 1, 1), 0.5, device=dev)

    def fn():
        with torch.no_grad():
            model(imgs, timestep)

    return fn, "1920x1152", "fp32", ""


def setup_kokoro(dev):
    """The two model halves and the alignment between them — not phonemisation.

    `export_kokoro`'s own split, so this times the same computation the graphs
    carry. fp32 because that is what the published artifact contains, however
    much the fp16 export exists.
    """
    import export_kokoro as EK
    from kokoro import KModel

    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(EK.WEIGHTS / "config.json"),
                   model=str(EK.WEIGHTS / "kokoro-v1_0.pth")).eval().to(dev)
    # The phonemisation of `bench_all.jl`'s sentence — "The quick brown fox jumps
    # over the lazy dog." — taken from `KokoroRunner.phonemize` so both sides run
    # the same 50 tokens. With `export_kokoro`'s default string instead this timed
    # 30 tokens against the Julia row's 50 and the ratio was meaningless.
    phonemes = "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ."
    ids = torch.tensor([[0] + [model.vocab[c] for c in phonemes if c in model.vocab] + [0]],
                       device=dev)
    voice = torch.load(EK.WEIGHTS / "af_heart.pt", map_location="cpu", weights_only=True)
    ref_s = voice[ids.shape[1] - 1].to(dev)

    ntokens = ids.shape[1]
    text = EK.TextHalf(model).eval()
    voc = EK.VocoderHalf(model).eval()

    # The whole path, exactly as `checkparity` composes it: the alignment between
    # the halves is a host gather, and leaving it out would time two thirds of a
    # model. `nframes = 0` keeps the true frame count, which is the only one that
    # is correct — see `align`.
    def fn():
        with torch.no_grad():
            d, t_en, duration = text(ids, ref_s)
            idx, _ = EK.align(duration, ntokens)
            en = d.transpose(-1, -2)[:, :, idx]
            asr = t_en[:, :, idx]
            voc(en, asr, ref_s)

    return fn, f"{ntokens} tokens", "fp32", "model only, no G2P"


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

    net, device = load_model(dev)
    core = InferenceCore(net, device=device)
    W, H = 512, 288
    img = torch.full((3, H, W), 0.5, device=device)
    msk = torch.zeros(H, W, device=device)
    msk[(H // 4):(3 * H // 4), (W // 4):(3 * W // 4)] = 1.0

    with torch.no_grad():
        core.step(img, msk, objects=[1])

    def fn():
        with torch.no_grad():
            core.step(img)

    return fn, "512x288", "fp16", "upstream memory bank"


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


def main(want, iters, ramp, out):
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA — there is nothing to compare against here")

    # TF32 off everywhere: its 10-bit mantissa is a different computation, and
    # every export in this tree was produced with it off.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    dev = "cuda"
    print(f"PyTorch eager, TF32 off — {torch.cuda.get_device_name(0)}")
    results = {}
    for name in want:
        if name not in SETUPS:
            print(f"{name:18s} no such model")
            continue
        print(f"{name:18s} ", end="", flush=True)
        try:
            fn, shape, dtype, note = SETUPS[name](dev)
            r = bench(fn, iters=iters, ramp=ramp)
        except Exception as e:                       # noqa: BLE001 — reported, not swallowed
            print(f"FAILED: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
            continue
        results[name] = {**r, "shape": shape, "dtype": dtype, "note": note,
                         "sm_clock_mhz": smclock()}
        print(f"{r['p50']:9.2f} ms  ±{100 * r['spread']:4.1f}%  {dtype:14s} "
              f"({shape}){'  ' + note if note else ''}")
        del fn
        torch.cuda.empty_cache()

    print("\n── summary (p50) ──")
    for n in want:
        if n in results:
            r = results[n]
            print(f"{n:18s} {r['p50']:9.2f} ms  {r['dtype']:14s} {r['shape']}")

    if out:
        Path(out).write_text(json.dumps(
            {"device": torch.cuda.get_device_name(0), "models": results}, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", default=None)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--ramp", type=float, default=2.0, help="seconds of load before timing")
    p.add_argument("--out", default=str(GEN / "graphs" / "pytorch_baseline_all.json"))
    a = p.parse_args()
    main(a.models or ORDER, a.iters, a.ramp, a.out)
