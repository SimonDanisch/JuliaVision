"""Run each candidate model in PyTorch and write something you can look at or
listen to, so the decision to port is made on evidence rather than a datasheet.

    uv run tools/demos.py --list
    uv run tools/demos.py kokoro whisper
    uv run tools/demos.py --all

Everything lands in `media/model-demos/<name>/` with a `manifest.json` per model
naming each artefact and what it is meant to show. `tools/demo_page.py` turns
those manifests into one page.

The point is the *failure* case as much as the success one: a TTS voice that
sounds synthetic, an upscaler that invents detail, a denoiser that eats
consonants. Each demo therefore keeps the input next to the output so they can
be compared directly, rather than showing the output alone.

Source audio in `_src/` is third-party sample content, fetched once:
`clean_speech.wav` and the two `noise_*.wav` come from DeepFilterNet's own
assets (freesound), `music.mp3` from Demucs' repo, `jfk.wav` from whisper.cpp
(public domain).
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "gen"
OUT = ROOT / "media" / "model-demos"
SRC = OUT / "_src"

DEMOS = {}


def demo(name):
    def wrap(fn):
        DEMOS[name] = fn
        return fn
    return wrap


def outdir(name):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(name, artifacts, notes, title, verdict_hint=""):
    """Write the manifest a demo produces. `artifacts` is [(file, caption)]."""
    d = outdir(name)
    (d / "manifest.json").write_text(json.dumps({
        "model": name, "title": title, "notes": notes,
        "verdict_hint": verdict_hint,
        "artifacts": [{"file": f, "caption": c} for f, c in artifacts],
    }, indent=1))
    print(f"  -> {d.relative_to(ROOT)}/manifest.json  ({len(artifacts)} artefacts)")


def torchaudio_shim():
    """DeepFilterNet imports `torchaudio.backend.common.AudioMetaData`, which
    torchaudio 2.11 removed. It is used as a return annotation and nothing else,
    so a stand-in dataclass is enough and is preferable to pinning the whole
    stack back to torchaudio 2.x for one import.
    """
    import dataclasses
    import types
    if "torchaudio.backend.common" in sys.modules:
        return

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    mod = types.ModuleType("torchaudio.backend.common")
    mod.AudioMetaData = AudioMetaData
    pkg = types.ModuleType("torchaudio.backend")
    pkg.common = mod
    sys.modules["torchaudio.backend"] = pkg
    sys.modules["torchaudio.backend.common"] = mod


# --------------------------------------------------------------------- audio

@demo("kokoro")
def kokoro_demo():
    """Several voices reading the same paragraph, so the question 'does this
    sound good enough to ship' can be answered by listening rather than by
    trusting a leaderboard."""
    import soundfile as sf
    from kokoro import KPipeline

    text = ("The quick cut lands on the downbeat, and the scene opens on a wide shot "
            "of the harbour. Wait — that take is soft. Use the second one, it holds "
            "focus all the way through the pan.")
    voices = [("af_heart", "American female (af_heart)"),
              ("am_michael", "American male (am_michael)"),
              ("bf_emma", "British female (bf_emma)"),
              ("bm_george", "British male (bm_george)")]

    d = outdir("kokoro")
    pipe = KPipeline(lang_code="a")
    arts, notes = [], []
    for vid, label in voices:
        t0 = time.perf_counter()
        chunks = [a for _, _, a in pipe(text, voice=vid)]
        dt = time.perf_counter() - t0
        import numpy as np
        audio = np.concatenate(chunks)
        f = f"{vid}.wav"
        sf.write(d / f, audio, 24000)
        secs = len(audio) / 24000
        arts.append((f, f"{label} — {secs:.1f}s of audio in {dt:.2f}s ({secs/dt:.1f}x realtime)"))
        notes.append(f"{vid}: {secs:.1f}s audio, {dt:.2f}s to synthesise, {secs/dt:.1f}x realtime")

    (d / "text.txt").write_text(text)
    save("kokoro", arts, notes, "Kokoro-82M — text to speech",
         "Listen for: synthetic timbre, wrong stress on 'that take is soft', "
         "and whether the em-dash pause sounds deliberate or like a glitch.")


@demo("deepfilternet")
def deepfilternet_demo():
    """Clean speech, the same speech buried in real noise, and the result. Two
    noise types and two levels, because a denoiser that works on steady hum can
    still fail on babble."""
    torchaudio_shim()
    import numpy as np
    import soundfile as sf
    import torch
    from df.enhance import enhance, init_df

    d = outdir("deepfilternet")
    model, state, _ = init_df()
    sr = state.sr()

    clean, _ = _load_mono(SRC / "clean_speech.wav", sr)
    arts = [("clean.wav", "the original clean speech (reference)")]
    sf.write(d / "clean.wav", clean, sr)
    notes = []

    for noisefile, label in [("noise_a.wav", "traffic"), ("noise_b.wav", "clatter")]:
        noise, _ = _load_mono(SRC / noisefile, sr)
        noise = np.resize(noise, len(clean))
        for snr in (5, 0):
            # Mix at a stated SNR so the difficulty is a number rather than
            # "some noise" — 0 dB means the noise is as loud as the speech.
            g = np.sqrt((clean @ clean) / max(noise @ noise, 1e-9) / (10 ** (snr / 10)))
            noisy = clean + g * noise
            peak = max(np.abs(noisy).max(), 1e-9)
            noisy = (noisy / peak * 0.9).astype(np.float32)

            t0 = time.perf_counter()
            out = enhance(model, state, torch.from_numpy(noisy)[None, :])
            dt = time.perf_counter() - t0
            out = out.squeeze(0).numpy()

            base = f"{label}_{snr}dB"
            sf.write(d / f"{base}_noisy.wav", noisy, sr)
            sf.write(d / f"{base}_clean.wav", out, sr)
            arts.append((f"{base}_noisy.wav", f"{label} noise at {snr} dB SNR — the input"))
            arts.append((f"{base}_clean.wav", f"{label} at {snr} dB — denoised"))
            secs = len(noisy) / sr
            notes.append(f"{label} {snr}dB: {secs:.1f}s in {dt:.3f}s ({secs/dt:.0f}x realtime)")

    save("deepfilternet", arts, notes, "DeepFilterNet3 — voice denoising",
         "Listen for: consonants eaten (s, t, k), a hollow or 'underwater' timbre, "
         "and pumping as the noise floor moves. 0 dB is the hard case.")


@demo("demucs")
def demucs_demo():
    """A real mix split into stems. For an edit the one that matters is whether
    'vocals' comes out clean enough to drop the rest and keep dialogue."""
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    d = outdir("demucs")
    model = get_model("htdemucs")
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    # ffmpeg rather than torchaudio.load: torchaudio 2.11 routes decoding through
    # TorchCodec, which is another dependency for something ffmpeg already does.
    sr = model.samplerate
    mono, _ = _load_mono(SRC / "music.mp3", sr)
    wav = torch.from_numpy(np.stack([mono, mono]))   # htdemucs wants stereo in
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    t0 = time.perf_counter()
    with torch.no_grad():
        stems = apply_model(model, wav[None].to(dev), device=dev, progress=False)[0]
    dt = time.perf_counter() - t0
    stems = stems * ref.std() + ref.mean()

    shutil.copy(SRC / "music.mp3", d / "mix.mp3")
    arts = [("mix.mp3", "the input mix")]
    for name, stem in zip(model.sources, stems):
        f = f"{name}.wav"
        sf.write(d / f, stem.cpu().numpy().T, sr)
        arts.append((f, f"separated: {name}"))

    secs = wav.shape[-1] / sr
    save("demucs", arts, [f"{secs:.1f}s in {dt:.2f}s ({secs/dt:.1f}x realtime) on {dev}"],
         "Demucs v4 (htdemucs) — stem separation",
         "Listen for: bleed of drums into 'vocals', and whether 'other' sounds "
         "like a leftover bucket. For an edit only the vocal/no-vocal split matters.")


@demo("whisper")
def whisper_demo():
    """Transcription on clean speech, on speech in noise, and on Kokoro's own
    output — the last one is a round trip and shows how the two models compose."""
    import torch
    from transformers import pipeline

    d = outdir("whisper")
    dev = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=str(GEN / "whisper"),
                   device=dev, torch_dtype=torch.float16 if dev == 0 else torch.float32,
                   return_timestamps="word")

    jobs = [("jfk.wav", SRC / "jfk.wav", "clean archival speech (public domain)")]
    noisy = OUT / "deepfilternet" / "traffic_0dB_noisy.wav"
    if noisy.exists():
        jobs.append(("noisy_0dB.wav", noisy, "the same DeepFilterNet input at 0 dB SNR — "
                                             "how far does it degrade?"))
    denoised = OUT / "deepfilternet" / "traffic_0dB_clean.wav"
    if denoised.exists():
        jobs.append(("denoised_0dB.wav", denoised, "…and after DeepFilterNet. Does "
                                                   "denoising actually help the ASR?"))
    kok = OUT / "kokoro" / "am_michael.wav"
    if kok.exists():
        jobs.append(("kokoro_roundtrip.wav", kok, "Kokoro's synthesis, transcribed back"))

    arts, notes, lines = [], [], []
    for label, path, why in jobs:
        shutil.copy(path, d / label)
        t0 = time.perf_counter()
        r = asr(str(path))
        dt = time.perf_counter() - t0
        import soundfile as sf
        secs = sf.info(str(path)).duration
        text = r["text"].strip()
        arts.append((label, f"{why} — transcribed in {dt:.2f}s ({secs/dt:.0f}x realtime)"))
        notes.append(f"{label}: {secs/dt:.0f}x realtime")
        lines.append(f"### {label}\n\n_{why}_\n\n> {text}\n")
    (d / "transcripts.md").write_text("\n".join(lines))

    save("whisper", arts, notes, "Whisper large-v3-turbo — speech to text",
         "Read the transcripts next to the audio. The question is not whether it "
         "is perfect but whether it is good enough to cut against.")


# -------------------------------------------------------------------- vision

CLIP = ROOT / "media" / "birds_export.mp4"    # real footage: sparrow on a nest box


@demo("depthanything")
def depth_demo():
    """Depth on stills, and then depth on a clip — because for an editor the
    question about a per-frame model is not accuracy, it is whether it flickers.
    A still depth map cannot answer that and a video can."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    d = outdir("depthanything")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # The `-hf` variant is the same Apache-2.0 Small checkpoint in transformers
    # layout, which saves a repo checkout. Base and Large are CC-BY-NC.
    mid = "depth-anything/Depth-Anything-V2-Small-hf"
    proc = AutoImageProcessor.from_pretrained(mid)
    model = AutoModelForDepthEstimation.from_pretrained(mid).to(dev).eval()

    frames = _frames(CLIP, start=3.0, n=48, width=640)
    arts, notes = [], []

    def infer(img):
        inp = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp).predicted_depth
        out = torch.nn.functional.interpolate(
            out[None], size=img.size[::-1], mode="bicubic", align_corners=False)[0, 0]
        return out.cpu().numpy()

    still = frames[0]
    t0 = time.perf_counter()
    dep = infer(still)
    dt = time.perf_counter() - t0
    still.save(d / "input.png")
    _save_depth(dep, d / "depth.png")
    arts += [("input.png", "the frame"), ("depth.png", "predicted depth (near = bright)")]
    notes.append(f"{still.size[0]}x{still.size[1]} in {dt*1000:.0f} ms on {dev}")

    # The use, not the map: depth-keyed defocus. This is what the feature buys.
    _save_dof(still, dep, d / "dof.png")
    arts.append(("dof.png", "what it buys: depth-keyed defocus, background only"))

    # And the flicker question.
    seq = []
    t0 = time.perf_counter()
    for f in frames:
        seq.append(_depth_rgb(infer(f)))
    dt = time.perf_counter() - t0
    _write_video([np.concatenate([np.asarray(f), s], axis=1) for f, s in zip(frames, seq)],
                 d / "depth_video.mp4", fps=24)
    arts.append(("depth_video.mp4", "48 frames, source next to depth — watch for flicker"))
    notes.append(f"{len(frames)} frames in {dt:.1f}s ({len(frames)/dt:.1f} fps)")

    save("depthanything", arts, notes, "Depth Anything V2 Small — monocular depth",
         "Look for: does the bird separate from the wall, and does the depth map "
         "boil between frames? Per-frame flicker is what would make this unusable "
         "for a grade without temporal smoothing.")


@demo("rife")
def rife_demo():
    """4x slow motion from real footage, plus the honest test: drop a frame the
    model has never seen, reconstruct it, and put it next to the truth."""
    import numpy as np
    import torch

    train_log = GEN / "rife" / "train_log"
    if not (train_log / "flownet.pkl").exists():
        sys.exit("no RIFE weights — uv run tools/models.py fetch rife")
    # `train_log/RIFE_HDv3.py` ships inside the weights archive but imports
    # `model.warplayer` from the repo layout, so both paths are needed: the
    # checkout for the shared modules, the archive for the architecture that
    # matches these particular weights.
    repo = ROOT / "dev" / "Practical-RIFE"
    if not repo.exists():
        sys.exit(f"no RIFE checkout at {repo} — "
                 "git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git")
    # It also does `from train_log.IFNet_HDv3 import *`, so `train_log` has to be
    # importable as a package — meaning its *parent* on the path, not itself.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(GEN / "rife"))
    from train_log.RIFE_HDv3 import Model

    d = outdir("rife")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model()
    model.load_model(str(train_log), -1)
    model.eval()
    model.device()

    frames = _frames(CLIP, start=3.0, n=25, width=640)
    arr = [np.asarray(f).astype(np.float32) / 255.0 for f in frames]
    h, w, _ = arr[0].shape
    # 128, not 32: IFNet runs a 5-level pyramid (scale_list 16..1) and
    # `inference_video.py:193` pads to `max(128, 128/scale)` for that reason. A
    # 32-multiple passes the first levels and then mismatches deeper in.
    tmp = 128
    ph, pw = ((h - 1) // tmp + 1) * tmp, ((w - 1) // tmp + 1) * tmp

    def to_t(a):
        t = torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)
        return torch.nn.functional.pad(t, (0, pw - w, 0, ph - h))

    def to_a(t):
        return (t[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # 4x: three synthesised frames between every real pair.
    t0 = time.perf_counter()
    out = []
    for a, b in zip(arr, arr[1:]):
        ta, tb = to_t(a), to_t(b)
        out.append((a * 255).astype(np.uint8))
        for k in (0.25, 0.5, 0.75):
            with torch.no_grad():
                out.append(to_a(model.inference(ta, tb, timestep=k)))
    out.append((arr[-1] * 255).astype(np.uint8))
    dt = time.perf_counter() - t0
    _write_video(out, d / "slowmo_4x.mp4", fps=24)
    _write_video([(a * 255).astype(np.uint8) for a in arr], d / "original.mp4", fps=24)

    # The honest one: reconstruct a real frame and show it beside the truth.
    a, truth, b = arr[10], arr[11], arr[12]
    with torch.no_grad():
        mid = to_a(model.inference(to_t(a), to_t(b), timestep=0.5))
    from PIL import Image
    Image.fromarray(np.concatenate([(truth * 255).astype(np.uint8), mid], axis=1)).save(
        d / "reconstruction.png")
    err = float(np.abs(mid.astype(np.float32) - truth * 255).mean())

    save("rife", [
        ("original.mp4", "the source clip, 24 fps"),
        ("slowmo_4x.mp4", "4x frames, played at 24 fps — i.e. quarter-speed slow motion"),
        ("reconstruction.png", "left: a real frame. right: the same frame synthesised "
                               "from its neighbours, never seen by the model"),
    ], [f"{len(out)} frames from {len(arr)} in {dt:.1f}s ({len(out)/dt:.1f} fps) on {dev}",
        f"reconstruction mean abs error {err:.1f}/255"],
        "RIFE 4.26 — frame interpolation",
        "Look for: warped edges around the bird's beak and tail during fast motion, "
        "and whether the reconstruction is distinguishable from the real frame.")


@demo("basicvsrpp")
def basicvsrpp_demo():
    """Downscale real footage 4x, put it back, and show it next to bicubic. The
    baseline matters: an upscaler that only beats nearest-neighbour is not worth
    a deformable-convolution kernel."""
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT / "tools"))
    from basicvsrpp import load_basicvsrpp

    d = outdir("basicvsrpp")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_basicvsrpp(GEN / "basicvsrpp" / "basicvsr_pp_reds4.pth").to(dev).eval()

    # Small window: this is recurrent over the clip, so VRAM goes with T, not
    # with parameter count. 12 frames at 160x196 is already ~4 GB at 4x out.
    hires = _frames(CLIP, start=3.0, n=12, width=640)
    w, h = hires[0].size
    w4, h4 = w // 4 * 4, h // 4 * 4
    hires = [im.crop((0, 0, w4, h4)) for im in hires]
    lores = [im.resize((w4 // 4, h4 // 4), Image.BICUBIC) for im in hires]

    lr = torch.from_numpy(np.stack([np.asarray(i) for i in lores])).permute(0, 3, 1, 2)
    lr = (lr.float() / 255.0)[None].to(dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        sr = model(lr)[0].clamp(0, 1).cpu()
    dt = time.perf_counter() - t0
    srf = [(s.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for s in sr]
    bic = [np.asarray(i.resize((w4, h4), Image.BICUBIC)) for i in lores]

    # Three-way strip, same frame, so the comparison is not from memory.
    k = 6
    strip = np.concatenate([bic[k], srf[k], np.asarray(hires[k])], axis=1)
    Image.fromarray(strip).save(d / "compare.png")
    Image.fromarray(np.asarray(lores[k].resize((w4, h4), Image.NEAREST))).save(d / "input_4x_nearest.png")
    _write_video([np.concatenate([b, s], axis=1) for b, s in zip(bic, srf)],
                 d / "bicubic_vs_basicvsr.mp4", fps=12)

    err_b = float(np.abs(np.stack(bic).astype(np.float32) -
                         np.stack([np.asarray(i) for i in hires]).astype(np.float32)).mean())
    err_s = float(np.abs(np.stack(srf).astype(np.float32) -
                         np.stack([np.asarray(i) for i in hires]).astype(np.float32)).mean())

    save("basicvsrpp", [
        ("input_4x_nearest.png", f"the input: {w4//4}x{h4//4}, shown at 4x nearest so you "
                                 "can see what it had to work with"),
        ("compare.png", "left: bicubic 4x. middle: BasicVSR++ 4x. right: the real frame"),
        ("bicubic_vs_basicvsr.mp4", "bicubic | BasicVSR++, moving — watch the wood grain"),
    ], [f"{len(hires)} frames {w4//4}x{h4//4} -> {w4}x{h4} in {dt:.2f}s on {dev}",
        f"mean abs error vs truth: bicubic {err_b:.1f}, BasicVSR++ {err_s:.1f} (out of 255)"],
        "BasicVSR++ — video upscaling",
        "Look for: recovered wood grain and feather detail against bicubic, and "
        "whether it invents texture that is not in the real frame on the right.")


@demo("basicvsrpp_4k")
def basicvsrpp_4k_demo():
    """The actual use, not the benchmark: take the bird clip at its native
    864x1056 and 4x it to 3456x4224 — 14.6 MP, more pixels than UHD — for five
    seconds from the middle at 60 fps.

    Two things differ from the `basicvsrpp` demo. That one downscaled first so
    it had a ground truth to score against; here the input is already as good as
    the footage gets, so there is nothing to compare to but bicubic. And it is
    300 frames at full resolution where that was 12 at 160x196.

    VRAM goes with sequence length, not parameters. The four propagation
    branches each hold a 64-channel feature map per frame, which at this
    resolution is 64*864*1056*4 = 233 MB per frame per branch — so a 300-frame
    shot cannot go in at once at any batch size. It runs in overlapping chunks
    and streams to the encoder rather than accumulating 13 GB of output frames.
    """
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT / "tools"))
    from basicvsrpp import load_basicvsrpp

    d = outdir("basicvsrpp_4k")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16, and not as an optimisation: measured with tools/probe_vsr.py, fp32
    # OOMs at this resolution for *any* sequence length, because the deform-conv
    # im2col buffer is (1, 128*9, 1056*864) = 4.2 GB per call and does not scale
    # with T. fp16 halves that and fits to T=12. It costs 0.15/255 mean against
    # fp32, an order of magnitude under the bicubic-vs-SR gap being measured.
    model = load_basicvsrpp(GEN / "basicvsrpp" / "basicvsr_pp_reds4.pth")
    model = model.to(dev).half().eval()

    start, secs, fps = 9.5, 5.0, 60
    src = _raw_frames(CLIP, start, secs)
    n, (h, w) = len(src), src[0].shape[:2]
    print(f"  {n} frames at {w}x{h} -> {w*4}x{h*4}", flush=True)
    _write_video([Image.fromarray(f) for f in src], d / "source.mp4", fps)

    # Crop the busiest region rather than guessing where the detail is: the
    # window with the highest mean gradient is where an upscaler is doing work.
    g = np.asarray(Image.fromarray(src[n // 2]).convert("L"), dtype=np.float32)
    gm = np.abs(np.gradient(g)[0]) + np.abs(np.gradient(g)[1])
    cw, ch = w // 4, h // 4
    best, bx, by = -1.0, 0, 0
    for y in range(0, h - ch + 1, 24):
        for x in range(0, w - cw + 1, 24):
            v = float(gm[y:y + ch, x:x + cw].mean())
            if v > best:
                best, bx, by = v, x, y
    print(f"  detail window {cw}x{ch} at ({bx},{by})", flush=True)

    sr_w = _ffmpeg_writer(d / "upscaled_4k.mp4", w * 4, h * 4, fps, preset="fast", crf=20)
    # 1:1 pixel crop, bicubic beside BasicVSR++. This is the only view that shows
    # anything: a 3456x4224 video scaled to fit a screen is indistinguishable
    # from its own input, which is the trap in every upscaler demo.
    cmp_w = _ffmpeg_writer(d / "crop_compare.mp4", cw * 4 * 2, ch * 4, fps)

    # T = chunk + 2*pad = 10 here. probe_vsr.py measured 14.2 GB at T=12 against
    # ~15 GB free, and this card also carries an interactive REPL, so leave room.
    sharp_b, sharp_s, chunk, pad, done = [], [], 6, 2, 0
    t0 = time.perf_counter()
    while done < n:
        lo, hi = max(0, done - pad), min(n, done + chunk + pad)
        x = torch.from_numpy(np.stack(src[lo:hi])).permute(0, 3, 1, 2)
        x = (x.float() / 255.0)[None].to(dev).half()
        oom = False
        try:
            with torch.no_grad():
                y = model(x)[0].clamp_(0, 1)
        except torch.OutOfMemoryError as e:
            # Clear the traceback first. It holds every frame's locals, so the
            # failed attempt's activations stay live and empty_cache() frees
            # nothing — which is why the earlier 8->4->2->1 halving still OOMed.
            import traceback as _tb
            _tb.clear_frames(e.__traceback__)
            oom = True
        if oom:
            del x
            torch.cuda.empty_cache()
            if chunk == 1:
                raise RuntimeError("OOM at chunk=1; nothing left to halve")
            chunk = max(1, chunk // 2)
            print(f"  OOM -> chunk {chunk}", flush=True)
            continue
        # Keep only the interior: the pad frames exist so every emitted frame had
        # propagation context on both sides, and are re-derived by the next chunk.
        s = done - lo
        for t in range(s, s + min(chunk, n - done)):
            f = (y[t].permute(1, 2, 0).float() * 255).round().byte().cpu().numpy()
            sr_w.stdin.write(np.ascontiguousarray(f).tobytes())
            j = lo + t
            bic = np.asarray(Image.fromarray(src[j]).resize((w * 4, h * 4), Image.BICUBIC))
            X, Y = bx * 4, by * 4
            cb, cs = bic[Y:Y + ch * 4, X:X + cw * 4], f[Y:Y + ch * 4, X:X + cw * 4]
            cmp_w.stdin.write(np.ascontiguousarray(
                np.concatenate([cb, cs], axis=1)).tobytes())
            if j % 30 == 0:
                for buf, acc in ((cb, sharp_b), (cs, sharp_s)):
                    l = np.asarray(Image.fromarray(buf).convert("L"), dtype=np.float32)
                    acc.append(float((np.abs(np.gradient(l)[0]) +
                                      np.abs(np.gradient(l)[1])).mean()))
        done += chunk
        del x, y
        torch.cuda.empty_cache()
        print(f"  {done}/{n}  {(time.perf_counter()-t0)/done:.2f}s/frame", flush=True)
    dt = time.perf_counter() - t0
    for p in (sr_w, cmp_w):
        p.stdin.close()
        p.wait()

    # Stills, so the difference survives any player scaling.
    k = n // 2
    Image.fromarray(np.concatenate([
        np.asarray(Image.fromarray(src[k]).resize((w * 4, h * 4), Image.BICUBIC))
        [by * 4:by * 4 + ch * 4, bx * 4:bx * 4 + cw * 4],
    ], axis=1)).save(d / "still_bicubic.png")
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
    sb, ss = float(np.mean(sharp_b)), float(np.mean(sharp_s))

    save("basicvsrpp_4k", [
        ("source.mp4", f"the source: {w}x{h}, {secs}s from {start}s, {fps} fps"),
        ("upscaled_4k.mp4", f"BasicVSR++ 4x -> {w*4}x{h*4} (14.6 MP) — open in a "
                            "real player, a browser will scale it back down"),
        ("crop_compare.mp4", f"1:1 crop {cw*4}x{ch*4} at ({bx*4},{by*4}): "
                             "bicubic 4x | BasicVSR++ 4x"),
    ], [
        f"{n} frames {w}x{h} -> {w*4}x{h*4} in {dt:.0f}s ({dt/n:.2f}s/frame) on {dev}",
        f"chunk {chunk} +/-{pad} overlap, peak VRAM {peak:.1f} GB",
        f"mean gradient magnitude in the crop: bicubic {sb:.2f}, BasicVSR++ {ss:.2f} "
        f"({ss/max(sb,1e-6):.2f}x)",
    ], "BasicVSR++ — real 4x upscale of the source footage",
       "There is no ground truth here, so the gradient ratio only says BasicVSR++ "
       "put more high-frequency content in — it cannot say the detail is real. "
       "Judge that by eye on the crop: recovered feather barbs and wood grain are "
       "a win, a crunchy over-sharpened edge or shimmering texture is not.")


def _chunked(model, src, chunk, pad, scale, dev, on_frame, label=""):
    """Run a recurrent VSR net over a long sequence, emitting frames as they land.

    `pad` frames of overlap on each side give every emitted frame propagation
    context that the chunk boundary would otherwise cut off. Halves `chunk` on
    OOM — and clears the traceback first, because it holds the failed attempt's
    activations and without that `empty_cache()` frees nothing.
    """
    import numpy as np
    import torch

    n, done, t0 = len(src), 0, time.perf_counter()
    while done < n:
        lo, hi = max(0, done - pad), min(n, done + chunk + pad)
        x = torch.from_numpy(np.stack(src[lo:hi])).permute(0, 3, 1, 2)
        x = (x.float() / 255.0)[None].to(dev).half()
        oom = False
        try:
            with torch.no_grad():
                y = model(x)[0].clamp_(0, 1)
        except torch.OutOfMemoryError as e:
            import traceback as _tb
            _tb.clear_frames(e.__traceback__)
            oom = True
        if oom:
            del x
            torch.cuda.empty_cache()
            if chunk == 1:
                raise RuntimeError(f"{label}: OOM at chunk=1")
            chunk = max(1, chunk // 2)
            print(f"  [{label}] OOM -> chunk {chunk}", flush=True)
            continue
        s = done - lo
        for t in range(s, s + min(chunk, n - done)):
            on_frame(lo + t, (y[t].permute(1, 2, 0).float() * 255
                              ).round().byte().cpu().numpy())
        done += chunk
        del x, y
        torch.cuda.empty_cache()
        if done % 30 < chunk:
            print(f"  [{label}] {done}/{n}  {(time.perf_counter()-t0)/done:.2f}s/frame",
                  flush=True)
    return time.perf_counter() - t0


@demo("upscale_compare")
def upscale_compare_demo():
    """Four ways to 4x the same compression-limited footage, on one crop.

    The `basicvsrpp_4k` run showed REDS4 adding no visible detail and a
    period-4 pixelshuffle grid. The diagnosis was that the model is wrong for the
    input: REDS4 inverts a clean bicubic downsample, while this clip is 0.065
    bits/pixel, so its detail was destroyed by the encoder rather than by
    resampling. Two models from the same authors target that case, and this puts
    both next to what we already ran.

      bicubic            the only baseline that matters
      BasicVSR++ REDS4   trained on clean bicubic degradation
      decompress -> SR   NTIRE c128n25 removes codec artifacts at 1x, then REDS4
      RealBasicVSR       one 4x pass, trained with a degradation pipeline that
                         includes compression

    Judged on a fixed crop containing the bird, at 1:1 output pixels — a 4x video
    scaled to fit a screen cannot show the difference, which is the trap the
    first run fell into.
    """
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(ROOT / "tools"))
    from basicvsrpp import load_basicvsrpp, load_realbasicvsr

    d = outdir("upscale_compare")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    CK = GEN / "basicvsrpp"

    start, secs, fps = 9.5, 5.0, 60
    src = _raw_frames(CLIP, start, secs)
    n, (h, w) = len(src), src[0].shape[:2]
    # Fixed rather than auto-selected: the gradient maximum lands on tree bark,
    # and the subject is what anyone will actually look at.
    cx, cy, cw, ch = 225, 300, 216, 264
    X, Y, CW, CH = cx * 4, cy * 4, cw * 4, ch * 4
    print(f"  {n} frames {w}x{h}; crop {CW}x{CH} at ({X},{Y}) of {w*4}x{h*4}", flush=True)

    def crop(f):
        return f[Y:Y + CH, X:X + CW].copy()

    lanes, notes = {}, []
    lanes["bicubic"] = [crop(np.asarray(Image.fromarray(f).resize((w * 4, h * 4),
                                                                  Image.BICUBIC)))
                        for f in src]

    def run(key, model, chunk, pad, source):
        buf = [None] * n
        dt = _chunked(model, source, chunk, pad, 4, dev,
                      lambda i, f: buf.__setitem__(i, crop(f)), key)
        lanes[key] = buf
        notes.append(f"{key}: {dt:.0f}s ({dt/n:.2f}s/frame)")
        del model
        torch.cuda.empty_cache()

    sr = load_basicvsrpp(CK / "basicvsr_pp_reds4.pth").to(dev).half().eval()
    run("basicvsrpp", sr, 6, 2, src)
    del sr
    torch.cuda.empty_cache()

    # decompress runs at 1x, so its output is the SR model's input.
    dec = load_basicvsrpp(CK / "decompress_track1.pth", 128, 25,
                          is_low_res_input=False).to(dev).half().eval()
    clean = [None] * n
    t0 = time.perf_counter()
    # probe_vsr2: 3.7 GB at T=24, so this one can take a much longer window than
    # the SR nets — it propagates at H/4.
    _chunked(dec, src, 16, 4, 1, dev, lambda i, f: clean.__setitem__(i, f), "decompress")
    dt_dec = time.perf_counter() - t0
    del dec
    torch.cuda.empty_cache()
    Image.fromarray(clean[n // 2][cy:cy + ch, cx:cx + cw]).save(d / "still_decompressed_1x.png")

    sr = load_basicvsrpp(CK / "basicvsr_pp_reds4.pth").to(dev).half().eval()
    run("decompress_sr", sr, 6, 2, clean)
    notes.append(f"  (of which decompress pass: {dt_dec:.0f}s, {dt_dec/n:.2f}s/frame)")
    del sr, clean
    torch.cuda.empty_cache()

    run("realbasicvsr", load_realbasicvsr(CK / "realbasicvsr.pth").to(dev).half().eval(),
        6, 2, src)

    order = ["bicubic", "basicvsrpp", "decompress_sr", "realbasicvsr"]
    titles = ["bicubic 4x", "BasicVSR++ REDS4", "decompress -> REDS4", "RealBasicVSR"]
    _write_video([np.concatenate([lanes[k][i] for k in order], axis=1)
                  for i in range(n)], d / "four_way.mp4", fps)
    k = n // 2
    Image.fromarray(np.concatenate([lanes[o][k] for o in order], axis=1)).save(
        d / "four_way_still.png")
    for o in order:
        Image.fromarray(lanes[o][k]).save(d / f"still_{o}.png")

    # Same two numbers as before: does the extra high-frequency sit at the
    # pixelshuffle period (artifact) or spread across the spectrum (detail)?
    def grid_and_sharp(im):
        p = np.asarray(Image.fromarray(im).convert("L"), np.float32)
        s = 256
        q = p[:s, :s] - p[:s, :s].mean()
        F = np.abs(np.fft.fftshift(np.fft.fft2(q)))
        c = s // 2
        p4 = (F[c, c + s // 4] + F[c, c - s // 4] + F[c + s // 4, c] + F[c - s // 4, c])
        g = (np.abs(np.gradient(p)[0]) + np.abs(np.gradient(p)[1])).mean()
        return float(p4 / (F.sum() + 1e-9) * 1e4), float(g)

    notes.append("lane: gradient (detail proxy) / period-4 energy (pixelshuffle grid)")
    for o, t in zip(order, titles):
        gr, sh = grid_and_sharp(lanes[o][k])
        notes.append(f"  {t:22} gradient {sh:5.2f}   grid {gr:5.2f}")

    save("upscale_compare", [
        ("four_way.mp4", "1:1 crop, " + " | ".join(titles)),
        ("four_way_still.png", "the same frame, " + " | ".join(titles)),
        ("still_decompressed_1x.png", "the NTIRE decompression output at 1x, before "
                                      "any upscaling — this is the codec cleanup alone"),
    ] + [(f"still_{o}.png", t) for o, t in zip(order, titles)], notes,
        "4x upscaling on compression-limited footage — four approaches",
        "Compare against bicubic, not against the source. The gradient number is "
        "a detail proxy and the grid number is the pixelshuffle artifact; a model "
        "that raises gradient without raising grid is recovering something real.")


@demo("neurallut")
def neurallut_demo():
    """The predicted grade, and then the same three basis LUTs driven by hand.

    The second part is the actual argument for this model over a style-transfer
    network: the output is three numbers and a lookup table, so it is a control
    surface the user can grab, not an image they have to accept.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    repo = ROOT / "dev" / "Image-Adaptive-3DLUT"
    if not repo.exists():
        sys.exit(f"no checkout at {repo} — "
                 "git clone --depth 1 https://github.com/HuiZeng/Image-Adaptive-3DLUT.git")
    d = outdir("neurallut")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load(kind):
        p = repo / "pretrained_models" / "sRGB"
        luts = torch.load(p / f"LUTs{kind}.pth", map_location="cpu", weights_only=False)
        # 3 basis LUTs of (3, 33, 33, 33). The classifier picks how much of each.
        return torch.stack([luts[str(i)]["LUT"] for i in range(3)]).to(dev)

    def classifier(kind):
        # `models.py` imports its compiled CUDA trilinear extension at module
        # level, and building it is a needless step when the only thing wanted
        # from that file is the classifier definition — the lookup itself is
        # grid_sample above. Same tactic as `basicvsrpp.py` uses for mmcv: stand
        # in for exactly the name that is imported, then load the file unchanged.
        import types
        if "trilinear_c._ext" not in sys.modules:
            pkg = types.ModuleType("trilinear_c")
            ext = types.ModuleType("trilinear_c._ext")
            ext.trilinear = None
            pkg._ext = ext
            sys.modules["trilinear_c"] = pkg
            sys.modules["trilinear_c._ext"] = ext
        sys.path.insert(0, str(repo))
        import models as M
        net = (M.Classifier() if kind == "" else M.Classifier_unpaired())
        sd = torch.load(repo / "pretrained_models" / "sRGB" / f"classifier{kind}.pth",
                        map_location="cpu")
        net.load_state_dict(sd)
        return net.to(dev).eval()

    def apply_lut(img, lut):
        """Trilinear lookup, written with grid_sample rather than the repo's CUDA
        extension. Same operation, no compile step — and it is the form the Lava
        port would take anyway, since a 3D LUT apply is a sampler, not a network.
        """
        # No channel flip, and this is worth stating because the obvious guess is
        # wrong. The stored LUT is indexed [c, b, g, r]: red varies along the LAST
        # axis, blue along the first. grid_sample's last grid axis is x -> W, so
        # (R, G, B) maps straight through. Checked against the repo's own
        # IdentityLUT33.txt: this ordering round-trips to exactly 0.0, the flipped
        # one to 0.22 — which reads as a plausible cool grade rather than as a bug.
        g = img.permute(0, 2, 3, 1)[:, None] * 2 - 1    # (N,1,H,W,3) as R,G,B
        out = F.grid_sample(lut[None], g, mode="bilinear",
                            padding_mode="border", align_corners=True)
        return out[:, :, 0]                             # (N,3,H,W)

    frame = _frames(CLIP, start=3.0, n=1, width=640)[0]
    x = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1)[None].float().to(dev) / 255

    def topil(t):
        return Image.fromarray((t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    frame.save(d / "original.png")
    arts = [("original.png", "the ungraded frame")]
    notes = []

    for kind, label in [("", "paired (trained against expert retouches)"),
                        ("_unpaired", "unpaired (trained adversarially)")]:
        luts, net = load(kind), classifier(kind)
        t0 = time.perf_counter()
        with torch.no_grad():
            w = net(x).squeeze()
            blended = (w[:, None, None, None, None] * luts).sum(0)
            out = apply_lut(x, blended)
        dt = time.perf_counter() - t0
        f = f"graded{kind or '_paired'}.png"
        topil(out).save(d / f)
        arts.append((f, f"{label} — weights {[round(v,2) for v in w.tolist()]}"))
        notes.append(f"{label}: {dt*1000:.1f} ms at {x.shape[-1]}x{x.shape[-2]}, "
                     f"predicted weights {[round(v,3) for v in w.tolist()]}")

    # Drive the same three LUTs by hand: this is the control surface.
    luts = load("")
    looks = [("warm", (1.4, 0.0, 0.2)), ("cool", (0.2, 1.2, 0.0)),
             ("flat", (0.5, 0.5, 0.5)), ("punchy", (1.8, -0.3, 0.6))]
    tiles = []
    for name, ws in looks:
        wt = torch.tensor(ws, device=dev)
        with torch.no_grad():
            out = apply_lut(x, (wt[:, None, None, None, None] * luts).sum(0))
        f = f"look_{name}.png"
        topil(out).save(d / f)
        tiles.append(np.asarray(topil(out)))
        arts.append((f, f"hand-set weights {ws} — the '{name}' look"))
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2], 1),
                                    np.concatenate(tiles[2:], 1)], 0)).save(d / "looks_grid.png")
    arts.insert(1, ("looks_grid.png", "four looks from the same three LUTs, weights set by hand"))

    save("neurallut", arts, notes, "Image-Adaptive 3D LUT — style / mood grading",
         "The question is not whether the auto grade is 'right' but whether the "
         "look space is useful and stays photographic. Also note the classifier "
         "uses InstanceNorm2d, which DNNKernels does not have yet.")


@demo("propainter")
def propainter_demo():
    """Object removal on ProPainter's own sample, which is the fair test: their
    masks, their footage, the model at its best. If it is unconvincing here it
    will not improve on ours.

    Run as a subprocess rather than imported. `inference_propainter.py` is a
    script with module-level argument parsing and relative weight paths, and
    wrapping it would mean reimplementing its pipeline for no gain at this stage.
    """
    repo = ROOT / "dev" / "ProPainter"
    if not repo.exists():
        sys.exit(f"no checkout at {repo} — "
                 "git clone --depth 1 https://github.com/sczhou/ProPainter.git")
    # Its loader downloads into `weights/`; point that at what we already fetched.
    wdir = repo / "weights"
    wdir.mkdir(exist_ok=True)
    for f in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"):
        link = wdir / f
        if not link.exists():
            link.symlink_to(GEN / "propainter" / f)

    d = outdir("propainter")
    arts, notes = [], []
    for clip in ("bmx-trees", "tennis"):
        tmp = Path("/tmp") / f"pp_{clip}"
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "inference_propainter.py",
             "--video", f"inputs/object_removal/{clip}",
             "--mask", f"inputs/object_removal/{clip}_mask",
             "--output", str(tmp), "--width", "432", "--height", "240", "--fp16"],
            cwd=repo, check=True, capture_output=True)
        dt = time.perf_counter() - t0
        got = tmp / clip
        # bmx-trees ships as jpg, tennis as png.
        src = repo / "inputs" / "object_removal" / clip
        n = len([p for p in src.iterdir() if p.suffix in (".jpg", ".png")])
        _side_by_side(got / "masked_in.mp4", got / "inpaint_out.mp4",
                      d / f"{clip}.mp4")
        arts.append((f"{clip}.mp4", f"{clip}: masked input on the left, "
                                    "ProPainter's reconstruction on the right"))
        notes.append(f"{clip}: {n} frames at 432x240 in {dt:.1f}s ({n/dt:.1f} fps)")

    save("propainter", arts, notes, "ProPainter — object removal",
         "Look for: ghosting where the object was, and whether the fill stays put "
         "as the camera moves. This is the model that pairs with SAM 2 — SAM 2 "
         "makes the mask, this fills the hole. Licence is non-commercial.")


@demo("fluxklein")
def fluxklein_demo():
    """Edits on a real frame from this project, because that is the use: fill
    behind a mask, extend the frame to reframe, change the light. Text-to-image
    is included last and is the least interesting part for an editor.

    Run at bf16 — the whole argument for this model over Qwen-Image-Edit is that
    it needs no quantisation here, so demoing it quantised would prove nothing.
    """
    import torch
    from PIL import Image

    mdir = ROOT / "reference" / "models" / "FLUX.2-klein-4B"
    if not (mdir / "model_index.json").exists():
        sys.exit(f"no weights at {mdir} — hf download black-forest-labs/FLUX.2-klein-4B "
                 f"--local-dir {mdir}")
    # DiffusionPipeline, not Flux2Pipeline: klein is its own class with its own
    # text encoder. `model_index.json` says Flux2KleinPipeline + Qwen3ForCausalLM,
    # where FLUX.2-dev is Flux2Pipeline + Mistral-3. Loading it as the dev class
    # gets far enough to fail inside the chat template, because the Qwen tokenizer
    # ships a text-only template and the Mistral path hands it list-shaped
    # multimodal content. The auto class reads the right one out of the checkpoint.
    from diffusers import DiffusionPipeline

    d = outdir("fluxklein")
    pipe = DiffusionPipeline.from_pretrained(str(mdir), torch_dtype=torch.bfloat16)
    # The text encoder is as big as the transformer and runs once, before it. CPU
    # offload keeps both out of VRAM at the same time, which is what makes 4B+4B
    # fit in a budget that has an editor in it already.
    pipe.enable_model_cpu_offload()

    frame = _frames(CLIP, start=3.0, n=1, width=1024)[0]
    frame.save(d / "source.png")
    arts = [("source.png", "the source frame from birds_export.mp4")]
    notes = []

    edits = [
        ("relight", "change the lighting to cold blue overcast morning light, "
                    "keep everything else identical"),
        ("season", "make it a snowy winter scene with snow on the ledges, "
                   "keep the building and birdhouse identical"),
        ("remove", "remove the birdhouse from the wall, show the plain wooden wall behind it"),
    ]
    for name, prompt in edits:
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pipe(image=[frame], prompt=prompt, num_inference_steps=4,
                       guidance_scale=4.0,
                       generator=torch.Generator("cpu").manual_seed(0)).images[0]
        dt = time.perf_counter() - t0
        f = f"edit_{name}.png"
        out.save(d / f)
        arts.append((f, f'"{prompt}" — {dt:.1f}s'))
        notes.append(f"{name}: {dt:.1f}s at {out.size[0]}x{out.size[1]}, 4 steps")

    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = pipe(prompt="a cinematic wide shot of a harbour at dawn, "
                          "fishing boats, volumetric fog, 35mm film",
                   num_inference_steps=4, guidance_scale=4.0,
                   generator=torch.Generator("cpu").manual_seed(0)).images[0]
    dt = time.perf_counter() - t0
    gen.save(d / "txt2img.png")
    arts.append(("txt2img.png", f"text-to-image, no reference — {dt:.1f}s"))
    notes.append(f"txt2img: {dt:.1f}s")

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    notes.append(f"peak VRAM {peak:.1f} GB (bf16, no quantisation, encoder offloaded)")

    save("fluxklein", arts, notes, "FLUX.2-klein-4B — generative fill / outpaint / edit",
         "Look for: does the edit hold the parts you did not ask to change? That "
         "is what matters for an edit — a model that redraws the whole frame is "
         "useless even if the frame is pretty. Peak VRAM is the other number: "
         "this is the only edit model in its class that runs here unquantised.")


@demo("fluxklein_video")
def fluxklein_video_demo():
    """The question the stills cannot answer: does a per-frame image edit hold up
    across a shot?

    Nothing in this model is temporal — each frame is an independent sample from
    the posterior, so wherever the prompt underdetermines the result (where each
    snowflake sits) consecutive frames are free to disagree. A fixed seed is the
    usual mitigation and is what is used here, so this is the *best* case, not a
    strawman: same noise, same prompt, only the conditioning image moves.

    Kept small (512 wide, 48 frames) because flicker is visible at any
    resolution and 4B at full res is ~18s/frame.
    """
    import torch
    from diffusers import DiffusionPipeline

    mdir = ROOT / "reference" / "models" / "FLUX.2-klein-4B"
    d = outdir("fluxklein_video")
    pipe = DiffusionPipeline.from_pretrained(str(mdir), torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    n, fps, prompt = 48, 24, ("make it a snowy winter scene with snow on the ledges, "
                              "keep the building and birdhouse identical")
    frames = _frames(CLIP, start=3.0, n=n, width=512)
    _write_video(frames, d / "source.mp4", fps)

    edited, t0 = [], time.perf_counter()
    for i, f in enumerate(frames):
        with torch.inference_mode():
            # Same seed every frame. Per-frame seeds are strictly worse here.
            edited.append(pipe(image=[f], prompt=prompt, num_inference_steps=4,
                               guidance_scale=4.0,
                               generator=torch.Generator("cpu").manual_seed(0)
                               ).images[0].resize(f.size))
        print(f"  frame {i+1}/{n}  {(time.perf_counter()-t0)/(i+1):.1f}s/frame", flush=True)
    dt = time.perf_counter() - t0
    _write_video(edited, d / "edited.mp4", fps)
    _side_by_side(d / "source.mp4", d / "edited.mp4", d / "compare.mp4")

    # Temporal delta of the edit vs of the source: how much of the frame-to-frame
    # change is the scene moving, and how much did the model invent? The ratio is
    # the number, not either alone.
    import numpy as np
    def tdiff(seq):
        a = [np.asarray(x, dtype=np.float32) for x in seq]
        return float(np.mean([np.abs(a[i+1] - a[i]).mean() for i in range(len(a) - 1)]))
    ds, de = tdiff(frames), tdiff(edited)

    save("fluxklein_video", [
        ("source.mp4", f"the source shot — {n} frames at {fps} fps"),
        ("edited.mp4", f'per-frame "{prompt}", fixed seed — {dt/n:.1f}s/frame'),
        ("compare.mp4", "source | edited, side by side"),
    ], [
        f"{n} frames at 512 wide, 4 steps, fixed seed: {dt:.0f}s total, {dt/n:.1f}s/frame",
        f"mean frame-to-frame abs delta: source {ds:.2f}/255, edited {de:.2f}/255 "
        f"({de/max(ds,1e-6):.1f}x)",
    ], "FLUX.2-klein-4B applied per-frame to a shot",
       "Watch the snow and the wall texture, not the composition. The ratio in the "
       "notes is the measurement: temporal delta of the edit over that of the "
       "source. 1.0x would mean the edit added no instability of its own.")


def _side_by_side(left, right, out):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(left),
                    "-i", str(right), "-filter_complex", "hstack=inputs=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(out)], check=True)


def _frames(video, start, n, width):
    """`n` frames from `video` starting at `start` seconds, scaled to `width`."""
    import io
    from PIL import Image
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "quiet", "-ss", str(start), "-i", str(video),
         "-frames:v", str(n), "-vf", f"scale={width}:-2", "-f", "image2pipe",
         "-vcodec", "png", "-"], capture_output=True, check=True)
    out, buf, data = [], b"", p.stdout
    sig = b"\x89PNG\r\n\x1a\n"
    starts = [i for i in range(len(data)) if data.startswith(sig, i)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(data)
        out.append(Image.open(io.BytesIO(data[s:e])).convert("RGB"))
    return out


def _raw_frames(video, start, secs):
    """`secs` of `video` from `start`, as a list of HxWx3 uint8 arrays.

    Raw rather than the PNG-pipe `_frames` uses: at 300 frames the PNG encode,
    the signature scan and the decode cost more than the model does.
    """
    import numpy as np
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, check=True)
    w, h = (int(v) for v in p.stdout.strip().split("x"))
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-t", str(secs),
         "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    return list(a[:len(a) // (w * h * 3) * w * h * 3].reshape(-1, h, w, 3))


def _ffmpeg_writer(path, w, h, fps, preset="medium", crf=18):
    """An open ffmpeg stdin to push rgb24 frames into, for output too big to hold."""
    return subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
         "-preset", preset, "-pix_fmt", "yuv420p", "-crf", str(crf),
         "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def _write_video(frames, path, fps):
    """h264 in mp4, because the page these end up on is a browser."""
    import numpy as np
    h, w, _ = np.asarray(frames[0]).shape
    p = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
         "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(np.asarray(f), dtype="uint8").tobytes())
    p.stdin.close()
    p.wait()


def _depth_rgb(dep):
    import numpy as np
    d = (dep - dep.min()) / max(dep.ptp(), 1e-6)
    import matplotlib.cm as cm
    return (cm.inferno(d)[..., :3] * 255).astype(np.uint8)


def _save_depth(dep, path):
    from PIL import Image
    Image.fromarray(_depth_rgb(dep)).save(path)


def _save_dof(img, dep, path):
    """Defocus keyed on depth: the subject stays sharp, everything behind it does
    not. The point is to show the feature rather than the tensor."""
    import numpy as np
    from PIL import Image, ImageFilter
    d = (dep - dep.min()) / max(dep.ptp(), 1e-6)
    sharp = np.asarray(img).astype(np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(6))).astype(np.float32)
    # Near = 1 in this convention, so the mask keeps the near subject sharp.
    m = np.clip((d - 0.45) / 0.25, 0, 1)[..., None]
    Image.fromarray((sharp * m + blur * (1 - m)).astype(np.uint8)).save(path)


def _load_mono(path, sr):
    """Read `path` as mono float32 at `sr`, via ffmpeg so any container works."""
    import numpy as np
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "quiet", "-i", str(path),
                        "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), sr


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        print("\n".join(sorted(DEMOS)))
        sys.exit()
    targets = sorted(DEMOS) if a.all else a.names
    if not targets:
        sys.exit("name a demo, or --all. --list shows them.")
    for n in targets:
        if n not in DEMOS:
            sys.exit(f"unknown demo {n!r}; --list shows them")
        print(f"{n} ...")
        DEMOS[n]()
