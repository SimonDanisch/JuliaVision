"""The registry of models being ported to DNNKernels: where each one comes from,
where its weights live, and what its runner package is called.

Dev-only. Nothing here ships. The tracking doc is `models-to-port.md`; this file
is the machine-readable half of it, so a URL is written down once and the
scaffolder, the fetcher and the exporters all read the same one.

    uv run tools/models.py list
    uv run tools/models.py fetch whisper
    uv run tools/models.py fetch --all

Checkpoints land in `gen/<name>/`, matching where `export_basicvsrpp.py` already
expects `gen/basicvsrpp/basicvsr_pp_reds4.pth`. They are not artifacts yet: a
Julia `Artifacts.toml` needs the sha256 of a tarball that has been uploaded to a
release, so the runner packages point at `gen/graphs/<name>` instead and
`DNNKernels.assetpath` falls through to it. Cutting the release is what turns a
working port into an installable one.
"""

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "gen"
DEV = ROOT / "dev"


@dataclass
class Model:
    name: str                      # registry key, and the gen/ subdirectory
    package: str                   # the Julia runner package in dev/JuliaVision
    uuid: str
    title: str                     # what a human calls it
    feature: str                   # the editor feature it provides — one each
    license: str
    upstream: str                  # the repo the architecture comes from
    files: list                    # [(url, filename)] — weights, verified reachable
    summary: str                   # one paragraph for the package docstring
    inputs: str                    # what the exported graph takes
    newops: list = field(default_factory=list)   # ATen ops DNNKernels lacks
    checkout: str = ""             # dev/ subdir needed for the export, if any
    pip: str = ""                  # or a package the export imports
    note: str = ""
    graphdir: str = ""             # gen/graphs/<this>, when it is not <name>
    gdrive: str = ""               # Google Drive file id, for upstreams with no direct URL
    unzip: str = ""                # archive in `files` to expand after fetching

    @property
    def dir(self):
        return GEN / self.name

    @property
    def graphs(self):
        return GEN / "graphs" / (self.graphdir or self.name)

    def paths(self):
        return [self.dir / f for _, f in self.files]

    def fetched(self):
        return all(p.exists() for p in self.paths())


# Ordered by the port order argued for in models-to-port.md, not alphabetically:
# the FFT/STFT models come first because one kernel unlocks four of them.
MODELS = [
    Model(
        name="whisper", package="WhisperRunner",
        uuid="6f1a5d52-5102-48dc-988e-ecb6f8c89f5e",
        title="Whisper large-v3-turbo", feature="speech -> text", license="MIT",
        upstream="https://github.com/openai/whisper",
        files=[
            ("https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/model.safetensors", "model.safetensors"),
            ("https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/config.json", "config.json"),
            ("https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/tokenizer.json", "tokenizer.json"),
            ("https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/preprocessor_config.json", "preprocessor_config.json"),
        ],
        summary=(
            "Transcription, and the transcript-driven editing that follows from it. "
            "809M parameters, ~1.6 GB in fp16.\n\n"
            "The reason this one is first is not the feature. It is the only model in "
            "the set that decodes autoregressively, so it forces two things the runtime "
            "does not have: an FFT for the log-mel front end, and a KV cache. The FFT is "
            "shared with every other audio model here; the KV cache puts the engine in a "
            "bandwidth-bound batch-1 GEMV regime that none of the GEMM tiling work in "
            "perf-plan.md applies to."
        ),
        inputs="log-mel spectrogram (1, 128, 3000) + decoder token ids",
        newops=["stft / fft (host-side mel is acceptable to start)",
                "KV-cache attention (incremental, not a new ATen op but a new execution mode)"],
        pip="transformers",
    ),
    Model(
        name="deepfilternet", package="DeepFilterRunner",
        uuid="76ab6b05-7ff6-4b1b-98e3-d56aa964097a",
        title="DeepFilterNet3", feature="voice denoising", license="MIT or Apache-2.0 (dual)",
        upstream="https://github.com/Rikorose/DeepFilterNet",
        files=[("https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip", "DeepFilterNet3.zip")],
        summary=(
            "Room tone, hum and hiss off a dialogue track, in real time. Roughly 2 MB of "
            "weights — the smallest model in the set by two orders of magnitude.\n\n"
            "Nearly free once Whisper's FFT exists: it works in the complex STFT domain, "
            "and `view_as_complex`/`view_as_real` are already ops the runtime has."
        ),
        inputs="complex STFT frames (1, T, F, 2)",
        newops=["ERB filterbank (a small matmul, not a new op)"],
        pip="deepfilternet",
    ),
    Model(
        name="demucs", package="DemucsRunner",
        uuid="a4515a7e-7709-4c13-8f3d-d3adf643ca1a",
        title="Demucs v4 (htdemucs)", feature="stem separation", license="MIT",
        upstream="https://github.com/adefossez/demucs",
        files=[("https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th", "htdemucs.th")],
        summary=(
            "Splits a mix into vocals / drums / bass / other. In an edit that means "
            "pulling dialogue off a music bed, or replacing the bed and keeping the "
            "dialogue. ~300 MB.\n\n"
            "A different feature from DeepFilterNet3, not a better one: separation, not "
            "denoising. It will not clean a noisy recording and DFN3 will not remove a "
            "song. Both are cheap and both want the same FFT.\n\n"
            "Hybrid Transformer Demucs runs two branches — waveform and spectrogram — and "
            "sums them, so the graph carries a real FFT *inside* it rather than only in a "
            "front end. That makes it the model that decides whether the FFT is a proper "
            "device kernel or a host convenience."
        ),
        inputs="waveform (1, 2, 343980) at 44.1 kHz",
        newops=["stft / istft on device", "lstm (the encoder's bottleneck)"],
        pip="demucs",
    ),
    Model(
        name="kokoro", package="KokoroRunner",
        uuid="31480c1c-f298-4848-9ae9-978a73fb95d8",
        title="Kokoro-82M", feature="text -> speech", license="Apache-2.0",
        upstream="https://github.com/hexgrad/kokoro",
        files=[
            ("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth", "kokoro-v1_0.pth"),
            ("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json", "config.json"),
            ("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt", "af_heart.pt"),
        ],
        summary=(
            "Voiceover and placeholder narration from 82M parameters — ~330 MB, 54 fixed "
            "voices, no cloning.\n\n"
            "The awkward one. Kokoro is StyleTTS2-derived, which means LSTMs: a sequential "
            "dependency along the time axis, which is the shape a GPU is worst at and the "
            "runtime has no answer for yet. Scheduled late deliberately — take that fight "
            "when the rest of the set is landing, not before."
        ),
        inputs="phoneme ids (1, T) + a 256-dim style vector",
        newops=["lstm", "istft vocoder", "weight_norm folding at export"],
        pip="kokoro",
    ),
    Model(
        name="neurallut", package="NeuralLUTRunner",
        uuid="0f237b2a-cdac-4549-9dcd-ab780cc4b910",
        title="Image-Adaptive 3D LUT", feature="style / mood grading", license="Apache-2.0",
        upstream="https://github.com/HuiZeng/Image-Adaptive-3DLUT",
        files=[],   # pretrained_models/ lives in the checkout, a few hundred KB
        summary=(
            "A look, applied as a grade. Under 600K parameters: a small CNN predicts blend "
            "weights over a handful of basis 3D LUTs, and the result is a LUT.\n\n"
            "That output shape is the whole argument for it. Per-frame diffusion style "
            "transfer flickers and cannot be edited; a predicted LUT is a first-class "
            "grading object — it drops into the inspector, the user can push it around, and "
            "it keyframes with the machinery already built. Temporal stability comes from "
            "smoothing the predicted weights across frames rather than from the network.\n\n"
            "Almost nothing new for the runtime, which is why it is second in the order: it "
            "is the fastest visible result in the set. The LUT apply is trilinear "
            "interpolation and belongs in GPUFiltering, not in the graph."
        ),
        inputs="image (1, 3, H, W) in 0..1",
        newops=["trilinear 3D LUT apply (a GPUFiltering kernel, not an ATen op)"],
        checkout="Image-Adaptive-3DLUT",
    ),
    Model(
        name="rife", package="RIFERunner",
        uuid="091c3b0f-d4d3-44b9-a2bb-d2bc125c3573",
        title="RIFE 4.x (Practical-RIFE)", feature="frame interpolation", license="MIT",
        upstream="https://github.com/hzwer/Practical-RIFE",
        summary=(
            "Slow motion, framerate conversion, and smoothing a retime — about 10 MB of "
            "weights for all of it.\n\n"
            "Cheap for the runtime: the warping is `grid_sampler_2d`, which is already "
            "implemented and already exercised by the optical-flow path in GPUFiltering."
        ),
        inputs="two frames (1, 3, H, W) + a timestep",
        newops=[],
        # RIFE 4.26, the newest full model. Upstream publishes only to Google Drive
        # and Baidu, so this goes through `gdown` rather than a direct URL — a
        # Drive file over 25 MB serves an HTML virus-scan interstitial to plain
        # HTTP, which is why `urllib` gets a web page where it expected weights.
        #
        # It still needs mirroring before it can be an artifact: `Pkg` speaks HTTP
        # and nothing else. The README states the model links carry the same MIT
        # licence as the code, so re-hosting `train_log/` on the JuliaVision assets
        # release is permitted and is what turns this into a normal download.
        gdrive="1gViYvvQrtETBgU1w8axZSsr7YUuw31uy",
        files=[(None, "rife-4.26.zip")],
        unzip="rife-4.26.zip",
        note="expands to train_log/{flownet.pkl, IFNet_HDv3.py, RIFE_HDv3.py, refine.py}",
    ),
    Model(
        name="depthanything", package="DepthAnythingRunner",
        uuid="acf050f5-8d8b-44a6-839a-ace43aa41d7c",
        title="Depth Anything V2 Small", feature="monocular depth", license="Apache-2.0",
        upstream="https://github.com/DepthAnything/Depth-Anything-V2",
        files=[("https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
                "depth_anything_v2_vits.pth")],
        summary=(
            "Depth per frame, which buys fake shallow depth of field, depth-keyed grading, "
            "parallax push-ins and sky masks. 25M parameters.\n\n"
            "**Small specifically.** The Base and Large checkpoints are CC-BY-NC-4.0; only "
            "Small is Apache-2.0. Large is better and can be added later as its own "
            "non-commercial package, the way MatAnyone already is.\n\n"
            "A ViT with scaled-dot-product attention and nothing else unusual, so the "
            "runtime needs nothing new. Pure editor value, no engine work — which is why "
            "it sits where it does in the order."
        ),
        inputs="image (1, 3, 518, 518)",
        newops=[],
        checkout="Depth-Anything-V2",
    ),
    Model(
        name="basicvsrpp", package="BasicVSRRunner",
        uuid="80334b24-06e6-4f50-a0e2-378c5490f15d",
        title="BasicVSR++", feature="video upscaling", license="Apache-2.0",
        upstream="https://github.com/open-mmlab/mmagic",
        files=[("https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
                "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth",
                "basicvsr_pp_reds4.pth")],
        summary=(
            "Temporally consistent upscaling. 7.3M parameters, but the footprint is "
            "activations rather than weights — it is recurrent over a clip, so VRAM scales "
            "with sequence length.\n\n"
            "The furthest along: `tools/export_basicvsrpp.py` already produces the graph "
            "into `gen/graphs/basicvsrpp-fp32`, and the runner package is what is missing.\n\n"
            "Engine-wise the interesting part is flow-guided deformable alignment — DCNv2 "
            "is an irregular per-pixel gather with no clean coopmat mapping, and it is the "
            "hardest kernel in this set."
        ),
        inputs="clip (1, T, 3, H, W)",
        newops=["deformable_conv2d (DCNv2)"],
        checkout="BasicVSR_PlusPlus",
        # The one model whose export predates this registry; it landed under the
        # precision-suffixed name `export_basicvsrpp.py --precision fp32` writes.
        graphdir="basicvsrpp-fp32",
    ),
    Model(
        name="propainter", package="ProPainterRunner",
        uuid="f30bad0e-6420-4ef0-ba68-abf723d394da",
        title="ProPainter", feature="object removal / inpainting", license="S-Lab 1.0 (NON-COMMERCIAL)",
        upstream="https://github.com/sczhou/ProPainter",
        files=[
            ("https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth", "ProPainter.pth"),
            ("https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth", "raft-things.pth"),
            ("https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
             "recurrent_flow_completion.pth"),
        ],
        summary=(
            "Remove a boom mic, a logo, a passer-by. SAM 2 produces the mask, this fills "
            "the hole — which is why it is in the set at all: it compounds a model that is "
            "already shipping rather than standing alone.\n\n"
            "**Non-commercial.** S-Lab License 1.0, same as MatAnyone, from the same group "
            "at NTU; commercial use is by arrangement with the authors. That is survivable "
            "because it is its own package, exactly as MatAnyoneRunner already is — but the "
            "editor must degrade gracefully without it rather than depend on it."
        ),
        inputs="masked frames (1, T, 3, H, W) + masks (1, T, 1, H, W)",
        newops=["flow-guided propagation (scatter/gather along flow)",
                "windowed temporal attention"],
        checkout="ProPainter",
    ),

    # ---- diffusion image editing; see the VRAM table in models-to-port.md ----
    Model(
        name="fluxklein", package="FluxKleinRunner",
        uuid="df7e011a-8407-4227-af4d-11f9843eaa40",
        title="FLUX.2-klein-4B", feature="generative fill / outpaint / instruction edit",
        license="Apache-2.0",
        upstream="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
        # A diffusers layout, not one checkpoint: fetched with `hf download` into
        # reference/models rather than through `files` here, because the pipeline
        # wants the whole tree (transformer/, text_encoder/, vae/, scheduler/).
        files=[],
        summary=(
            "Generative fill behind a SAM 2 mask, outpainting to reframe without "
            "cropping, replacing a sign, relighting a subject. 4B parameters, 4 steps.\n\n"
            "**The only edit-capable diffusion model that fits this card at full "
            "precision.** 7.8 GB of transformer in bf16, against 40.9 GB for "
            "Qwen-Image-Edit-2511, 34.2 for HiDream-E1-1 and 23.8 for FLUX.1-Kontext. "
            "Everything else in the category needs 3-4 bit quantisation to fit, which "
            "means judging the model well below its best.\n\n"
            "Not a lesser feature set for being small: the card states text-to-image "
            "and image-to-image multi-reference editing in one unified model."
        ),
        inputs="prompt + 0..n reference images -> latent, denoised 4 steps, VAE decoded",
        newops=["group_norm", "VAE decoder", "sampler loop (host-side)"],
        note="weights in reference/models/FLUX.2-klein-4B via `hf download`; ~16 GB with the text encoder",
    ),
    Model(
        name="qwenimageedit", package="QwenImageEditRunner",
        uuid="dd5f9249-a9bb-4bd8-a662-2fbe122e846d",
        title="Qwen-Image-Edit-2511", feature="instruction image editing (blocked)",
        license="Apache-2.0",
        upstream="https://huggingface.co/Qwen/Qwen-Image-Edit-2511",
        files=[],
        summary=(
            "The model we wanted first, and it does not fit. 40.9 GB of transformer "
            "plus a 16.6 GB Qwen2.5-VL text encoder against a ~16 GB budget.\n\n"
            "Only Q4_0 (11.9 GB) and below leave room for the encoder and the editor, "
            "and Q4_K_M — where quantised quality usually stops hurting — is 13.2 GB "
            "and already too big. So this is blocked on an int4 dequant epilogue in "
            "the GEMM, which is real engine work rather than a model port.\n\n"
            "Kept in the registry because it is the target once quantisation exists."
        ),
        inputs="prompt + image -> edited image",
        newops=["int4 dequant in the GEMM epilogue", "group_norm", "VAE decoder"],
        note="BLOCKED: needs int4. See the VRAM table in models-to-port.md",
    ),
    Model(
        name="zimage", package="ZImageRunner",
        uuid="fc477ab7-e249-4da5-849a-327b5efffa1f",
        title="Z-Image-Turbo (6B)", feature="text to image", license="Apache-2.0",
        upstream="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        files=[],
        summary=(
            "8-step text-to-image, 6B parameters. Published fp32 (24.6 GB), so ~12 GB "
            "converted to bf16 — fits, but only just, and it is generation rather than "
            "editing so it buys title cards and background plates rather than an edit "
            "operation. Second to FLUX.2-klein on both counts."
        ),
        inputs="prompt -> image",
        newops=["group_norm", "VAE decoder"],
        note="fp32 upstream; convert to bf16 before judging VRAM",
    ),
]

BY_NAME = {m.name: m for m in MODELS}


def fetch(model: Model, force=False):
    """Download `model`'s weights into gen/<name>/, skipping what is there."""
    if not model.files:
        print(f"{model.name}: no weight URL — {model.note or 'weights come from the checkout'}")
        return
    model.dir.mkdir(parents=True, exist_ok=True)
    for url, fname in model.files:
        dest = model.dir / fname
        if dest.exists() and not force:
            print(f"  have {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        if url is None:
            # Google Drive. `gdown` exists to handle the interstitial that Drive
            # serves for anything over 25 MB — plain urllib gets that HTML page
            # and writes it out as if it were the weights.
            import gdown
            print(f"  get  {dest.relative_to(ROOT)} <- drive:{model.gdrive}")
            gdown.download(id=model.gdrive, output=str(dest), quiet=True)
        else:
            print(f"  get  {dest.relative_to(ROOT)} <- {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
        print(f"       {dest.stat().st_size / 1e6:.1f} MB")
    if model.unzip:
        import zipfile
        with zipfile.ZipFile(model.dir / model.unzip) as z:
            # The archive was made on a Mac, so it carries __MACOSX resource forks
            # and .DS_Store alongside the real files. Unpacking those would put
            # AppleDouble stubs next to every weight.
            members = [n for n in z.namelist()
                       if not n.startswith("__MACOSX/") and ".DS_Store" not in n
                       and "__pycache__" not in n]
            z.extractall(model.dir, members=members)
        print(f"       unpacked {model.unzip} -> {len(members)} files")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def cmd_list(args):
    for m in MODELS:
        have = "fetched" if m.fetched() and m.files else ("no-url" if not m.files else "missing")
        print(f"{m.name:14} {m.package:20} {have:8} {m.feature:26} {m.license}")


def cmd_fetch(args):
    targets = MODELS if args.all else [BY_NAME[n] for n in args.names]
    for m in targets:
        print(f"{m.name} ({m.title})")
        fetch(m, force=args.force)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="*", choices=[m.name for m in MODELS] + [[]])
    f.add_argument("--all", action="store_true")
    f.add_argument("--force", action="store_true")
    f.set_defaults(fn=cmd_fetch)
    a = p.parse_args()
    if a.cmd == "fetch" and not a.all and not a.names:
        sys.exit("name a model or pass --all")
    a.fn(a)
