"""Per-node reference activations for Whisper large-v3-turbo's encoder.

Same contract as `dump_sam2_refs.py` — `"<graph>/in<i>"` and `"<graph>/node/<name>"`
in one safetensors file — so `DNNKernels.verifygraph` reads it without knowing
which model produced it.

    uv run tools/dump_whisper_refs.py

**The input is a real log-mel, not `randn`.** The encoder's first op is a
convolution over 128 mel bins and everything after it is a residual stream whose
scale is set by that first layer, so a Gaussian input puts every activation in a
range the model never sees and makes the tolerance meaningless. The audio is
synthesised from a fixed seed rather than read from a file — a reference you
cannot regenerate identically is not a reference, and this keeps the script
runnable with none of the editor's media installed. It is speech-*shaped*: a
drifting f0 with harmonics, three moving formants, pauses, and a noise floor.

The audio and the mel are both saved (`whisper/audio`, `whisper/mel`). That is
deliberate: step 3 of this port is the log-mel front end, and this file is
already the reference it has to match.

**What gets recorded.** All 617 nodes at (1, 1500, 1280) is ~6 GB, so the
default is a slice that still localises a bug: every node of encoder blocks 0
and 1, the residual stream after every block (all 65 `add.Tensor`), and the
output. `--nodes all` overrides it when a middle block needs bisecting.

Recorded from CUDA with TF32 off, matching `export_whisper.py`'s device — see
that module's docstring for why a CPU export is not equivalent, and note it
applies to the *reference* too: cuDNN's TF32 path reads as ~2e-4 relative error,
the same order as a real bug in a fused kernel.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common

ROOT = common.ROOT
GEN = common.GEN
SR = 16000
WINDOW = 30 * SR


class Recorder(torch.fx.Interpreter):
    """Runs a graph and keeps the intermediates named in `keep`.

    Every result is cloned: safetensors refuses to write aliased tensors, and a
    `.contiguous()` on a view hands back the same storage.
    """

    def __init__(self, gm, keep=None):
        super().__init__(gm)
        self.keep = keep
        self.results = {}

    def store(self, name, t):
        if self.keep is None or name in self.keep:
            self.results[name] = t.detach().cpu().contiguous().clone()

    def run_node(self, n):
        out = super().run_node(n)
        if isinstance(out, torch.Tensor):
            self.store(n.name, out)
        elif isinstance(out, (tuple, list)):
            for i, v in enumerate(out):
                if isinstance(v, torch.Tensor):
                    self.store(f"{n.name}.{i}", v)
        return out


def speechlike(seconds=30.0, seed=0):
    """A deterministic, speech-shaped mono signal at 16 kHz, peak-normalised.

    Not speech — nothing here is transcribable — but it occupies the same part
    of the mel plane that speech does: energy under ~4 kHz, harmonic structure,
    silences, and a floor well below the peak. That is what the encoder's
    activation scale is calibrated against, and the reason this is not `randn`.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR

    # f0 drifts 110 -> 180 Hz over a few seconds, as an utterance does
    f0 = 130.0 + 40.0 * np.sin(2 * np.pi * 0.23 * t) + 15.0 * np.sin(2 * np.pi * 1.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros_like(t)
    for k in range(1, 13):                      # harmonics, rolling off
        sig += np.sin(k * phase) / k**1.3

    # three formants sweeping, which is what puts structure across mel bins
    for fc, bw in ((650.0, 90.0), (1200.0, 120.0), (2600.0, 180.0)):
        f = fc * (1.0 + 0.25 * np.sin(2 * np.pi * 0.31 * t + fc))
        sig += 0.35 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-bw / 4000.0)

    # syllable-rate envelope with real pauses: silence is a distinct regime for
    # the mel (it clamps against the dynamic-range floor) and must be covered
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.3 * t - np.pi / 2)
    env *= (np.sin(2 * np.pi * 0.17 * t) > -0.55)
    sig *= env

    sig += 0.004 * rng.standard_normal(sig.shape)     # noise floor
    sig /= np.abs(sig).max()
    return sig.astype(np.float32)


def blockrange(spec):
    """`"2"` -> blocks 0..1, `"20-22"` -> blocks 20..22."""
    if "-" in str(spec):
        lo, hi = str(spec).split("-")
        return int(lo), int(hi)
    return 0, int(spec) - 1


def keepset(graph, mode, blocks):
    """fx node names to record, as a slice of the exported graph's op ids.

    A *range* rather than a prefix, because bisection does not always start at
    the front: Whisper's residual stream picks up an outlier feature of
    magnitude ~290 around block 21 and that is where an fp16 run diverges, ten
    thousand ops past anything a leading slice records.
    """
    ops = graph["ops"]
    if mode == "all":
        return None
    if mode == "outputs":
        return set(graph["outputs"])
    # `native_layer_norm` returns a tuple, so the graph's output is a getitem
    # *view* rather than an op; the residual stream is the `add.Tensor` chain and
    # is always recorded — it is what localises a block, cheaply, for all 32.
    keep = set(graph["outputs"]) | {o["out"] for o in ops if o["aten"] == "add.Tensor"}
    lo, hi = blockrange(blocks)
    if hi >= lo:
        # `add.Tensor` closes both halves of a block, and one extra `add` ahead of
        # block 0 carries the positional embedding — so block b runs from just
        # past `adds[2b]` to `adds[2b+2]`.
        adds = [i for i, o in enumerate(ops) if o["aten"] == "add.Tensor"]
        a = adds[min(2 * lo, len(adds) - 1)]
        b = adds[min(2 * hi + 2, len(adds) - 1)]
        keep |= {o["out"] for o in ops[a:b + 1]}
    for o in ops:
        if o["out"] in keep and o["aten"].startswith("native_layer_norm"):
            keep.add(o["out"] + ".0")
    return keep


def run(out_path, gdir, nodes, blocks, seed, precision, device):
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA — the graph this diffs against was exported "
                         "from CUDA, and a CPU trace decomposes attention "
                         "differently (see export_whisper.py)")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    import export_graphs as EG
    import export_whisper as EW

    # Under the same policy the graph was exported with. Recording an fp32
    # reference for an fp16 graph reports the dtype difference as a bug — see
    # `dump_sam2_refs.run`, which learned that on the decoder.
    weights = GEN / "whisper"
    model = WhisperForConditionalGeneration.from_pretrained(
        str(weights), torch_dtype=EW.STORAGE[precision], attn_implementation="sdpa",
        local_files_only=True).to(device)

    enc = EW.Encoder(model.model.encoder).eval()

    audio = speechlike(seed=seed)
    fe = WhisperFeatureExtractor.from_pretrained(str(weights), local_files_only=True)
    mel = fe(audio, sampling_rate=SR, return_tensors="pt").input_features.float()
    assert mel.shape[1] == model.config.num_mel_bins, mel.shape
    print(f"mel {tuple(mel.shape)}  min {mel.min():.4f} max {mel.max():.4f} "
          f"mean {mel.mean():.4f} std {mel.std():.4f}")
    args = (mel.to(device=device, dtype=EW.STORAGE[precision]),)

    graph = json.loads((gdir / "whisper.json").read_text())
    keep = keepset(graph, nodes, blocks)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(enc, args, strict=False).run_decompositions()
        rec = Recorder(ep.module(), keep)
        t0 = time.perf_counter()
        rec.run(*args)
        dt = time.perf_counter() - t0

    # The audio and the fp32 mel stay fp32 whatever the graph's precision: they
    # are the front end's reference, and the front end is not the thing being
    # quantised.
    tensors = {"whisper/in0": args[0].cpu().contiguous().clone(),
               "whisper/audio": torch.from_numpy(audio).clone(),
               "whisper/mel": mel.cpu().contiguous().clone()}
    for k, v in rec.results.items():
        tensors[f"whisper/node/{k}"] = v

    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path),
              metadata={"seed": str(seed), "nodes": nodes, "precision": precision})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    manifest = {
        "seed": seed,
        "precision": precision,
        "nodes": nodes,
        "blocks": blocks,
        "sample_rate": SR,
        "input": {"name": graph["inputs"][0], "shape": list(mel.shape)},
        "outputs": graph["outputs"],
        "n_nodes_recorded": len(rec.results),
        "n_ops": len(graph["ops"]),
        "seconds": round(dt, 4),
    }
    (out_path.parent / "refs_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"  {len(rec.results)} intermediates of {len(graph['ops'])} ops, "
          f"forward {dt * 1e3:.1f} ms")
    print(f"{len(tensors)} tensors, {total / 1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", default="slice", choices=["slice", "outputs", "all"])
    ap.add_argument("--blocks", default="2",
                    help="with --nodes slice: leading block count (\"2\") or an inclusive range (\"20-22\")")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "autocast"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gdir = GEN / "graphs" / ("whisper" if a.precision == "fp32" else f"whisper-{a.precision}")
    out = Path(a.out) if a.out else gdir / "refs.safetensors"
    run(out, gdir, a.nodes, a.blocks, a.seed, a.precision, a.device)
