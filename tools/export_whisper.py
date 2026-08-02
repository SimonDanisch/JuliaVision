"""
Export Whisper large-v3-turbo's encoder to DNNKernels' graph JSON + weights.

Same shape as `export_basicvsrpp.py`: load the module, name the inputs, export,
hand the `ExportedProgram` to `export_graphs.convert`, which is model-agnostic.

**Encoder only, and that is the honest half of the model.** The encoder is one
static-shaped forward over a 30 s window and it is where the compute is. The
decoder is autoregressive with a KV cache, which `torch.export` will not capture
as one graph and the runtime cannot execute yet — that is the piece of engine
work Whisper exists to force, and it needs a design, not a trace. Getting the
encoder running and matching first means the decoder work starts against a
runtime that already has the weights loaded and the mel front end settled.

`sdpa` rather than eager attention on purpose: it stays one
`_scaled_dot_product_*` op, which DNNKernels already implements, so the encoder
needs no new ops at all.

**Export from CUDA.** `export_sam2.device` documents why and it bites here just
as hard: `run_decompositions()` keeps attention whole on CUDA, but on CPU there
is no fused kernel to dispatch to, so it decomposes into the math form — an
explicit softmax over a materialised attention matrix, plus the
`logical_not`/`where` pair guarding fully-masked rows. Whisper's encoder is 32
blocks of (1, 20, 1500, 1500) attention; materialised that is ~180 MB apiece.
Exported from CPU this graph has 64 `bmm` and no `sdpa`; from CUDA it has 32
`sdpa` and no `bmm`.

    uv run tools/export_whisper.py
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_graphs as EG

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "gen"
WEIGHTS = GEN / "whisper"


class Encoder(torch.nn.Module):
    """The encoder as a plain callable returning one tensor.

    `WhisperEncoder.forward` returns a `BaseModelOutput` dataclass, which
    `torch.export` treats as a pytree with a non-tensor structure. Unwrapping it
    here keeps the graph's outputs a flat tuple, which is what `convert` expects
    and what the Julia side reads.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        return self.encoder(input_features).last_hidden_state


def main(out: Path, precision: str, dev: str):
    from transformers import WhisperForConditionalGeneration

    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA — see the module docstring; a CPU export "
                         "silently decomposes attention. Pass --device cpu to force it.")

    model = WhisperForConditionalGeneration.from_pretrained(
        str(WEIGHTS), torch_dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=True).to(dev)
    enc = Encoder(model.model.encoder).eval()

    # (batch, mel bins, frames): large-v3 uses 128 mel bins and a fixed 3000-frame
    # window, which is 30 s at 100 frames/s. Static — Whisper's encoder has no
    # variable length, audio is padded to the window.
    n_mels = model.config.num_mel_bins
    args = (torch.randn(1, n_mels, 3000, device=dev),)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(enc, args, strict=False)
        ep = ep.run_decompositions()   # must stay inside the precision context

    g = EG.convert(ep, ({},), "whisper")   # {} = all dims static
    out.mkdir(parents=True, exist_ok=True)
    (out / "whisper.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(enc.named_parameters()),
                            **dict(enc.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"whisper encoder: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights")
    print(f"  mel bins {n_mels}, output {tuple(ep.example_outputs[0].shape) if hasattr(ep,'example_outputs') else '(1,1500,d)'}")
    print("  ops:", json.dumps(hist, sort_keys=True))
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "whisper")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = p.parse_args()
    main(a.out, a.precision, a.device)
