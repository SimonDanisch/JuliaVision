"""SAM 2.1's encoder as StableHLO, for the XLA baseline.

    uv run tools/export_sam2_stablehlo.py --size large

**The same `torch.export` call `export_sam2.py` makes**, under the same
precision policy, on the same wrapper — only the lowering differs: `torchax`
turns the ExportedProgram into StableHLO where `EG.convert` turns it into our
JSON. That is what makes the three numbers comparable. Lava, PyTorch eager and
XLA then all run the graph this one export produced, and a difference between
them is a difference between runtimes rather than between traces.

Exported ON THE GPU and not on the CPU, because `precision_ctx("autocast")` is
`autocast(device_type="cuda")`: with CPU tensors it applies to nothing, the
`_to_copy` casts never appear, and what came out would be an fp32 graph wearing
an autocast label. The weights are moved to the host afterwards, which is only
a transfer.

Writes, next to the graphs:

    sam2_encoder.mlir                the module — weights are arguments, not
                                     inlined constants, so this stays small
    sam2_encoder_states.safetensors  those weights, keyed by their position in
                                     the argument list @main declares
    sam2_encoder_xla.json            every argument and result, in order, with
                                     shape and dtype

`@main` takes the weights first and the image last (`in_avals` order); the JSON
records that rather than leaving the caller to rediscover it.

**Attention is the one place the graph cannot be identical, and `--sdpa` is
where that is decided.** `export_sam2.py` exports from CUDA so that
`scaled_dot_product_attention` survives as a single
`aten._scaled_dot_product_flash_attention`, which is what DNNKernels implements
as `fused.sdpa`. torchax registers 434 aten lowerings and not one of them is an
attention op, so that graph cannot be lowered to StableHLO at all — the export
stops with `AssertionError: aten._scaled_dot_product_flash_attention.default`.

`--sdpa math` forces the decomposition instead: an explicit softmax over a
materialised attention matrix, plus the `logical_not`/`where` pair that guards
fully-masked rows. Those are ops torchax does lower. It is also a DIFFERENT
algorithm for the attention blocks than Lava and PyTorch run here, so a number
measured this way says "what XLA achieves on the attention it can be given on
this device", not "what XLA does with the same kernels" — and it must be
labelled that way wherever it is reported.

What makes it possible here at all is the memory: the math form materialises a
`(1, 8, 4096, 4096)` fp16 tensor, 512 MiB, per global-attention block, and
`export_sam2.py`'s `device()` notes that this needs ~15 GB and does not fit on
their 20 GB card. This APU has 112 GB of unified memory, so the objection that
made it impractical there does not apply.
"""

import argparse
import json
from pathlib import Path

import torch

import export_graphs as EG
import export_sam2 as ES

from common import find_root
ROOT = find_root()


def avaldesc(aval):
    return {"shape": list(aval.shape), "dtype": str(aval.dtype)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(ES.CKPTS))
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"])
    ap.add_argument("--sdpa", default="math", choices=["math", "fused"],
                    help="`math` decomposes attention into ops torchax can "
                         "lower; `fused` keeps the op DNNKernels runs and fails "
                         "to lower. See this file's docstring")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "no GPU visible to torch. The export itself would run on the CPU, but "
            "autocast would then apply to nothing and the graph would come out "
            "fp32 — see this file's docstring.")

    import torchax.export as tex

    out = Path(a.out) if a.out else ROOT / "gen" / "graphs" / f"sam2-{a.size}"
    out.mkdir(parents=True, exist_ok=True)

    model = ES.build(a.size).to("cuda")
    res = model.image_size
    enc = ES.Encoder(model).to("cuda").eval()
    image = torch.zeros(1, 3, res, res, device="cuda")

    # `run_decompositions()` inside the same context, for the reason
    # `EG.precision_ctx` gives: called outside it re-traces and drops every cast.
    import contextlib
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sdpa = sdpa_kernel(SDPBackend.MATH) if a.sdpa == "math" else contextlib.nullcontext()
    with torch.no_grad(), EG.precision_ctx(a.precision), sdpa:
        ep = torch.export.export(enc, (image,), strict=False).run_decompositions()

    # torchax reads the program's tensors through jax, which is host-side here,
    # so hand it a program whose weights live on the host.
    ep = ep._apply_to_tensors(lambda t: t.cpu()) if hasattr(ep, "_apply_to_tensors") else ep
    states, exported = tex.exported_program_to_stablehlo(ep)

    mlir = exported.mlir_module
    mlir = mlir if isinstance(mlir, str) else mlir()
    (out / "sam2_encoder.mlir").write_text(mlir)

    from safetensors.torch import save_file
    import numpy as np
    save_file({f"states/{i:04d}": torch.from_numpy(np.asarray(s).copy())
               for i, s in enumerate(states)},
              str(out / "sam2_encoder_states.safetensors"))

    meta = {"size": a.size, "res": res, "precision": a.precision, "sdpa": a.sdpa,
            "nstates": len(states),
            "in_avals": [avaldesc(x) for x in exported.in_avals],
            "out_avals": [avaldesc(x) for x in exported.out_avals],
            "argument_order": "states in file order, then the image",
            "torch": torch.__version__}
    (out / "sam2_encoder_xla.json").write_text(json.dumps(meta, indent=1))
    print(f"mlir {len(mlir)/2**20:.1f} MiB, {len(states)} states, "
          f"{len(exported.out_avals)} results -> {out}")


if __name__ == "__main__":
    main()
