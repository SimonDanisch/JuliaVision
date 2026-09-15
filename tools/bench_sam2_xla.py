"""SAM 2.1's encoder through XLA, as the third baseline.

    LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/torch/lib \
        uv run tools/bench_sam2_xla.py --size large

Same wrapper, same precision policy, same `torch.export` call as
`export_sam2_stablehlo.py` makes — that export IS this script's front half, so
the graph XLA runs is the one PyTorch traced. Parity against the refs artifact
first, on the refs' own input, with the band `sam2_pytorch_baseline` pins;
timing afterwards and only if parity holds.

**Attention is math-decomposed here and it is not in the other two.** torchax
lowers no attention op at all, so the fused
`aten._scaled_dot_product_flash_attention` that Lava and PyTorch both run cannot
be expressed — see `export_sam2_stablehlo.py`. What this measures is XLA on the
attention it can be given on this device. Reported next to the other two it
needs that label attached, every time.

**The ROCm plugin needs two sonames the wheels do not ship**, which is
ROCm/TheRock#8163: `librocprofiler-sdk.so.1` and `librocm_smi64.so.1`. Both
exist unversioned in the torch wheel in the same venv, so `LD_LIBRARY_PATH`
pointed at `torch/lib` plus a `.so.1` symlink beside each is enough. Without it
`jax.devices()` silently returns `[CpuDevice(id=0)]` — which is why this script
refuses to run on CPU rather than reporting the number it would get.

# The torchax bug this had to route around

The first six tensors that came back were not the encoder's six: the positional
pyramid arrived shifted a level, `pos` at 128/64/32 where the refs have
256/128/64, with no error raised. `ep.module()(image)` was correct, which said
the export was fine and the fault was downstream, and the interpreter's own
meta-vs-real check found it — every mismatch was on a lifted CONSTANT, rotated
by one, with SAM 2's prompt-encoder buffer sitting where a cached positional
encoding belonged. `orderedstates` below has the cause and the fix.

Worth keeping in mind for the other six models: any program whose lifted
constants do not all sort after its buffers hits this, silently, and a parity
gate is the only thing between that and a published number.
"""

import argparse
import contextlib
import json
import time

import torch

import export_graphs as EG
import export_sam2 as ES
from sam2_pytorch_baseline import KNOWN_DRIFT, outofband, refspath, smclock


def orderedstates(ep):
    """The program's parameters, buffers and constants in PLACEHOLDER order.

    torchax builds this list as `parameters + buffers` and then appends the
    lifted tensor constants at the end
    (`_extract_states_from_exported_program`). The graph does not declare them
    in that order: `input_specs` interleaves them, and `torch.fx.Interpreter`
    binds positionally, so for any program whose constants come before a buffer
    every state lands one place off.

    SAM 2.1's encoder is such a program, and the symptom is not an error. Its
    cached positional encodings are lifted constants, so each one receives its
    neighbour's value and the positional pyramid comes back shifted a level —
    `pos` at 128/64/32 instead of 256/128/64 — while the FPN maps, whose
    constants happen to sit elsewhere, stay right. Reading the states off
    `input_specs` instead fixes it; `ep.module()` was correct all along, which
    is what said the fault was here and not in the export.
    """
    lookup = dict(ep.state_dict)
    lookup.update(getattr(ep, "constants", None) or {})
    lookup.update(getattr(ep, "tensor_constants", None) or {})
    out = []
    for spec in ep.graph_signature.input_specs:
        if str(spec.kind).split(".")[-1] == "USER_INPUT":
            continue
        out.append(lookup[spec.target])
    return out


def pairup(got, names, refs):
    """Each reference output against the XLA result that is actually it.

    **Not by position.** The encoder returns a flat 6-tuple — three FPN maps
    then three positional encodings — and PyTorch hands those back in exactly
    the order the reference manifest lists. `jax.export` does not: it flattens
    the same tuple to a different order, and comparing by index pairs a
    `(1, 256, 128, 128)` result with a `(1, 256, 256, 256)` reference and dies
    on the broadcast.

    Matched on shape, then on content, because shape alone is ambiguous: the
    deepest FPN map and the last positional encoding are both
    `(1, 256, 64, 64)`. A feature map and a positional encoding are nothing like
    each other numerically, so the smallest-difference assignment is the right
    one — and if it were not, every difference would blow up and parity would
    fail, which is the behaviour to want from a guess.
    """
    import numpy as np

    arrays = [np.asarray(g, dtype=np.float32) for g in got]
    taken, per = set(), {}
    for name in names:
        want = refs[f"sam2_encoder/node/{name}"].float().numpy()
        cands = [(float(np.abs(a - want).max()), i) for i, a in enumerate(arrays)
                 if i not in taken and a.shape == want.shape]
        if not cands:
            raise SystemExit(
                f"no XLA output has {name}'s shape {want.shape}; the export "
                f"produced {[a.shape for a in arrays]}")
        d, i = min(cands)
        taken.add(i)
        per[name] = d
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(ES.CKPTS))
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--sdpa", default="flash", choices=["flash", "math"],
                    help="`flash` registers the Pallas lowering and keeps the "
                         "fused op the other runtimes use; `math` decomposes it")
    ap.add_argument("--ramp", type=int, default=50,
                    help="untimed calls before measuring, to bring the clock up")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torchax.export as tex
    from safetensors.torch import load_file

    # `platform` is `"gpu"` on ROCm exactly as it is on CUDA — the ROCm-ness is
    # in `device_kind` ("AMD Radeon 8060S") and the repr. Checking for "rocm"
    # here rejected a perfectly good `RocmDevice`; what matters is only that it
    # is not the CPU fallback, because that fallback is silent.
    devs = jax.devices()
    dev = next((d for d in devs if d.platform != "cpu"), None)
    if dev is None:
        raise SystemExit(
            f"jax is on {devs} — the ROCm plugin did not load, and a CPU number "
            "is not a GPU baseline. See this file's docstring for the two "
            "sonames it needs.")

    if not torch.cuda.is_available():
        raise SystemExit("the export needs a GPU for autocast to apply; see "
                         "export_sam2_stablehlo.py")

    import contextlib

    from torch.nn.attention import SDPBackend, sdpa_kernel
    if a.sdpa == "flash":
        import torchax_flash  # noqa: F401  (registering the lowering is the point)
        sdpactx = contextlib.nullcontext()
    else:
        sdpactx = sdpa_kernel(SDPBackend.MATH)
    model = ES.build(a.size).to("cuda")
    res = model.image_size
    enc = ES.Encoder(model).to("cuda").eval()
    with torch.no_grad(), EG.precision_ctx(a.precision), sdpactx:
        ep = torch.export.export(
            enc, (torch.zeros(1, 3, res, res, device="cuda"),), strict=False
        ).run_decompositions()
    del model, enc
    torch.cuda.empty_cache()

    # `exported_program_to_jax`, not `..._to_stablehlo`: both go through the same
    # interpreter and the same XLA compile, and this one hands back the callable
    # so the states can be supplied in the order the graph actually wants them.
    _, callable_ = tex.exported_program_to_jax(ep)
    states = [jax.device_put(t.detach().cpu().numpy(), dev) for t in orderedstates(ep)]

    path = refspath(a.size)
    if path is None:
        raise SystemExit("no refs.safetensors — parity cannot be checked, so no "
                         "timing is reported")
    manifest = json.loads((path.parent / "refs_manifest.json").read_text())
    names = manifest["graphs"]["sam2_encoder"]["outputs"]
    refs = load_file(str(path))
    image = jax.device_put(refs["sam2_encoder/in0"].numpy(), dev)

    run = jax.jit(lambda w, x: callable_(w, (x,)))

    got = jax.block_until_ready(run(states, image))
    per = pairup(got, names, refs)
    failed = [(n, d) for n, d in per.items() if outofband(n, d)]
    if failed:
        detail = ", ".join(f"{n} {d:.3e}" for n, d in failed)
        raise SystemExit(
            f"refs parity failed: {detail}\n"
            f"  expected every output under 2e-2 except {sorted(KNOWN_DRIFT)}, "
            f"pinned to {KNOWN_DRIFT}.\n"
            "  Note this graph's attention is math-decomposed, so a drift here "
            "need not mean the setup is wrong — but it does mean the number "
            "would not be comparable.")

    for _ in range(a.ramp):
        run(states, image)
    jax.block_until_ready(run(states, image))
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        jax.block_until_ready(run(states, image))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    enc_ms = {"min": ts[0], "p50": ts[len(ts) // 2], "mean": sum(ts) / len(ts)}

    out = {"size": a.size, "res": res, "precision": a.precision, "sdpa": a.sdpa,
           "runtime": "xla", "jax": jax.__version__, "device": dev.device_kind,
           "sm_clock_mhz": smclock(), "parity": {"per_output": per, "failed": []},
           "encode_ms": enc_ms}
    print(json.dumps(out, indent=1))
    dest = path.parent if path.parent.name.startswith("sam2-") else None
    from common import find_root
    dest = find_root() / "gen" / "graphs" / f"sam2-{a.size}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"xla_baseline{'_math' if a.sdpa == 'math' else ''}.json").write_text(json.dumps(out, indent=1))
    print(f"\nencode {enc_ms['p50']:.1f} ms   (XLA, {a.sdpa} attention, p50)")


if __name__ == "__main__":
    main()
