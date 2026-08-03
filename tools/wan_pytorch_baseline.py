"""
PyTorch reference generation with Wan's own code, for a machine without flash-attn.

    uv run tools/wan_pytorch_baseline.py --frame_num 5 --sample_steps 50 \
        --prompt "..." --save_file gen/wan_pytorch_baseline.mp4

Everything after the script name goes straight to `Wan2.2/generate.py`; this
wrapper only fixes the dispatch and then hands over.

Why a wrapper at all: `wan/modules/attention.py` has a perfectly good
`attention()` that falls back to `scaled_dot_product_attention` when flash-attn
is missing — but `model.py` never calls it. It does
`from .attention import flash_attention` and calls *that* directly (lines 145 and
175), and `flash_attention` opens with `assert FLASH_ATTN_2_AVAILABLE`. So the
fallback exists and is unreachable, and the run dies 2.5 minutes in, at the first
self-attention of the first step, after loading 20 GB of weights.

SDPA is not an approximation here. Both compute exact attention; they differ in
tiling and memory traffic, not in what they compute. This is also the same
substitution the DNNKernels export was verified against, so the reference this
produces is comparable to the DNNKernels numbers rather than a second baseline.

The patch has to land on the *binding in model.py*, not on `attention.py` — by
the time this runs, `from .attention import flash_attention` has already copied
the name into model.py's namespace, and rebinding the source module would not
touch it.
"""
import runpy
import sys
from pathlib import Path

WAN = Path(__file__).resolve().parent.parent / "dev" / "Wan2.2"


def patch_attention():
    """Point every direct `flash_attention` caller at the SDPA fallback.

    Returns the modules patched, so the caller can say what it did rather than
    assume it worked — a silent no-op here reappears as the same AssertionError
    much later.
    """
    import wan.modules.attention as att

    sdpa = att.attention
    patched = []
    for name in ("wan.modules.model", "wan.modules", "wan.distributed.ulysses"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "flash_attention", None) is not None:
            mod.flash_attention = sdpa
            patched.append(name)
    # `attention()` dispatches to `flash_attention` when it believes flash-attn
    # is present. It is not, but say so explicitly rather than rely on the probe.
    att.FLASH_ATTN_2_AVAILABLE = False
    att.FLASH_ATTN_3_AVAILABLE = False
    return patched


def patch_vae_cpu():
    """Run the VAE decode on the host.

    720P is where this card runs out: the transformer is offloaded before the
    decode (`textimage2video.py:396`), so the ~14 GB still resident is the
    decoder's own activations, and with the desktop holding 3.3 GB there is not
    enough left. Fragmentation was a red herring — `expandable_segments:True`
    cut reserved-but-unallocated from 1.30 GB to 250 MB and it still died, just
    deeper in.

    Decoding on the host trades minutes for the 720P frames the GPU cannot hold.
    It is one decode per generation, against 50 sampling steps that stay on the
    GPU, so it does not touch the number worth measuring.
    """
    import torch
    from wan.modules.vae2_2 import Wan2_2_VAE

    def decode(self, zs):
        self.model.to("cpu").float()
        self.scale = [s.to("cpu").float() for s in self.scale]
        torch.cuda.empty_cache()
        # fp32 on the host: autocast here is CUDA's, so it would not apply, and
        # bf16 convolutions on CPU are slower than fp32 rather than faster.
        return [
            self.model.decode(u.unsqueeze(0).to("cpu").float(), self.scale)
            .float().clamp_(-1, 1).squeeze(0)
            for u in zs
        ]

    Wan2_2_VAE.decode = decode


def register_size(size):
    """Allow a size Wan does not ship a preset for.

    `generate.py` asserts `args.size in SUPPORTED_SIZES[task]`, and TI2V-5B ships
    only the two 720P orientations. Nothing in the model requires that — it is
    convolutional with rotary position embeddings, so it takes any grid — but
    720P is what it was trained at, and away from that resolution the pictures
    get worse. This is the knob for fitting on a smaller card, not a free win.

    Both extents must be divisible by 32: the VAE strides by 16 and the patch
    embedding takes 2x2 of the latent, so anything else silently truncates a row
    of patches.
    """
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

    w, h = (int(v) for v in size.split("*"))
    for name, v in (("width", w), ("height", h)):
        v % 32 == 0 or sys.exit(f"[baseline] {name} {v} is not a multiple of 32 "
                                "(VAE stride 16 x patch 2)")
    SIZE_CONFIGS[size] = (w, h)
    MAX_AREA_CONFIGS[size] = w * h
    for task, sizes in list(SUPPORTED_SIZES.items()):
        if size not in sizes:
            SUPPORTED_SIZES[task] = tuple(sizes) + (size,)
    return w, h


def main():
    sys.path.insert(0, str(WAN))
    argv = sys.argv[1:]
    vae_cpu = "--vae_cpu" in argv
    argv = [a for a in argv if a != "--vae_cpu"]

    import wan  # noqa: F401  - populates the submodules patch_attention looks for

    patched = patch_attention()
    print(f"[baseline] flash_attention -> scaled_dot_product_attention in: {patched}",
          flush=True)
    if not patched:
        raise SystemExit("[baseline] nothing patched - generate.py would die at the "
                         "first attention call; check the module names above")
    if vae_cpu:
        patch_vae_cpu()
        print("[baseline] VAE decode on CPU", flush=True)
    if "--size" in argv:
        size = argv[argv.index("--size") + 1]
        w, h = register_size(size)
        print(f"[baseline] size {w}x{h} registered", flush=True)

    sys.argv = ["generate.py"] + argv
    runpy.run_path(str(WAN / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
