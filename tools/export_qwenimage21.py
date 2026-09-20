"""Export the Qwen-Image 2.1 denoiser to DNNKernels JSON and safetensors.

This targets the *2.1* single-stream transformer.  It is deliberately a static
text-to-image graph: prompt length and latent resolution are bound at export
time, while latents, prompt embeddings, and timestep remain graph inputs.  The
static wrapper also removes Python/data-dependent mask construction from the
Diffusers forward; the resulting graph contains the model arithmetic rather
than host-side preparation that ``torch.export`` cannot represent.

Current Diffusers releases may not yet contain ``QwenImage21Transformer2DModel``.
Point ``--diffusers-source`` at a current Diffusers checkout in that case:

    .venv/bin/python tools/export_qwenimage21.py --smoke \
        --diffusers-source /path/to/diffusers/src

The smoke configuration is a one-block, small-width numerical/export test.  A
real export downloads the official BF16 transformer (about 14 GB):

    .venv/bin/python tools/export_qwenimage21.py \
        --diffusers-source /path/to/diffusers/src --height 1024 --width 1024

The Comfy-Org INT8+ConvRot file is not accepted here: it is not an ordinary
Diffusers state dict and must retain its quantization metadata through a native
DNNKernels weight path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_graphs as EG
from common import find_root


ROOT = find_root()


def import_qwen21(diffusers_source: Path | None):
    if diffusers_source is not None:
        source = diffusers_source.expanduser().resolve()
        if not (source / "diffusers").is_dir():
            raise SystemExit(f"--diffusers-source must contain diffusers/, got {source}")
        sys.path.insert(0, str(source))
    try:
        import diffusers.models.transformers.transformer_qwenimage21 as module
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Qwen-Image 2.1 needs a Diffusers build containing "
            "QwenImage21Transformer2DModel; use --diffusers-source with a "
            "current Diffusers src/ checkout. That checkout currently needs "
            "huggingface-hub>=1.31 and transformers>=5.17. Original error: "
            f"{exc}"
        ) from exc
    return module


def real_rotary(x, freqs_cis, **_kwargs):
    """Complex-free equivalent of Diffusers' Qwen 2.1 RoPE.

    DNNKernels tensors are real.  Diffusers' default implementation temporarily
    views adjacent lanes as complex numbers; spelling out that multiplication
    produces bit-identical fp32 output and leaves ordinary real ATen ops in the
    exported graph.
    """
    cos, sin = freqs_cis
    real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out_real = real * cos[None, :, None] - imag * sin[None, :, None]
    out_imag = imag * cos[None, :, None] + real * sin[None, :, None]
    return torch.stack((out_real, out_imag), dim=-1).flatten(3).to(x.dtype)


class StaticTextToImageTransformer(torch.nn.Module):
    """Qwen 2.1 denoiser with static host metadata and tensor-only inputs."""

    def __init__(self, model, context_tokens: int, latent_height: int, latent_width: int):
        super().__init__()
        self.model = model
        target_tokens = latent_height * latent_width

        # Text-to-image has one text run followed by the target image.  Keep
        # these as plain attributes: export lifts them as constants, and
        # EG.save_constants serializes the corresponding graph weights.
        target_mask = torch.cat(
            [torch.zeros(context_tokens, dtype=torch.bool), torch.ones(target_tokens, dtype=torch.bool)]
        )
        complex_rope = model.pos_embed(
            [(1, latent_height, latent_width)], target_mask, torch.device("cpu")
        )
        self.rotary = (complex_rope.real.contiguous(), complex_rope.imag.contiguous())
        self.target_mask = target_mask
        self.segments = [(0, context_tokens, True)]

    def forward(self, latents, prompt_embeddings, timestep):
        hidden = self.model.img_in(latents)
        context = self.model.txt_in(prompt_embeddings)
        joint = torch.cat((context, hidden), dim=1)

        # causal_condition=True: the target uses the sampled time and the text
        # prefix uses the appended t=0 row.
        timesteps = torch.cat((timestep, timestep.new_zeros(1)))
        temb = self.model.time_text_embed(timesteps, hidden)
        modulation = self.model.modulation(temb)

        for block in self.model.transformer_blocks:
            joint = block(
                hidden_states=joint,
                modulation=modulation,
                rotary_emb=self.rotary,
                attention_mask=None,
                target_token_mask=self.target_mask,
                layer_cache=None,
                kv_cache_mode=None,
                cache_write_slice=None,
                segments=self.segments,
                key_valid=None,
            )

        joint = self.model.norm_out(joint, temb, self.target_mask)
        output = self.model.proj_out(joint)
        return output[:, -latents.shape[1] :]


def build(module, args):
    if args.smoke:
        model = module.QwenImage21Transformer2DModel(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=24,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        ).eval()
        latent_height, latent_width, context_tokens = 2, 2, 3
    else:
        model = module.QwenImage21Transformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        if args.layers is not None:
            if not 1 <= args.layers <= len(model.transformer_blocks):
                raise SystemExit(f"--layers must be in 1..{len(model.transformer_blocks)}")
            model.transformer_blocks = torch.nn.ModuleList(model.transformer_blocks[: args.layers])
        if args.height % 16 or args.width % 16:
            raise SystemExit("--height and --width must be divisible by the VAE stride, 16")
        latent_height, latent_width = args.height // 16, args.width // 16
        context_tokens = args.context_tokens

    # Replace only the representation of complex multiplication.  This was
    # checked against Diffusers exactly; it does not alter model arithmetic.
    module.apply_rotary_emb_qwen = real_rotary
    wrapper = StaticTextToImageTransformer(model, context_tokens, latent_height, latent_width).eval()
    dtype = next(model.parameters()).dtype
    examples = (
        torch.randn(1, latent_height * latent_width, model.config.in_channels, dtype=dtype),
        torch.randn(1, context_tokens, model.config.context_in_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
    )
    return wrapper, examples, (latent_height, latent_width, context_tokens)


def export_transformer(module, args):
    torch.manual_seed(0)
    wrapper, examples, shape = build(module, args)
    with torch.no_grad():
        reference = wrapper(*examples)
        program = torch.export.export(wrapper, examples, strict=False).run_decompositions()

    graph = EG.convert(program, ({}, {}, {}), "qwenimage21_transformer")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwenimage21_transformer.json").write_text(json.dumps(graph, indent=1))

    tensors = {key: value.detach().cpu().contiguous() for key, value in wrapper.state_dict().items()}
    tensors.update(
        {key: value.detach().cpu().contiguous() for key, value in EG.save_constants(program).items()}
    )
    save_file(tensors, str(out / "transformer.safetensors"))
    save_file(
        {
            "latents": examples[0].cpu().contiguous(),
            "prompt_embeddings": examples[1].cpu().contiguous(),
            "timestep": examples[2].cpu().contiguous(),
            "output": reference.cpu().contiguous(),
        },
        str(out / "transformer_reference.safetensors"),
    )

    requested = {buffer["key"] for buffer in graph["buffers"] if buffer["kind"] == "weight"}
    missing = sorted(requested - tensors.keys())
    if missing:
        raise SystemExit(f"exported graph references {len(missing)} unsaved weights: {missing[:5]}")

    histogram = {}
    for op in graph["ops"]:
        histogram[op["aten"]] = histogram.get(op["aten"], 0) + 1
    (out / "transformer_op_histogram.json").write_text(json.dumps(histogram, indent=1, sort_keys=True))
    lh, lw, context = shape
    (out / "qwenimage21_export.json").write_text(
        json.dumps(
            {
                "model": "random-smoke" if args.smoke else args.model,
                "image_height": lh * 16,
                "image_width": lw * 16,
                "latent_height": lh,
                "latent_width": lw,
                "image_tokens": lh * lw,
                "context_tokens": context,
                "layers": len(wrapper.model.transformer_blocks),
                "dtype": str(next(wrapper.parameters()).dtype).removeprefix("torch."),
                "patch_size": 1,
            },
            indent=1,
        )
    )
    print(
        f"qwenimage21_transformer: {len(graph['ops'])} ops, "
        f"{sum(p.numel() for p in wrapper.parameters()) / 1e6:.2f}M parameters"
    )
    print(f"  latent {lh}x{lw} ({lh * lw} tokens), context {context}, output {tuple(reference.shape)}")
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "gen" / "graphs" / "qwenimage21")
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--layers", type=int, default=None, help="export only the first N real blocks for bring-up")
    parser.add_argument("--smoke", action="store_true", help="small random one-block export; downloads no weights")
    parser.add_argument("--diffusers-source", type=Path, default=None, help="path to a Diffusers src/ checkout")
    args = parser.parse_args()
    export_transformer(import_qwen21(args.diffusers_source), args)


if __name__ == "__main__":
    main()
