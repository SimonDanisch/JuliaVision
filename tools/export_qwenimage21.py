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
        import diffusers.models.transformers.transformer_qwenimage21 as transformer_module
        import diffusers.models.autoencoders.autoencoder_kl_qwenimage21 as vae_module
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Qwen-Image 2.1 needs a Diffusers build containing "
            "QwenImage21Transformer2DModel; use --diffusers-source with a "
            "current Diffusers src/ checkout. That checkout currently needs "
            "huggingface-hub>=1.31 and transformers>=5.17. Original error: "
            f"{exc}"
        ) from exc
    return transformer_module, vae_module


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
        target_mask = self.target_mask.to(latents.device)
        rotary = tuple(value.to(latents.device) for value in self.rotary)
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
                rotary_emb=rotary,
                attention_mask=None,
                target_token_mask=target_mask,
                layer_cache=None,
                kv_cache_mode=None,
                cache_write_slice=None,
                segments=self.segments,
                key_valid=None,
            )

        joint = self.model.norm_out(joint, temb, target_mask)
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
    elif args.graph_only:
        config_path = args.config_dir / "transformer" / "config.json"
        if not config_path.is_file():
            raise SystemExit(f"missing transformer config at {config_path}")
        config = json.loads(config_path.read_text())
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        with torch.device("meta"):
            # Mantle's cooperative-matrix path is Float16. Compact quantized
            # weights decode into that compute type, so bind the allocation-free
            # graph to Float16 rather than emitting unsupported BF16 buffers.
            model = module.QwenImage21Transformer2DModel(**config).to(dtype=torch.float16).eval()
        # Rope's frequency tables are constants, not parameters. Construct them
        # on CPU so the graph-only export can serialize them while the 7B
        # parameter set remains allocation-free on the meta device.
        model.pos_embed = module.QwenImage21Rope(
            theta=10000, axes_dim=list(model.config.axes_dims_rope)
        )
        model.time_text_embed.time_proj = module.QwenImage21TemporalTimesteps(timestep_dim=256)
        if args.layers is not None:
            if not 1 <= args.layers <= len(model.transformer_blocks):
                raise SystemExit(f"--layers must be in 1..{len(model.transformer_blocks)}")
            model.transformer_blocks = torch.nn.ModuleList(model.transformer_blocks[: args.layers])
        if args.height % 16 or args.width % 16:
            raise SystemExit("--height and --width must be divisible by the VAE stride, 16")
        latent_height, latent_width = args.height // 16, args.width // 16
        context_tokens = args.context_tokens
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
    device = "meta" if args.graph_only else "cpu"
    examples = (
        torch.randn(1, latent_height * latent_width, model.config.in_channels, dtype=dtype, device=device),
        torch.randn(1, context_tokens, model.config.context_in_dim, dtype=dtype, device=device),
        torch.tensor([0.5], dtype=dtype, device=device),
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

    if args.graph_only:
        tensors = {
            key: value.detach().cpu().contiguous()
            for key, value in EG.save_constants(program).items()
            if value.device.type != "meta"
        }
        save_file(tensors, str(out / "transformer_constants.safetensors"))
    else:
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
    if args.graph_only:
        missing = [key for key in missing if not key.startswith("model.")]
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


class StaticRotary(torch.nn.Module):
    """Qwen3-VL's rotary tables for one fixed prompt length, as constants.

    The real module multiplies its inverse frequencies by the position ids and
    recomposes the mRoPE sections; at a bound length that is a constant, and
    computing it on CPU is what lets a meta-device export serialize it. Same
    values as the module it replaces — they come *from* it.
    """

    def __init__(self, rotary, position_ids, dtype):
        super().__init__()
        with torch.no_grad():
            cos, sin = rotary(torch.zeros(1, dtype=dtype), position_ids)
        self.tables = (cos.contiguous(), sin.contiguous())

    def forward(self, x, position_ids):
        return tuple(table.to(x.device) for table in self.tables)


class StaticQwen3VLTextEncoder(torch.nn.Module):
    """Qwen3-VL's language model over a fixed-length text prompt.

    Three things are bound here rather than computed per call.

    **The final RMSNorm is dropped.**  What the denoiser was trained on is the
    last *decoder layer's* output; the pipeline neutralizes ``model.norm`` with
    a forward hook for exactly this reason (see ``_get_qwen_prompt_embeds``).
    Removing it is the same value and one fewer op.

    **The input is ``inputs_embeds``, not ``input_ids``.**  The embedding table
    is 151936x4096 - 622M entries, a fifth of the checkpoint - and a prompt
    reads forty rows of it.  The runner gathers those rows on the host and the
    table never reaches the device.

    **Positions are static.**  Qwen3-VL's mRoPE splits the head dimension into
    (24, 20, 20) lanes fed by three position streams; for text the three agree,
    so the rotary tables fold to constants at export.
    """

    def __init__(self, text_model, tokens: int):
        super().__init__()
        text_model.norm = torch.nn.Identity()
        self.model = text_model
        positions = torch.arange(tokens).view(1, 1, tokens).expand(3, 1, tokens)
        self.position_ids = positions.contiguous()

    def forward(self, inputs_embeds):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=self.position_ids.to(inputs_embeds.device),
            attention_mask=None,
            use_cache=False,
        )
        return outputs.last_hidden_state


def build_text_encoder(args):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3vl

    config_path = args.config_dir / "text_encoder" / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing text encoder config at {config_path}")
    config = json.loads(config_path.read_text())
    text_config = qwen3vl.Qwen3VLTextConfig(**config["text_config"])
    # `sdpa`, and NOT the default: `flash_attention_2` is not exportable and
    # `eager` writes the causal mask as an additive fp16 tensor whose `-inf`
    # rows the graph then carries. Both reach the same numbers here.
    text_config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = qwen3vl.Qwen3VLTextModel(text_config).to(dtype=torch.float16).eval()
    if args.layers is not None:
        if not 1 <= args.layers <= len(model.layers):
            raise SystemExit(f"--layers must be in 1..{len(model.layers)}")
        model.layers = torch.nn.ModuleList(model.layers[: args.layers])
    # The rotary tables are a constant, not a parameter: build them on CPU so a
    # meta-device export can serialize them.
    positions = torch.arange(args.prompt_tokens).view(1, 1, args.prompt_tokens).expand(3, 1, -1)
    model.rotary_emb = StaticRotary(qwen3vl.Qwen3VLTextRotaryEmbedding(text_config),
                                    positions.contiguous(), torch.float16)
    wrapper = StaticQwen3VLTextEncoder(model, args.prompt_tokens).eval()
    example = torch.randn(1, args.prompt_tokens, text_config.hidden_size,
                          dtype=torch.float16, device="meta")
    return wrapper, (example,), text_config


def export_text_encoder(args):
    torch.manual_seed(0)
    wrapper, examples, text_config = build_text_encoder(args)
    with torch.no_grad():
        program = torch.export.export(wrapper, examples, strict=False).run_decompositions()

    graph = EG.convert(program, ({},), "qwenimage21_text_encoder")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwenimage21_text_encoder.json").write_text(json.dumps(graph, indent=1))
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in EG.save_constants(program).items()
        if value.device.type != "meta"
    }
    save_file(tensors, str(out / "text_encoder_constants.safetensors"))

    requested = {buffer["key"] for buffer in graph["buffers"] if buffer["kind"] == "weight"}
    missing = [key for key in sorted(requested - tensors.keys()) if not key.startswith("model.")]
    if missing:
        raise SystemExit(f"exported graph references {len(missing)} unsaved constants: {missing[:5]}")

    histogram = {}
    for op in graph["ops"]:
        histogram[op["aten"]] = histogram.get(op["aten"], 0) + 1
    (out / "text_encoder_op_histogram.json").write_text(json.dumps(histogram, indent=1, sort_keys=True))
    (out / "qwenimage21_text_encoder_export.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt_tokens": args.prompt_tokens,
                "layers": len(wrapper.model.layers),
                "hidden_size": text_config.hidden_size,
                "num_attention_heads": text_config.num_attention_heads,
                "num_key_value_heads": text_config.num_key_value_heads,
                "dtype": "float16",
            },
            indent=1,
        )
    )
    print(f"qwenimage21_text_encoder: {len(graph['ops'])} ops, "
          f"{args.prompt_tokens} tokens, {len(wrapper.model.layers)} layers")
    print(f"  wrote {out}")


class NormalizedVAEDecoder(torch.nn.Module):
    """Decode scheduler-space latents and return the single image frame."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        dtype = next(vae.parameters()).dtype
        self.latents_mean = torch.tensor(vae.config.latents_mean, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
        self.latents_std = torch.tensor(vae.config.latents_std, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)

    def forward(self, latents):
        latents = latents * self.latents_std + self.latents_mean
        return self.vae.decode(latents, return_dict=False)[0][:, :, 0]


def build_vae(module, args):
    if args.smoke:
        vae = module.AutoencoderKLQwenImage21(
            base_dim=32,
            decoder_base_dim=32,
            z_dim=4,
            dim_mult=[1, 2],
            num_res_blocks=1,
            temperal_downsample=[False],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
            is_residual=False,
            in_channels=3,
            out_channels=3,
            scale_factor_temporal=1,
            scale_factor_spatial=2,
        ).eval()
        latent_height = latent_width = 2
    else:
        vae = module.AutoencoderKLQwenImage21.from_pretrained(
            args.model,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        if args.height % 16 or args.width % 16:
            raise SystemExit("--height and --width must be divisible by the VAE stride, 16")
        latent_height, latent_width = args.height // 16, args.width // 16
    # The reference runs in the checkpoint's own BF16 and the graph is exported
    # in Float16: Mantle's cooperative-matrix path is fp16 and DNNKernels has no
    # bfloat16 buffer, but torch has no fp16 CPU convolution either — a 1024²
    # decode in fp16 on the host does not finish in twenty minutes, while the
    # same decode in bf16 takes seconds. Same weights, one rounding apart.
    reference = None
    if not args.no_reference:
        with torch.no_grad():
            example = torch.randn(1, vae.config.z_dim, 1, latent_height, latent_width,
                                  dtype=next(vae.parameters()).dtype)
            reference = NormalizedVAEDecoder(vae).eval()(example).float()
    vae = vae.to(torch.float16)
    wrapper = NormalizedVAEDecoder(vae).eval()
    example = torch.randn(1, vae.config.z_dim, 1, latent_height, latent_width,
                          dtype=torch.float16) if reference is None else example.half()
    return wrapper, example, (latent_height, latent_width), reference


def export_vae(module, args):
    torch.manual_seed(0)
    wrapper, example, shape, reference = build_vae(module, args)
    with torch.no_grad():
        program = torch.export.export(wrapper, (example,), strict=False).run_decompositions()

    graph = EG.convert(program, ({},), "qwenimage21_vae_decoder")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwenimage21_vae_decoder.json").write_text(json.dumps(graph, indent=1))

    available = {key: value.detach().cpu().contiguous() for key, value in wrapper.state_dict().items()}
    available.update(
        {key: value.detach().cpu().contiguous() for key, value in EG.save_constants(program).items()}
    )
    requested = {buffer["key"] for buffer in graph["buffers"] if buffer["kind"] == "weight"}
    missing = sorted(requested - available.keys())
    if missing:
        raise SystemExit(f"exported VAE graph references {len(missing)} unsaved weights: {missing[:5]}")
    tensors = {key: available[key] for key in requested}
    save_file(tensors, str(out / "vae.safetensors"))
    if reference is not None:
        save_file(
            {"latents": example.float().cpu().contiguous(),
             "image": reference.cpu().contiguous()},
            str(out / "vae_reference.safetensors"),
        )

    histogram = {}
    for op in graph["ops"]:
        histogram[op["aten"]] = histogram.get(op["aten"], 0) + 1
    (out / "vae_op_histogram.json").write_text(json.dumps(histogram, indent=1, sort_keys=True))
    lh, lw = shape
    print(
        f"qwenimage21_vae_decoder: {len(graph['ops'])} ops, "
        f"{sum(value.numel() for value in tensors.values()) / 1e6:.2f}M saved parameters"
    )
    print(f"  latent {lh}x{lw}" +
          ("" if reference is None else f", output {tuple(reference.shape)}"))
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "gen" / "graphs" / "qwenimage21")
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--component", choices=("transformer", "vae", "text_encoder", "all"),
                        default="all")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the host reference decode for the VAE")
    parser.add_argument("--prompt-tokens", type=int, default=64,
                        help="static prompt length for the text encoder export")
    parser.add_argument("--layers", type=int, default=None, help="export only the first N real blocks for bring-up")
    parser.add_argument("--smoke", action="store_true", help="small random one-block export; downloads no weights")
    parser.add_argument("--graph-only", action="store_true", help="export from a meta model without downloading BF16 weights")
    parser.add_argument("--config-dir", type=Path, default=Path("/home/sim/.cache/JuliaVision/Qwen-Image-2.1-config"))
    parser.add_argument("--diffusers-source", type=Path, default=None, help="path to a Diffusers src/ checkout")
    args = parser.parse_args()
    transformer_module, vae_module = import_qwen21(args.diffusers_source)
    if args.component in ("transformer", "all"):
        export_transformer(transformer_module, args)
    if args.component in ("vae", "all"):
        export_vae(vae_module, args)
    if args.component in ("text_encoder", "all"):
        export_text_encoder(args)


if __name__ == "__main__":
    main()
