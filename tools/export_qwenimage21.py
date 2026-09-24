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
from torch.export import Dim

import export_graphs as EG
from common import find_root


ROOT = find_root()

# `max_prompt_tokens` from the checkpoint config, which is what the text
# encoder itself is capped at.
QWEN_MAX_PROMPT_TOKENS = 1024

# The image axis is a symbol too, and these are the bounds it is checked
# against. One image token is a 16x16 pixel block, so the range is a 64x64
# thumbnail up to 2048x2048 -- 16 to 16384 tokens. They are BOUNDS and not a
# menu: any latent grid whose token count lands inside gets the same graph, and
# the runner binds the symbol per plan.
#
# Both ends are load-bearing. `min` may not be 1: `torch.export` specialises a
# dimension it can prove is 0 or 1, which silently turns the symbol back into a
# constant. `max` is what the export is CHECKED at, so a graph is only as
# general as this number claims.
#
# **A bound is not a recommendation.** The checkpoint is trained around 1024x1024
# and the rotary grid is centred on the latent extent, so a 64x64 picture is far
# off-distribution and will look like it. The floor is here so the graph does
# not stand in the way, not because the output at the floor is any good.
QWEN_MIN_IMAGE_TOKENS = 16
QWEN_MAX_IMAGE_TOKENS = 16384


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
    """Qwen 2.1 denoiser with host metadata as inputs and tensor-only arguments.

    The prompt length is a graph SYMBOL, not a constant. That matches the
    reference: ``QwenImagePipeline`` pads nothing for a single prompt, and
    ``encode_prompt`` drops the mask outright when it is all ones
    (``pipeline_qwenimage.py``), so ``QwenImage21Transformer2DModel.forward``
    runs at whatever length the prompt has.

    The rotary tables are inputs for the same reason. They are not free of the
    prompt length: ``QwenImage21Rope`` advances a shared position one step per
    text token and then FREEZES the image block's frame axis at the position
    the text reached, so the image's rotary depends on how long the text was.
    Baking them in is what bound the graph to one prompt. They cannot be
    computed inside the graph either -- ``QwenImage21Rope.forward`` uses
    ``.tolist()`` and ``list.index`` -- which is exactly the "host-side
    preparation ``torch.export`` cannot represent" this wrapper exists to
    lift out. The reference calls ``self.pos_embed(...)`` per forward and hands
    the result to the blocks; here the host does it and hands it in.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, prompt_embeddings, timestep, rotary_real, rotary_imag):
        # Both derive from the input shapes, so both follow the symbol rather
        # than pinning it. Text-to-image is one text run followed by the target
        # image, which is what `segments` and `target_mask` say.
        context_tokens = prompt_embeddings.shape[1]
        target_tokens = latents.shape[1]
        target_mask = torch.cat(
            [
                torch.zeros(context_tokens, dtype=torch.bool, device=latents.device),
                torch.ones(target_tokens, dtype=torch.bool, device=latents.device),
            ]
        )
        segments = [(0, context_tokens, True)]
        rotary = (rotary_real, rotary_imag)
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
                segments=segments,
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
    wrapper = StaticTextToImageTransformer(model).eval()
    dtype = next(model.parameters()).dtype
    device = "meta" if args.graph_only else "cpu"
    target_tokens = latent_height * latent_width

    # The example rotary is the real one for `context_tokens`, built on CPU the
    # way the reference builds it. At runtime the host recomputes it for the
    # prompt actually given; `QwenImageRunner.rotarytables` is the port of the
    # same loop.
    example_mask = torch.cat(
        [torch.zeros(context_tokens, dtype=torch.bool), torch.ones(target_tokens, dtype=torch.bool)]
    )
    complex_rope = model.pos_embed([(1, latent_height, latent_width)], example_mask, torch.device("cpu"))
    rope_dtype = torch.float32
    rotary_real = complex_rope.real.contiguous().to(dtype=rope_dtype, device=device)
    rotary_imag = complex_rope.imag.contiguous().to(dtype=rope_dtype, device=device)

    examples = (
        torch.randn(1, target_tokens, model.config.in_channels, dtype=dtype, device=device),
        torch.randn(1, context_tokens, model.config.context_in_dim, dtype=dtype, device=device),
        torch.tensor([0.5], dtype=dtype, device=device),
        rotary_real,
        rotary_imag,
    )
    return wrapper, examples, (latent_height, latent_width, context_tokens)


def export_transformer(module, args):
    torch.manual_seed(0)
    wrapper, examples, shape = build(module, args)
    _, _, context_tokens = shape

    # TWO symbols: the prompt axis and the image axis. The rotary tables span
    # the JOINT sequence, so their length is the SUM of the two -- declared as
    # the affine expression rather than a third free symbol, so export knows the
    # three move together.
    #
    # The image axis used to be the fixed `latent_height * latent_width` of
    # whatever resolution the export was run at, which bound the graph to one
    # picture size and meant a new resolution needed a re-export. Nothing in
    # `StaticTextToImageTransformer.forward` wanted that: it already reads
    # `target_tokens = latents.shape[1]` and builds its mask from it, so the
    # body followed the symbol as soon as the spec stopped pinning it.
    target_tokens = examples[0].shape[1]
    # The bounds have to admit the tracing example: `--smoke` traces a 2x2
    # latent, which is four image tokens and well under the real floor, and
    # `torch.export` rejects a `Dim` its own example violates. Widening for that
    # case rather than raising the smoke resolution keeps the smoke export as
    # cheap as it is meant to be.
    img_lo = min(QWEN_MIN_IMAGE_TOKENS, target_tokens)
    img_hi = max(QWEN_MAX_IMAGE_TOKENS, target_tokens)
    #
    # THREE symbols and not two-plus-an-expression, which is what this wanted to
    # be. `t + target_tokens` worked while the image axis was an integer;
    # `Dim + Dim` is not expressible — `torch.export` supports only "increasing
    # linear operations with integer coefficients" on a `Dim`, so a sum of two
    # free symbols raises. The joint axis therefore gets its own symbol and the
    # graph does not encode that it is the sum. Nothing needs it to: the rotary
    # tables are INPUTS, built host-side by `QwenImageRunner.rotarytables` for
    # the prompt and grid actually being run, so `tj == t + i` is guaranteed by
    # the caller rather than by the graph. The runner binds all three together
    # for that reason.
    t = Dim("t", min=2, max=QWEN_MAX_PROMPT_TOKENS)
    i = Dim("i", min=img_lo, max=img_hi)
    tj = Dim("tj", min=2 + img_lo, max=QWEN_MAX_PROMPT_TOKENS + img_hi)
    dynamic_shapes = ({1: i}, {1: t}, {}, {0: tj}, {0: tj})
    specs = ({1: "i"}, {1: "t"}, {}, {0: "tj"}, {0: "tj"})

    with torch.no_grad():
        reference = wrapper(*examples)
        program = torch.export.export(
            wrapper, examples, dynamic_shapes=dynamic_shapes, strict=False
        ).run_decompositions()

    graph = EG.convert(program, specs, "qwenimage21_transformer")
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
                # Neither length is a number any more: both are symbols, and
                # the values above are only the shapes of the tracing example.
                # The `*_symbol` names are what the runner binds and the
                # `max_*` / `min_*` are the `Dim` bounds the graph was checked
                # against, which is how far it may be bound.
                "image_tokens": None,
                "image_symbol": "i",
                "min_image_tokens": img_lo,
                "max_image_tokens": img_hi,
                "context_tokens": None,
                "context_symbol": "t",
                "max_context_tokens": QWEN_MAX_PROMPT_TOKENS,
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
    # same decode in bf16 takes seconds.
    #
    # "Same weights, one rounding apart" is what this comment used to say next,
    # and it is WRONG. bf16 and fp16 are not one rounding apart: they differ by
    # eight bits of EXPONENT, so bf16 carries fp32's range and fp16 saturates at
    # 65504. Measured 2026-09-23 on a "transparent background" prompt at 1024²:
    # the decode is 60.5% NaN, and the NaN is exactly the background — the object
    # in the middle survives. Scaling the same latents by 0.75 makes it 0% NaN,
    # which is the signature of an overflow rather than a logic fault. A prompt
    # that asks for an opaque background decodes clean, which is why this went
    # unnoticed: the alpha channel is then 1.0 everywhere and nobody looked.
    #
    # So the fp16 cast below costs this port the model's transparent-background
    # output, which is a real capability of the checkpoint (the decoder has four
    # channels and the fourth is a genuine soft matte). Exporting the VAE at
    # fp32 is the fix — its weights are 0.3 GB and the decode is 2.5 s, so
    # neither the artifact nor the runtime is the reason this is fp16.
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

    # The decoder's spatial axes are symbols too, because a resolution-generic
    # denoiser in front of a fixed-size decoder still cannot change resolution.
    # `EG.dims()` rather than two fresh `Dim`s: the decoder's upsampling stages
    # are exact multiples of the latent grid, and deriving them from `h` and `w`
    # is what lets export know that rather than leaving it three unrelated
    # symbols to reconcile.
    #
    # `(1, z, 1, h, w)` — batch, latent channels, the single temporal frame this
    # wrapper already squeezes, then the grid.
    # `Dim.AUTO` and not the `dims()` algebra, which is what this wanted to be.
    # The decoder's upsampling stages compute `min(2*w, 4*w)`-shaped guards, and
    # `torch.export`'s solver will not assume `w > 0`, so it cannot discharge
    # them and rejects an explicit `Dim` with "Not all values of w in the
    # specified range satisfy the generated guard". AUTO asks export to make the
    # axis dynamic where it can and specialise where it must, which is exactly
    # the judgement being asked for. The result is CHECKED below rather than
    # assumed: a silent specialisation here would put the fixed-resolution
    # decoder back without saying so.
    spatial = ({3: torch.export.Dim.AUTO, 4: torch.export.Dim.AUTO},)
    with torch.no_grad():
        program = torch.export.export(
            wrapper, (example,), dynamic_shapes=spatial, strict=False
        ).run_decompositions()

    graph = EG.convert(program, ({3: "h", 4: "w"},), "qwenimage21_vae_decoder")
    # Did AUTO actually keep them dynamic? A decoder that specialised back to the
    # traced grid is the bug this whole change exists to remove, and it would
    # otherwise be discovered by a wrong-sized picture much later.
    latent_shape = next(b["shape"] for b in graph["buffers"] if b["id"] in graph["inputs"])
    dynamic = [d for d in latent_shape if isinstance(d, str)]
    if len(dynamic) < 2:
        raise SystemExit(
            f"the VAE decoder's spatial axes did not stay dynamic: latents are "
            f"{latent_shape}. torch.export specialised them, so this graph is "
            f"pinned to {shape} and `Dim.AUTO` is not enough here."
        )
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
