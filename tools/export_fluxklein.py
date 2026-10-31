"""Export FLUX.2-klein (4B or 9B) to DNNKernels JSON and safetensors.

Three graphs, each with the host-side preparation lifted out the way
`export_qwenimage21.py` does it:

* `fluxklein_encoder`: the Qwen3 text encoder's decoder layers on token
  embeddings, with the rotary tables and the attention mask as inputs; returns
  the hidden states after layers 9, 18 and 27 side by side, which is what the
  pipeline hands the transformer as `prompt_embeds` (`_get_qwen3_prompt_embeds`).
* `fluxklein_transformer`: one denoiser evaluation. The 256-channel sinusoidal
  timestep projection and the 4-axis rotary tables are inputs; text and image
  token counts are symbols.
* `fluxklein_vae_decoder`: latents (de-normalised, unpatchified) to pixels.
* `fluxklein_vae_encoder`: a reference image to its latent mean, for editing
  and multi-reference generation.

    .venv/bin/python tools/export_fluxklein.py --smoke --out /tmp/fluxklein_smoke
    .venv/bin/python tools/export_fluxklein.py --model 4B --graph-only --out gen/fluxklein4b

`--smoke` builds one-block, narrow models with random weights on the CPU,
exports them, and writes reference inputs/outputs for the numerical test.
`--graph-only` traces the real configuration on the meta device, so the 8 GB
of weights never enter host RAM; the Julia side reads the Hugging Face
safetensors directly.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.export import Dim

import export_graphs as EG

REPOS = {"4B": "black-forest-labs/FLUX.2-klein-4B", "9B": "black-forest-labs/FLUX.2-klein-9B"}
TEXT_TOKENS = 512               # `tokenizer_max_length`: every prompt is padded to this
HIDDEN_TAPS = (9, 18, 27)       # `text_encoder_out_layers`
TIMESTEP_CHANNELS = 256         # `timestep_guidance_channels`
MAX_IMAGE_TOKENS = 16384        # 2048x2048 at 16 px per token; the model is trained around 1024²
MIN_IMAGE_TOKENS = 16


# ── text encoder ─────────────────────────────────────────────────────────────

class StaticQwen3Encoder(torch.nn.Module):
    """Qwen3's decoder layers on `inputs_embeds`, with the rotary tables and the
    attention mask as inputs (the host builds both), returning the hidden states
    after the `taps` layers concatenated along features. `hidden_states[k]` in
    transformers is the output of layer `k` (index 0 is the embedding), so a tap
    `k` is read after layer index `k - 1`. The final RMSNorm is never applied to
    a tap, so it is not part of the graph."""

    def __init__(self, model, taps):
        super().__init__()
        model.embed_tokens = torch.nn.Identity()
        model.rotary_emb = torch.nn.Identity()
        model.norm = torch.nn.Identity()
        self.model = model
        self.taps = taps

    def forward(self, inputs_embeds, cos, sin, mask):
        hidden = inputs_embeds
        outs = []
        for index, layer in enumerate(self.model.layers):
            hidden = layer(hidden, attention_mask=mask, position_ids=None, past_key_values=None,
                           position_embeddings=(cos, sin))
            if index + 1 in self.taps:
                outs.append(hidden)
        return torch.cat(outs, dim=-1)


def encoder_mask(real_tokens, total=TEXT_TOKENS, dtype=torch.float32):
    """The pipeline's mask as one additive `(1, 1, T, T)` tensor: causal, and the
    padding keys (right padding, `pad_token` after the prompt) masked for every
    query, padding queries included."""
    q = torch.arange(total)[:, None]
    k = torch.arange(total)[None, :]
    allowed = (k <= q) & (k < real_tokens)
    return torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), float("-inf"), dtype=dtype))[None, None]


def export_encoder(args):
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as q3

    torch.manual_seed(0)
    if args.smoke:
        config = Qwen3Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=16, intermediate_size=96, vocab_size=128, max_position_embeddings=1024,
                             rope_theta=1000000.0, rms_norm_eps=1e-6, attention_bias=False)
        taps = (1, 2, 3)
        device = "cpu"
    else:
        config = Qwen3Config.from_pretrained(REPOS[args.model], subfolder="text_encoder")
        taps = HIDDEN_TAPS
        device = "meta"
    config._attn_implementation = "sdpa"
    tokens = TEXT_TOKENS if not args.smoke else 16
    with torch.device(device):
        model = q3.Qwen3Model(config).to(torch.float16).eval()
    hidden = config.hidden_size
    embeds = torch.randn(1, tokens, hidden, dtype=torch.float16, device=device)
    # The rotary tables, built the way the model builds them, on the CPU.
    rotary = q3.Qwen3RotaryEmbedding(config)
    positions = torch.arange(tokens)[None]
    cos, sin = rotary(torch.zeros(1, tokens, hidden), positions)
    cos, sin = cos.to(torch.float16).to(device), sin.to(torch.float16).to(device)
    real = tokens - 5
    mask = encoder_mask(real, tokens, torch.float16).to(device)

    full = None
    if args.smoke:
        with torch.no_grad():
            ids_mask = torch.ones(1, tokens, dtype=torch.long)
            ids_mask[:, real:] = 0
            out = model.float()(inputs_embeds=embeds.float(), attention_mask=ids_mask, output_hidden_states=True,
                                use_cache=False)
            full = torch.cat([out.hidden_states[k] for k in taps], dim=-1)
        model.to(torch.float16)

    wrapper = StaticQwen3Encoder(model, taps).eval()
    with torch.no_grad():
        program = torch.export.export(wrapper, (embeds, cos, sin, mask), strict=False).run_decompositions()
    graph = EG.convert(program, ({}, {}, {}, {}), "fluxklein_encoder")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fluxklein_encoder.json").write_text(json.dumps(graph, indent=1))
    ops = sorted({op["aten"] if "aten" in op else op.get("op", "?") for op in graph["ops"]})
    (args.out / "fluxklein_encoder_ops.txt").write_text("\n".join(ops))
    if args.smoke:
        with torch.no_grad():
            reference = wrapper.float()(embeds.float(), cos.float(), sin.float(), mask.float())
        print(f"encoder wrapper against Qwen3Model: max |diff| = {(reference - full).abs().max().item():.3e}")
        save_file({"inputs_embeds": embeds.float().contiguous(), "cos": cos.float().contiguous(),
                   "sin": sin.float().contiguous(), "mask": mask.float().contiguous(),
                   "prompt_embeds": reference.contiguous()}, str(args.out / "encoder_reference.safetensors"))
        save_file({f"model.{k}": v.to(torch.float16).contiguous() for k, v in model.state_dict().items()},
                  str(args.out / "encoder_smoke_weights.safetensors"))
    return ops


# ── transformer ──────────────────────────────────────────────────────────────

class StaticFluxTransformer(torch.nn.Module):
    """One denoiser evaluation with tensor-only inputs.

    `timestep_proj` is the 256-channel sinusoidal projection of `t * 1000`
    (`Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`), built on the
    host: inside the graph it is an `arange`/`exp` chain over a scalar, which
    buys nothing on the device. `rope_cos`/`rope_sin` are the 4-axis tables for
    the joint text+image sequence (`Flux2PosEmbed` on the text ids then the
    image ids), `(T + I, head_dim)`, built on the host for the same reason as in
    Qwen-Image 2.1: the reference computes them in float64 from position ids.
    The klein models have no guidance embedder (`guidance_embeds=False`)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, prompt_embeddings, timestep_proj, rope_cos, rope_sin):
        m = self.model
        temb = m.time_guidance_embed.timestep_embedder(timestep_proj)
        mod_img = m.double_stream_modulation_img(temb)
        mod_txt = m.double_stream_modulation_txt(temb)
        mod_single = m.single_stream_modulation(temb)
        hidden = m.x_embedder(latents)
        context = m.context_embedder(prompt_embeddings)
        rotary = (rope_cos, rope_sin)
        for block in m.transformer_blocks:
            context, hidden = block(hidden, context, mod_img, mod_txt, image_rotary_emb=rotary)
        text_tokens = context.shape[1]
        hidden = torch.cat([context, hidden], dim=1)
        for block in m.single_transformer_blocks:
            hidden = block(hidden, None, mod_single, image_rotary_emb=rotary)
        hidden = hidden[:, text_tokens:]
        hidden = m.norm_out(hidden, temb)
        return m.proj_out(hidden)


def transformer_rope(model, text_tokens, latent_height, latent_width):
    """The joint rotary tables the pipeline builds: text ids `(0, 0, 0, l)`, image
    ids `(0, h, w, 0)`, each axis through `Flux2PosEmbed`; text first."""
    text_ids = torch.stack([torch.zeros(text_tokens), torch.zeros(text_tokens), torch.zeros(text_tokens),
                            torch.arange(text_tokens)], dim=-1)
    hh, ww = torch.meshgrid(torch.arange(latent_height), torch.arange(latent_width), indexing="ij")
    image_ids = torch.stack([torch.zeros_like(hh), hh, ww, torch.zeros_like(hh)], dim=-1).reshape(-1, 4)
    cos_t, sin_t = model.pos_embed(text_ids)
    cos_i, sin_i = model.pos_embed(image_ids)
    return torch.cat([cos_t, cos_i]), torch.cat([sin_t, sin_i])


def export_transformer(args):
    from diffusers.models.transformers import transformer_flux2 as tf2

    torch.manual_seed(0)
    if args.smoke:
        model = tf2.Flux2Transformer2DModel(
            patch_size=1, in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16,
            num_attention_heads=2, joint_attention_dim=24, timestep_guidance_channels=TIMESTEP_CHANNELS,
            mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, guidance_embeds=False).eval()
        device = "cpu"
        latent_height, latent_width, text_tokens = 2, 3, 5
    else:
        config = json.loads((hf_snapshot(args.model) / "transformer" / "config.json").read_text())
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        with torch.device("meta"):
            model = tf2.Flux2Transformer2DModel(**config).to(torch.float16).eval()
        model.pos_embed = tf2.Flux2PosEmbed(theta=config["rope_theta"], axes_dim=config["axes_dims_rope"])
        device = "meta"
        latent_height, latent_width, text_tokens = 64, 64, TEXT_TOKENS
    dtype = next(model.parameters()).dtype
    image_tokens = latent_height * latent_width
    cos, sin = transformer_rope(model, text_tokens, latent_height, latent_width)
    # `time_proj` of the reference, on the CPU: the host port is `FluxKleinRunner.timestepproj`.
    t = torch.tensor([0.75])
    timestep_proj = model.time_guidance_embed.time_proj(t * 1000).to(dtype)
    examples = (
        torch.randn(1, image_tokens, model.config.in_channels, dtype=dtype, device=device),
        torch.randn(1, text_tokens, model.config.joint_attention_dim, dtype=dtype, device=device),
        timestep_proj.to(device),
        cos.to(dtype).to(device),
        sin.to(dtype).to(device),
    )
    wrapper = StaticFluxTransformer(model).eval()

    img_lo, img_hi = min(MIN_IMAGE_TOKENS, image_tokens), max(MAX_IMAGE_TOKENS, image_tokens)
    txt = Dim("t", min=2, max=TEXT_TOKENS)
    img = Dim("i", min=img_lo, max=img_hi)
    joint = Dim("tj", min=2 + img_lo, max=TEXT_TOKENS + img_hi)
    dynamic_shapes = ({1: img}, {1: txt}, {}, {0: joint}, {0: joint})
    specs = ({1: "i"}, {1: "t"}, {}, {0: "tj"}, {0: "tj"})

    reference = None
    if args.smoke:
        with torch.no_grad():
            reference = wrapper(*examples)
            # The model's own forward, for the wrapper check.
            full = model(hidden_states=examples[0], encoder_hidden_states=examples[1], timestep=t.to(dtype),
                         img_ids=image_ids_like(latent_height, latent_width), txt_ids=text_ids_like(text_tokens),
                         guidance=None, return_dict=False)[0]
            print(f"transformer wrapper against Flux2Transformer2DModel: max |diff| = "
                  f"{(reference - full).abs().max().item():.3e}")
    with torch.no_grad():
        program = torch.export.export(wrapper, examples, dynamic_shapes=dynamic_shapes, strict=False).run_decompositions()
    graph = EG.convert(program, specs, "fluxklein_transformer")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fluxklein_transformer.json").write_text(json.dumps(graph, indent=1))
    ops = sorted({op["aten"] if "aten" in op else op.get("op", "?") for op in graph["ops"]})
    (args.out / "fluxklein_transformer_ops.txt").write_text("\n".join(ops))
    constants = {k: v.detach().cpu().contiguous() for k, v in EG.save_constants(program).items()
                 if v.device.type != "meta"}
    if args.smoke:
        tensors = {k: v.detach().cpu().contiguous() for k, v in wrapper.state_dict().items()}
        tensors.update(constants)
        save_file(tensors, str(args.out / "transformer_smoke_weights.safetensors"))
        save_file({"latents": examples[0].contiguous(), "prompt_embeddings": examples[1].contiguous(),
                   "timestep_proj": examples[2].contiguous(), "rope_cos": examples[3].contiguous(),
                   "rope_sin": examples[4].contiguous(), "output": reference.contiguous()},
                  str(args.out / "transformer_reference.safetensors"))
    elif constants:
        save_file(constants, str(args.out / "transformer_constants.safetensors"))
    return ops


def image_ids_like(latent_height, latent_width):
    hh, ww = torch.meshgrid(torch.arange(latent_height), torch.arange(latent_width), indexing="ij")
    return torch.stack([torch.zeros_like(hh), hh, ww, torch.zeros_like(hh)], dim=-1).reshape(1, -1, 4)


def text_ids_like(text_tokens):
    return torch.stack([torch.zeros(text_tokens), torch.zeros(text_tokens), torch.zeros(text_tokens),
                        torch.arange(text_tokens)], dim=-1).reshape(1, -1, 4).long()


# ── VAE decoder ──────────────────────────────────────────────────────────────

class StaticVAEDecoder(torch.nn.Module):
    """`AutoencoderKLFlux2._decode`: `post_quant_conv` then the decoder. The
    batch-norm de-normalisation and the 2x2 unpatchify are the host's
    (`_unpack_latents_with_ids`, `_unpatchify_latents`); the graph takes the
    `(1, latent_channels, h, w)` latent grid and returns `(1, 3, 8h, 8w)` in
    [-1, 1]. Float32 throughout: the config says `force_upcast`."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae._decode(latents, return_dict=False)[0]


def export_vae_decoder(args):
    from diffusers.models.autoencoders import autoencoder_kl_flux2 as akl

    torch.manual_seed(0)
    if args.smoke:
        vae = akl.AutoencoderKLFlux2(block_out_channels=(8, 16, 16, 16), layers_per_block=1, latent_channels=4,
                                     norm_num_groups=4, sample_size=64).eval()
        device = "cpu"
        h, w = 3, 2
    else:
        config = json.loads((hf_snapshot(args.model) / "vae" / "config.json").read_text())
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        with torch.device("meta"):
            vae = akl.AutoencoderKLFlux2(**config).eval()
        device = "meta"
        h, w = 64, 64
    latents = torch.randn(1, vae.config.latent_channels, h, w, dtype=torch.float32, device=device)
    wrapper = StaticVAEDecoder(vae).eval()
    hd = Dim("h", min=2, max=256)
    wd = Dim("w", min=2, max=256)
    with torch.no_grad():
        reference = wrapper(latents) if args.smoke else None
        program = torch.export.export(wrapper, (latents,), dynamic_shapes=({2: hd, 3: wd},),
                                      strict=False).run_decompositions()
    graph = EG.convert(program, ({2: "h", 3: "w"},), "fluxklein_vae_decoder")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fluxklein_vae_decoder.json").write_text(json.dumps(graph, indent=1))
    ops = sorted({op["aten"] if "aten" in op else op.get("op", "?") for op in graph["ops"]})
    (args.out / "fluxklein_vae_decoder_ops.txt").write_text("\n".join(ops))
    if args.smoke:
        tensors = {k: v.detach().cpu().contiguous() for k, v in wrapper.state_dict().items()}
        tensors.update({k: v.detach().cpu().contiguous() for k, v in EG.save_constants(program).items()})
        save_file(tensors, str(args.out / "vae_decoder_smoke_weights.safetensors"))
        save_file({"latents": latents.contiguous(), "image": reference.contiguous()},
                  str(args.out / "vae_decoder_reference.safetensors"))
    return ops


# ── VAE encoder (reference images) ───────────────────────────────────────────

class StaticVAEEncoder(torch.nn.Module):
    """`AutoencoderKLFlux2._encode` and the posterior's `mode()`: the encoder,
    `quant_conv`, and the first `latent_channels` channels of the result (the
    mean; `retrieve_latents(..., sample_mode="argmax")` is what the pipeline
    asks for). `(1, 3, 8h, 8w)` in [-1, 1] to `(1, latent_channels, h, w)`.
    The 2x2 patchify and the batch-norm normalisation are the host's, mirroring
    the decoder's side."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        moments = self.vae._encode(image)
        return moments[:, : self.vae.config.latent_channels]


def export_vae_encoder(args):
    from diffusers.models.autoencoders import autoencoder_kl_flux2 as akl

    torch.manual_seed(0)
    if args.smoke:
        vae = akl.AutoencoderKLFlux2(block_out_channels=(8, 16, 16, 16), layers_per_block=1, latent_channels=4,
                                     norm_num_groups=4, sample_size=64).eval()
        device = "cpu"
        h, w = 3, 2
    else:
        config = json.loads((hf_snapshot(args.model) / "vae" / "config.json").read_text())
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        with torch.device("meta"):
            vae = akl.AutoencoderKLFlux2(**config).eval()
        device = "meta"
        h, w = 64, 64
    image = torch.rand(1, 3, 8 * h, 8 * w, dtype=torch.float32, device=device) * 2 - 1
    wrapper = StaticVAEEncoder(vae).eval()
    hd = Dim("h", min=2, max=256)
    wd = Dim("w", min=2, max=256)
    with torch.no_grad():
        reference = wrapper(image) if args.smoke else None
        program = torch.export.export(wrapper, (image,), dynamic_shapes=({2: 8 * hd, 3: 8 * wd},),
                                      strict=False).run_decompositions()
    graph = EG.convert(program, ({2: "h", 3: "w"},), "fluxklein_vae_encoder")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fluxklein_vae_encoder.json").write_text(json.dumps(graph, indent=1))
    ops = sorted({op["aten"] if "aten" in op else op.get("op", "?") for op in graph["ops"]})
    (args.out / "fluxklein_vae_encoder_ops.txt").write_text("\n".join(ops))
    if args.smoke:
        tensors = {k: v.detach().cpu().contiguous() for k, v in wrapper.state_dict().items()}
        tensors.update({k: v.detach().cpu().contiguous() for k, v in EG.save_constants(program).items()})
        save_file(tensors, str(args.out / "vae_encoder_smoke_weights.safetensors"))
        save_file({"image": image.contiguous(), "latents": reference.contiguous()},
                  str(args.out / "vae_encoder_reference.safetensors"))
    return ops


# ── driver ───────────────────────────────────────────────────────────────────

def hf_snapshot(model):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPOS[model], allow_patterns=["*/config.json", "model_index.json"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(REPOS), default="4B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny random models on the CPU, with references")
    ap.add_argument("--graph-only", action="store_true", help="real configuration on the meta device")
    ap.add_argument("--part", choices=["encoder", "transformer", "vae_decoder", "vae_encoder", "all"], default="all")
    args = ap.parse_args()
    if not (args.smoke or args.graph_only):
        raise SystemExit("pick --smoke or --graph-only; the real weights are read from the Hugging Face files")
    parts = ["encoder", "transformer", "vae_decoder", "vae_encoder"] if args.part == "all" else [args.part]
    exported = {}
    for part in parts:
        exported[part] = {"encoder": export_encoder, "transformer": export_transformer,
                          "vae_decoder": export_vae_decoder, "vae_encoder": export_vae_encoder}[part](args)
        print(f"{part}: {len(exported[part])} distinct ops")
    (args.out / "ops.json").write_text(json.dumps(exported, indent=1))


if __name__ == "__main__":
    main()
