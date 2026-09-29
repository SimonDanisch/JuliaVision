"""Export the parts of Qwen-Image 2.1 that condition on reference images.

`export_qwenimage21.py` covers text to image: a prompt run through Qwen3-VL's
language model, a denoiser that sees one text run followed by the target image,
and the VAE decoder. A reference image touches all three stages, so this adds:

* ``vision``: Qwen3-VL's vision tower over ONE image's patches, returning the
  merged tokens that replace the prompt's ``<|image_pad|>`` slots and the three
  DeepStack features the language model adds after its first three layers.
* ``vl_encoder``: the language model with M-RoPE position ids and the DeepStack
  features as inputs. The text-only encoder bakes positions ``0..T-1`` in as a
  constant table, which is wrong as soon as an image sits in the prompt.
* ``vae_encoder``: the VAE encoder over an RGBA image, returning normalised
  latents, the ``_encode_vae_image`` path of the pipeline.
* ``prefix`` and ``step``: the denoiser, split the way the reference runs it
  with ``use_kv_cache=True`` (its default). Text and reference-image tokens are
  modulated from ``t = 0`` and never attend to the target, so their keys and
  values are the same at every step. The prefix is computed once, one segment
  at a time, and each denoising step runs only the target's tokens against the
  cached keys and values.

Why segments. The prefix mask is block-causal: text runs are causal, each
reference image is one bidirectional block, and every token sees everything
before it. The reference's default processor (`QwenImage21AttnProcessor`) runs
exactly this as one attention per segment over the keys ``[0, end)``. Doing the
same here with the whole stack per segment, and handing each segment the keys
and values of the segments before it, is the same arithmetic in a different
loop order: no segment reads anything that comes after it, at any layer. It
keeps the graphs free of data-dependent structure, and it keeps the reference
images out of the unmasked flash path's way: an image segment and every
denoising step attend with no mask at all, and only the short text runs carry
a causal one.

    .venv/bin/python tools/export_qwenimage21_condition.py --check
    .venv/bin/python tools/export_qwenimage21_condition.py --smoke
    .venv/bin/python tools/export_qwenimage21_condition.py --component all
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.export import Dim

import export_graphs as EG
import export_qwenimage21 as Q
from artifacts import artifact
from common import find_root

ROOT = find_root()
CONFIG_DIR = Path("/home/sim/.cache/JuliaVision/Qwen-Image-2.1-config")

# The model card: "Support up to 10 reference images". Twelve is what the runner
# admits; the bounds below are sized for it.
MAX_REFERENCE_IMAGES = 12
# A reference image is resized to about 1024x1024 in area: at most 64x64 = 4096
# latent tokens and 1024 vision-language tokens. The bounds are the area with
# room for rounding to multiples of 32 at extreme aspect ratios.
MAX_IMAGE_TOKENS = 16384
MAX_VL_TOKENS_PER_IMAGE = MAX_IMAGE_TOKENS // 4
# The longest joint prefix: every reference image at the bound plus the prompt.
MAX_PREFIX_TOKENS = MAX_REFERENCE_IMAGES * 4096 + 4096


# ---------------------------------------------------------------------------
# Denoiser
# ---------------------------------------------------------------------------

def attend_block(module, block, joint, modulation, rotary, past_k, past_v, causal):
    """One `QwenImage21TransformerBlock` over `joint`, attending to `past_*` first.

    The reference block's own arithmetic, spelled out so the attention can take
    the cached keys and values as tensors: `_modulate`, then
    `_qwenimage21_prepare_qkv`, then the segment attention of
    `QwenImage21AttnProcessor` (keys `[past; own]`, a causal triangle over the
    own keys for text), then the MLP. `modulation` is one row for every token,
    so `_select_modulation_rows` reduces to a broadcast.

    Returns the block's output and the rotated keys and values of `joint`, which
    are what a later segment reads.
    """
    mod1, mod2 = modulation.chunk(2, dim=-1)
    img_modulated, gate1 = block._modulate(block.img_norm1(joint), mod1, None)

    attn = block.attn
    query = attn.to_q(img_modulated).unflatten(-1, (attn.heads, -1))
    key = attn.to_k(img_modulated).unflatten(-1, (attn.heads, -1))
    value = attn.to_v(img_modulated).unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)
    query = Q.real_rotary(query, rotary)
    key = Q.real_rotary(key, rotary)

    keys = key if past_k is None else torch.cat([past_k, key], dim=1)
    values = value if past_v is None else torch.cat([past_v, value], dim=1)
    mask = None
    if causal:
        n = query.shape[1]
        own = torch.tril(torch.ones(n, n, dtype=torch.bool, device=query.device))
        if past_k is not None:
            before = torch.ones(n, past_k.shape[1], dtype=torch.bool, device=query.device)
            own = torch.cat([before, own], dim=1)
        mask = own[None, None]
    out = module.dispatch_attention_fn(query, keys, values, attn_mask=mask, dropout_p=0.0, backend=None)
    out = out.flatten(2, 3).type_as(query)
    out = attn.to_out[0](out)
    joint = joint + gate1.tanh() * out

    img_modulated2, gate2 = block._modulate(block.img_norm2(joint), mod2, None)
    joint = joint + gate2.tanh() * block.img_mlp(img_modulated2)
    if joint.dtype == torch.float16:
        joint = joint.clip(-65504, 65504)
    return joint, key, value


class PrefixSegment(torch.nn.Module):
    """One prefix segment through every block, returning its keys and values.

    `kind` is `"text"` (prompt embeddings, `txt_in`, causal) or `"image"`
    (reference latents, `img_in`, bidirectional). `past` is the flat list
    `(k1, v1, ..., kL, vL)` of every earlier segment's keys and values, empty
    for the first segment, which is always text: the template opens with
    `<|im_start|>user\\n` before any image.

    Every prefix token is modulated from `t = 0` (`causal_condition`), so the
    timestep is not an input: the zero row is computed here, as the reference
    computes it for the trailing row of its `temb`.
    """

    def __init__(self, module, model, kind, has_past):
        super().__init__()
        self.module = module
        self.model = model
        self.kind = kind
        self.has_past = has_past

    def forward(self, x, rotary_real, rotary_imag, past=()):
        m = self.model
        joint = m.txt_in(x) if self.kind == "text" else m.img_in(x)
        zero = torch.zeros(1, dtype=joint.dtype, device=joint.device)
        modulation = m.modulation(m.time_text_embed(zero, joint))
        rotary = (rotary_real, rotary_imag)
        out = []
        for layer, block in enumerate(m.transformer_blocks):
            past_k = past[2 * layer] if self.has_past else None
            past_v = past[2 * layer + 1] if self.has_past else None
            joint, key, value = attend_block(self.module, block, joint, modulation, rotary,
                                             past_k, past_v, causal=self.kind == "text")
            out += [key, value]
        return tuple(out)


class TargetStep(torch.nn.Module):
    """One denoising step over the target's tokens against the cached prefix.

    The reference's `kv_cache_mode="cached"` forward: the target attends to the
    whole prefix and to itself with no mask, and every token takes the sampled
    timestep's modulation row.
    """

    def __init__(self, module, model):
        super().__init__()
        self.module = module
        self.model = model

    def forward(self, latents, timestep, rotary_real, rotary_imag, past):
        m = self.model
        joint = m.img_in(latents)
        temb = m.time_text_embed(timestep.to(joint.dtype), joint)
        modulation = m.modulation(temb)
        rotary = (rotary_real, rotary_imag)
        for layer, block in enumerate(m.transformer_blocks):
            joint, _, _ = attend_block(self.module, block, joint, modulation, rotary,
                                       past[2 * layer], past[2 * layer + 1], causal=False)
        joint = m.norm_out(joint, temb, None)
        return m.proj_out(joint)


def smoke_transformer(module):
    return module.QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, context_in_dim=24, mlp_ratio=2, axes_dims_rope=(4, 6, 6),
    ).eval()


def run_segmented(module, model, latents, prompt, timestep, img_shapes, img_mask):
    """The reference forward, recomputed as prefix segments plus one target step."""
    m = model
    target_tokens = math.prod(img_shapes[-1])
    # The joint sequence's image positions, as the reference's `forward` expands
    # them: every vision-language image slot stands for 2x2 latent tokens.
    repeats = torch.where(img_mask, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
    rotary = m.pos_embed(img_shapes, image_pad_mask, device=latents.device)
    rotary_real, rotary_imag = rotary.real.contiguous(), rotary.imag.contiguous()
    image_ids, target_mask = m.build_token_metadata(image_pad_mask, img_shapes)
    prefix_len = int((~target_mask).sum())

    # The prefix's tokens in joint order: text rows from the prompt, image rows
    # from the reference latents, in the order the slots appear.
    text_rows = prompt[0][~img_mask[0][: prompt.shape[1]]]
    past = []
    text_cursor, latent_cursor = 0, 0
    for start, end, is_text in module._qwenimage21_prefix_segments(image_ids, prefix_len):
        n = end - start
        if is_text:
            x = text_rows[text_cursor : text_cursor + n][None]
            text_cursor += n
        else:
            x = latents[:, latent_cursor : latent_cursor + n]
            latent_cursor += n
        seg = PrefixSegment(module, m, "text" if is_text else "image", bool(past))
        kv = seg(x, rotary_real[start:end], rotary_imag[start:end], tuple(past))
        past = list(kv) if not past else [torch.cat([p, s], dim=1) for p, s in zip(past, kv)]
    step = TargetStep(module, m)
    target = latents[:, latent_cursor:]
    assert target.shape[1] == target_tokens
    return step(target, timestep, rotary_real[prefix_len:], rotary_imag[prefix_len:], tuple(past))


def check_decomposition(module):
    """The segmented prefix plus the target step against the reference forward,
    on a random two-block model with two reference images between text runs."""
    torch.manual_seed(0)
    # The reference keeps its complex rotary; `attend_block` calls the real
    # spelling itself, so nothing is patched for either side.
    model = smoke_transformer(module).float()
    # Text, an image of 2x4 latents (two slots), text, an image of 4x2, text,
    # then the target's slots appended the way the pipeline appends them.
    text_before, text_between, text_after = 5, 3, 7
    shapes = [(1, 2, 4), (1, 4, 2), (1, 4, 4)]
    vl = ([False] * text_before + [True] * 2 + [False] * text_between + [True] * 2 +
          [False] * text_after)
    prompt = torch.randn(1, len(vl), 24)
    img_mask = torch.tensor([vl + [True] * (16 // 4)])
    latents = torch.randn(1, 8 + 8 + 16, 4)
    timestep = torch.tensor([0.37])
    with torch.no_grad():
        reference = model(hidden_states=latents, encoder_hidden_states=prompt, timestep=timestep,
                          img_shapes=[shapes], img_mask=img_mask, return_dict=False)[0][:, -16:]
        ours = run_segmented(module, model, latents, prompt, timestep, shapes, img_mask)
    err = (reference - ours).abs().max().item()
    print(f"segmented prefix + step against the reference forward: max |diff| = {err:.3e}")
    if err > 1e-5:
        raise SystemExit("the segmented denoiser does not reproduce the reference")


def real_transformer(module, args):
    """The 2.1 denoiser on the meta device at Float16, as `export_qwenimage21.py
    --graph-only` builds it: the weights come from the compact checkpoint at
    load time, and only the constants are real."""
    config_path = args.config_dir / "transformer" / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"missing transformer config at {config_path}")
    config = {k: v for k, v in json.loads(config_path.read_text()).items() if not k.startswith("_")}
    with torch.device("meta"):
        model = module.QwenImage21Transformer2DModel(**config).to(dtype=torch.float16).eval()
    model.pos_embed = module.QwenImage21Rope(theta=10000, axes_dim=list(model.config.axes_dims_rope))
    model.time_text_embed.time_proj = module.QwenImage21TemporalTimesteps(timestep_dim=256)
    return model


def write_graph(out, program, specs, name, constants_file):
    """Convert, write the JSON, and write the lifted constants.

    The constants are renamed under the graph's own name. The prefix and step
    graphs are loaded into one `Model` with one weight dictionary, and torch
    numbers lifted constants per program, so two graphs' `lifted_tensor_0` would
    otherwise be one key with two values.
    """
    graph = EG.convert(program, specs, name)
    constants = EG.save_constants(program)
    for buffer in graph["buffers"]:
        if buffer["kind"] == "weight" and buffer["key"] in constants:
            buffer["key"] = f"{name}.{buffer['key']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(graph, indent=1))
    tensors = {f"{name}.{k}": v.detach().cpu().contiguous() for k, v in constants.items()
               if v.device.type != "meta"}
    if tensors:
        save_file(tensors, str(out / constants_file))
    histogram = {}
    for op in graph["ops"]:
        histogram[op["aten"]] = histogram.get(op["aten"], 0) + 1
    print(f"{name}: {len(graph['ops'])} ops, inputs {len(graph['inputs'])}, "
          f"outputs {len(graph['outputs'])}, {len(tensors)} constants")
    return graph, tensors


def export_denoiser(module, args):
    """The four denoiser graphs: the first text segment, a later text segment, a
    reference-image segment, and the denoising step."""
    torch.manual_seed(0)
    if args.smoke:
        model = smoke_transformer(module).to(torch.float16)
        device, dtype = "cpu", torch.float16
    else:
        model = real_transformer(module, args)
        device, dtype = "meta", torch.float16
    cfg = model.config
    layers = len(model.transformer_blocks)
    heads, head_dim = cfg.num_attention_heads, cfg.attention_head_dim
    half = head_dim // 2

    # Distinct example lengths, so no two symbols are unified by an accident of
    # the trace, and none of them 0 or 1, which `torch.export` specialises.
    n_text, n_image, n_past, n_target = (7, 24, 13, 16) if args.smoke else (37, 256, 300, 1024)

    def rotary(n):
        return (torch.randn(n, half, device=device), torch.randn(n, half, device=device))

    def past(p):
        return tuple(torch.randn(1, p, heads, head_dim, dtype=dtype, device=device) for _ in range(2 * layers))

    t = Dim("n_text", min=2, max=Q.QWEN_MAX_PROMPT_TOKENS + 64)
    s = Dim("n_image", min=Q.QWEN_MIN_IMAGE_TOKENS, max=MAX_IMAGE_TOKENS)
    p = Dim("p", min=2, max=MAX_PREFIX_TOKENS)
    i = Dim("i", min=Q.QWEN_MIN_IMAGE_TOKENS, max=Q.QWEN_MAX_IMAGE_TOKENS)
    past_dims = tuple({1: p} for _ in range(2 * layers))
    past_specs = tuple({1: "p"} for _ in range(2 * layers))

    cases = [
        ("qwenimage21_prefix_text0", PrefixSegment(module, model, "text", False),
         (torch.randn(1, n_text, cfg.context_in_dim, dtype=dtype, device=device), *rotary(n_text)),
         ({1: t}, {0: t}, {0: t}), ({1: "n"}, {0: "n"}, {0: "n"})),
        ("qwenimage21_prefix_text", PrefixSegment(module, model, "text", True),
         (torch.randn(1, n_text, cfg.context_in_dim, dtype=dtype, device=device), *rotary(n_text),
          past(n_past)),
         ({1: t}, {0: t}, {0: t}, past_dims), ({1: "n"}, {0: "n"}, {0: "n"}, *past_specs)),
        ("qwenimage21_prefix_image", PrefixSegment(module, model, "image", True),
         (torch.randn(1, n_image, cfg.in_channels, dtype=dtype, device=device), *rotary(n_image),
          past(n_past)),
         ({1: s}, {0: s}, {0: s}, past_dims), ({1: "n"}, {0: "n"}, {0: "n"}, *past_specs)),
        ("qwenimage21_step", TargetStep(module, model),
         (torch.randn(1, n_target, cfg.in_channels, dtype=dtype, device=device),
          torch.tensor([0.5], dtype=dtype, device=device), *rotary(n_target), past(n_past)),
         ({1: i}, {}, {0: i}, {0: i}, past_dims), ({1: "i"}, {}, {0: "i"}, {0: "i"}, *past_specs)),
    ]
    out = args.out
    for name, wrapper, examples, dynamic, specs in cases:
        with torch.no_grad():
            program = torch.export.export(wrapper.eval(), examples, dynamic_shapes=dynamic,
                                          strict=False).run_decompositions()
        write_graph(out, program, specs, name, f"{name}_constants.safetensors")
    meta = {
        "model": "random-smoke" if args.smoke else args.model,
        "layers": layers, "heads": heads, "head_dim": head_dim,
        "max_prefix_tokens": MAX_PREFIX_TOKENS, "max_image_tokens": MAX_IMAGE_TOKENS,
        "max_reference_images": MAX_REFERENCE_IMAGES,
        "graphs": [c[0] for c in cases],
    }
    (out / "qwenimage21_condition_export.json").write_text(json.dumps(meta, indent=1))
    if args.smoke:
        write_denoiser_reference(module, model, out)


def write_denoiser_reference(module, model, out):
    """A full reference forward on the smoke model, for the runner's parity test:
    two reference images between text runs, the layout a real edit prompt has.

    The reference runs in Float32 on the Float16 weights, so it is the exact
    value the Float16 graphs approximate rather than a second rounding of it.
    """
    torch.manual_seed(1)
    text_before, text_between, text_after = 5, 3, 7
    shapes = [(1, 4, 8), (1, 8, 4), (1, 4, 4)]
    vl = ([False] * text_before + [True] * 8 + [False] * text_between + [True] * 8 +
          [False] * text_after)
    prompt = torch.randn(1, len(vl), model.config.context_in_dim)
    img_mask = torch.tensor([vl + [True] * 4])
    latents = torch.randn(1, 32 + 32 + 16, model.config.in_channels)
    timestep = torch.tensor([0.37])
    reference_model = model.float()
    with torch.no_grad():
        noise = reference_model(hidden_states=latents, encoder_hidden_states=prompt, timestep=timestep,
                                img_shapes=[shapes], img_mask=img_mask, return_dict=False)[0][:, -16:]
        repeats = torch.where(img_mask, 4, 1)[0]
        pad = torch.repeat_interleave(img_mask[0], repeats)
        rope = reference_model.pos_embed(shapes, pad, device=latents.device)
    save_file({
        "prompt": prompt.contiguous(), "latents": latents.contiguous(), "timestep": timestep,
        "img_mask": img_mask.to(torch.uint8).contiguous(),
        "img_shapes": torch.tensor(shapes, dtype=torch.int64),
        "rotary_real": rope.real.contiguous(), "rotary_imag": rope.imag.contiguous(),
        "noise": noise.contiguous(),
    }, str(out / "condition_reference.safetensors"))
    # The smoke weights, which the parity test loads into the same graphs.
    save_file({k: v.detach().to(torch.float16).contiguous() for k, v in model.state_dict().items()},
              str(out / "condition_smoke_weights.safetensors"))
    model.to(torch.float16)
    print(f"  wrote the smoke reference: noise {tuple(noise.shape)}")


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------

# The compact encoder checkpoint, which carries the vision tower's BF16 weights
# under `model.visual.*` next to the language model. The same five artifacts the
# runner merges.
ENCODER_SHARDS = [
    "96083ec98dc539c839f8c8733bc6fc9befb98d3d", "75db2d83c0009d49e085689207b6cac0b64354b6",
    "b339514b80597881715488c4c788ef2f6d654ab1", "1a4aec105a559a9ab2922eed87f43f6c95b9cc49",
    "6ba234ff4813ac171b462dcade9cb92c79a49e3b",
]


def encoder_config(args):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    path = args.config_dir / "text_encoder" / "config.json"
    if not path.is_file():
        raise SystemExit(f"missing text encoder config at {path}")
    return Qwen3VLConfig(**json.loads(path.read_text()))


def compact_visual_state():
    """The vision tower's weights out of the compact checkpoint, as Float32."""
    from safetensors import safe_open

    state = {}
    for h in ENCODER_SHARDS:
        for f in (Path.home() / ".julia" / "artifacts" / h).glob("*.safetensors"):
            with safe_open(str(f), "pt") as fh:
                for k in fh.keys():
                    if k.startswith("model.visual."):
                        state[k[len("model.visual."):]] = fh.get_tensor(k).float()
    if not state:
        raise SystemExit("no `model.visual.*` tensors found in the compact encoder artifacts")
    return state


def vision_block(block, x, cos, sin):
    """`Qwen3VLVisionBlock` over one image: full attention, no `cu_seqlens`.

    The residual stream stays Float32: on a real picture it reaches 1.5e4 after
    the last block, a quarter of Float16's range, and a LayerNorm squares it.
    The attention's operands are normalised and rotated, and go through the
    attention in Float16, where the fused kernel is.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

    n = x.shape[0]
    attn = block.attn
    q, k, v = attn.qkv(block.norm1(x)).reshape(n, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(0, 1).unsqueeze(0).to(torch.float16) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
    o = o.squeeze(0).transpose(0, 1).reshape(n, -1).to(x.dtype)
    x = x + attn.proj(o)
    return x + block.mlp(block.norm2(x))


class StaticQwen3VLVision(torch.nn.Module):
    """Qwen3-VL's vision tower over one image's patches.

    The inputs are what `Qwen3VLVisionModel.forward` derives from `grid_thw` on
    the host (`get_vision_interpolation_indices_and_weights`,
    `get_vision_position_ids`), because both are integer bookkeeping over the
    grid that `torch.export` cannot trace; the runner computes them.

    The patch embedding is a `Conv3d` whose kernel equals its stride over a
    patch already cut to the kernel's shape, which is a matrix product with the
    kernel flattened; spelled that way here.
    """

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, patches, interp_indices, interp_weights, position_ids):
        v = self.visual
        w = v.patch_embed.proj.weight
        x = F.linear(patches, w.reshape(w.shape[0], -1), v.patch_embed.proj.bias)
        pos = (v.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        x = x + pos.to(x.dtype)
        cos, sin = v.rotary_pos_emb(x, position_ids)
        features = []
        for index, block in enumerate(v.blocks):
            x = vision_block(block, x, cos, sin)
            if index in v.deepstack_visual_indexes:
                merger = v.deepstack_merger_list[v.deepstack_visual_indexes.index(index)]
                features.append(merger(x))
        return (v.merger(x), *features)


def vision_inputs(visual, grid_h, grid_w):
    """The host-side inputs for one `grid_h x grid_w` patch grid, from the
    reference's own helpers."""
    from transformers.vision_utils import (get_vision_interpolation_indices_and_weights,
                                           get_vision_position_ids)

    grid = torch.tensor([[1, grid_h, grid_w]])
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=visual.num_grid_per_side, mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=visual.config.spatial_merge_size)
    positions = get_vision_position_ids(grid, visual.spatial_merge_size)
    return indices, weights, positions


def export_vision(args):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as q3

    torch.manual_seed(0)
    config = encoder_config(args).vision_config
    config._attn_implementation = "sdpa"
    if args.smoke:
        config.depth, config.hidden_size, config.num_heads = 3, 64, 4
        config.intermediate_size, config.out_hidden_size = 96, 48
        config.deepstack_visual_indexes = [0, 2]
    visual = q3.Qwen3VLVisionModel(config).float().eval()
    if not args.smoke:
        missing, unexpected = visual.load_state_dict(compact_visual_state(), strict=False)
        if missing or unexpected:
            raise SystemExit(f"vision weights: missing {missing[:5]}, unexpected {unexpected[:5]}")
    wrapper = StaticQwen3VLVision(visual).eval()

    # A 16x24-patch grid: 384 patches, 96 merged tokens. Small enough to ship as
    # a test fixture, and not square, so a transposed grid shows.
    grid_h, grid_w = 16, 24
    indices, weights, positions = vision_inputs(visual, grid_h, grid_w)
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size ** 2
    patches = torch.randn(grid_h * grid_w, patch_dim)
    with torch.no_grad():
        reference = wrapper(patches, indices, weights, positions)
        # The wrapper against the reference model itself, before either is exported.
        full = visual(patches, grid_thw=torch.tensor([[1, grid_h, grid_w]]))
    err = max((reference[0] - full.pooler_output).abs().max().item(),
              *[(a - b).abs().max().item() for a, b in zip(reference[1:], full.deepstack_features)])
    scale = full.pooler_output.abs().max().item()
    print(f"vision wrapper against Qwen3VLVisionModel: max |diff| = {err:.3e} (outputs up to {scale:.2f})")

    n = Dim("n", min=4, max=MAX_VL_TOKENS_PER_IMAGE)
    dynamic = ({0: 4 * n}, {0: 4 * n}, {0: 4 * n}, {0: 4 * n})
    specs = ({0: "n"}, {0: "n"}, {0: "n"}, {0: "n"})
    with torch.no_grad():
        program = torch.export.export(wrapper, (patches, indices, weights, positions),
                                      dynamic_shapes=dynamic, strict=False).run_decompositions()
    write_graph(args.out, program, specs, "qwenimage21_vision", "qwenimage21_vision_constants.safetensors")
    fixture = {"patches": patches, "interp_indices": indices, "interp_weights": weights,
               "position_ids": positions, "grid": torch.tensor([grid_h, grid_w]),
               "merged": reference[0]}
    fixture.update({f"deepstack_{i}": d for i, d in enumerate(reference[1:])})
    save_file({k: v.contiguous() for k, v in fixture.items()}, str(args.out / "vision_reference.safetensors"))
    if args.smoke:
        save_file({f"visual.{k}": v.contiguous() for k, v in visual.state_dict().items()},
                  str(args.out / "vision_smoke_weights.safetensors"))


# ---------------------------------------------------------------------------
# Language model over a prompt with images in it
# ---------------------------------------------------------------------------

class InterleavedMRoPE(torch.nn.Module):
    """`Qwen3VLTextRotaryEmbedding.forward`, with its recomposition as a select.

    The reference computes the angles of all three position streams and then
    writes the H and W streams into every third frequency of the T stream
    (`freqs_thw[..., offset:length:3] = freq[dim, ..., offset:length:3]`). That
    exports as a `slice_scatter` with step 3, which has no kernel. Which stream
    each frequency reads is a constant, so the same values come out of two
    `where`s over it.
    """

    def __init__(self, rotary):
        super().__init__()
        half = rotary.inv_freq.shape[0]
        stream = torch.zeros(half, dtype=torch.long)
        for dim, offset in enumerate((1, 2), start=1):
            stream[offset:rotary.mrope_section[dim] * 3:3] = dim
        self.register_buffer("inv_freq", rotary.inv_freq.clone(), persistent=False)
        self.register_buffer("is_h", stream == 1, persistent=False)
        self.register_buffer("is_w", stream == 2, persistent=False)
        self.scaling = rotary.attention_scaling

    def forward(self, x, position_ids):
        # (3, B, T) positions times (half,) frequencies -> (3, B, T, half)
        # `.to(device)`: the constants are real CPU tensors while a graph-only
        # export traces on the meta device.
        device = position_ids.device
        freqs = position_ids[:, :, :, None].float() * self.inv_freq.float().to(device)
        f = torch.where(self.is_h.to(device), freqs[1],
                        torch.where(self.is_w.to(device), freqs[2], freqs[0]))
        emb = torch.cat((f, f), dim=-1)
        return (emb.cos() * self.scaling).to(x.dtype), (emb.sin() * self.scaling).to(x.dtype)


class StaticQwen3VLConditionEncoder(torch.nn.Module):
    """Qwen3-VL's language model with M-RoPE positions and DeepStack as inputs.

    `Qwen3VLTextModel.forward` with the reference's two deviations for this
    pipeline, as in the text-only encoder: `inputs_embeds` instead of token ids
    (the table stays on the host), and no final RMSNorm (the pipeline reads the
    last decoder layer's output). What differs from the text-only encoder:

    * `position_ids` is `(3, 1, T)`: text advances all three streams together,
      an image's tokens take `(t, h, w)` grid positions (`get_rope_index`). The
      rotary tables are computed from them here, by the model's own module.
    * `deepstack` is `(3, 1, T, hidden)`, the vision tower's three DeepStack
      features scattered to the image positions and zero elsewhere. The model
      adds the `i`-th to the hidden states after layer `i` at the image
      positions only (`_deepstack_process`); adding zeros everywhere else is the
      same sum without a boolean index.
    """

    def __init__(self, text_model, rotary):
        super().__init__()
        text_model.norm = torch.nn.Identity()
        # The model's own rotary module is replaced by `rotary`; left in place,
        # its buffers are lifted into the graph as inputs nothing reads, and on
        # the meta device they have no value to ship.
        text_model.rotary_emb = torch.nn.Identity()
        # The same for the embedding table, which the host reads instead
        # (`token_embeddings`): 622M entries the graph would declare and never use.
        text_model.embed_tokens = torch.nn.Identity()
        self.model = text_model
        self.rotary = rotary

    def forward(self, inputs_embeds, position_ids, deepstack):
        position_embeddings = self.rotary(inputs_embeds, position_ids)
        hidden = inputs_embeds
        for index, layer in enumerate(self.model.layers):
            hidden = layer(hidden, attention_mask=None, position_ids=None, past_key_values=None,
                           position_embeddings=position_embeddings)
            if index < deepstack.shape[0]:
                hidden = hidden + deepstack[index]
        return hidden


def export_vl_encoder(args):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as q3

    torch.manual_seed(0)
    text_config = encoder_config(args).text_config
    text_config._attn_implementation = "sdpa"
    if args.smoke:
        text_config.num_hidden_layers, text_config.hidden_size = 4, 64
        text_config.intermediate_size, text_config.num_attention_heads = 96, 4
        text_config.num_key_value_heads, text_config.head_dim = 2, 16
        text_config.rope_parameters = dict(text_config.rope_parameters, mrope_section=[2, 3, 3])
        model = q3.Qwen3VLTextModel(text_config).to(torch.float16).eval()
        device = "cpu"
    else:
        with torch.device("meta"):
            model = q3.Qwen3VLTextModel(text_config).to(dtype=torch.float16).eval()
        device = "meta"
    hidden = text_config.hidden_size
    tokens = 61
    embeds = torch.randn(1, tokens, hidden, dtype=torch.float16, device=device)
    # Text, then a 4x6 grid of merged image tokens (24), then text: the layout
    # `get_rope_index` gives one image in a prompt.
    positions = example_positions(9, (4, 6), tokens - 9 - 24)
    deepstack = torch.zeros(3, 1, tokens, hidden, dtype=torch.float16, device=device)
    full = None
    if args.smoke:
        deepstack[:, :, 9:33] = torch.randn(3, 1, 24, hidden, dtype=torch.float16)
        # The model's own forward, deepstack and all, before the wrapper takes
        # its rotary module and its final norm away. The final norm is dropped
        # here too: what the pipeline reads is the last layer's output.
        norm = model.norm
        model.norm = torch.nn.Identity()
        mask = torch.zeros(1, tokens, dtype=torch.bool)
        mask[:, 9:33] = True
        with torch.no_grad():
            full = model.float()(inputs_embeds=embeds.float(), position_ids=positions,
                                 visual_pos_masks=mask,
                                 deepstack_visual_embeds=[d[0, 9:33] for d in deepstack.float()],
                                 use_cache=False).last_hidden_state
        model.to(torch.float16)
        model.norm = norm
    # The inverse frequencies are a constant: built on the CPU so a meta export
    # can serialise them.
    rotary = q3.Qwen3VLTextRotaryEmbedding(text_config)
    wrapper = StaticQwen3VLConditionEncoder(model, InterleavedMRoPE(rotary)).eval()
    maxtokens = 16384
    t = Dim.DYNAMIC(min=2, max=maxtokens)
    with torch.no_grad():
        program = torch.export.export(
            wrapper, (embeds, positions.to(device), deepstack),
            dynamic_shapes=({1: t}, {2: t}, {2: t}), strict=False).run_decompositions()
    write_graph(args.out, program, ({1: "t"}, {2: "t"}, {2: "t"}), "qwenimage21_vl_encoder",
                "qwenimage21_vl_encoder_constants.safetensors")
    (args.out / "qwenimage21_vl_encoder_export.json").write_text(json.dumps({
        "model": "random-smoke" if args.smoke else args.model, "token_symbol": "t",
        "min_tokens": 2, "max_tokens": maxtokens, "layers": len(model.layers),
        "hidden_size": hidden, "deepstack": 3, "dtype": "float16"}, indent=1))
    if args.smoke:
        with torch.no_grad():
            reference = wrapper.float()(embeds.float(), positions, deepstack.float())
        print(f"vl encoder wrapper against Qwen3VLTextModel: max |diff| = "
              f"{(reference - full).abs().max().item():.3e}")
        save_file({"inputs_embeds": embeds.float().contiguous(), "position_ids": positions.contiguous(),
                   "deepstack": deepstack.float().contiguous(), "hidden": reference.contiguous()},
                  str(args.out / "vl_encoder_reference.safetensors"))
        save_file({f"model.{k}": v.to(torch.float16).contiguous() for k, v in model.state_dict().items()},
                  str(args.out / "vl_encoder_smoke_weights.safetensors"))


def example_positions(before, grid, after):
    """`get_rope_index`'s positions for `before` text tokens, one image of
    `grid` merged tokens, and `after` text tokens."""
    gh, gw = grid
    text0 = torch.arange(before).view(1, -1).expand(3, -1)
    h = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1) + before
    w = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1) + before
    t = torch.full_like(h, before)
    image = torch.stack([t, h, w])
    start = before + max(gh, gw)
    text1 = torch.arange(after).view(1, -1).expand(3, -1) + start
    return torch.cat([text0, image, text1], dim=1).view(3, 1, -1).contiguous()


# ---------------------------------------------------------------------------
# VAE encoder
# ---------------------------------------------------------------------------

class NormalizedVAEEncoder(torch.nn.Module):
    """`_encode_vae_image`: an RGBA image in [-1, 1] to normalised latents.

    `retrieve_latents(..., sample_mode="argmax")` is the posterior's mode, which
    for a diagonal Gaussian is its mean, the first half of `quant_conv`'s
    channels. The single frame is kept as the temporal axis, as the pipeline
    keeps it (`unsqueeze(2)`).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        dtype = next(vae.parameters()).dtype
        self.latents_mean = torch.tensor(vae.config.latents_mean, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
        self.latents_std = torch.tensor(vae.config.latents_std, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)

    def forward(self, image):
        moments = self.vae._encode(image)
        mean, _ = torch.chunk(moments, 2, dim=1)
        return (mean - self.latents_mean) / self.latents_std


def export_vae_encoder(vae_module, args):
    torch.manual_seed(0)
    if args.smoke:
        vae = vae_module.AutoencoderKLQwenImage21(
            base_dim=32, decoder_base_dim=32, z_dim=4, dim_mult=[1, 2], num_res_blocks=1,
            temperal_downsample=[False], latents_mean=[0.1, -0.2, 0.3, 0.0],
            latents_std=[1.5, 0.5, 2.0, 1.0], is_residual=False, in_channels=4, out_channels=4,
            scale_factor_temporal=1, scale_factor_spatial=2).float().eval()
        height, width = 16, 24
    else:
        vae = vae_module.AutoencoderKLQwenImage21.from_pretrained(
            args.model, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
        height, width = 64, 96
    wrapper = NormalizedVAEEncoder(vae).eval()
    image = torch.rand(1, 4, 1, height, width) * 2 - 1
    with torch.no_grad():
        reference = wrapper(image)
        # Against the pipeline's own `_encode_vae_image` path.
        mode = vae.encode(image).latent_dist.mode()
        direct = (mode - wrapper.latents_mean) / wrapper.latents_std
    print(f"vae encoder wrapper against encode().mode(): max |diff| = {(reference - direct).abs().max().item():.3e}")
    spatial = ({3: torch.export.Dim.AUTO, 4: torch.export.Dim.AUTO},)
    with torch.no_grad():
        program = torch.export.export(wrapper, (image,), dynamic_shapes=spatial,
                                      strict=False).run_decompositions()
    graph, _ = write_graph(args.out, program, ({3: "h", 4: "w"},), "qwenimage21_vae_encoder",
                           "qwenimage21_vae_encoder_constants.safetensors")
    shape = next(b["shape"] for b in graph["buffers"] if b["id"] in graph["inputs"])
    if sum(isinstance(d, str) for d in shape) < 2:
        raise SystemExit(f"the VAE encoder's spatial axes did not stay dynamic: the image is {shape}")
    requested = {b["key"] for b in graph["buffers"] if b["kind"] == "weight"}
    state = {k: v for k, v in wrapper.state_dict().items()}
    print(f"  input {shape}, {len(requested)} weights, "
          f"{len([k for k in requested if k not in state])} of them constants")
    save_file({"image": image.contiguous(), "latents": reference.contiguous()},
              str(args.out / "vae_encoder_reference.safetensors"))
    if args.smoke:
        save_file({k: v.contiguous() for k, v in state.items() if k in requested},
                  str(args.out / "vae_encoder_smoke_weights.safetensors"))


# ---------------------------------------------------------------------------
# Test fixtures for the runner's host-side bookkeeping
# ---------------------------------------------------------------------------

def synthetic_images():
    """Two reference images of different aspect ratios, made here so the
    fixtures carry nothing anyone else owns: an RGBA gradient with a soft round
    matte (so the premultiplied resize is exercised at partial alpha) and an
    opaque RGB pattern."""
    import numpy as np
    from PIL import Image

    h, w = 190, 300
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.clip(x / w * 255, 0, 255)
    g = np.clip(y / h * 255, 0, 255)
    b = np.clip(128 + 100 * np.sin(x / 7) * np.cos(y / 11), 0, 255)
    d = np.sqrt((x - w * 0.45) ** 2 + (y - h * 0.55) ** 2)
    a = np.clip((80 - d) / 12 * 255, 0, 255)
    first = Image.fromarray(np.stack([r, g, b, a], -1).astype(np.uint8), "RGBA")
    h, w = 200, 128
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    pattern = np.stack([(x * 3 + y) % 256, (x * y / 7) % 256, (255 - y) % 256], -1)
    second = Image.fromarray(pattern.astype(np.uint8), "RGB")
    return [first, second]


def export_fixtures(module, args):
    """What the runner's host code has to reproduce, from the reference's own
    code: the resize, the white composite, the processor's patches and tokens,
    `get_rope_index`, the vision tower's grid inputs, and the denoiser's joint
    rotary tables. At `output_resolution = 320`, a pipeline option like any
    other, so the fixture stays small. Not 256: its rounding lands the first
    image at 320x192, under the processor's 65536-pixel floor, and the processor
    then upsamples it past its latent grid (see `vision_patches` in the runner).
    320 also rounds the second image's 12.5 blocks to 12, half to even."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import modeling_qwen3_vl as q3
    from transformers.vision_utils import (get_vision_interpolation_indices_and_weights,
                                           get_vision_position_ids)

    resolution = 320
    prompt = "Put the object from the first picture onto the second one."
    out, whites, sizes = {}, [], []
    for k, img in enumerate(synthetic_images()):
        img = img.convert("RGBA")  # what `QwenImage21Pipeline.__call__` does first
        out[f"orig_{k}"] = torch.from_numpy(np.array(img)).contiguous()
        w, h = calc_dimensions(resolution * resolution, img.size[0] / img.size[1])
        sizes.append((w, h))
        resized = img.resize((w, h), resample=Image.LANCZOS)
        out[f"resized_{k}"] = torch.from_numpy(np.array(resized)).contiguous()
        white = Image.new("RGB", resized.size, (255, 255, 255))
        white.paste(resized, mask=resized.getchannel("A"))
        out[f"white_{k}"] = torch.from_numpy(np.array(white)).contiguous()
        whites.append(white)
    out["sizes"] = torch.tensor(sizes)
    processor = AutoProcessor.from_pretrained(args.model, subfolder="processor")
    slots = "".join(f"{'' if i == 0 else ' '}<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(len(whites)))
    text = (f"<|im_start|>system\n{Q.QWEN_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{slots}{prompt}"
            f"<|im_end|>\n<|im_start|>assistant\n")
    inputs = processor(text=[text], images=whites, padding=True, padding_side="left", return_tensors="pt")
    out["input_ids"] = inputs["input_ids"][0].contiguous()
    # Float16: 800 patches of 1536 values is 4.9 MB at Float32, and the test
    # compares at a tolerance a transposed patch would miss by orders.
    out["pixel_values"] = inputs["pixel_values"].half().contiguous()
    out["grid"] = inputs["image_grid_thw"].contiguous()
    config = encoder_config(args)
    with torch.device("meta"):
        vl = q3.Qwen3VLModel(config)
    positions, _ = vl.get_rope_index(inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                                     image_grid_thw=inputs["image_grid_thw"])
    out["mrope"] = positions[:, 0].contiguous()
    for k in range(len(whites)):
        grid = inputs["image_grid_thw"][k:k + 1]
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=48, mode="bilinear", align_corners=True, spatial_merge_size=2)
        out[f"interp_idx_{k}"] = indices.contiguous()
        out[f"interp_w_{k}"] = weights.contiguous()
        out[f"vispos_{k}"] = get_vision_position_ids(grid, 2).contiguous()
    # The denoiser's joint sequence for that prompt, with the target the size of
    # the last reference image, as the pipeline defaults it.
    drop = 14
    img_mask = (inputs["input_ids"][0] == 151655)[drop:]
    shapes = [(1, h // 16, w // 16) for (w, h) in sizes]
    shapes.append(shapes[-1])
    target = shapes[-1][1] * shapes[-1][2]
    joint = torch.cat([img_mask, torch.ones(target // 4, dtype=torch.bool)])
    pad = torch.repeat_interleave(joint, torch.where(joint, 4, 1))
    rope = module.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])(shapes, pad, torch.device("cpu"))
    out["rope_real"] = rope.real.contiguous()
    out["rope_imag"] = rope.imag.contiguous()
    out["img_mask"] = img_mask.to(torch.uint8).contiguous()
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out / "condition_host.safetensors"),
              metadata={"prompt": prompt, "resolution": str(resolution)})
    print(f"condition_host: sizes {sizes}, {inputs['input_ids'].shape[1]} tokens, joint {rope.shape[0]}")


def calc_dimensions(target_area, ratio):
    """`calculate_dimensions` from `pipeline_qwenimage21.py`."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the segmented denoiser against the reference forward and stop")
    parser.add_argument("--component",
                        choices=("denoiser", "vision", "vl_encoder", "vae_encoder", "fixtures", "all"),
                        default="all",
                        help="`fixtures` writes the host-side test fixture to --out "
                             "(default gen/graphs/qwenimage21-refs); `all` is the four graphs")
    parser.add_argument("--smoke", action="store_true",
                        help="small random models, and reference outputs for the runner's parity tests")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--diffusers-source", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        args.out = ROOT / "gen" / "graphs" / (
            "qwenimage21-refs" if args.component == "fixtures" else
            "qwenimage21-condition-smoke" if args.smoke else "qwenimage21")
    module, vae_module = Q.import_qwen21(args.diffusers_source or artifact("diffusers-src") / "src")
    if args.check:
        check_decomposition(module)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.component in ("denoiser", "all"):
        export_denoiser(module, args)
    if args.component in ("vision", "all"):
        export_vision(args)
    if args.component in ("vl_encoder", "all"):
        export_vl_encoder(args)
    if args.component in ("vae_encoder", "all"):
        export_vae_encoder(vae_module, args)
    if args.component == "fixtures":
        export_fixtures(module, args)


if __name__ == "__main__":
    main()
