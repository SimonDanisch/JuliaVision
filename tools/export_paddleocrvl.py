"""Export PaddleOCR-VL (0.9B: a NaViT vision encoder and ERNIE-4.5-0.3B) to
DNNKernels' graph JSON + weights.

    uv run tools/export_paddleocrvl.py --reference-image page.png --box 236,1393,2241,1978

The model reads one element of a page (a table, a formula, a chart, a block of
text) and writes its content: a table as cells, a formula as LaTeX. Four
graphs, one weight file:

  - `paddleocrvl_vision`: a crop's patches -> its image tokens. The crop is cut
    into 14x14 patches on the device, in 2x2-block order (the four patches each
    image token merges are adjacent), `(4n, 588)` with `n` symbolic. The
    position-embedding taps (bilinear over the learned 27x27 grid) and the
    patches' (row, col) for the 2D rotary are integer bookkeeping over the grid
    and come in computed on the host, as for Qwen3-VL here. -> `(n, 1024)`.
  - `paddleocrvl_embed`: token ids `(1, t)` -> embeddings `(1, t, 1024)`, for
    the prompt around the image tokens, `t` symbolic.
  - `paddleocrvl_prefill`: the prompt's embeddings `(1, t, 1024)` and their 3D
    rotary positions `(3, 1, t)` -> the next token, writing the KV cache's
    first `t` slots. Causal attention over the prompt alone.
  - `paddleocrvl_step`: one token `(1, 1)`, its rotary position and its cache
    slot -> the next token, attending to the cache up to the slot.

The cache is `(18, 1, 2, MAX_CACHE, 128)` for keys and values, passed in and
out and written in place (`foldcacheupdate`). The next token is picked in the
graph, so a step returns one id.

`prompts.json` holds the token ids around the image tokens for each task;
`vocab.json` every id's piece, for decoding (byte fallback pieces as
`<0xNN>`, spaces as `▁`).

From Hugging Face Transformers (`paddleocr_vl`), `PaddlePaddle/PaddleOCR-VL`.
Apache-2.0, code and weights.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.export import Dim
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration
from transformers.models.paddleocr_vl import modeling_paddleocr_vl as M
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

import export_graphs as EG
from common import GEN

REPO = "PaddlePaddle/PaddleOCR-VL"
MAX_CACHE = 8192
TASKS = {"ocr": "OCR:", "table": "Table Recognition:", "formula": "Formula Recognition:", "chart": "Chart Recognition:"}


def one_image_attention(self, hidden_states, cu_seqlens, position_embeddings, max_seqlen=None, **kwargs):
    """The vision attention for one image: the whole sequence is one chunk, so
    the per-image split (data-dependent, not exportable) is not needed."""
    n = hidden_states.shape[0]
    q = self.q_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    k = self.k_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(n, self.num_heads, self.head_dim)
    cos, sin = position_embeddings
    q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out, _ = M.eager_attention_forward(self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0)
    return self.out_proj(out.reshape(n, -1).contiguous()), None


class Vision(nn.Module):
    """Patches in 2x2-block order -> image tokens."""

    def __init__(self, model):
        super().__init__()
        vm = model.model.visual.vision_model
        self.embeddings, self.encoder, self.post = vm.embeddings, vm.encoder, vm.post_layernorm
        self.projector = model.model.projector

    def forward(self, patches, taps, weights, positions):
        e = self.embeddings
        w = e.patch_embedding.weight
        x = F.linear(patches, w.reshape(w.shape[0], -1), e.patch_embedding.bias)
        x = x + (e.position_embedding(taps) * weights[:, :, None]).sum(1)
        rope = self.encoder.rotary_pos_emb(x, positions)
        for layer in self.encoder.layers:
            x = layer(x, cu_seqlens=None, position_embeddings=rope)
        x = self.post(x)
        p = self.projector
        x = p.pre_norm(x).reshape(-1, 4 * x.shape[-1])
        return p.linear_2(p.act(p.linear_1(x)))


class Embed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.model.language_model.embed_tokens

    def forward(self, ids):
        return self.embed(ids)


def decoder(lm, h, cos, sin, mask, kcache, vcache, slots):
    """The 18 layers over `h`, writing each layer's keys and values into the
    cache at `slots`; attention over the keys `mask` lets through, of the
    tokens' own (prefill) or the whole cache (step)."""
    a0 = lm.layers[0].self_attn
    H, Hkv, hd = a0.num_heads, a0.num_key_value_heads, a0.head_dim
    B, T = h.shape[:2]
    newk, newv = [], []
    for i, layer in enumerate(lm.layers):
        r = h
        x = layer.input_layernorm(h)
        a = layer.self_attn
        q = a.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = a.k_proj(x).view(B, T, Hkv, hd).transpose(1, 2)
        v = a.v_proj(x).view(B, T, Hkv, hd).transpose(1, 2)
        q, k = M.apply_rotary_pos_emb(q, k, cos, sin)
        K = kcache[i].index_copy(2, slots, k)
        V = vcache[i].index_copy(2, slots, v)
        newk.append(K)
        newv.append(V)
        keys, values = (k, v) if mask.shape[-1] == T else (K, V)
        o, _ = M.eager_attention_forward(a, q, keys, values, mask, scaling=a.scaling)
        h = r + a.o_proj(o.reshape(B, T, H * hd))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, torch.stack(newk), torch.stack(newv)


class Prefill(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm, self.head = model.model.language_model, model.lm_head

    def forward(self, embeds, positions, kcache, vcache):
        T = embeds.shape[1]
        cos, sin = self.lm.rotary_emb(embeds, positions)
        idx = torch.arange(T, device=embeds.device)
        zero = torch.zeros((), dtype=embeds.dtype, device=embeds.device)
        neg = torch.full((), torch.finfo(embeds.dtype).min, dtype=embeds.dtype, device=embeds.device)
        mask = torch.where(idx[None, :] <= idx[:, None], zero, neg).view(1, 1, T, T)
        h, K, V = decoder(self.lm, embeds, cos, sin, mask, kcache, vcache, idx)
        logits = self.head(self.lm.norm(h[:, -1:]))
        return logits.max(-1).indices, K, V


class Step(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm, self.head = model.model.language_model, model.lm_head

    def forward(self, token, position, slot, kcache, vcache):
        h = self.lm.embed_tokens(token)
        cos, sin = self.lm.rotary_emb(h, position.view(1, 1, 1).expand(3, 1, 1))
        idx = torch.arange(kcache.shape[3], device=h.device)
        zero = torch.zeros((), dtype=h.dtype, device=h.device)
        neg = torch.full((), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device)
        mask = torch.where(idx <= slot, zero, neg).view(1, 1, 1, -1)
        h, K, V = decoder(self.lm, h, cos, sin, mask, kcache, vcache, slot)
        logits = self.head(self.lm.norm(h))
        return logits.max(-1).indices, K, V


def blockorder(grid):
    """The raster index of each patch in 2x2-block order, the order the
    projector merges in."""
    t, h, w = grid
    idx = torch.arange(h * w).view(h // 2, 2, w // 2, 2).transpose(1, 2).flatten()
    return idx


def inputs(processor, crop, task):
    messages = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": TASKS[task]}]}]
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                         return_tensors="pt")


def vision_inputs(model, x):
    grid = x.image_grid_thw
    order = blockorder(grid[0].tolist())
    patches = x.pixel_values.reshape(x.pixel_values.shape[0], -1)[order]
    e = model.model.visual.vision_model.embeddings
    taps, weights = get_vision_interpolation_indices_and_weights(grid, num_grid_per_side=e.num_grid_per_side,
        mode=e.interpolation_mode, align_corners=e.interpolation_align_corners, spatial_merge_size=2)
    positions = get_vision_position_ids(grid, 2)
    return patches.contiguous(), taps.contiguous(), weights.float().contiguous(), positions.contiguous()


def generate(model, embed, prefill, step, vision, x, image_token, steps):
    """Greedy decoding through the four graphs' modules, as the runner will."""
    with torch.no_grad():
        image = vision(*vision_inputs(model, x))
        ids = x.input_ids[0]
        embeds = embed(ids[None])[0].clone()
        embeds[ids == image_token] = image
        mm = x["mm_token_type_ids"] if "mm_token_type_ids" in x else (x.input_ids == image_token).int()
        positions = model.model.compute_3d_position_ids(input_ids=x.input_ids, image_grid_thw=x.image_grid_thw,
            inputs_embeds=embeds[None], attention_mask=x.attention_mask, past_key_values=None,
            mm_token_type_ids=mm)
        cfg = model.config.get_text_config()
        shape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, MAX_CACHE, cfg.head_dim)
        kc, vc = torch.zeros(shape), torch.zeros(shape)
        token, kc, vc = prefill(embeds[None], positions, kc, vc)
        out = [int(token)]
        slot, position = ids.shape[0], int(positions.max()) + 1
        for _ in range(steps - 1):
            token, kc, vc = step(token.view(1, 1), torch.tensor([position]), torch.tensor([slot]), kc, vc)
            out.append(int(token))
            slot += 1; position += 1
        return out, image, embeds, positions


def main(args):
    torch.set_num_threads(16)
    model = PaddleOCRVLForConditionalGeneration.from_pretrained(REPO, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(REPO)
    M.PaddleOCRVisionAttention.forward = one_image_attention
    page = Image.open(args.reference_image).convert("RGB")
    crop = page.crop(tuple(int(v) for v in args.box.split(","))) if args.box else page
    image_token = model.config.image_token_id
    vision, embed, prefill, step = Vision(model).eval(), Embed(model).eval(), Prefill(model).eval(), Step(model).eval()

    # Parity: the four modules decode what the model's own generate does.
    x = inputs(processor, crop, "table")
    with torch.no_grad():
        expected = model.generate(**x, max_new_tokens=args.check_tokens, do_sample=False)[0, x.input_ids.shape[1]:].tolist()
        reference_image = model.model.get_image_features(x.pixel_values, x.image_grid_thw).pooler_output[0]
    ours, image, embeds, positions = generate(model, embed, prefill, step, vision, x, image_token, args.check_tokens)
    err = (image - reference_image).abs().max().item()
    same = ours[:len(expected)] == expected
    print(f"vision vs the model's own: max |diff| {err:.2e}; first {len(expected)} tokens the same: {same}")
    assert same, (ours, expected)

    cfg = model.config.get_text_config()
    shape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, MAX_CACHE, cfg.head_dim)
    out = args.out_root / "paddleocrvl"
    graphs, constants = {}, {}
    with torch.no_grad():
        v_in = vision_inputs(model, x)
        n = Dim("n", min=2, max=MAX_CACHE // 2)
        ep = torch.export.export(vision, v_in, dynamic_shapes=({0: 4 * n}, {0: 4 * n}, {0: 4 * n}, {0: 4 * n}),
                                 strict=False).run_decompositions()
        graphs["paddleocrvl_vision"] = EG.convert(ep, ({0: "n"}, {}, {}, {}), "paddleocrvl_vision")
        constants.update(EG.save_constants(ep))

        t = Dim("t", min=2, max=MAX_CACHE)
        ep = torch.export.export(embed, (x.input_ids,), dynamic_shapes=({1: t},), strict=False).run_decompositions()
        graphs["paddleocrvl_embed"] = EG.convert(ep, ({1: "t"},), "paddleocrvl_embed")
        constants.update(EG.save_constants(ep))

        z = torch.zeros(shape)
        ep = torch.export.export(prefill, (embeds[None], positions, z, z),
                                 dynamic_shapes=({1: t}, {2: t}, None, None), strict=False).run_decompositions()
        graphs["paddleocrvl_prefill"] = EG.convert(ep, ({1: "t"}, {}, {}, {}), "paddleocrvl_prefill")
        constants.update(EG.save_constants(ep))

        ep = torch.export.export(step, (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                                        torch.zeros(1, dtype=torch.long), z, z), strict=False).run_decompositions()
        graphs["paddleocrvl_step"] = EG.convert(ep, ({},), "paddleocrvl_step")
        constants.update(EG.save_constants(ep))

    out.mkdir(parents=True, exist_ok=True)
    histogram = {}
    for name, g in graphs.items():
        (out / f"{name}.json").write_text(json.dumps(g, indent=1))
        for o in g["ops"]:
            histogram[o["aten"]] = histogram.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(histogram, indent=1, sort_keys=True))
    ops = sum(len(g["ops"]) for g in graphs.values())
    # Each graph names its weights by their path in the module it was exported
    # from (`embeddings.patch_embedding.weight` for the vision, `lm.layers.0...`
    # for prefill and step, `embed.weight`); one file holds every module's names.
    # Shared tensors are cloned, as safetensors refuses shared memory.
    named = {}
    for module in (vision, embed, prefill, step):
        for k, v in {**dict(module.named_parameters(remove_duplicate=False)),
                     **dict(module.named_buffers(remove_duplicate=False))}.items():
            named.setdefault(k, v)
    named.update(constants)
    tensors, seen = {}, set()
    for k, v in named.items():
        v = v.detach().contiguous().cpu()
        tensors[k] = v.clone() if v.data_ptr() in seen else v
        seen.add(v.data_ptr())
    from safetensors.torch import save_file
    save_file(tensors, str(out / "weights.safetensors"))

    prompts = {"image_token": image_token, "eos": model.generation_config.eos_token_id, "max_cache": MAX_CACHE,
               "tasks": {}}
    for task in TASKS:
        ids = inputs(processor, crop, task).input_ids[0].tolist()
        first, last = ids.index(image_token), len(ids) - 1 - ids[::-1].index(image_token)
        prompts["tasks"][task] = {"prefix": ids[:first], "suffix": ids[last + 1:]}
    (out / "prompts.json").write_text(json.dumps(prompts, indent=1))
    tok = json.load(open(hf_hub_download(REPO, "tokenizer.json")))
    vocab = [""] * cfg.vocab_size
    for piece, i in tok["model"]["vocab"].items():
        vocab[i] = piece
    for added in tok["added_tokens"]:
        vocab[added["id"]] = added["content"]
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    save_file({"patches": v_in[0], "taps": v_in[1], "weights": v_in[2], "positions": v_in[3],
               "image": image.contiguous(), "embeds": embeds.contiguous(), "rope": positions.contiguous(),
               "tokens": torch.tensor(ours)}, str(out / "reference.safetensors"))
    (out / "provenance.json").write_text(json.dumps({"repo": REPO,
        "revision": Path(hf_hub_download(REPO, "config.json")).parent.name, "torch": torch.__version__,
        "graphs": list(graphs), "max_cache": MAX_CACHE, "check_tokens": args.check_tokens}, indent=2))
    print(f"paddleocrvl: {len(graphs)} graphs, {ops} ops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--box", default=None, help="x0,y0,x1,y1 of the element to read")
    parser.add_argument("--check-tokens", type=int, default=48)
    parser.add_argument("--out-root", type=Path, default=GEN / "graphs")
    main(parser.parse_args())
