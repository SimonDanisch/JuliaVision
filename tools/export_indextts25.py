"""
Export IndexTTS-2.5's networks to DNNKernels' graph JSON + weights.

    ~/tools/index-tts/.venv/bin/python tools/export_indextts25.py --part w2v
    ~/tools/index-tts/.venv/bin/python tools/export_indextts25.py --part all

**Run it with index-tts's own interpreter**, as the reference dump does
(`dump_indextts25_refs.py`): the modules come from `~/tools/index-tts` at
d9e41aa, the weights from `~/models/IndexTTS-2.5`.

Every part is a wrapper module around upstream's, checked against upstream's
own module on the inputs `dump_indextts25_refs.py` recorded (case `en_greedy`)
before it is traced, so an export that changed the math fails here and not in
Julia. Writes `gen/indextts25/graphs/<part>/{<part>.json, weights.safetensors,
op_histogram.json, reference.safetensors}`.

## The parts

| part | upstream | input -> output |
|---|---|---|
| `w2v` | w2v-BERT 2.0, layers 0..16, normalised | features (1,t,160) -> (1,t,1024) |
| `campplus` | CAMPPlus speaker embedding | fbank (1,t,80) -> style (1,192) |
| `emo` | `UnifiedVoice.get_emovec` | w2v (1,t,1024) -> (1,1280) |
| `regulator` | `InterpolateRegulator` | w2v/codec (1,t,1024), source index (y,) -> (1,y,512) |
| `codec` | `EnhancedCodec.decode` | codes (1,n) -> (1,2n,1024) |
| `dit` | the CFM estimator, one call | x, prompt, t, style, mu at batch 2 -> (2,80,t) |
| `bigvgan` | BigVGAN v2 22 kHz | mel (1,80,t) -> (1,1,256t) |

What each wrapper changes, and why it is the same function:

  * `w2v`: upstream runs all 24 layers and keeps `hidden_states[17]`, the
    output of layer 16. The wrapper stops there. `relative_key` attention adds
    `einsum("bhld,lrd->bhlr", q, E[clamp(r - l, -64, 8) + 64])`, which
    materialises a (t,t,64) table per layer; the wrapper takes the 73 distinct
    dot products `q @ E^T` and gathers them by distance. Same products.
  * `regulator`: upstream's `F.interpolate(mode="nearest", size=ylens.max())`
    takes its output size from a tensor VALUE, which a graph cannot. The
    nearest source index of every output frame is computed on the host instead,
    as ATen does (`floorf(dst * (float)in / out)`, clamped), and is an input.
  * `codec`: upsampling by 2 with `nearest` is `index >> 1`, written as a
    repeat.
  * `dit`: inference always passes `x_lens` equal to the length, so the key
    mask is all true; the wrapper drops it. Upstream's `Transformer` would
    otherwise build a causal mask when given none.
"""

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

import export_graphs as EG
from common import find_root

REPO = Path.home() / "tools" / "index-tts"
BASE = Path.home() / "models" / "IndexTTS-2.5"
GERMAN = Path.home() / "models" / "index-tts-2.5-german"
sys.path.insert(0, str(REPO))

GEN = find_root() / "gen" / "indextts25"
REFS = GEN / "refs"
EG.ROOT.update(t="t", y="y", n="n", b="b", kv="kv")


# ── loading upstream ─────────────────────────────────────────────────────────

def config():
    from omegaconf import OmegaConf
    return OmegaConf.load(BASE / "config.yaml")


def load_w2v(dev):
    from transformers import Wav2Vec2BertModel
    m = Wav2Vec2BertModel.from_pretrained(BASE / "hf_cache" / "w2v-bert-2.0", local_files_only=True)
    stats = torch.load(BASE / "wav2vec2bert_stats.pt")
    return m.eval().to(dev), stats["mean"].to(dev), torch.sqrt(stats["var"]).to(dev)


def load_campplus(dev):
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(BASE / "hf_cache" / "campplus_cn_common.bin", map_location="cpu"))
    # The embedding's last BatchNorm is `affine=False`. The graph converter
    # drops absent operands, so its op would arrive as (x, mean, var) and the
    # emitter reads operands by position. A scale of one and a shift of zero
    # are the same arithmetic exactly.
    for bn in m.modules():
        if isinstance(bn, torch.nn.BatchNorm1d) and not bn.affine:
            bn.affine = True
            bn.weight = torch.nn.Parameter(torch.ones(bn.num_features))
            bn.bias = torch.nn.Parameter(torch.zeros(bn.num_features))
    return m.eval().to(dev)


def load_gpt(dev, checkpoint=None):
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    cfg = config()
    m = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    load_checkpoint(m, str(checkpoint or BASE / cfg.gpt_checkpoint))
    return m.eval().to(dev)


def load_s2mel(dev):
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    cfg = config()
    m = MyModel(cfg.s2mel)
    m, _, _, _ = load_checkpoint2(m, None, str(BASE / cfg.s2mel_checkpoint), load_only_params=True,
                                  ignore_modules=[], is_distributed=False)
    # After the move, as upstream does: `freqs_cis` is a plain attribute, so
    # `.to` would leave a table built before it on the host.
    m = m.eval().to(dev)
    m.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    return m


def load_codec(dev):
    from indextts.codec.models import EnhancedCodec
    cfg = config()
    m = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
    m.load_checkpoint(str(BASE / "codec.pth"))
    return m.eval().to(dev)


def load_bigvgan(dev):
    from indextts.s2mel.modules.bigvgan import bigvgan
    m = bigvgan.BigVGAN.from_pretrained(str(BASE / "hf_cache" / "bigvgan"), use_cuda_kernel=False)
    m.remove_weight_norm()
    return m.eval().to(dev)


# ── the wrappers ─────────────────────────────────────────────────────────────

def relative_key_attention(att, h, bias):
    """`Wav2Vec2BertSelfAttention.forward` for `relative_key`, with the distance
    term as a gather over the 73 distinct `q . E[k]` products. `bias` is HF's
    extended mask, `(1 - mask) * finfo.min` per key, `(1, 1, 1, t)`."""
    b, t, _ = h.shape
    nh, hs = att.num_heads, att.head_size
    q = att.linear_q(h).view(b, t, nh, hs).transpose(1, 2)
    k = att.linear_k(h).view(b, t, nh, hs).transpose(1, 2)
    v = att.linear_v(h).view(b, t, nh, hs).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hs)
    left, right = att.left_max_position_embeddings, att.right_max_position_embeddings
    pos = torch.arange(t, device=h.device)
    dist = torch.clamp(pos.view(1, t) - pos.view(t, 1), -left, right) + left      # (l, r)
    qe = torch.matmul(q, att.distance_embedding.weight.t())                        # (b,h,l,73)
    rel = torch.gather(qe, -1, dist.view(1, 1, t, t).expand(b, nh, t, t))
    scores = scores + rel / math.sqrt(hs) + bias
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v).transpose(1, 2).reshape(b, t, nh * hs)
    return att.linear_out(out)


def conv_module(conv, h, keep):
    """`Wav2Vec2BertConvolutionModule.forward` with the padded frames zeroed by a
    multiply instead of `masked_fill`."""
    h = conv.layer_norm(h)
    h = h * keep
    h = h.transpose(1, 2)
    h = conv.pointwise_conv1(h)
    h = conv.glu(h)
    h = F.pad(h, (conv.depthwise_conv.kernel_size[0] - 1, 0))
    h = conv.depthwise_conv(h)
    h = conv.depthwise_layer_norm(h.transpose(1, 2)).transpose(1, 2)
    h = conv.activation(h)
    h = conv.pointwise_conv2(h)
    h = conv.dropout(h)
    return h.transpose(1, 2)


def conformer_layer(layer, h, bias, keep):
    """`Wav2Vec2BertEncoderLayer.forward` with HF's attention mask, through the
    attention above: keys masked by `bias`, frames zeroed by `keep` before the
    depthwise convolution."""
    r = h
    h = layer.ffn1_layer_norm(h)
    h = layer.ffn1(h)
    h = h * 0.5 + r
    r = h
    h = layer.self_attn_layer_norm(h)
    h = relative_key_attention(layer.self_attn, h, bias)
    h = h + r
    r = h
    h = conv_module(layer.conv_module, h, keep)
    h = r + h
    r = h
    h = layer.ffn2_layer_norm(h)
    h = layer.ffn2(h)
    h = h * 0.5 + r
    return layer.final_layer_norm(h)


class W2V(torch.nn.Module):
    """Normalised `hidden_states[17]` of w2v-BERT 2.0: `IndexTTS2.get_emb`.

    `mask` is the feature extractor's attention mask as float, `(1, t)`: an odd
    number of Kaldi frames is padded with a zero frame before pairing, and the
    last pair is then masked (`attention_mask[:, indices % 2 == 1]`). HF zeroes
    masked frames on the way in and before every depthwise convolution, and
    adds `(1 - mask) * finfo.min` to the attention scores; the masked frame's
    own output is kept, as `get_emb` keeps it."""

    def __init__(self, model, mean, std, layers=17):
        super().__init__()
        self.feature_projection = model.feature_projection
        self.layers = model.encoder.layers[:layers]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, feats, mask):
        h, _ = self.feature_projection(feats)
        keep = mask.unsqueeze(-1)
        h = h * keep
        bias = ((1.0 - mask) * torch.finfo(h.dtype).min)[:, None, None, :]
        for layer in self.layers:
            h = conformer_layer(layer, h, bias, keep)
        return (h - self.mean) / self.std


class Emo(torch.nn.Module):
    """`UnifiedVoice.get_emovec` on one unpadded clip."""

    def __init__(self, gpt):
        super().__init__()
        self.emo_conditioning_encoder = gpt.emo_conditioning_encoder
        self.emo_cond_mask_pad = gpt.emo_cond_mask_pad
        self.emo_perceiver_encoder = gpt.emo_perceiver_encoder
        self.emovec_layer = gpt.emovec_layer
        self.emo_layer = gpt.emo_layer

    def forward(self, x):
        # `get_emo_conditioning(x.transpose(1, 2), lens)` transposes back.
        lens = torch.full((1,), x.shape[1], dtype=torch.long, device=x.device)
        h, mask = self.emo_conditioning_encoder(x, lens)
        v = self.emo_perceiver_encoder(h, self.emo_cond_mask_pad(mask.squeeze(1))).squeeze(1)
        return self.emo_layer(self.emovec_layer(v))


class OneGroupNorm(torch.nn.Module):
    """`nn.GroupNorm(1, C)` written out: mean and biased variance over (C, t),
    two passes. DNNKernels has no `native_group_norm`, and the regulator is
    its only user here; ATen's CUDA kernel uses Welford, which rounds
    differently in the last bits."""

    def __init__(self, gn):
        super().__init__()
        assert gn.num_groups == 1 and gn.affine
        self.weight, self.bias, self.eps = gn.weight, gn.bias, gn.eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        d = x - mean
        var = (d * d).mean(dim=(1, 2), keepdim=True)
        return d * torch.rsqrt(var + self.eps) * self.weight[:, None] + self.bias[:, None]


class Regulator(torch.nn.Module):
    """`InterpolateRegulator.forward` for continuous input, f0 off, one clip,
    with the nearest-neighbour source index as an input."""

    def __init__(self, reg):
        super().__init__()
        self.content_in_proj = reg.content_in_proj
        self.model = torch.nn.Sequential(*[OneGroupNorm(m) if isinstance(m, torch.nn.GroupNorm) else m
                                           for m in reg.model])

    def forward(self, x, index):
        x = self.content_in_proj(x)                    # (1, t, 512)
        x = x[:, index, :].transpose(1, 2)             # (1, 512, y)
        return self.model(x).transpose(1, 2)           # mask is all true


def nearest_index(t_in, t_out):
    """ATen's `nearest_idx` for upsample_nearest1d given an output size."""
    if t_out == t_in:
        return torch.arange(t_out)
    if t_out == 2 * t_in:
        return torch.arange(t_out) >> 1
    scale = torch.tensor(t_in, dtype=torch.float32) / t_out
    return torch.clamp(torch.floor(torch.arange(t_out, dtype=torch.float32) * scale).long(), max=t_in - 1)


class CodecDecode(torch.nn.Module):
    """`EnhancedCodec.decode` for one quantizer."""

    def __init__(self, codec):
        super().__init__()
        self.quantizer = codec.quantizer
        self.decoder = codec.decoder
        self.up = codec.up

    def forward(self, codes):
        q = self.quantizer.vq2emb(codes.unsqueeze(0))  # (1, 1024, n)
        x = self.decoder(q).transpose(1, 2)            # (1, 1024, n)
        b, c, n = x.shape
        x = x.unsqueeze(-1).expand(b, c, n, 2).reshape(b, c, 2 * n)
        return self.up(x).transpose(1, 2)


def dit_attention(att, x, freqs_cis):
    """gpt_fast `Attention.forward`, self-attention, no mask."""
    from indextts.s2mel.modules.gpt_fast.model import apply_rotary_emb
    b, t, _ = x.shape
    kv = att.n_local_heads * att.head_dim
    q, k, v = att.wqkv(x).split([kv, kv, kv], dim=-1)
    q = q.view(b, t, att.n_head, att.head_dim)
    k = k.view(b, t, att.n_local_heads, att.head_dim)
    v = v.view(b, t, att.n_local_heads, att.head_dim)
    q = apply_rotary_emb(q, freqs_cis)
    k = apply_rotary_emb(k, freqs_cis)
    q, k, v = (a.transpose(1, 2) for a in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    y = y.transpose(1, 2).contiguous().view(b, t, att.head_dim * att.n_head)
    return att.wo(y)


class DiTStep(torch.nn.Module):
    """`DiT.forward` (s2mel `diffusion_transformer.py`) for inference: no
    class dropout, style and time not as tokens, wavenet final layer, every
    frame valid."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, prompt_x, t, style, cond):
        d = self.dit
        tr = d.transformer
        T = x.shape[2]
        t1 = d.t_embedder(t)
        cond = d.cond_projection(cond)
        x = x.transpose(1, 2)
        prompt_x = prompt_x.transpose(1, 2)
        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        x_in = torch.cat([x_in, style[:, None, :].expand(-1, T, -1)], dim=-1)
        x_in = d.cond_x_merge_linear(x_in)
        freqs_cis = tr.freqs_cis[:T]
        c = t1.unsqueeze(1)
        h = x_in
        skips = []
        for i, layer in enumerate(tr.layers):
            if tr.uvit_skip_connection and i in tr.layers_receive_skip:
                h = layer.skip_in_linear(torch.cat([h, skips.pop(-1)], dim=-1))
            a = h + dit_attention(layer.attention, layer.attention_norm(h, c), freqs_cis)
            h = a + layer.feed_forward(layer.ffn_norm(a, c))
            if tr.uvit_skip_connection and i in tr.layers_emit_skip:
                skips.append(h)
        x_res = tr.norm(h, c)
        x_res = d.skip_linear(torch.cat([x_res, x], dim=-1))
        y = d.conv1(x_res).transpose(1, 2)
        t2 = d.t_embedder2(t)
        ones = torch.ones(x.shape[0], 1, T, dtype=x.dtype, device=x.device)
        y = d.wavenet(y, ones, g=t2.unsqueeze(2)).transpose(1, 2) + d.res_projection(x_res)
        y = d.final_layer(y, t1).transpose(1, 2)
        return d.conv2(y)


class GPTBlocks(torch.nn.Module):
    """The GPT-2 stack of `UnifiedVoice` and its heads, shared by prefill and
    decode so the two graphs cannot drift (as `export_horizon32b.py`'s
    `HorizonBlocks`).

    The attention is written out rather than HF's: `q k^T / 8` plus an additive
    mask, softmax, `@ v`. HF's eager GPT-2 divides by `sqrt(head_dim)` and masks
    with `finfo.min` the same way; SDPA computes the same function.
    """

    def __init__(self, uv):
        super().__init__()
        self.gpt = uv.gpt
        self.final_norm = uv.final_norm
        self.mel_head = uv.mel_head
        self.mel_embedding = uv.mel_embedding
        self.mel_pos_embedding = uv.mel_pos_embedding
        self.text_embedding = uv.text_embedding
        self.text_pos_embedding = uv.text_pos_embedding
        self.lang_embedding = uv.lang_embedding
        self.spk_emb_proj = uv.spk_emb_proj
        self.start_mel_token = uv.start_mel_token
        self.heads = uv.gpt.config.n_head
        self.dim = uv.gpt.config.n_embd

    def layer(self, i, h, attend):
        blk = self.gpt.h[i]
        B, T, D = h.shape
        H, hd = self.heads, D // self.heads
        x = blk.ln_1(h)
        q, k, v = blk.attn.c_attn(x).split(D, dim=2)
        q, k, v = (a.view(B, T, H, hd).transpose(1, 2) for a in (q, k, v))
        o, k, v = attend(i, q, k, v)
        h = h + blk.attn.c_proj(o.transpose(1, 2).reshape(B, T, D))
        h = h + blk.mlp(blk.ln_2(h))
        return h, k, v

    def head(self, h):
        return self.mel_head(self.final_norm(self.gpt.ln_f(h)))


def attention(q, k, v, mask):
    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]) + mask
    return torch.matmul(torch.softmax(s, dim=-1), v)


class GPTPrefill(torch.nn.Module):
    """`prepare_gpt_inputs` plus the first forward of `inference_speech`.

    Inputs: the CAMPPlus style (1,192), the merged emotion vector (1,1280), the
    text ids with upstream's start (0) and stop (1) already placed and every
    other 0/1 removed (as `prepare_gpt_inputs`' `valid_mask` does), and the
    language id. The prefix is `[spk + emo, 0, 0][text + pos + lang][start_mel
    + mel_pos[0]]`, P = 3 + n + 1 rows, all visible causally.

    Returns the logits of the last row (1, 8194) and the K and V of all P rows,
    (layers, 1, heads, P, 64), which the decode cache is seeded from.
    """

    def __init__(self, blocks):
        super().__init__()
        self.b = blocks

    def forward(self, style, emovec, text, lang):
        b = self.b
        D = b.dim
        spk = b.spk_emb_proj(style) + emovec                          # (1, D)
        cond = torch.cat([spk.unsqueeze(1), spk.new_zeros(1, 2, D)], dim=1)
        n = text.shape[1]
        temb = b.text_embedding(text) + b.text_pos_embedding.emb(torch.arange(n, device=text.device)) \
            + b.lang_embedding(lang).unsqueeze(1)
        start = b.mel_embedding(torch.full((1, 1), b.start_mel_token, dtype=torch.long, device=text.device)) \
            + b.mel_pos_embedding.emb(torch.zeros(1, dtype=torch.long, device=text.device))
        h = torch.cat([cond, temb, start], dim=1)                      # (1, P, D)
        P = h.shape[1]
        i = torch.arange(P, device=h.device)
        neg = torch.full((), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device)
        mask = torch.where(i.view(1, P) <= i.view(P, 1), torch.zeros((), dtype=h.dtype, device=h.device), neg)
        ks, vs = [], []
        for layer in range(len(b.gpt.h)):
            h, k, v = b.layer(layer, h, lambda _, q, k, v: (attention(q, k, v, mask), k, v))
            ks.append(k)
            vs.append(v)
        return b.head(h[:, -1]), torch.stack(ks), torch.stack(vs)


class GPTDecode(torch.nn.Module):
    """One token per beam against a fixed-capacity cache.

    `beam_idx` reorders the cache first, as HF's `_reorder_cache` does after
    every beam step (`index_select` on the batch axis); for one beam it is
    `[0]`. `mel_pos` is the token's mel position, `slot` the cache row it is
    written to. Upstream skips mel position 1: the start token is 0 and the
    j-th generated token gets `attention_mask.shape[1] - mel_len = j + 1`
    (`GPT2InferenceModel.forward`), because the mask has already grown by one
    when the first generated token goes in.
    `mask` is `(1, 1, 1, kv)`: zero where a slot is filled, `finfo.min` after,
    and its width `kv` is how much of the cache the attention reads.
    """

    def __init__(self, blocks):
        super().__init__()
        self.b = blocks

    def forward(self, tokens, mel_pos, slot, beam_idx, k_cache, v_cache, mask):
        b = self.b
        # `k_cache[:, beam_idx]`, i.e. `index_select(1, beam_idx)`, written as the
        # indexing DNNKernels runs (`index.Tensor`; it has no `index_select`).
        K0 = k_cache[:, beam_idx]
        V0 = v_cache[:, beam_idx]
        h = b.mel_embedding(tokens) + b.mel_pos_embedding.emb(mel_pos)   # (B, 1, D)
        kv = mask.shape[-1]
        ks, vs = [], []

        def attend(i, q, k, v):
            K = K0[i].index_copy(2, slot, k)
            V = V0[i].index_copy(2, slot, v)
            ks.append(K)
            vs.append(V)
            return attention(q, K[:, :, :kv], V[:, :, :kv], mask), K, V

        for layer in range(len(b.gpt.h)):
            h, _, _ = b.layer(layer, h, attend)
        return b.head(h[:, -1]), torch.stack(ks), torch.stack(vs)


def prefix_text(text_tokens, start=0, stop=1):
    """`prepare_gpt_inputs`' text row: every 0 and 1 dropped, then wrapped."""
    t = text_tokens[0]
    t = t[(t != start) & (t != stop)]
    return torch.cat([t.new_tensor([start]), t, t.new_tensor([stop])]).unsqueeze(0).long()


# ── tracing as DNNKernels runs fastest ───────────────────────────────────────

def polyphase_transpose(x, w, bias, s, p, op, groups, dilation):
    """`conv_transpose1d` as `s` ordinary convolutions over every `s`-th tap,
    their outputs interleaved: the polyphase form of a strided transposed
    convolution. Output `n = q s + c` sums `x[i] W[k]` over `i s + k - p = n`;
    with `r = (c + p) % s` and `carry = (c + p) ÷ s` those are the taps
    `k = r + j s` at `i = q + carry - j`, a correlation of `x` with the taps
    reversed. The same products as the scatter form, and it reaches the 2-D
    implicit GEMM instead of DNNKernels' transposed-convolution kernel (85 ms of
    BigVGAN's 255 were its two stride-4 upsamplers)."""
    assert dilation == 1
    B, Cin, L = x.shape
    K = w.shape[-1]
    Cout = w.shape[1] * groups
    Lout = (L - 1) * s - 2 * p + K + op
    phases = []
    Q = -(-Lout // s)
    for c in range(s):
        r, carry = (c + p) % s, (c + p) // s
        taps = list(range(r, K, s))
        J = len(taps)
        if J == 0:
            phases.append(None)
            continue
        # (Cin, Cout/g, J) taps reversed -> conv weight (Cout, Cin/g, J)
        wt = w[:, :, taps[::-1]]
        if groups == 1:
            wc = wt.transpose(0, 1)
        else:
            cin_g, cout_g = Cin // groups, w.shape[1]
            wc = wt.reshape(groups, cin_g, cout_g, J).transpose(1, 2).reshape(groups * cout_g, cin_g, J)
        left = J - 1 - carry
        # Outputs q = 0..Q-1 read x[q + carry - (J-1) + t]: pad left by J-1-carry
        # (negative: drop), right so that Q outputs exist.
        need = Q + J - 1
        xl = F.pad(x, (left, 0)) if left >= 0 else x[:, :, -left:]
        right = need - xl.shape[-1]
        xr = F.pad(xl, (0, right)) if right >= 0 else xl[:, :, :need]
        yc = F.conv2d(xr.unsqueeze(2), wc.unsqueeze(2), None, 1, 0, 1, groups).squeeze(2)
        phases.append(yc)
    y = torch.stack([ph if ph is not None else torch.zeros_like(phases[0]) for ph in phases], dim=-1)
    y = y.reshape(B, Cout, Q * s)[:, :, :Lout]
    if bias is not None:
        y = y + bias[None, :, None]
    return y


@contextlib.contextmanager
def conv1d_as_conv2d():
    """Trace every `conv1d` and `conv_transpose1d` as the same 2-D convolution
    over a height of 1. DNNKernels lowers a 2-D convolution to a GEMM (1x1) or an
    implicit GEMM; a 1-D one, on a device without a library convolution, runs a
    gather kernel: 86 of the DiT's 122 ms were its 18 one-dimensional
    convolutions. The arithmetic is the same: each output is the same sum over
    the same taps."""
    c1, t1 = F.conv1d, F.conv_transpose1d

    def conv1d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
        one = lambda v: (1, v[0] if isinstance(v, (tuple, list)) else v)
        if isinstance(padding, str):
            raise ValueError("conv1d_as_conv2d: string padding is not handled")
        y = F.conv2d(x.unsqueeze(2), w.unsqueeze(2), bias, one(stride), (0, padding[0] if isinstance(padding, (tuple, list)) else padding),
                     one(dilation), groups)
        return y.squeeze(2)

    def conv_transpose1d(x, w, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        sc = lambda v: v[0] if isinstance(v, (tuple, list)) else v
        return polyphase_transpose(x, w, bias, sc(stride), sc(padding), sc(output_padding), groups, sc(dilation))

    F.conv1d, F.conv_transpose1d = conv1d, conv_transpose1d
    try:
        yield
    finally:
        F.conv1d, F.conv_transpose1d = c1, t1


@contextlib.contextmanager
def fused_rmsnorm():
    """gpt_fast's `RMSNorm` (`x * rsqrt(mean(x * x) + eps)` in float32, then the
    weight) spelled `pow(x, 2)`, the chain DNNKernels' `fusegroupedrms` rewrites
    into one op (pow, mean, add, rsqrt, scale, gain); `x * x` and `pow(x, 2)`
    are the same value. Unfused, its mean was 27 reductions at 0.36 ms each in
    the DiT."""
    from indextts.s2mel.modules.gpt_fast import model as gf
    orig = gf.RMSNorm.forward

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight

    gf.RMSNorm.forward = forward
    try:
        yield
    finally:
        gf.RMSNorm.forward = orig


# ── export plumbing ──────────────────────────────────────────────────────────

def refs(case="en_greedy"):
    return load_file(str(REFS / f"{case}.safetensors"))


def agree(name, got, want):
    got, want = got.detach().float().cpu(), want.detach().float().cpu()
    assert got.shape == want.shape, f"{name}: shape {tuple(got.shape)} vs {tuple(want.shape)}"
    err = (got - want).abs().max().item()
    scale = want.abs().max().item()
    print(f"  {name}: max |delta| {err:.3e} of absmax {scale:.3e}")
    return err / max(scale, 1e-12)


def prune_unused_weights(graph):
    """Drop unreferenced lifted buffers, following view parents as well as ops.

    torch.export lifts GPT-2's cached causal masks even though our attention
    wrappers never read them: 24 boolean 2420-by-2420 tables per GPT export.
    """
    buffers = {b["id"]: b for b in graph["buffers"]}
    used = set(graph["inputs"]) | set(graph["outputs"])
    used.update(i for op in graph["ops"] for i in op["in"])
    stack = list(used)
    while stack:
        parent = buffers.get(stack.pop(), {}).get("of")
        if parent and parent not in used:
            used.add(parent)
            stack.append(parent)
    graph["buffers"] = [b for b in graph["buffers"]
                        if b.get("kind") != "weight" or b["id"] in used]
    return graph


def write(name, ep, module, args, argnames, y, specs):
    out = GEN / "graphs" / name
    out.mkdir(parents=True, exist_ok=True)
    g = prune_unused_weights(EG.convert(ep, specs, name))
    (out / f"{name}.json").write_text(json.dumps(g, indent=1))
    # Every name a weight is reachable by: torch.export names it by the path
    # the trace took, which for a module held twice is not the deduplicated one.
    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(module.named_parameters(remove_duplicate=False)),
                            **dict(module.named_buffers(remove_duplicate=False))}.items()}
    tensors.update({k: v.detach().contiguous().cpu() for k, v in ep.constants.items()
                    if isinstance(v, torch.Tensor)})
    keys = {b["key"] for b in g["buffers"] if b.get("kind") == "weight" and b.get("key") is not None}
    missing = keys - set(tensors)
    assert not missing, f"{name}: graph wants weights the module does not have: {sorted(missing)[:5]}"
    save_file({k: tensors[k] for k in sorted(keys)}, str(out / "weights.safetensors"))
    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))
    # Small inputs only: the decode's caches are 800 MB each and say nothing a
    # zero tensor of the declared shape would not.
    ref = {n: a.detach().contiguous().cpu() for n, a in zip(argnames, args)
           if a.numel() * a.element_size() < 64 * 2**20}
    ys = y if isinstance(y, tuple) else (y,)
    ref.update({("out" if i == 0 else f"out{i}"): t.detach().contiguous().cpu() for i, t in enumerate(ys)
                if t.numel() * t.element_size() < 64 * 2**20})
    save_file(ref, str(out / "reference.safetensors"))
    nb = sum(tensors[k].numel() * tensors[k].element_size() for k in keys)
    print(f"  {name}: {len(g['ops'])} ops, {len(keys)} weights, {nb / 2**20:.1f} MiB -> {out}")
    print("  ops:", json.dumps(hist, sort_keys=True))


class Recorder(torch.fx.Interpreter):
    """Runs an exported program and keeps every intermediate (as `dump_refs.py`)."""

    def __init__(self, gm):
        super().__init__(gm)
        self.results = {}

    def run_node(self, n):
        out = super().run_node(n)
        if isinstance(out, torch.Tensor):
            self.results[n.name] = out.detach().contiguous().cpu()
        elif isinstance(out, (tuple, list)):
            for i, v in enumerate(out):
                if isinstance(v, torch.Tensor):
                    self.results[f"{n.name}.{i}"] = v.detach().contiguous().cpu()
        return out


NODES = {"on": False}


def lifted_inputs(ep, args):
    """The flat placeholder values of `ep.graph_module`: parameters, buffers and
    constants as the signature lifted them, then the user inputs."""
    from torch.export.graph_signature import InputKind
    flat, user = [], iter(args)
    for spec in ep.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            flat.append(next(user))
        elif spec.kind == InputKind.PARAMETER:
            flat.append(ep.state_dict[spec.target])
        elif spec.kind == InputKind.BUFFER:
            flat.append(ep.state_dict[spec.target] if spec.persistent else ep.constants[spec.target])
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            flat.append(ep.constants[spec.target])
        else:
            raise ValueError(f"input kind {spec.kind} of {spec.arg.name}")
    return flat


def write_nodes(name, ep, args):
    """`graphs/<name>/nodes.safetensors`: `"<name>/in<i>"` and every node's value
    as `"<name>/node/<node>"`, the contract `DNNKernels.verifygraph` reads.

    Over `ep.graph_module`, the graph `EG.convert` reads, with the lifted
    weights as placeholders. `ep.module()` would unlift them and rename nodes,
    and from the first renamed node on every value keys to the wrong op."""
    rec = Recorder(ep.graph_module)
    with torch.no_grad():
        rec.run(*lifted_inputs(ep, args))
    refs = {f"{name}/in{i}": a.detach().contiguous().cpu() for i, a in enumerate(args)}
    refs.update({f"{name}/node/{k}": v for k, v in rec.results.items()})
    save_file(refs, str(GEN / "graphs" / name / "nodes.safetensors"))
    print(f"  {name}: {len(rec.results)} node values")


PRECISION = {"fp16": False}


def precision_ctx():
    """fp16 autocast when `--fp16`: matmuls and convolutions in fp16, norms and
    reductions in fp32, the casts recorded in the graph (`export_graphs.py`'s
    `precision_ctx`). The graph is then `<part>_fp16`."""
    if PRECISION["fp16"]:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def export(name, module, args, argnames, specs, shapes):
    """Trace, decompose and write one part.

    `shapes` uses `Dim.DYNAMIC` rather than named ranges. Strided convolutions
    leave guards like `max(1, 1 + (t-1)//2) == 1 + (t-1)//2`, true for every
    t >= 1, which the solver cannot prove over a declared range and reports as
    a violation; DYNAMIC lets it refine the range instead. The symbol names in
    the JSON come from `specs`, which `EG.convert` reads off the placeholders.
    """
    if PRECISION["fp16"]:
        with torch.no_grad():
            y32 = module(*args)
    with torch.no_grad(), precision_ctx():
        y = module(*args)
        ep = torch.export.export(module, args, dynamic_shapes=shapes, strict=False)
        ep = ep.run_decompositions()
    if PRECISION["fp16"]:
        y0 = y[0] if isinstance(y, tuple) else y
        r0 = y32[0] if isinstance(y32, tuple) else y32
        d = (y0.float() - r0.float()).abs()
        print(f"  {name} fp16 vs fp32: max {d.max().item() / r0.abs().max().item():.3e}, "
              f"mean {d.mean().item() / r0.abs().max().item():.3e} of absmax")
        name = name + "_fp16"
    write(name, ep, module, args, argnames, y, specs)
    if NODES["on"]:
        write_nodes(name, ep, args)
    return y


# ── the parts ────────────────────────────────────────────────────────────────

def part_w2v(dev):
    model, mean, std = load_w2v(dev)
    w = W2V(model, mean, std).eval()
    # Both parities of the Kaldi frame count: en_greedy's speaker clip has an
    # even count, en_emoclip's emotion clip an odd one, so its last pair masked.
    for case, key in (("en_greedy", ""), ("en_emoclip", "/1")):
        r = refs(case)
        feats = r["ref/features" + key].to(dev)
        mask = r["ref/features_mask" + key].to(dev)
        with torch.no_grad():
            want = (model(input_features=feats, attention_mask=mask,
                          output_hidden_states=True).hidden_states[17] - mean) / std
            got = w(feats, mask.float())
            print(f"  {case}{key}: {int(mask.sum())} of {mask.shape[1]} pairs valid")
            assert agree("w2v vs upstream module", got, want) < 1e-5
            assert agree("w2v vs recorded", got, r["ref/w2v" + key].to(dev)) < 1e-5
    r = refs("en_emoclip")
    feats, mask = r["ref/features/1"].to(dev), r["ref/features_mask/1"].to(dev).float()
    t = torch.export.Dim.DYNAMIC
    export("w2v", w, (feats, mask), ("features", "mask"), ({1: "t"}, {1: "t"}), ({1: t}, {1: t}))


def seg_pooling(self, x, seg_len=100, stype="avg"):
    """`CAMLayer.seg_pooling` without `avg_pool1d(ceil_mode=True)`.

    The ceil-mode output size is computed with branches on the length, and
    torch.export specialises every one of them to the traced clip; so does any
    reshape into a symbolic number of segments. This is the same mean from a
    running sum, indexed by `arange` alone: frame `i` lies in the segment
    `[s, min(s + 100, t))`, `s = i - i % 100`, whose sum is `c[e] - c[s]` with
    `c` the cumulative sum behind a zero. Past the last frame `c` is extended
    by its own last value, so `c[s + 100]` is `c[min(s + 100, t)]` without a
    `min` against the length; the frame count is the same difference over ones.
    ATen's avg_pool divides by exactly that count (`count_include_pad`, no
    padding). The difference of two running sums rounds differently from the
    window sum, by ~1e-7 of the sum."""
    assert stype == "avg"
    t = x.shape[-1]
    i = torch.arange(t, device=x.device)
    s = i - torch.remainder(i, seg_len)

    def windows(v):
        c = torch.cat([torch.zeros_like(v[..., :1]), torch.cumsum(v, dim=-1)], dim=-1)
        c = torch.cat([c, c[..., -1:].expand(*c.shape[:-1], seg_len)], dim=-1)
        return c[..., s + seg_len] - c[..., s]

    return windows(x) / windows(torch.ones_like(x[..., :1, :]))


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    """`campplus.layers.statistics_pooling` with the unbiased std written out
    (DNNKernels has no `var.correction`): two passes, the count as a tensor so
    no symbolic scalar reaches the graph."""
    assert dim == -1 and not keepdim and unbiased
    mean = x.mean(dim=-1)
    d = x - mean.unsqueeze(-1)
    n = torch.ones_like(x[..., :1, :]).sum(dim=-1)
    std = torch.sqrt((d * d).sum(dim=-1) / (n - 1))
    return torch.cat([mean, std], dim=-1)


def part_campplus(dev):
    from indextts.s2mel.modules.campplus import layers
    r = refs()
    m = load_campplus(dev)
    x = r["ref/campplus_in"].to(dev)
    with torch.no_grad():
        want = m(x)
        assert agree("campplus vs recorded", want, r["ref/style"].to(dev)) < 1e-5
    original = layers.CAMLayer.seg_pooling
    original_stats = layers.statistics_pooling
    layers.CAMLayer.seg_pooling = seg_pooling
    layers.statistics_pooling = statistics_pooling
    try:
        with torch.no_grad():
            assert agree("campplus, pooling rewritten, vs upstream", m(x), want) < 1e-5
            # A second length, one past a whole segment, through both.
            x2 = torch.randn(1, 401, 80, device=dev)
            got2 = m(x2)
            layers.CAMLayer.seg_pooling, layers.statistics_pooling = original, original_stats
            want2 = m(x2)
            layers.CAMLayer.seg_pooling, layers.statistics_pooling = seg_pooling, statistics_pooling
            assert agree("campplus, 401 frames, rewritten vs upstream", got2, want2) < 1e-5
        export("campplus", m, (x,), ("fbank",), ({1: "t"},), ({1: torch.export.Dim.DYNAMIC},))
    finally:
        layers.CAMLayer.seg_pooling = original
        layers.statistics_pooling = original_stats


def part_emo(dev):
    r = refs()
    gpt = load_gpt(dev)
    e = Emo(gpt).eval()
    x = r["emo/get_emovec_in"].to(dev)
    with torch.no_grad():
        want = gpt.get_emovec(x, torch.tensor([x.shape[1]], device=dev))
        assert agree("emo vs upstream", e(x), want) < 1e-5
        assert agree("emo vs recorded", e(x), r["emo/get_emovec"].to(dev)) < 1e-5
    t = torch.export.Dim.DYNAMIC
    export("emo", e, (x,), ("w2v",), ({1: "t"},), ({1: t},))


def part_regulator(dev):
    r = refs()
    s2mel = load_s2mel(dev)
    reg = s2mel.models["length_regulator"]
    w = Regulator(reg).eval()
    for i, suffix in enumerate(("", "/1")):
        x = r["regulator/in" + suffix].to(dev)
        y = int(r["regulator/ylens" + suffix][0])
        idx = nearest_index(x.shape[1], y).to(dev)
        with torch.no_grad():
            want = reg(x, ylens=torch.tensor([y], device=dev), n_quantizers=3, f0=None)[0]
            assert agree(f"regulator[{i}] vs upstream", w(x, idx), want) < 1e-5
            assert agree(f"regulator[{i}] vs recorded", w(x, idx), r["regulator/out" + suffix].to(dev)) < 1e-5
    x = r["regulator/in/1"].to(dev)
    idx = nearest_index(x.shape[1], int(r["regulator/ylens/1"][0])).to(dev)
    t = torch.export.Dim.DYNAMIC
    y = torch.export.Dim.DYNAMIC
    export("regulator", w, (x, idx), ("x", "index"), ({1: "t"}, {0: "y"}), ({1: t}, {0: y}))


def part_codec(dev):
    r = refs()
    codec = load_codec(dev)
    w = CodecDecode(codec).eval()
    codes = r["codec/codes"].to(dev)
    with torch.no_grad():
        assert agree("codec vs upstream", w(codes), codec.decode(codes)) < 1e-5
        assert agree("codec vs recorded", w(codes), r["codec/out"].to(dev)) < 1e-5
    n = torch.export.Dim.DYNAMIC
    export("codec", w, (codes,), ("codes",), ({1: "n"},), ({1: n},))


def fold_weight_norm(module):
    """Replace every weight-normalised layer's `g * v / |v|` by the weight it
    computes, once. Upstream recomputes it on every forward; folded, the graph
    reads a plain weight. Old-style hooks (`weight_g`/`weight_v`) and
    parametrizations are both handled."""
    from torch.nn.utils import parametrize
    n = 0
    for m in module.modules():
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            torch.nn.utils.remove_weight_norm(m)
            n += 1
        elif parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
            n += 1
    return n


def reflect_pad1d(x, paddings, mode="zero", value=0.0):
    """`encodec.pad1d` for clips longer than the pad, reflect mode as flipped
    slices. ATen decomposes `reflection_pad1d` into `(t-1) - |(t-1) - |i||`
    index arithmetic on the symbolic length, which the graph converter records
    with the length both as an input and as an attribute. The flipped slices are
    the same values: reflect padding repeats `x[1:l+1]` reversed on the left
    and `x[-r-1:-1]` reversed on the right."""
    left, right = paddings
    if mode != "reflect":
        return F.pad(x, paddings, mode, value)
    parts = []
    if left > 0:
        parts.append(x[..., 1:left + 1].flip(-1))
    parts.append(x)
    if right > 0:
        parts.append(x[..., -right - 1:-1].flip(-1))
    return torch.cat(parts, dim=-1)


def part_dit(dev):
    from indextts.s2mel.modules import encodec
    r = refs()
    s2mel = load_s2mel(dev)
    dit = s2mel.models["cfm"].estimator
    print(f"  folded {fold_weight_norm(dit)} weight norms")
    w = DiTStep(dit).eval()
    args = tuple(r[f"dit/{k}"].to(dev) for k in ("x", "prompt_x", "t", "style", "mu"))
    with torch.no_grad():
        x, px, t, style, mu = args
        want = dit(x, px, r["dit/x_lens"].to(dev), t, style, mu)
        assert agree("dit vs upstream", w(*args), want) < 1e-5
        assert agree("dit vs recorded", w(*args), r["dit/out"].to(dev)) < 1e-5
    original = encodec.pad1d
    encodec.pad1d = reflect_pad1d
    try:
        with torch.no_grad(), conv1d_as_conv2d(), fused_rmsnorm():
            assert agree("dit, flipped-slice reflect pad, 2-D convolutions, fused RMS norm, vs upstream",
                         w(*args), want) < 1e-5
            T = torch.export.Dim.DYNAMIC
            export("dit", w, args, ("x", "prompt_x", "t", "style", "mu"),
                   ({2: "t"}, {2: "t"}, {}, {}, {1: "t"}), ({2: T}, {2: T}, {}, {}, {1: T}))
    finally:
        encodec.pad1d = original


def part_bigvgan(dev):
    r = refs()
    m = load_bigvgan(dev)
    mel = r["bigvgan/in"].to(dev)
    with torch.no_grad():
        want = m(mel)
        assert agree("bigvgan vs recorded", want, r["bigvgan/out"].to(dev)) < 1e-4
        with conv1d_as_conv2d():
            # The polyphase upsamplers sum the same products in another order
            # than cuDNN's transposed convolution: ~3e-4 of range at the
            # worst sample, ~2e-6 on average, in fp32.
            got = m(mel)
            print(f"  mean |delta| {(got - want).abs().mean().item() / want.abs().max().item():.3e} of absmax")
            assert agree("bigvgan, 2-D convolutions, vs upstream", got, want) < 1e-3
            t = torch.export.Dim.DYNAMIC
            export("bigvgan", m, (mel,), ("mel",), ({2: "t"},), ({2: t},))


MAXLEN = 2176      # cache rows: a 130-row prefix and 1500 mel tokens fit, kv buckets reach 2048


def part_gpt(dev, checkpoint=None, case="en_greedy", name="gpt"):
    """`gpt_prefill` and `gpt_decode`, checked against the recorded greedy decode:
    the prefill's logits against step 0, then every later step teacher-forced
    with the recorded tokens, against that step's recorded logits."""
    r = refs(case)
    uv = load_gpt(dev, checkpoint)
    blocks = GPTBlocks(uv).eval()
    pre, dec = GPTPrefill(blocks).eval(), GPTDecode(blocks).eval()
    style, emovec = r["gpt/campplus"].to(dev), r["gpt/emovec"].to(dev)
    text = prefix_text(r["gpt/text_tokens"]).to(dev)
    lang = r["gpt/lang"].to(dev)
    codes = r["gpt/codes"][0]
    L, H, hd = len(uv.gpt.h), blocks.heads, blocks.dim // blocks.heads
    with torch.no_grad():
        logits, k, v = pre(style, emovec, text, lang)
        # Upstream's prefix is left-padded with masked rows when the text had
        # 0/1 ids stripped (its trailing pad id always is); masked, they change
        # nothing for the other rows, so this prefix has none.
        P = k.shape[3]
        worst = agree("prefill logits vs recorded step 0", logits, r["gpt/logits"].to(dev))
        kc = torch.zeros(L, 1, H, MAXLEN, hd, device=dev)
        vc = torch.zeros_like(kc)
        kc[:, :, :, :P], vc[:, :, :, :P] = k, v
        nsteps = len(codes) - 1
        for j in range(1, nsteps + 1):
            slot = P - 1 + j
            kvw = 256 * ((slot + 1 + 255) // 256)
            mask = torch.where(torch.arange(kvw, device=dev) <= slot, 0.0, torch.finfo(torch.float32).min).view(1, 1, 1, kvw)
            tok = codes[j - 1].view(1, 1).to(dev)
            lj, kc, vc = dec(tok, torch.tensor([j + 1], device=dev), torch.tensor([slot], device=dev),
                             torch.tensor([0], device=dev), kc, vc, mask)
            want = r[f"gpt/logits/{j}"].to(dev)
            e = (lj - want).abs().max().item() / want.abs().max().item()
            worst = max(worst, e)
        print(f"  decode: {nsteps} teacher-forced steps, worst max|delta| {worst:.3e} of range")
        assert worst < 1e-4
    out = GEN / "graphs" / name
    pargs = (style, emovec, text, lang)
    export(f"{name}_prefill", pre, pargs, ("style", "emovec", "text", "lang"),
           ({}, {}, {1: "n"}, {}), ({}, {}, {1: torch.export.Dim.DYNAMIC}, {}))
    B = 3
    slot = P
    kvw = 256
    mask = torch.where(torch.arange(kvw, device=dev) <= slot, 0.0, torch.finfo(torch.float32).min).view(1, 1, 1, kvw)
    dargs = (torch.full((B, 1), int(codes[0]), device=dev), torch.tensor([2], device=dev),
             torch.tensor([slot], device=dev), torch.zeros(B, dtype=torch.long, device=dev),
             torch.zeros(L, B, H, MAXLEN, hd, device=dev), torch.zeros(L, B, H, MAXLEN, hd, device=dev), mask)
    kv = torch.export.Dim("kv", min=2, max=MAXLEN - 1)
    b = torch.export.Dim.DYNAMIC
    export(f"{name}_decode", dec, dargs, ("tokens", "mel_pos", "slot", "beam_idx", "k_cache", "v_cache", "mask"),
           ({0: "b"}, {}, {}, {0: "b"}, {1: "b"}, {1: "b"}, {3: "kv"}),
           ({0: b}, {}, {}, {0: b}, {1: b}, {1: b}, {3: kv}))


PARTS = {"w2v": part_w2v, "campplus": part_campplus, "emo": part_emo, "regulator": part_regulator,
         "codec": part_codec, "dit": part_dit, "bigvgan": part_bigvgan, "gpt": part_gpt,
         "gpt_de": lambda dev: part_gpt(dev, GERMAN / "gpt.pth", "de_greedy", "gpt_de")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", action="append", required=True, help=f"one of {list(PARTS)} or all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--nodes", action="store_true",
                   help="also write every node's value, for DNNKernels.verifygraph")
    p.add_argument("--fp16", action="store_true", help="trace under fp16 autocast, as <part>_fp16")
    args = p.parse_args()
    NODES["on"] = args.nodes
    PRECISION["fp16"] = args.fp16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parts = list(PARTS) if "all" in args.part else args.part
    for name in parts:
        print(f"== {name}", flush=True)
        PARTS[name](args.device)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
