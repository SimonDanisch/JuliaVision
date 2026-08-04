"""Export ONE decoder step of Whisper large-v3-turbo to DNNKernels' graph JSON.

    uv run tools/export_whisper_decoder.py

## The decoder is a static graph plus a host loop

`export_whisper.py` says the decoder "is autoregressive with a KV cache, which
`torch.export` will not capture". That is true of the *generation loop* and not
of a *step*: with the cache passed in and out as ordinary tensors, one step is
static-shaped and exports cleanly. That is the same decomposition llama.cpp and
whisper.cpp use — the graph is one token's worth of work, and the loop, the
sampling and the cache rotation live on the host.

So the signature is

    (input_ids, self_k, self_v, cross_k, cross_v, cache_position)
        -> (logits, new_self_k, new_self_v)

with `self_k/v` shaped `(layers, 1, heads, max_target, head_dim)` and `cross_k/v`
`(layers, 1, heads, 1500, head_dim)`.

Note what is NOT in the signature: `encoder_hidden`. Per token, cross-attention
only projects the *query* — its K and V come from the encoder output, which does
not change while decoding a window, so they are computed once by the separate
`whispercross` graph and handed in. Passing the encoder hidden state to the step
would invite recomputing 1500 positions for every one of ~200 tokens.

## Why the step is written out rather than driven through `DynamicCache`

The obvious export is `decoder(..., past_key_values=EncoderDecoderCache(...))`,
and it produces a graph that is quietly useless. Three things go wrong at once,
and only the third is fatal:

  1. `DynamicCache` seeds every layer with `torch.tensor([])`, which
     `torch.export` lifts into the signature as `lifted_tensor_0..15` — sixteen
     zero-element weights that no `safetensors` file has, so the load fails
     naming a tensor that exists in no module.
  2. Feeding it the cache as a legacy tuple makes it `cat` the empty seed onto
     the real cache, so every step copies all 80 MB of cache twice for nothing.
  3. **The updated cache never leaves the graph.** `DynamicCache` accumulates
     into a Python object; the exported outputs are `[logits]` and nothing else.
     A host loop driving that graph attends to a zeroed cache at every step and
     produces fluent nonsense — the failure is silent.

So the step below is written against the model's own submodules
(`layer.self_attn.q_proj`, `layer.final_layer_norm`, ...) with the cache write
made explicit. It is the same arithmetic — `checkparity` asserts that against
the real `WhisperDecoder` before anything is written to disk — with the cache as
data instead of as state.

## The cache is a fixed buffer, written at `cache_position`

`self_k` is the whole `max_target`-slot buffer, the new key is written into slot
`cache_position`, and attention runs over all `max_target` slots with everything
past `cache_position` masked out. That is `StaticCache`'s shape, and it is the
only shape that is static: `cat`-ing the cache to `position + 1` makes the
sequence axis a function of how many tokens have been generated.

The write is `index_copy`, which is functional, so the new cache is an output and
the host copies it back — ~18 MB per token, ~0.12 ms at this card's bandwidth
against a 2.07 ms floor. An in-place write would save that; it needs DNNKernels
to express output-aliases-input, which it does not today. Correctness first, and
the op is the one a later in-place path would specialise.

## Export on CUDA, not CPU

`export_whisper.py`'s own note: from CPU the graph has `bmm` and no `sdpa`, from
CUDA it has `sdpa` and no `bmm`. The same holds here and it is worth far more
for the decoder — a CPU export decomposes every attention into
`bmm + _softmax + logical_not + where + any.dim + full_like`, roughly 90 ops of
masking scaffolding that a fused `sdpa` expresses in one. Measured: 531 ops from
CPU. Export from CUDA.

## 4 layers, not 32

large-v3-turbo is distilled to **4** decoder layers against the encoder's 32.
Per generated token the decoder reads ~158M parameters — 92M of decoder weights
plus the 66M output projection over the 51866-token vocabulary — so at this
card's bandwidth a token costs roughly 2 ms and essentially all of it is weight
traffic. Every matmul is `M = 1`: this is a GEMV regime, and none of the GEMM
tiling in `perf-plan.md` applies to it.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

import export_graphs as EG

from common import find_root
ROOT = find_root()
GEN = ROOT / "gen"
WEIGHTS = GEN / "whisper"

SRC_LEN = 1500          # encoder output positions for a 30 s window

# Same table as `export_whisper.py`: `autocast` stores fp32 and runs the matmuls
# in fp16, which is a different thing from an fp16 export.
STORAGE = {"fp32": torch.float32, "fp16": torch.float16, "autocast": torch.float32}


def sdpa(q, k, v, mask):
    """`scaled_dot_product_attention` with the scale already folded into `q`.

    `scale=1.0` and not the default, deliberately, and it is HF's own choice —
    `WhisperAttention.forward` multiplies the query by `head_dim ** -0.5` before
    the attention call and then passes `scaling=1.0`, with the comment that
    Whisper is one of the models whose outputs move when the scaling is
    reassociated. Matching the reference bit for bit means matching where the
    multiply happens.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=1.0)


class DecoderStep(torch.nn.Module):
    """One decoder token, cache in and cache out as plain tensors.

    Holds the whole `WhisperForConditionalGeneration` rather than the pieces it
    uses, because `torch.export` names weights by their path in the exported
    module: holding `model` makes the graph ask for
    `model.model.decoder.layers.0.fc1.weight`, which is exactly the key
    `model.named_parameters()` yields. Reaching in for submodules and stashing
    them in a fresh `ModuleList` renames every tensor and the load fails on a
    weight that is right there under another name.
    """

    def __init__(self, model, heads, headdim):
        super().__init__()
        # `proj_out` is TIED to `embed_tokens` — one 51866x1280 tensor, 265 MB,
        # reached by two names, and `torch.export` picks one of them for both
        # uses (here `model.proj_out.weight`, for the *embedding* as well).
        # `forward` writes `linear(h, embed_tokens.weight)` so that only the tied
        # tensor is ever named, and `main` looks weights up in `state_dict()`,
        # which unlike `named_parameters()` carries both names.
        #
        # Checked rather than assumed: a checkpoint that unties them would decode
        # with the wrong output projection and no error at all.
        assert (model.proj_out.weight.data_ptr()
                == model.model.decoder.embed_tokens.weight.data_ptr()), \
            "proj_out is not tied to embed_tokens; this export would use the wrong weight"
        self.model = model
        self.heads, self.headdim = heads, headdim

    def forward(self, input_ids, self_k, self_v, cross_k, cross_v, cache_position):
        dec = self.model.model.decoder
        H, hd = self.heads, self.headdim
        B, T = self_k.shape[1], self_k.shape[3]
        D = H * hd

        h = dec.embed_tokens(input_ids) + dec.embed_positions.weight[cache_position]

        # Slots at or before `cache_position` are live; the rest are the previous
        # window's, or zeros. `finfo.min` and not `-inf` — the same value HF's own
        # mask builder uses, and it survives an fp16 cast, which `-inf` does not.
        idx = torch.arange(T, device=h.device)
        neg = torch.full((), torch.finfo(h.dtype).min, dtype=h.dtype, device=h.device)
        mask = torch.where(idx <= cache_position,
                           torch.zeros((), dtype=h.dtype, device=h.device),
                           neg).view(1, 1, 1, T).expand(B, H, 1, T)

        newk, newv = [], []
        for i, layer in enumerate(dec.layers):
            # ---- self-attention, over the cache
            r = h
            h = layer.self_attn_layer_norm(h)
            a = layer.self_attn
            q = (a.q_proj(h) * a.scaling).view(B, 1, H, hd).transpose(1, 2)
            k = a.k_proj(h).view(B, 1, H, hd).transpose(1, 2)
            v = a.v_proj(h).view(B, 1, H, hd).transpose(1, 2)
            # the whole cache, with this token's k/v written into slot `position`
            K = self_k[i].index_copy(2, cache_position, k)
            V = self_v[i].index_copy(2, cache_position, v)
            newk.append(K)
            newv.append(V)
            o = sdpa(q, K, V, mask)
            h = r + a.out_proj(o.transpose(1, 2).reshape(B, 1, D))

            # ---- cross-attention: K/V are the window's, only Q is per token
            r = h
            h = layer.encoder_attn_layer_norm(h)
            c = layer.encoder_attn
            q = (c.q_proj(h) * c.scaling).view(B, 1, H, hd).transpose(1, 2)
            o = sdpa(q, cross_k[i], cross_v[i], None)
            h = r + c.out_proj(o.transpose(1, 2).reshape(B, 1, D))

            # ---- feed-forward
            r = h
            h = layer.final_layer_norm(h)
            h = r + layer.fc2(layer.activation_fn(layer.fc1(h)))

        h = dec.layer_norm(h)
        logits = torch.nn.functional.linear(h, dec.embed_tokens.weight)
        return logits, torch.stack(newk), torch.stack(newv)


class CrossKV(torch.nn.Module):
    """The cross-attention K/V projections, for the whole window at once.

    `cross_k`/`cross_v` are *inputs* to the decoder step, and something has to
    produce them. They are `encoder_attn.{k,v}_proj` applied to the encoder
    output — 8 matmuls of (1500,1280)@(1280,1280) for the 4 turbo layers — and
    they are computed ONCE per 30 s window, not per token. That is most of why
    the turbo decoder is cheap.

    Holds the same `model` object as `DecoderStep` for the same reason: both
    graphs read one `weights.safetensors`, so both must name these tensors
    `model.model.decoder.layers.0.encoder_attn.k_proj.weight`.
    """

    def __init__(self, model, heads, head_dim):
        super().__init__()
        self.model = model
        self.heads, self.head_dim = heads, head_dim

    def forward(self, encoder_hidden):
        B, S, _ = encoder_hidden.shape
        ks, vs = [], []
        for layer in self.model.model.decoder.layers:
            attn = layer.encoder_attn
            k = attn.k_proj(encoder_hidden)
            v = attn.v_proj(encoder_hidden)
            k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)
            ks.append(k)
            vs.append(v)
        return torch.stack(ks), torch.stack(vs)


def checkparity(model, step, cross, L, H, hd, D, dev, dt, maxtarget, ntok=6):
    """Assert the hand-written step is the real decoder, before exporting it.

    The step below is written out by hand (see the module docstring for why), and
    a hand-written transformer that is subtly wrong — a layer norm on the wrong
    side of a residual, a mask off by one slot, the query scaled twice — still
    produces plausible logits and a plausible transcript. So it is checked
    against `WhisperDecoder` itself, run the way generation runs it, for several
    tokens: a mask that is off by one only diverges once there is more than one
    cached token, so a single-token check would pass on a broken cache.

    Runs before anything is written. An export that does not match is not worth
    the 656 MiB it would take to find out on the GPU.
    """
    torch.manual_seed(0)
    enc = torch.randn(1, SRC_LEN, D, device=dev, dtype=dt)
    ids = torch.randint(0, 50000, (ntok,), device=dev)

    # reference: the real decoder, one token at a time, HF's own cache
    cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
    ref = []
    for t in range(ntok):
        out = model.model.decoder(
            input_ids=ids[t].view(1, 1),
            encoder_hidden_states=enc,
            past_key_values=cache,
            cache_position=torch.tensor([t], device=dev),
            use_cache=True)
        ref.append(model.proj_out(out.last_hidden_state))

    # ours: the exported decomposition, cache as tensors
    ck, cv = cross(enc)
    sk = torch.zeros(L, 1, H, maxtarget, hd, device=dev, dtype=dt)
    sv = torch.zeros_like(sk)
    worst = 0.0
    for t in range(ntok):
        logits, sk, sv = step(ids[t].view(1, 1), sk, sv, ck, cv,
                              torch.tensor([t], device=dev))
        d = (logits - ref[t]).float()
        rel = (d.pow(2).mean().sqrt() / ref[t].float().pow(2).mean().sqrt()).item()
        worst = max(worst, rel)
        print(f"    token {t}: rel rms {rel:.3e}  "
              f"argmax {'==' if logits.argmax() == ref[t].argmax() else '!='}")
        assert logits.argmax() == ref[t].argmax(), f"token {t}: argmax differs"
    # 1e-4 and not 1e-6: `index_copy` + a full-length masked attention is a
    # different reduction ORDER from cat-to-length, and fp32 attention over 448
    # slots of which 440 are masked accumulates a little differently. The argmax
    # assert above is the one that would catch a real error.
    assert worst < 1e-4, f"hand-written step diverges: worst rel rms {worst:.3e}"
    print(f"  parity vs WhisperDecoder over {ntok} tokens: worst rel rms {worst:.2e}")


def main(out: Path, precision: str, dev: str, maxtarget: int):
    model = WhisperForConditionalGeneration.from_pretrained(
        WEIGHTS, dtype=STORAGE[precision]).eval().to(dev)
    cfg = model.config
    L = cfg.decoder_layers
    H = cfg.decoder_attention_heads
    D = cfg.d_model
    hd = D // H
    dt = STORAGE[precision]

    step = DecoderStep(model, H, hd).eval()
    cross = CrossKV(model, H, hd).eval()

    with torch.no_grad(), EG.precision_ctx(precision):
        checkparity(model, step, cross, L, H, hd, D, dev, dt, maxtarget)

    def z(*shape):
        return torch.zeros(*shape, device=dev, dtype=dt)

    args = (
        torch.zeros(1, 1, dtype=torch.long, device=dev),      # input_ids
        z(L, 1, H, maxtarget, hd), z(L, 1, H, maxtarget, hd), # self KV cache
        z(L, 1, H, SRC_LEN, hd), z(L, 1, H, SRC_LEN, hd),     # cross KV cache
        torch.zeros(1, dtype=torch.long, device=dev),         # cache_position
    )

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(step, args, strict=False)
        ep = ep.run_decompositions()      # must stay inside the precision context

    g = EG.convert(ep, ({},), "whisperdec")
    out.mkdir(parents=True, exist_ok=True)

    # From the WRAPPER, not from `model.model.decoder`. `torch.export` names a
    # buffer by its path in the exported module, so a graph built on the wrapper
    # asks for `model.model.decoder.embed_positions.weight` while
    # `decoder.named_parameters()` yields `embed_positions.weight` — and the load
    # fails with "missing weight" naming a tensor that is right there under a
    # different key.
    #
    # `ep.constants` too: `torch.export` LIFTS tensor constants that are neither
    # parameters nor buffers into the signature under names like
    # `lifted_tensor_0`. There should be none of those now — the sixteen the
    # `DynamicCache` export produced were its empty per-layer seeds — but a
    # constant that appears later would otherwise fail at load rather than here.
    # `state_dict()` and not `named_parameters()`: `proj_out.weight` and
    # `embed_tokens.weight` are ONE tensor under two names, `named_parameters()`
    # deduplicates it, and `torch.export` canonicalises the placeholder to
    # whichever name it found — here `model.proj_out.weight`, even though
    # `forward` writes `dec.embed_tokens.weight`. `state_dict()` carries both, so
    # the lookup succeeds whichever one the exporter picked; `prune` then makes
    # sure only the one the graph actually reads is written.
    available = {**dict(step.state_dict()),
                 **dict(step.named_buffers()),
                 **{k: v for k, v in ep.constants.items()
                    if isinstance(v, torch.Tensor)}}
    tensors = {}

    def prune(graph):
        """Drop weight buffers no op reads, and return the keys that survive.

        `EG.convert` emits a buffer entry for every placeholder `torch.export`
        lifted, which is every parameter of the exported module. `DecoderStep`
        holds the whole `WhisperForConditionalGeneration` — it has to, so that
        weights are named the way `state_dict()` names them — so the raw graph
        declares 588 weights, 808.9M parameters, 3086 MiB, of which the 32-layer
        *encoder* is the large majority and no decoder step touches it.

        Reachability is computed from what the ops name, not from what the
        signature declares: `in`, `out`, `$name` references inside `attrs`, the
        `of` back-pointer of a view, and the graph's own inputs and outputs.
        """
        live = set(graph["inputs"]) | set(graph["outputs"])
        for o in graph["ops"]:
            live.update(o["in"])
            live.add(o["out"])
            def walk(v):
                if isinstance(v, str) and v.startswith("$"):
                    live.add(v[1:])
                elif isinstance(v, list):
                    for e in v:
                        walk(e)
                elif isinstance(v, dict):
                    for e in v.values():
                        walk(e)
            walk(o.get("attrs", {}))
        for b in graph["buffers"]:
            if b.get("of"):
                live.add(b["of"])
        keep, dropped = [], 0
        for b in graph["buffers"]:
            if b.get("kind") == "weight" and b["id"] not in live:
                dropped += 1
                continue
            keep.append(b)
        graph["buffers"] = keep
        for b in keep:
            k = b.get("key")
            if b.get("kind") == "weight" and k is not None and k not in tensors:
                if k not in available:
                    raise KeyError(f"graph {graph['name']} wants {k}, which is "
                                   f"neither a parameter, a buffer nor a lifted constant")
                tensors[k] = available[k].detach().contiguous().cpu()
        print(f"  {graph['name']}: dropped {dropped} unread weight buffers, "
              f"{len(keep)} left")

    prune(g)
    (out / "whisperdec.json").write_text(json.dumps(g, indent=1))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    # ── the cross-attention K/V graph, same directory, its own name
    with torch.no_grad(), EG.precision_ctx(precision):
        epc = torch.export.export(cross, (z(1, SRC_LEN, D),), strict=False)
        epc = epc.run_decompositions()
    gc = EG.convert(epc, ({},), "whispercross")
    available.update({k: v for k, v in epc.constants.items()
                      if isinstance(v, torch.Tensor)})
    prune(gc)
    (out / "whispercross.json").write_text(json.dumps(gc, indent=1))

    save_file(tensors, str(out / "weights.safetensors"))
    print(f"  cross-KV graph: {len(gc['ops'])} ops")

    nb = sum(v.numel() for v in tensors.values())
    print(f"whisper decoder step: {len(g['ops'])} ops, {len(g['buffers'])} buffers, "
          f"{nb / 1e6:.1f}M params, "
          f"{sum(v.numel() * v.element_size() for v in tensors.values()) / 2**20:.0f} MiB")
    print(f"  inputs:  {g['inputs']}")
    print(f"  outputs: {g['outputs']}")
    for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "whisperdec")
    p.add_argument("--precision", choices=sorted(STORAGE), default="fp32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-target", type=int, default=448,
                   help="self-attention cache slots; Whisper's position table is 448")
    a = p.parse_args()
    main(a.out, a.precision, a.device, a.max_target)
