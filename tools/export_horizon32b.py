"""
Export K2 Horizon 32B to DNNKernels' graph JSON + weights.

    uv run tools/export_horizon32b.py                 # prefill(512) + decode(1)
    uv run tools/export_horizon32b.py --prefill 1024  # a different prompt bucket
    uv run tools/export_horizon32b.py --maxlen 8192   # KV capacity
    uv run tools/export_horizon32b.py --graphs-only --dynamic  # optional buckets

**The default exports two fixed graphs.** `horizon32b_decode` is one token against
the cache and is genuinely static — `T = 1` forever. `horizon32b_prefill` runs a
whole prompt in one pass at a bucket length; a caller pads up to the nearest
bucket, the way `whispercross` works against its fixed 1500-frame window.

`--dynamic` instead emits a decode graph and fixed prompt buckets whose attention
width `kv` is symbolic, while their cache allocations retain `maxlen` capacity.
Prefill buckets take `logits_indices`, selecting one position before the final
normalization and vocabulary projection; the KV caches still cover the prompt.
`--graphs-only` uses meta weights and leaves the existing weights file untouched.
Keep the original two graphs alongside these exports: the runner uses them at
full cache capacity, where PyTorch specializes the contiguous slice differently.
Prompt lengths remain static because torch.export cannot prove the reshape
constraints when both prompt length and attention width are symbolic.

**The cache is plain tensors, not an HF `Cache`.** `K2HorizonAttention.forward`
branches on `past_key_values is not None` and calls `.update()`; exporting
through `DynamicCache` lifts its empty per-layer seeds into the signature as
constants. `DecoderStep` in `export_whisper_decoder.py` hit this first and the
answer is the same here: reimplement the forward against the module's weights,
take `k_cache`/`v_cache` as arguments, and write with `index_copy`.

**The wrapper holds the whole model.** `torch.export` names a weight by its path
in the exported module, so holding `K2HorizonForCausalLM` makes the graph ask
for `model.model.layers.0.self_attn.q_proj.weight`, which is exactly what
`state_dict()` yields. Reaching in for submodules renames every tensor and the
load then fails on a weight that is present under another name.

Two properties of this config the code below relies on, asserted rather than
assumed because a sibling in the family breaks each one:

  * `rope_head_dim == head_dim`, so RoPE is the plain `rotate_half` path and the
    `split_to_interleaved` branch is dead. The 36B-A4B may differ.
  * `num_experts == 0` and `attention_gate_func is None`: this model is dense
    and ungated. The MoE and MoVA paths are not implemented here at all.

`layernorm_num_groups` is 2, so RMSNorm is NOT the standard one and will not
fold into `_fused_rms_norm`. It decomposes to reshape/pow/mean/rsqrt/mul, which
DNNKernels has; a fused grouped variant is worth having later, since it runs
twice per layer across 64 layers.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_graphs as EG
from common import find_root

ROOT = find_root()
GEN = ROOT / "gen"
WEIGHTS = GEN / "horizon32b"


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def groupq(q, nkv, n_rep):
    """(B, nkv*n_rep, T, hd) -> (B, nkv, n_rep*T, hd): GQA by regrouping Q.

    The alternative is `repeat_kv`, which expands each KV head `n_rep` times and
    `reshape`s — and that reshape is a real copy of the WHOLE cache. On this
    model it materialised `[1, 64, 4096, 128]` twice per layer, **8 GiB written
    per token**, and it was 46% of a decode step at ~7.6 GB/s.

    Regrouping Q the other way is exactly equivalent — each KV head still meets
    its own `n_rep` query heads — and copies Q instead: 16 KiB per layer at
    decode, 8 MiB at prefill. Row `r` of the grouped Q is query head
    `r // T` within the group, position `r % T`, which is what `groupmask`
    below has to match.
    """
    if n_rep == 1:
        return q
    b, _, t, d = q.shape
    return q.view(b, nkv, n_rep, t, d).reshape(b, nkv, n_rep * t, d)


def causalmask(T, maxlen, dtype, device, cache_position):
    """`(1, 1, T, maxlen)` additive mask: slot j visible to query i iff
    `j <= cache_position[i]`.

    `finfo.min` rather than `-inf` — it survives an fp16 cast, which `-inf` does
    not, and it is the value HF's own mask builder uses.
    """
    slots = torch.arange(maxlen, device=device)
    neg = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    visible = slots.view(1, maxlen) <= cache_position.view(T, 1)
    return torch.where(visible, zero, neg).view(1, 1, T, maxlen)


def groupmask(mask, n_rep):
    """`(1, 1, T, maxlen)` -> `(1, 1, n_rep*T, maxlen)`, tiled to match `groupq`.

    Row `r` of the grouped scores is position `r % T`, so the mask repeats along
    the row axis rather than broadcasting.
    """
    if n_rep == 1:
        return mask
    _, _, T, maxlen = mask.shape
    return mask.expand(1, 1, T, maxlen).repeat(1, 1, n_rep, 1).view(1, 1, n_rep * T, maxlen)


class HorizonBlocks(torch.nn.Module):
    """The whole model for one call: `T` tokens against a fixed-capacity cache.

    Used for both graphs — prefill and decode differ only in `T` and in the mask,
    so they are the same forward exported at two shapes rather than two bodies
    that can drift apart.
    """

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        assert cfg.num_experts == 0, "this exporter is the DENSE path; 36B-A4B needs MoE"
        assert getattr(cfg, "attention_gate_func", None) in (None, "None"), \
            "attention gating (MoVA) is not implemented here"
        assert cfg.rope_head_dim == cfg.head_dim, \
            "rope_head_dim != head_dim takes the split_to_interleaved path, not implemented"
        assert not cfg.query_key_norm, "query_key_norm is not implemented here"
        self.model = model
        self.nkv = cfg.num_key_value_heads
        self.nq = cfg.num_attention_heads
        self.hd = cfg.head_dim
        self.n_rep = self.nq // self.nkv

    def forward(self, input_ids, k_cache, v_cache, cache_position, mask, logits_indices=None):
        """`mask` is an ARGUMENT, not built here, and that is the whole point.

        Built inside, `torch.where(...).view(1, 1, T, maxlen)` is broadcast by
        SDPA to `(B, H, T, maxlen)` and `torch.export` captures the result as a
        materialised constant PER LAYER. At `T = 512` that is 64 tensors of
        `[1, 64, 512, 4096]` fp32 — 512 MiB each, **32 GiB** for one prompt
        bucket, which `hoistconstants` then turns into real host arrays before a
        single weight is uploaded. Together with 64.78 GiB of weights it does not
        fit in this machine's 122 GiB at all, and the loader is OOM-killed.
        
        As an argument it is one `(1, 1, T, maxlen)` fp16 buffer — 4 MiB at
        T = 512 — passed once and broadcast on the device. It is data, not a
        constant, so this is also what it should have been.
        """
        m = self.model.model
        B, T = input_ids.shape
        hd, nq, nkv = self.hd, self.nq, self.nkv

        h = m.embed_tokens(input_ids)
        position_ids = cache_position.unsqueeze(0).expand(B, T)
        cos, sin = m.rotary_emb(h, position_ids)

        newk, newv = [], []
        for i, layer in enumerate(m.layers):
            r = h
            x = layer.input_layernorm(h)
            a = layer.self_attn

            q = a.q_proj(x).view(B, T, nq, hd).transpose(1, 2)
            k = a.k_proj(x).view(B, T, nkv, hd).transpose(1, 2)
            v = a.v_proj(x).view(B, T, nkv, hd).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)

            K = k_cache[i].index_copy(2, cache_position, k)
            V = v_cache[i].index_copy(2, cache_position, v)
            newk.append(K)
            newv.append(V)

            # The attention math WRITTEN OUT rather than `scaled_dot_product_attention`.
            # SDPA decomposes through `_safe_softmax`, which guards against a row
            # that is entirely masked by testing `scores == -inf`, building a
            # `full_like(scores, 0)` and selecting with `where`. That guard is
            # dead code here — every query sees at least its own slot — but it
            # costs three extra passes over the scores, and the `eq`/`where` pair
            # sitting between the softmax and the second matmul is also what stops
            # `fuseattention` from recognising the block at all.
            qg = groupq(q, nkv, self.n_rep)
            # The mask names the occupied attention bucket; cache storage keeps
            # its full capacity as this bucket grows between decode steps.
            active = mask.shape[-1]
            Ka, Va = K[:, :, :active, :], V[:, :, :active, :]
            scores = torch.matmul(qg, Ka.transpose(-1, -2)) * a.scaling + mask
            # fp32 softmax, as SDPA's own decomposition does; the cast back is
            # what the following matmul wants.
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(qg.dtype)
            o = torch.matmul(attn, Va).view(B, nq, T, hd)
            h = r + a.o_proj(o.transpose(1, 2).reshape(B, T, nq * hd))

            r = h
            x = layer.post_attention_layernorm(h)
            h = r + layer.mlp(x)

        if logits_indices is not None:
            h = h[:, logits_indices, :]
        h = m.norm(h)
        return self.model.lm_head(h), torch.stack(newk), torch.stack(newv)


def build(step, name, T, maxlen, dev, dt, out, available, tensors,
          *, dynamic=False, activekv=None, collect_weights=True, lastlogit=False):
    """Export one shape, prune, and write the JSON."""
    cfg = step.model.config
    L, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    args = (
        torch.zeros(1, T, dtype=torch.long, device=dev),
        torch.zeros(L, 1, nkv, maxlen, hd, dtype=dt, device=dev),
        torch.zeros(L, 1, nkv, maxlen, hd, dtype=dt, device=dev),
        torch.arange(T, dtype=torch.long, device=dev),
        groupmask(causalmask(T, activekv or maxlen, dt, dev, torch.arange(T, device=dev)),
                  step.n_rep),
    )
    shapes, specs = None, ({}, {}, {}, {}, {})
    if dynamic:
        # A full-width slice is a different stride specialization in torch.
        # The runner uses the original fixed graph at full cache capacity.
        kv = torch.export.Dim("kv", min=2, max=maxlen - 1)
        shapes = ({}, {}, {}, {}, {3: kv})
        specs = ({}, {}, {}, {}, {3: "kv"})
        EG.ROOT.update(kv="kv")
    if lastlogit:
        args += (torch.tensor([T - 1], dtype=torch.long, device=dev),)
        specs += ({},)
        if shapes is not None:
            shapes += ({},)
    with torch.no_grad():
        ep = torch.export.export(step, args, strict=False, dynamic_shapes=shapes)
        ep = ep.run_decompositions()
    g = EG.convert(ep, specs, name)
    available.update({k: v for k, v in ep.constants.items()
                      if isinstance(v, torch.Tensor)})
    prune(g, available, tensors, collect_weights=collect_weights)
    (out / f"{name}.json").write_text(json.dumps(g, indent=1))
    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / f"{name}_op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))
    return g


def prune(graph, available, tensors, *, collect_weights=True):
    """Drop weight buffers no op reads; collect the ones that survive.

    Same reachability rule as `export_whisper_decoder.py`: computed from what the
    ops NAME, not from what the signature declares, because holding the whole
    model means every parameter is lifted into the signature whether the graph
    reads it or not.
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
                raise KeyError(f"graph {graph['name']} wants {k}, which is neither "
                               f"a parameter, a buffer nor a lifted constant")
            if collect_weights:
                tensors[k] = available[k].detach().contiguous().cpu()
    print(f"  {graph['name']}: dropped {dropped} unread weight buffers, {len(keep)} left")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefill", type=int, default=512, help="prompt bucket length")
    p.add_argument("--maxlen", type=int, default=4096, help="KV cache capacity")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dynamic", action="store_true", help="export prompt/attention bucket dimensions")
    p.add_argument("--graphs-only", action="store_true", help="trace on meta without loading or writing weights")
    p.add_argument("--prompt-buckets", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "horizon32b")
    args = p.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM
    dt = torch.float16
    print(f"loading {WEIGHTS} as {dt} ...", flush=True)
    if args.graphs_only:
        args.device = "meta"
        cfg = AutoConfig.from_pretrained(WEIGHTS, trust_remote_code=True)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                    attn_implementation="eager").eval().to(dt)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            WEIGHTS, dtype=dt, trust_remote_code=True,
            attn_implementation="eager").eval().to(args.device)

    step = HorizonBlocks(model).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    available = {**dict(step.state_dict()), **dict(step.named_buffers())}
    tensors = {}

    print(f"exporting decode  (T=1, maxlen={args.maxlen}) ...", flush=True)
    suffix = "_bucket" if args.dynamic else ""
    kwargs = dict(dynamic=args.dynamic, collect_weights=not args.graphs_only,
                  activekv=min(128, args.maxlen // 2) if args.dynamic else None)
    build(step, "horizon32b_decode" + suffix, 1, args.maxlen, args.device, dt,
          args.out, available, tensors, **kwargs)
    prompts = sorted(set(args.prompt_buckets)) if args.dynamic else [args.prefill]
    for T in prompts:
        if not 1 < T <= args.maxlen:
            raise ValueError(f"prompt bucket {T} must be between 2 and maxlen")
        name = f"horizon32b_prefill_bucket_{T}" if args.dynamic else "horizon32b_prefill"
        print(f"exporting prefill (T={T}, maxlen={args.maxlen}) ...", flush=True)
        build(step, name, T, args.maxlen, args.device, dt,
              args.out, available, tensors, lastlogit=args.dynamic, **kwargs)

    if args.graphs_only:
        return
    save_file(tensors, str(args.out / "weights.safetensors"))
    nb = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors, {nb / 2**30:.2f} GiB -> {args.out}")


if __name__ == "__main__":
    main()
