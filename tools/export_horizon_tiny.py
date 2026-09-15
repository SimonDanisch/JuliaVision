"""A TINY random K2Horizon through the real exporter, for end-to-end checking.

`/tmp/parity_horizon.py` checks the wrapper against HF in PyTorch. It cannot
catch a wrong op in DNNKernels — a grouped RMSNorm that reduces over the wrong
axis, an `index_copy` with a transposed cache, a `where` that broadcasts the
mask the other way. Those only show up when the exported graph actually runs.

So: same `HorizonBlocks`, same `build()`, same dtype, ~4 MB of random weights.
Export here, run in Julia, compare logits to PyTorch's. A mismatch is then a
15-second edit-and-retry instead of a 30-minute reload of a 64 GiB checkpoint.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, "tools")
sys.path.insert(0, "gen/horizon32b")

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from export_horizon32b import HorizonBlocks, build, causalmask, groupmask

OUT = Path("gen/graphs/horizontiny")
T, MAXLEN = 6, 32
# fp32: this checks SEMANTICS, and fp16 rounding would blur a real sign error
# into the tolerance. The 32B graph differs only in dtype and extents.
DT = torch.float32


def tinymodel():
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained("gen/horizon32b", trust_remote_code=True)
    cfg = type(cfg)(
        hidden_size=128, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=2, head_dim=16, rope_head_dim=16,
        vocab_size=512, max_position_embeddings=256, layernorm_num_groups=2,
        num_experts=0, mova_num_experts=0, mova_num_experts_per_tok=0,
        num_experts_per_tok=0, attention_gate_func=None, query_key_norm=False,
        mlp_only_layers=[0, 1], tie_word_embeddings=False,
    )
    Cls = get_class_from_dynamic_module(
        "modeling_k2_horizon.K2HorizonForCausalLM", "gen/horizon32b")
    return cfg, Cls(cfg).eval().to(DT)


def main():
    cfg, model = tinymodel()
    step = HorizonBlocks(model).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    available = {**dict(step.state_dict()), **dict(step.named_buffers())}
    tensors = {}
    # The SAME graph names as the real export, so `horizon32b(; dir = ...)` runs
    # this model verbatim: the runner reads every extent off the graph, so it is
    # the runner under test here, not a copy of it.
    build(step, "horizon32b_decode", 1, MAXLEN, "cpu", DT, OUT, available, tensors)
    build(step, "horizon32b_prefill", T, MAXLEN, "cpu", DT, OUT, available, tensors)
    build(step, "horizon32b_decode_bucket", 1, MAXLEN, "cpu", DT, OUT,
          available, tensors, dynamic=True, activekv=8)
    for prompt in (2, 4, 6):
        build(step, f"horizon32b_prefill_bucket_{prompt}", prompt, MAXLEN, "cpu", DT, OUT,
              available, tensors, dynamic=True, activekv=8, lastlogit=True)
    save_file(tensors, str(OUT / "weights.safetensors"))

    # The reference, and the inputs Julia must be handed verbatim.
    L, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))
    with torch.no_grad():
        k = torch.zeros(L, 1, nkv, MAXLEN, hd, dtype=DT)
        v = torch.zeros(L, 1, nkv, MAXLEN, hd, dtype=DT)
        cp = torch.arange(T)
        mp = groupmask(causalmask(T, MAXLEN, DT, ids.device, cp), step.n_rep)
        lp, nk, nv = step(ids, k, v, cp, mp)
        cp2 = torch.tensor([T])
        md = groupmask(causalmask(1, MAXLEN, DT, ids.device, cp2), step.n_rep)
        ld, _, _ = step(nxt, nk, nv, cp2, md)

    # A greedy continuation from a SHORT prompt (4 < bucket 6), so the runner's
    # padding and its position bookkeeping across decode steps are under test
    # too, not just one step. Reference is plain HF, no cache, re-running the
    # whole prefix each time: the slowest possible way and the least likely to
    # share a bug with the thing it checks.
    NPROMPT, NNEW = 4, 8
    prompt = ids[:, :NPROMPT]
    greedy = []
    with torch.no_grad():
        cur = prompt
        for _ in range(NNEW):
            t = int(model(input_ids=cur, use_cache=False).logits[0, -1].argmax())
            greedy.append(t)
            cur = torch.cat([cur, torch.tensor([[t]])], dim=1)
    print("greedy reference:", greedy)

    save_file({
        "ids": ids.to(torch.int32), "nxt": nxt.to(torch.int32),
        "prompt": prompt.to(torch.int32),
        "greedy": torch.tensor(greedy, dtype=torch.int32),
        "mask_prefill": mp,
        "mask_decode": md,
        "logits_prefill": lp, "logits_decode": ld,
        "k_after": nk, "v_after": nv,
    }, str(OUT / "reference.safetensors"))
    (OUT / "meta.json").write_text(json.dumps(
        {"T": T, "maxlen": MAXLEN, "layers": L, "nkv": nkv, "hd": hd,
         "vocab": cfg.vocab_size}, indent=1))
    print(f"tiny model + reference -> {OUT}")


if __name__ == "__main__":
    main()
