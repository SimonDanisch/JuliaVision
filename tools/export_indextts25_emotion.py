"""Export the bundled Qwen3-0.6B emotion classifier and record its greedy cases."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import export_graphs as EG
from export_indextts25 import GEN, BASE, prune_unused_weights
from export_horizon32b import apply_rope, groupq, groupmask


class EmotionStep(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, k_cache, v_cache, positions, mask):
        m, cfg = self.model.model, self.model.config
        batch, n = ids.shape
        nq, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        h = m.embed_tokens(ids)
        cos, sin = m.rotary_emb(h, positions.unsqueeze(0))
        ks, vs = [], []
        for i, layer in enumerate(m.layers):
            x = layer.input_layernorm(h)
            a = layer.self_attn
            q = a.q_norm(a.q_proj(x).view(batch, n, nq, hd)).transpose(1, 2)
            k = a.k_norm(a.k_proj(x).view(batch, n, nkv, hd)).transpose(1, 2)
            v = a.v_proj(x).view(batch, n, nkv, hd).transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
            K = k_cache[i].index_copy(2, positions, k)
            V = v_cache[i].index_copy(2, positions, v)
            ks.append(K); vs.append(V)
            q = groupq(q, nkv, nq // nkv)
            scores = (q @ K.transpose(-1, -2)) * a.scaling + groupmask(mask, nq // nkv)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attention = (probs @ V).view(batch, nq, n, hd).transpose(1, 2).reshape(batch, n, nq * hd)
            h = h + a.o_proj(attention)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.model.lm_head(m.norm(h[:, -1:, :])), torch.stack(ks), torch.stack(vs)


def main():
    directory = BASE / "qwen0.6bemo4-merge"
    dev, dtype, capacity = "cuda:0", torch.float16, 1024
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(directory, torch_dtype=dtype,
                                                attn_implementation="eager").eval().to(dev)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    step = EmotionStep(model).eval()
    cfg = model.config
    cache = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, capacity,
                        cfg.head_dim, dtype=dtype, device=dev)
    out = GEN / "graphs" / "emotion_text"
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    for name, n in [("emotion_prefill", 16), ("emotion_decode", 1)]:
        positions = torch.arange(n, dtype=torch.long, device=dev)
        mask = torch.where(torch.arange(capacity, device=dev)[None, :] <= positions[:, None],
                           0.0, torch.finfo(dtype).min).to(dtype).view(1, 1, n, capacity)
        ids = torch.full((1, n), 1000, dtype=torch.long, device=dev)
        args = (ids, cache, cache.clone(), positions, mask)
        shapes = specs = ({}, {}, {}, {}, {})
        if n > 1:
            symbol = torch.export.Dim("n", min=2, max=capacity - 1)
            shapes = ({1: symbol}, {}, {}, {0: symbol}, {2: symbol})
            specs = ({1: "n"}, {}, {}, {0: "n"}, {2: "n"})
            EG.ROOT.update(n="n")
        with torch.inference_mode():
            got = step(*args)
            want = model(ids, attention_mask=None, use_cache=True).logits[:, -1:, :]
            torch.testing.assert_close(got[0], want, rtol=0.01, atol=0.02)
            ep = torch.export.export(step, args, strict=False, dynamic_shapes=shapes)
            ep = ep.run_decompositions()
        graph = prune_unused_weights(EG.convert(ep, specs, name))
        (out / f"{name}.json").write_text(json.dumps(graph, indent=1))
        available = dict(step.named_parameters(remove_duplicate=False))
        available.update(dict(step.named_buffers(remove_duplicate=False)))
        available.update({k: v for k, v in ep.constants.items() if isinstance(v, torch.Tensor)})
        for b in graph["buffers"]:
            if b.get("kind") == "weight":
                weights[b["key"]] = available[b["key"]].detach().contiguous().cpu()
        save_file(dict(ids=ids.cpu(), positions=positions.cpu(), mask=mask.cpu(),
                       logits=got[0].cpu()), str(out / f"{name}_reference.safetensors"))
        print(f"{name}: {len(graph['ops'])} ops", flush=True)
    save_file(weights, str(out / "weights.safetensors"))
    for file in ("vocab.json", "merges.txt", "added_tokens.json", "tokenizer.json", "chat_template.jinja"):
        (out / file).write_bytes((directory / file).read_bytes())
    (out / "config.json").write_text(json.dumps(dict(capacity=capacity, eos=tokenizer.eos_token_id)))
    references = []
    for text in ["I am so scared, something is in the dark.", "I am very happy!",
                 "Verschwinde aus meinem Haus!", "I feel gloomy and depressed."]:
        messages = [dict(role="system", content="文本情感分类"), dict(role="user", content=text)]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
        generated = result[0, inputs.input_ids.shape[1]:].tolist()
        references.append(dict(text=text, prompt=prompt, ids=inputs.input_ids[0].tolist(),
                               generated=generated, response=tokenizer.decode(generated, skip_special_tokens=True)))
        print(text, references[-1]["response"], flush=True)
    (out / "reference.json").write_text(json.dumps(references, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
