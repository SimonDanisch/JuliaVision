# BonsaiRunner

Text-only Mantle runtime for Prism's
[Ternary Bonsai 2 27B PTQ1 GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
The checkpoint stays in its 1.58-bit PTQ1 representation on the GPU; it is not
expanded into a dense 27B matrix set.

```julia
using BonsaiRunner, Mantle

model = Bonsai2("Ternary-Bonsai-2-27B-PTQ1_0.gguf";
                device=Mantle.device("nvidia"))
s = session(model)
prompt = chatprompt(["user" => "Explain Vulkan barriers."])
print(generate(s, prompt; max_tokens=256))
```

The device selector is optional and uses `Mantle.device()` by default. Another
session can use another device in the same process because the backend and all
state belong to the model:

```julia
amd = Bonsai2(path; device=Mantle.device("radeon"), context=4096)
cpu = Bonsai2(path; device=Mantle.device("lavapipe"), context=256)
```

The runtime has batch-one greedy decode, the 48 Gated DeltaNet states, the
checkpoint's byte-level Qwen tokenizer, and the complete 64-layer forward pass.
`session(model)` records the fixed decode graph once; each `step!` updates the
token and position GPU references and replays all 1,527 passes in one Vulkan
submission. `prefill!` ingests prompts in recorded chunks, and `generate` uses
8-token chunks by default. This keeps each submission below desktop GPU
watchdogs while still batching the expensive projections:

```julia
logits = prefill!(s, encode(model.tokenizer, prompt); chunk=8)
text = generate(session(model), prompt; max_tokens=256, prefill_chunk=8)
```

Sampling policies and vision input remain follow-up work.

`context=nothing`, the default, exposes the checkpoint's native 262,144-token
limit. A session initially allocates one 4,096-token KV page and doubles its
capacity when needed. K and V are symmetric Q8 with one Float32 scale per head
and token, costing 33,280 bytes per context token across all 16 attention
layers: about 130 MiB for the initial page and 7.75 GiB at 250,000 tokens. The
recurrent state is about 144 MiB per session. Use `session(model; kv_page=...)`
to change the allocation granularity or `context=...` to impose a smaller cap.
