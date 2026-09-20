# BonsaiRunner

Text-only Mantle runtime for Prism's
[Ternary Bonsai 2 27B PTQ1 GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
The checkpoint stays in its 1.58-bit PTQ1 representation on the GPU; it is not
expanded into a dense 27B matrix set.

```julia
using BonsaiRunner, Mantle

model = Bonsai2(device=Mantle.device("nvidia"))
s = session(model)
prompt = chatprompt(["user" => "Explain Vulkan barriers."])
print(generate(s, prompt; max_tokens=256))
```

The device selector is optional and uses `Mantle.device()` by default. Another
session can use another device in the same process because the backend and all
state belong to the model:

```julia
amd = Bonsai2(device=Mantle.device("radeon"), context=4096)
cpu = Bonsai2(device=Mantle.device("lavapipe"), context=256)
```

The no-path constructor resolves the official PTQ1 GGUF from four lazy Julia
artifacts attached to JuliaVision's `assets-v1` release. Four parts are required
because the 5.95 GB checkpoint exceeds GitHub's 2 GiB per-asset limit. On first
use BonsaiRunner downloads and verifies each part, assembles the original GGUF
in its Scratch.jl cache, and verifies the upstream SHA-256. Allow about 12 GB of
local storage for the artifact parts plus the assembled mmap-ready file. An
explicit `Bonsai2(path; ...)` still selects a local checkpoint.

The runtime has batch-one greedy decode, the 48 Gated DeltaNet states, the
checkpoint's byte-level Qwen tokenizer, and the complete 64-layer forward pass.
`session(model)` records the fixed decode graph once; each `step!` updates the
token and position GPU references and replays all 1,527 passes in one Vulkan
submission. `prefill!` ingests prompts in 512-token chunks by default. Each
chunk uses wide tiled matrix kernels and is split into recorded four-layer
submissions, keeping individual submissions below desktop GPU watchdog limits
without giving up prompt-wide weight and activation reuse:

```julia
logits = prefill!(s, encode(model.tokenizer, prompt))
text = generate(session(model), prompt; max_tokens=256)
```

Sampling policies and vision input remain follow-up work.

`context=nothing`, the default, exposes the checkpoint's native 262,144-token
limit. A session initially allocates one 4,096-token KV page and doubles its
capacity when needed. K and V are symmetric Q8 with one Float32 scale per head
and token, costing 33,280 bytes per context token across all 16 attention
layers: about 130 MiB for the initial page and 7.75 GiB at 250,000 tokens. The
recurrent state is about 144 MiB per session. Use `session(model; kv_page=...)`
to change the allocation granularity or `context=...` to impose a smaller cap.
