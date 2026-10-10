# BonsaiRunner

Text-only Mantle runtime for Prism's
[Ternary Bonsai 2 27B PTQ1 GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
The checkpoint stays in its 1.58-bit PTQ1 representation on the GPU; it is not
expanded into a dense 27B matrix set.

```julia
using BonsaiRunner, Mantle

model = Bonsai2(device=Mantle.Device("nvidia"))
s = session(model)
prompt = chatprompt(["user" => "Explain Vulkan barriers."])
print(generate(s, prompt; max_tokens=256))
```

## Example

[`docs/examples/bonsai.jl`](../docs/examples/bonsai.jl), `thinking = false`,
greedy:

> **In three sentences: why does a Vulkan compute shader need a pipeline barrier
> between two dispatches that share a buffer?**
>
> Vulkan requires explicit synchronization because the driver does not guarantee
> that the results of a previous compute dispatch are visible to subsequent
> commands until a barrier is issued. Without a pipeline barrier, the second
> dispatch may begin executing before the first has finished writing to the
> shared buffer, leading to undefined behavior or data races. The barrier ensures
> that all previous writes to the buffer are complete and visible before the next
> dispatch reads from or writes to it.

86 tokens at **4.3 tokens/s**, after a 35-token prompt at 8.1 tokens/s, on a
Radeon 8060S (RADV, 2026-09-30).

## Devices and checkpoints

The device selector is optional and uses `Mantle.Device()` by default. Another
session can use another device in the same process because the backend and all
state belong to the model:

```julia
amd = Bonsai2(device=Mantle.Device("radeon"), context=4096)
cpu = Bonsai2(device=Mantle.Device("lavapipe"), context=256)
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
token and position GPU references and replays its ~900 passes in one
submission, ending with the greedy pick so only one token id is read back.
`prefill!` ingests prompts in 2048-token chunks by default. Each chunk is one
recorded graph of wide matrix kernels, which Mantle cuts into submissions below
desktop GPU watchdog limits without giving up prompt-wide weight and activation
reuse:

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
