# BonsaiRunner

Text-only Mantle runtime for Prism's
[Ternary Bonsai 2 27B PTQ1 GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
The checkpoint stays in its 1.58-bit PTQ1 representation on the GPU; it is not
expanded into a dense 27B matrix set.

```julia
using BonsaiRunner, Mantle

model = Bonsai2("Ternary-Bonsai-2-27B-PTQ1_0.gguf";
                device=Mantle.device("nvidia"), context=8192)
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

The initial implementation is batch-one greedy decode. It has a real KV cache,
the 48 Gated DeltaNet states, the checkpoint's byte-level Qwen tokenizer, and
the complete 64-layer forward pass. Prompt ingestion currently calls the same
one-token step repeatedly. Chunked prefill, sampling policies, vision input,
and quantized long-context KV storage remain follow-up work.

An F16 KV cache costs 64 KiB per context token: 512 MiB at 8192 tokens and 2
GiB at 32768. The recurrent state is about 144 MiB per session.
