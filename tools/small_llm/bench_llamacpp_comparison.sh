#!/usr/bin/env bash
set -euo pipefail

# Run from the VulkanDev root. Timeout the whole script externally as well.

bench=tmp/llama-vulkan-build/bin/llama-bench
quantize=tmp/llama-vulkan-build/bin/llama-quantize
results=dev/JuliaVision/tools/small_llm
model=gen/qwen35-4b-comparison-q4k-q6head.gguf

# Match the runner's broad policy, not its exact quantized bytes. llama.cpp
# shares quantized embeddings with the head; our runner keeps input embeddings
# FP32 and a separate Q6_K head. Tiny recurrent projections remain FP32.
if [[ ! -f "$model" ]]; then
    "$quantize" --pure --token-embedding-type q6_k \
        --tensor-type 'ssm_(alpha|beta)\.weight=f32' \
        gen/qwen35-4b-f16.gguf "$model" Q4_K 8 \
        >"$results/llamacpp_qwen4b_quantize.log" 2>&1
fi

# This llama-bench revision rejects f32 for -ctk/-ctv. Keep the FP16
# comparison explicit rather than silently claiming equal cache precision.
for cache in f16; do
    "$bench" -m "$model" -p 0 -n 128 -d 26,4096 -r 3 \
        -ngl 99 -fa on -ctk "$cache" -ctv "$cache" -t 8 -o json \
        >"$results/llamacpp_qwen4b_${cache}_benchmark.json" \
        2>"$results/llamacpp_qwen4b_${cache}_benchmark.log"
done

echo 'llama.cpp comparison complete; lock released on exit'
