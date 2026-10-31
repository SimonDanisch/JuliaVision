# Long-context inference for Qwen3.6-35B-A3B (hybrid GDN + gated attention) on Strix Halo and Apple M5

Research notes, 2026-10-27. Sources are linked inline. Tags used throughout:

- **[derived]**: my own arithmetic from published numbers or model dimensions, not a measured result.
- **[unverified]**: seen only in a search snippet or secondary write-up, primary source not opened, or numbers that conflict between sources.
- **[source read]**: taken from reading the implementation's source code (main branches as of 2026-10-27), not from documentation.

---

## 0. Cost model for our setup [derived]

Model facts (from the [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)): 40 layers laid out as 10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE)); gated attention has 16 Q heads, 2 KV heads, head dim 256, RoPE on only 64 of the 256 dims (`partial_rotary_factor` 0.25, `rope_theta` 1e7); Gated DeltaNet has 32 V heads, 16 QK heads, head dim 128; 256 experts, 8 routed + 1 shared, expert intermediate 512; hidden 2048; vocab 248,320; native context 262,144.

**KV bytes per position** (10 attention layers x K,V x 2 KV heads x 256 = 10,240 elements):

| KV type | bytes/position | 64k (65,536) | 128k | 256k (262,144) |
|---|---:|---:|---:|---:|
| F32 (ours today) | 40,960 | 2.68 GB | 5.37 GB | 10.7 GB |
| F16/BF16 | 20,480 | 1.34 GB | 2.68 GB | 5.37 GB |
| q8_0 (8.5 bit) | 10,880 | 0.713 GB | 1.43 GB | 2.85 GB |
| q4_0 (4.5 bit) | 5,760 | 0.378 GB | 0.755 GB | 1.51 GB |

**Where our 64k time goes.** 22 tok/s = 45.5 ms/token. Short-context cost is 12.5 ms (80 tok/s). F32 KV at 64k is 2.68 GB; at the measured ~100 GB/s that is 26.8 ms. That leaves **~6 ms of depth-dependent time that KV reads do not explain** (mask building, KV append, a separate combine/softmax pass, or a second read of K). Profile it before anything else; it alone is worth 14%.

**Achievable bandwidth on the same chip.** Measured GPU bandwidth on Strix Halo is ~212-215 GB/s of 256 GB/s theoretical ([Framework forum](https://community.frame.work/t/amd-strix-halo-ryzen-ai-max-395-gpu-llm-performance-tests/72521), rocm_bandwidth_test per a [r/LocalLLaMA post, unverified]). llama.cpp's Vulkan decode attention on a Radeon 8060S reaches **~175-198 GB/s effective** [derived from the llama-bench tables in [PR #19625](https://github.com/ggml-org/llama.cpp/pull/19625): Llama-3-8B Q4_0 tg128 44.86 -> 30.13 tok/s at depth 16384 with F16 KV (131,072 B/position) gives 10.9 ms for 2.16 GB = 198 GB/s; Qwen3-30B-A3B Q2_K 91.60 -> 49.64 tok/s gives 175 GB/s]. These are lower bounds, because all depth-dependent overhead is attributed to KV reads.

**Projected decode** (12.5 ms non-attention + KV read at 190 GB/s, assuming the 6 ms is removed) [derived]:

| context | F16 | q8_0 | q4_0 |
|---|---:|---:|---:|
| 64k | 51 tok/s | 62 tok/s | 69 tok/s |
| 128k | 38 tok/s | 50 tok/s | 61 tok/s |
| 256k | 25 tok/s | 36 tok/s | 49 tok/s |

Conclusion: KV format + kernel efficiency alone reach the 50-80 tok/s goal at 64k. Reaching it at 128k-256k needs speculative decoding (MTP) and/or sparse attention on top.

**Prefill FLOPs** [derived]: per token at depth D, attention costs 10 layers x 16 heads x 4 x 256 x D = 163,840 x D FLOPs; weights cost ~2 x 2.9e9 = 5.8 GFLOP (active-parameter count recomputed from the card's dims: routed experts ~1.0B, shared expert 0.13B, GDN projections ~1.0B, attention projections 0.27B, LM head 0.51B). GDN recurrence is ~60 MFLOP/token (negligible). A 64k prompt from scratch averages 32k depth, so ~11.2 GFLOP/token. At 6-10 TFLOP/s achieved (10-17% of the 59.4 TFLOPS FP16 WMMA peak), that is **~540-900 tok/s**. Our 34 tok/s means ~29 ms per prompt token, which is the same as one decode step at mid-depth: prefill is currently running as N decode steps.

---

## 1. Ranked techniques for our setup

Ranking is by expected gain for our engine, not by novelty. "Decode gain" is at 64k unless noted.

| # | Technique | Expected gain | Quality risk | Complexity |
|---|---|---|---|---|
| 1 | Batched prefill (ubatch of 512-2048 tokens: MoE GEMM, FA-2 causal attention over cache + chunk, fused GDN scan per layer) | prefill 34 -> ~400-900 tok/s (12-25x) [derived]; llama.cpp gets 1047 tok/s pp512 and 322.7 tok/s at 153k on this chip | none (exact) | high, but mostly reuse of matmul kernels |
| 2 | KV cache F32 -> F16 -> q8_0 | attention bytes /2 (F16), /3.8 (q8_0); 22 -> ~30-35 tok/s at our current 100 GB/s, more with #3 | F16: none. q8_0: negligible on Qwen3.5/3.6 (1/500 ARC answer flips) | low to medium |
| 3 | Bandwidth-efficient decode attention (split-K to 2x CU count, GQA packing of 8 Q heads per KV head, K/V staged through LDS, LSE combine) | 100 -> ~190 GB/s on attention; with #2 gives the 51-69 tok/s table above | none (exact) | medium |
| 4 | MTP speculative decoding (Qwen3.6 ships the MTP head) | 1.2-2.3x measured on Strix Halo at depth; ~1.45x at 64k q8_0, ~1.8x at 256k [derived] | none if verification is exact | high (MTP layer + own KV, GDN state rollback, multi-row verify attention) |
| 5 | N-gram drafting (prompt lookup, ngram-mod, suffix trees) on top of #4 | +73% combined with MTP on agent workloads (Strix Halo, llama.cpp); 2.4x PLD on summarization | none | low once #4 exists |
| 6 | Prefix caching with GDN state checkpoints | returning to a 65k chat: >5 min -> 13.6 s (llama.cpp) | none | medium |
| 7 | Chunkwise-parallel GDN prefill (WY/UT, chunk 64) | +3-21% prefill on CUDA over a fused recurrent scan; "not at model level yet" on Vulkan | none (numerics) | high |
| 8 | q4_0 V (and K) or Hadamard-rotated q4 KV | 256k: 36 -> 49 tok/s [derived] | small on Qwen3.6 (KLD 0.0047 vs 0.0023 for q8_0); catastrophic on non-hybrid Qwen2.5 K | low after #2 |
| 9 | Query-aware sparse decode attention (Quest-style page selection) for >=128k | up to ~2x at 256k after #2/#3 [derived]; Quest: 7.03x attention, 2.23x end-to-end at 32k | real: hybrids lose retrieval when the full layers are sparsified without training | medium |
| 10 | Sparse prefill (MInference patterns, DuoAttention streaming heads) | <=1.5-2x prefill at 64k [derived]; MInference 10x at 1M on A100 | real, needs per-head calibration | high |

Not recommended for this setup (details in section 3.4): StreamingLLM / sink-only windows (gated attention removes sinks, and evicted tokens are unrecoverable), CPU offloading designs (RetrievalAttention, ShadowKV; they solve capacity, and 256k of F16 KV is only 5.4 GB on 128 GB UMA), FP8 KV (RDNA3.5 has no FP8 math, so q8_0 is the same bytes with cheaper decode), FlashDecoding++ unified max (saves the rescale, which is a small cost in split-K decode), YaRN below 262,144 tokens.

Minor items worth doing when touching the code: GDN state in F16 (llama.cpp [PR #20376](https://github.com/ggml-org/llama.cpp/pull/20376), open) halves the 2 x 60 MiB of state traffic per token (~0.3 ms) [derived]; write the GDN state straight into the recurrent cache instead of a separate copy ([PR #27973](https://github.com/ggml-org/llama.cpp/pull/27973), open; removes one copy per GDN layer per token).

---

## 2. Technique details (in rank order)

### R1. Batched prefill

**Mechanism.** Process the prompt in ubatches of 512-2048 tokens. Every linear and MoE layer then reads its weights once per ubatch instead of once per token (decode is weight-bandwidth bound, prefill should be compute bound). Attention for the ubatch is a FlashAttention-2 style kernel over [KV cache prefix + ubatch], causal within the ubatch; Gated DeltaNet runs one fused kernel per layer that walks the ubatch tokens sequentially with the state held on-chip.

**Reported numbers.**
- Qwen3.6-35B-A3B Q8_0 on Strix Halo, llama.cpp Vulkan: pp512 ~1047-1049 tok/s (ROCm ~980-1014); live server at a 153,562-token context with q8_0 KV: 322.7 tok/s prompt processing ([Level1Techs](https://forum.level1techs.com/t/strix-halo-llm-inference-notes/249466)).
- llama.cpp Metal GDN kernel ([PR #20361](https://github.com/ggml-org/llama.cpp/pull/20361), hardware not stated in the PR): qwen35moe 35B.A3B Q4_0 pp512 1612 -> 2058, pp2048 1880 -> 2462 tok/s; tg32 57 -> 77.7. The PR uses the fused *recurrent* kernel "for BS == 1 and BS > 1".
- llama.cpp Vulkan still uses the autoregressive GDN kernel for all token counts: [PR #20377](https://github.com/ggml-org/llama.cpp/pull/20377) says chunked dispatch is disabled (`GDN_CHUNK_THRESHOLD = UINT32_MAX`) and "the autoregressive path handles all token counts". The per-token recurrence is therefore not what limits prefill, provided it is one dispatch per layer per ubatch.
- ubatch size: one Strix Halo guide found `-ub 2048` 23-28% slower at prefill than `-b 2048 -ub 1024` ([joanmedina guide](https://github.com/joanmedina/qwen3-llama-cpp-inference-tuning)); a Framework forum commenter found 2048 better than 512 for long context. Tune it.

**Kernel design points from llama.cpp Vulkan [source read]** ([ggml-vulkan.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp)):
- N > 1 uses the cooperative-matrix path (coopmat1 = KHR_cooperative_matrix, i.e. WMMA on RDNA3): Br = 16 rows, Bc = 64 (16 x 4 subgroups), workgroup = 4 subgroups. coopmat2 (NVIDIA only): Br 64 / Bc 32 for head dim >= 256.
- With a quantized KV cache and >= 64 query rows, K/V are dequantized once into an F16 scratch buffer, then the F16 FA kernel runs (`use_dequant_kv`). Metal does the same ([PR #27390 reference in ggml-metal-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp)).
- Causal masks are pre-scanned so fully masked KV blocks are skipped ([#17186](https://github.com/ggml-org/llama.cpp/pull/17186), [#19281](https://github.com/ggml-org/llama.cpp/pull/19281)).
- Staging K/V through shared memory in the scalar FA path gave +12% (d0) to +50% (d8192) pp2048 and +29.8% / +62.4% at d8192 / d16384 on RDNA2 for head sizes 256/512 ([PR #28507](https://github.com/ggml-org/llama.cpp/pull/28507), open; RDNA3 not tested there).

**FA-2/3 reference efficiency.** FlashAttention-2 reaches 50-73% of peak FLOPs on A100 ([arXiv 2307.08691](https://arxiv.org/abs/2307.08691)); FlashAttention-3 reaches 740 TFLOPs/s FP16 (75%) on H100 ([arXiv 2407.08608](https://arxiv.org/abs/2407.08608)). On Strix Halo, hipBLASLt BF16 GEMM reaches 36.9 of 59.4 TFLOPS (>60%), plain rocBLAS 5.1 TFLOPS [unverified, r/LocalLLaMA post via search].

**Chunked prefill scheduling.** Sarathi-Serve splits prefills into equal chunks and mixes them with ongoing decodes: 2.6x serving capacity (Mistral-7B, A100) up to 5.6x (Falcon-180B) vs vLLM ([arXiv 2403.02310](https://arxiv.org/abs/2403.02310)). For a single user this matters less; what matters is the ubatch size and that each chunk attends to the cache written by earlier chunks.

**Risk / complexity.** Exact, no quality risk. High complexity, but it is the single largest gain available (prefill is 12-25x below what the hardware does today in llama.cpp).

### R2. KV cache format (F16, then q8_0)

**Mechanism.** Store K and V in F16 (halves bytes vs F32), or block-quantized q8_0 (int8 + one F16 scale per 32 values, 8.5 bit) or q4_0 (4.5 bit). Decode attention is a pure stream over the cache, so time scales with bytes as long as the dequant ALU work stays hidden behind memory.

**Reported numbers (quality).**
- Qwen3.5-4B (same hybrid family), wikitext PPL, F16 baseline 8.3266: q8_0 8.3272, q5_0 8.3359, q4_0 8.3475; with Hadamard rotation 8.3261 / 8.3339 / 8.3448. Non-hybrid Qwen3-8B: q4_0 7.6451 vs 7.3203 (7.5012 rotated). Qwen3-0.6B: q4_0 62.0 vs 13.67 ([llama.cpp PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)).
- Qwen3.6-27B, 500 ARC-Challenge questions, answers changed vs F16 cache: q8_0 1, q8_0 K / q4_0 V 1, q4_0 / q4_0 2. Qwen2.5-7B: q4_0 K alone collapses accuracy (92.0% -> 24.2%, 375 flips) ([discussion #23470](https://github.com/ggml-org/llama.cpp/discussions/23470)).
- Qwen3.6 vs bf16 cache, mean KLD: q8_0 0.0023, q8_0/q5_0 0.0027, q8_0/q4_0 0.0033, q4_0 0.0047, "turbo2" 0.036 (an article by Anbeeld, numbers as reproduced in the same discussion; article itself not opened, [unverified]). Qwen3.8-27B at 65,536 context, 99% KLD: q8_0 0.0064, q8_0/q4_0 0.0123, q4_0 0.0221.
- KVTuner: Qwen2.5-7B-Instruct has many retrieval heads sensitive to key quantization; K4 configurations collapse GSM8K to ~0.004-0.006 while K8V4 keeps 0.7167 vs 0.7294 BF16 ([arXiv 2502.04420](https://arxiv.org/abs/2502.04420)).
- Long-context benchmarks of quantized KV: KIVI-4bit matches baseline LongBench (Llama-3-8B-Instruct 45.2 -> 45.3), KIVI-2bit loses ~1 point, needle stays 100; token-dropping methods at 4x compression collapse on needle (StreamingLLM 22.3, InfLLM 20.7) ([arXiv 2407.01527](https://arxiv.org/abs/2407.01527)). KIVI-2 (G = 32, residual 128 tokens in full precision): LongBench Llama2-7B 44.52 -> 44.27, Mistral-7B 46.58 -> 45.85 ([arXiv 2402.02750](https://arxiv.org/abs/2402.02750)). KVQuant: < 0.1 PPL at 3 bit with per-channel pre-RoPE K, non-uniform types and dense-and-sparse outliers ([arXiv 2401.18079](https://arxiv.org/abs/2401.18079)).
- No RULER or needle measurement of q8_0/q4_0 KV on Qwen3.5/3.6 at >= 64k was found [unverified for long-context retrieval]. One fork user reports 6/6 needle retrieval with a 3-bit "turbo3" cache on Qwen3.5-27B ([discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)).

**Reported numbers (speed).**
- Strix Halo, llama.cpp Vulkan, Qwen3.6: q8_0 K+V gives "half the KV memory at the same speed" ([joanmedina guide](https://github.com/joanmedina/qwen3-llama-cpp-inference-tuning); no f16 numbers given).
- Counter-example on a high-bandwidth GPU: Qwen3-8B on A100, llama.cpp CUDA, 32k: F16 101.3, q8_0 66.7, q4_0 62.6 tok/s, i.e. dequant made decode slower ([sotaaz](https://sotaaz.com/post/llamacpp-server-kv-cache-quantization-en)). With 8x less bandwidth per FLOP on Strix Halo this is less likely, but it shows the kernel must keep dequant cheap.
- llama.cpp Vulkan has a DP4A path for q8_0/q4_0 KV: Q is quantized to int8 in the shader and QK uses integer dot products ([PR #20797](https://github.com/ggml-org/llama.cpp/pull/20797); 8060S graphs only as images). FlashInfer reports up to 2x decode speedup for FP8 vs FP16 KV ([FlashInfer blog](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)).
- Compute balance [derived]: per position and KV head, 8 Q heads x 512 MACs = 8192 FLOPs. At 190 GB/s, q8_0 (544 B per position per KV head) needs ~2.9 TFLOP/s and q4_0 ~5.4 TFLOP/s of FP32 work plus dequant, which fits the FP32 rate. With 4 verify tokens (MTP) it becomes 12-22 TFLOP/s, which needs WMMA/coopmat or DP4A.

**Quality risk.** F16: none (BF16-trained model; check for overflow in K, llama.cpp clamps). q8_0: negligible on this family. q4_0 V is the next safe step; q4_0 K is safe on Qwen3.5/3.6 per the data above but catastrophic on Qwen2.5, so validate on our model with a long-context retrieval test before shipping.

**Notes specific to Qwen3.6** [derived, not measured]: only 64 of 256 K dims carry RoPE, so the KVQuant "quantize before RoPE" trick buys little; per-channel K outliers in the 192 NoPE dims could still matter, which is what Hadamard rotation (PR #21038) addresses at near-zero cost (rotate Q, K, V, attend, rotate the output back).

### R3. Bandwidth-efficient decode attention (split-K / Flash-Decoding)

**Mechanism.** At batch 1 there are only 2 KV heads, so a kernel parallel over heads launches 2 workgroups. Flash-Decoding splits the KV sequence into chunks, each workgroup computes a partial softmax (O, m, l or log-sum-exp) for its chunk, and a small second kernel merges the partials with the LSE weights ([PyTorch blog](https://pytorch.org/blog/flash-decoding/)). For GQA, all 8 Q heads that share a KV head are processed by the same workgroup so each K/V row is loaded once and reused 8 times.

**How many splits, block sizes, combine (by implementation).**
- **FlashAttention-2** ([flash_api.cpp, source read](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp)): no split if batch x heads x m-blocks >= 0.8 x (2 x SMs); otherwise the smallest split count reaching >= 85% of the best "wave efficiency" (n_waves / ceil(n_waves)), capped at 128. KV block_n = 64 for head dim > 128. FA3 ([heuristics.h](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/heuristics.h)) adds PackGQA (Q heads of a group packed into the M dimension when that raises tile efficiency) and also splits when one KV head exceeds a 50 MB L2.
- **llama.cpp Vulkan** [source read]: for N <= 8 and GQA ratio <= min(Br, 32), the GQA group becomes the row dimension (N = 8 for us, workgroups_y /= 8). split_k = 2 x shader_core_count / workgroups; for our shape on a 40-CU 8060S that is 2 x 40 / 2 = **40 splits, 80 workgroups**; split length is rounded to a multiple of Bc (at 64k: 1664 positions per split) [derived from the code]. Scalar path for this shape: Br = 8, Bc = 32 (head dim 256), row_split 4, 4 subgroups per workgroup, wave32 on RDNA when rows < 4. Combine ([flash_attn_split_k_reduce.comp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp)): one workgroup per row; max of m over splits, L = sum exp(m_k - m_max) l_k, O = sum exp(m_k - m_max) O_k / L; head dim split over workgroup y; partials stored in F32 (D x rows + 2 x rows per split). Split-K for GQA small batches gave 6.11 -> 14.48 TFLOPS-equivalent on a test shape ([PR #18938](https://github.com/ggml-org/llama.cpp/pull/18938)). Scalar FA refactor ([PR #19625](https://github.com/ggml-org/llama.cpp/pull/19625)) on 8060S without coopmat: Qwen3-30B-A3B tg128 @ d16384 43.2 -> 51.6 tok/s.
- **llama.cpp Metal** [source read] ([ggml-metal-ops.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp)): the "vec" kernel is used for < 20 query rows and head dim % 32 == 0. Fixed **nwg = 32 workgroups per head** along KV, 1-4 simdgroups each (grown while 2 x nwg x nsg x 32 < n_kv), 32 cache values per simdgroup per iteration, **1 head per threadgroup** (no GQA packing in this kernel; K/V reuse across the 8 heads relies on the cache), 1/2/4 queries per threadgroup from a per-device tuning table, then a reduce kernel over the 32 partials. On M5/M6/A19/A20 the prefill path uses the Metal 4 tensor API (MetalPerformancePrimitives, the GPU neural accelerators); it is disabled on older chips because it was not faster (M2 Ultra ~5% slower) ([ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)).
- **SGLang Triton decode** [source read] ([decode_attention.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/decode_attention.py)): default max **8 KV splits** (`--triton-attention-num-kv-splits`, help text: "Larger value is better in longer context scenarios"), per-sequence split count computed on device from seq_len and core count; splits aligned to 32 tokens; grouped kernel packs up to 16 Q heads (BLOCK_H) of a KV group per program; BLOCK_N 64.
- **FlashInfer** ([blog](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)): single-request long-sequence decode "close to 100%" bandwidth utilization; GQA decode on CUDA cores only reached "40%+", so FlashInfer runs GQA decode through its tensor-core prefill kernel (2-3x vs vLLM). It also notes split-KV did not help on RTX Ada 6000 / 4090 for its batch cases. The LSE merge operator is associative and commutative ([cascade post](https://flashinfer.ai/2024/02/02/cascade-inference.html)).
- **vLLM**: uses the FlashAttention / FlashInfer backends above; for Qwen3-Next it enables full CUDA graphs because Triton launch overhead hurt decode ([vLLM blog](https://vllm.ai/blog/2025-09-11-qwen3-next)).
- **MLC-LLM / TVM** [source read] ([_decode_kernels.py](https://github.com/apache/tvm/blob/main/python/tvm/relax/frontend/nn/llm/_decode_kernels.py)): on non-CUDA targets the TIR decode kernel packs the GQA group into threadIdx.y and stages K/V pages in shared memory, but the grid is batch x KV heads x (group / bdy) with **no KV split**; for our shape that is 2 workgroups at batch 1. On CUDA MLC uses FlashInfer.
- **Lean Attention** (stream-K style decomposition of decode attention): 2.6x average over FA-2, up to 8.33x at 512k ([arXiv 2405.10480](https://arxiv.org/abs/2405.10480)).
- **FlashDecoding++**: one shared max value instead of per-split maxima (fallback recompute if any logit leaves the safe range); 1.37x vs FlashDecoding on A100 on average, 1.21x across platforms; the synchronized rescale was 18.8% of attention time for Llama2-7B at 1024 tokens ([arXiv 2311.01282](https://arxiv.org/html/2311.01282v4)). For long-context split-K decode the rescale is amortized over 1600+ positions per split, so expected gain is small [derived].

**Bandwidth fractions reached.**
- Flash-Decoding on A100, batch 1, 16 Q / 2 KV heads, head dim 128: 64.4 us at 64k, 106.6 us at 128k. That is 67 MB / 64.4 us = 1.04 TB/s and 1.26 TB/s [derived], i.e. ~52-63% of an A100-80GB's 2.0 TB/s.
- llama.cpp Vulkan on 8060S: ~175-198 GB/s effective, 82-93% of the measured 212-215 GB/s [derived, section 0].
- Our kernel: ~100 GB/s, ~47% of measured peak.

**Design for our shape** [derived from the above]: one workgroup per (KV head, split) with the 8 Q rows of the group in the workgroup; 2 x CU count workgroups total (40 splits on the 8060S, read the CU count as a device capability rather than a vendor table); K/V tile (Bc = 32-64 rows x 256) loaded coalesced into LDS once and used by all 8 rows; F32 accumulation; partial (O, m, l) in F32 to a scratch buffer; one reduce dispatch. When MTP lands, the same kernel must accept 8 x (k+1) rows so the KV stream is read once per verify step, and the QK/PV products should move to WMMA (coopmat) or DP4A because the arithmetic intensity rises 3-4x.

**Risk / complexity.** Exact. Medium complexity. Note llama.cpp's tuning is heavily vendor-gated (AMD/Intel/NVIDIA branches); the portable parts are split count from CU count, GQA packing, LDS staging, and the LSE reduce.

### R4. MTP speculative decoding

**Mechanism.** Qwen3-Next/3.5/3.6 checkpoints contain a multi-token-prediction module trained with multiple steps so its drafts stay consistent with the main model ([Qwen3-Next blog](https://www.alibabacloud.com/blog/qwen3-next-towards-ultimate-training-%26-inference-efficiency_602580)). Each step drafts k tokens with the MTP module, then the main model verifies k+1 positions in one forward pass. At long context the KV stream (the dominant cost) is read once for all k+1 positions, so the gain grows with depth; for MoE the verify pass touches up to 8 x (k+1) distinct experts, so it costs more than one decode step.

**Official settings** ([Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3-Next card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)): vLLM `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`; SGLang `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. No official acceptance numbers are published.

**Reported numbers.**
- llama.cpp MTP ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), merged; GGUF [ggml-org/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-MTP-GGUF)): Qwen3.6-27B Q8_0 on DGX Spark, aggregate acceptance 72.2% with 3 drafts and 82.6% with 2 drafts; 7.0-7.7 -> 13.9-21.6 tok/s per prompt (~2.4x; code highest at 0.91-0.92 acceptance, translation lowest). The MTP module has its own KV cache. Prompt processing gets slower with MTP enabled because of device-to-host embedding transfers.
- Strix Halo, Qwen3.6-35B-A3B Q8, 76k prompt / 5k output ([Level1Techs](https://forum.level1techs.com/t/strix-halo-llm-inference-notes/249466)): Vulkan 32.7 -> 46.8 tok/s empty (1.43x), 28.9 -> 34.3 at full context (1.19x); ROCm 46.2 -> 63.7 (1.38x) and 16.6 -> 37.5 (2.26x). Self-reported, single poster.
- Strix Halo, Qwen3.6-35B-A3B Q4_K_M, llama.cpp Vulkan, MTP + ngram-mod with `--spec-draft-n-max 2`: 70.3 -> 121.6 tok/s (+73%) on mixed agent workloads; copy x2.5, edit x2.0, reasoning +14%; draft length 4 was -11% to -21% on prose/agent/reasoning; `--repeat-penalty 1.0` alone +9% ([joanmedina guide](https://github.com/joanmedina/qwen3-llama-cpp-inference-tuning)). Identical passes vary by 15-20% on the iGPU per the same author.
- vLLM 0.19.1, 2x RTX 3090, AWQ: +27.5% TPOT with k in {1,2,3} ([community report](https://hackmd.io/@thc1006/SJly6IE6Wx)).
- Apple M5 Max, llama.cpp Q8: 93 -> 105 tok/s (short), 87 -> 97 tok/s (8k) with MTP ([benchmark repo](https://github.com/stared/benching-local-llms-on-apple-silicon)).
- Learned external drafters for this model: DFlash (0.4B block-diffusion drafter, block 16) on B200 at concurrency 1: 2.24x (MT-Bench) to 3.61x (Math500), accept length 5.1-7.6 ([z-lab card](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)); on an Intel Arc B390 iGPU with W4A16 weights: 41 -> 54.7-89.8 tok/s (1.3-2.2x), accept 4.0-6.4, with the note that "verifying several drafted tokens may require loading far more experts than a single-token decoding step" ([Intel/HF blog](https://huggingface.co/blog/ofirzaf/intel-dflash-ptl)). DFlash 3.4 vs MTP 3.0 committed tokens per round [unverified, Doubleword blog via search]. EAGLE-3: up to 6.5x, ~1.4x over EAGLE-2 ([arXiv 2503.01840](https://arxiv.org/abs/2503.01840)); no official EAGLE-3 head for Qwen3.6 found [unverified].

**Estimate for our engine** [derived; assumptions: routed experts ~1.0B of ~2.9B active params, 3 verify tokens touch ~3x the experts, per-draft acceptance 0.83 with k = 2 so ~2.65 tokens per step, 2 draft passes each costing ~1/40 of the model plus the 0.51B-param LM head, KV read once per verify]: verify ~1.69 x 12.5 ms + attention + ~5 ms drafting. At 64k q8_0: ~30 ms per 2.65 tokens, ~89 tok/s vs 62 without (1.45x). At 256k q8_0: ~64 vs 36 tok/s (1.8x). At short context: ~102 vs 80 (1.27x). The LM head is a large share of each draft pass; a reduced draft vocabulary would cut it [inference, not measured here].

**Hybrid-specific requirement.** GDN state cannot be truncated, so rejected drafts need state rollback. vLLM stores the intermediate GDN state per draft position; a llama.cpp PR doing the same measured 1.8x over restarting from a checkpoint (70.95 -> 126.06 tok/s, Qwen3.5-27B + 0.8B draft on a 5090) but was closed ([PR #22400](https://github.com/ggml-org/llama.cpp/pull/22400)). Cost [derived]: one state snapshot is 30 layers x 32 heads x 128 x 128 x 4 B = 63 MB, so storing k = 2 snapshots is ~126 MB of writes (~0.6 ms at 200 GB/s). vLLM has a reported prefix-cache hit-rate drop (92% -> 71%) with MTP on Qwen3.5/3.6 A3B (vLLM issue #38182, as cited in the community report).

**Risk / complexity.** Output distribution unchanged if verification is exact (greedy, or rejection sampling for temperature > 0; llama.cpp added rejection sampling for MTP in [#27694](https://github.com/ggml-org/llama.cpp/pull/27694)). High complexity.

### R5. N-gram drafting (prompt lookup, ngram-mod, suffix decoding)

**Mechanism.** Draft tokens by matching the last n tokens against earlier text (prompt and prior output) and proposing what followed; no model, near-zero draft cost. Works when the output copies from context (code edits, refactors, summaries, reasoning restating itself), which is common at long context.

**Reported numbers.**
- Prompt lookup decoding: ~2.4x average on summarization and context QA, Mistral-7B, A100, max n-gram 3, 10 draft tokens ([repo](https://github.com/apoorvumang/prompt-lookup-decoding)).
- SuffixDecoding (suffix trees over prompts and past outputs, adaptive draft length): up to 5.3x on SWE-Bench / Text-to-SQL, 2.8x faster than EAGLE-2/3 ([arXiv 2411.04975](https://arxiv.org/abs/2411.04975)).
- llama.cpp ngram-mod ([PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164)): LCG rolling hash of n-grams -> next token, ~16 MB, constant memory; suggested `--spec-ngram-size-n 24 --draft-min 48 --draft-max 64`; "MoEs require long drafts" (because the verify pass loads more experts). Combined with MTP: +73% on the Strix Halo agent workload above.

**Risk / complexity.** None for quality; low once the verify path from R4 exists. Low-overlap prompts gain nothing and can lose throughput if drafts are always attempted.

### R6. Prefix caching with GDN state checkpoints

**Mechanism.** Hybrid models cannot reuse an arbitrary prefix from the KV cache alone, because the GDN state at the prefix end is also needed. Keep snapshots of the GDN (and conv) state at a few positions so a returning conversation re-prefills only the new suffix.

**Reported numbers.** Qwen3.8-27B returning to a 65K-token chat: >5 min with full reprocessing, 13.6 s with `--cache-ram 8192 --ctx-checkpoints 8`; Qwen3.6 compacting a 126K context: 313 s with 4 checkpoints, 14.7 s with >= 8 ([joanmedina guide](https://github.com/joanmedina/qwen3-llama-cpp-inference-tuning)). vLLM implements a hybrid KV cache manager that sizes attention blocks to match the linear-state page size ([vLLM blog](https://vllm.ai/blog/2025-09-11-qwen3-next)).

**Risk / complexity.** Exact. Medium complexity. Memory: 63 MB per snapshot [derived].

### R7. Chunkwise-parallel Gated DeltaNet prefill (WY / UT transform)

**Mechanism.** Split the sequence into chunks of C tokens. Within a chunk, the product of generalized Householder updates is written in WY form: with Gamma_ij = gamma_i / gamma_j (cumulative decays), U~ = [I + strictLower(diag(beta) (Gamma o K K^T))]^-1 diag(beta) V, then S_{t+1} = gamma_C S_t + (U~ - W<- S_t^T)^T K->, and O = Q<- S_t^T + (Q K^T o M)(U~ - W<- S_t^T) ([Gated DeltaNet, arXiv 2412.06464](https://arxiv.org/abs/2412.06464); WY idea from [arXiv 2406.06484](https://arxiv.org/abs/2406.06484)). Everything except the inter-chunk state carry is matmul-shaped, so it maps to WMMA.

**Implementation details** ([FLA chunk.py, source read](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)): chunk size 64; pipeline = chunk-local cumsum of log gates -> fused K K^T + triangular solve + recompute W, U (`solve_tril` inverts 16x16 diagonal blocks by forward substitution and merges them to 32x32 / 64x64) -> one kernel that carries the F32 state sequentially over chunks (parallel over V-blocks of 32/64 x heads; state tile 64 x BV per K half) and writes per-chunk states -> output kernel. Qwen's FlashQLA (TileLang, Hopper/Blackwell) fuses this into 2 kernels plus a context-parallel preprocessing kernel, and splits long sequences across SMs using gate decay (each split warms up from zero over ~6-8 chunks); 2-3x forward vs FLA Triton on H200, enabled when batch x heads <= 40 ([Alibaba Cloud blog](https://alibabacloud.com/blog/flashqla-cp-bwd-friendly-fused-linear-attention-kernels-for-gdn_603084)).

**Reported numbers vs a fused recurrent scan.** llama.cpp CUDA chunked GDN ([PR #29353](https://github.com/ggml-org/llama.cpp/pull/29353), merged): Qwen3.5-35B-A3B prefill +10.8% (4090) to +20.6% (RTX PRO 6000), Qwen3.6-35B-A3B on GB10 +3.4-4.1%, Strix Halo +6.75-6.98% on Qwen3.8-27B (earlier master); mean KLD 0.00046 vs the recurrent path. Vulkan chunked port ([PR #30207](https://github.com/ggml-org/llama.cpp/pull/30207), draft): "significant op-level speedup, but not on the model-level yet", needs coopmat2 or KHR_cooperative_matrix maintenance1 (RADV upstream only). Absolute: Qwen3.5-35B-A3B Q4_K_M pp512 7851 tok/s on an RTX 5090 with the recurrent Vulkan kernel ([PR #20662](https://github.com/ggml-org/llama.cpp/pull/20662)).

**Risk / complexity.** Numerics only (triangular inverse in F32). High complexity for a modest gain; do it after R1.

### R8. q4_0 (or rotated q4) KV for 128k-256k

See R2 for numbers. Rotation (Hadamard on Q/K/V, inverse on the output) improves low-bit KV at almost no cost ([PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)). TurboQuant (random rotation + PolarQuant + 1-bit QJL residual) claims 3-bit KV with no accuracy loss on LongBench / needle / RULER for Gemma and Mistral and up to 8x faster attention logits at 4 bit on H100 vs 32-bit keys ([Google Research blog, 2026-03-24](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)); llama.cpp fork results are mixed (one tester, ubergarm, found turbo3/turbo4 worse than q4_0 in perplexity) and it is not upstream ([discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)).

### R9. Query-aware sparse decode attention (Quest-style)

**Mechanism.** Keep per-page (16 tokens) channel-wise min and max of K. For each query, score each page as sum_i max(q_i M_i, q_i m_i) (an upper bound on any q.k in the page), then attend exactly over the top pages only ([Quest, arXiv 2406.10774](https://arxiv.org/abs/2406.10774)).

**Reported numbers.** 7.03x self-attention speedup at 32k with a 2048-token budget (RTX 4090 kernel), 1.74x end-to-end FP16 and 2.23x with 4-bit weights; passkey at 100k (Yarn-Llama-2-7b): 88/92/96/100/100% at budgets 256/512/1024/2048/4096 vs H2O/TOVA 2-10%; first two layers kept dense. Evaluated on MHA models only.

**Why the risk is higher on our model.** In the hybrid, the 10 full layers are the only exact-retrieval path. Qwen's own successor replaces them with a trained sparse attention (QSA: 4-token blocks, top 512 blocks = 2048 tokens, 4-head MQA indexer) and reports that "directly applying the indexer for sparse attention leads to a clear performance drop" before joint training, and that "periodic full attention is particularly important for long-context performance" ([arXiv 2608.30320](https://arxiv.org/html/2608.30320)). Quest's selection must also be shared by the 8 Q heads of a KV head (union or summed scores), which lowers selectivity; Quest did not evaluate GQA [inference]. Metadata cost: min+max per page is 2/16 of K, ~6% of KV bytes [derived].

**Expected gain** [derived]: after R2/R3, attention is 3.75 ms of ~16 ms at 64k (q8_0), so <= 1.3x; at 256k 15 ms of 27.5 ms, so ~2x with a 4k budget. Only worth it past 128k, and only after a RULER/needle run on our model with the exact selection rule.

### R10. Sparse prefill (MInference, DuoAttention)

- MInference: offline per-head pattern choice (A-shape, vertical-slash, block-sparse), up to 10x prefill latency reduction on A100 for 1M-token prompts; tested on Qwen2-128K among others ([arXiv 2407.02490](https://arxiv.org/abs/2407.02490)).
- DuoAttention: identify retrieval heads (full KV) vs streaming heads (sink + recent window) by optimization on synthetic data; for GQA models up to 1.67x memory, 1.50x decode, 1.63x prefill; 3.3M context for Llama-3-8B on one A100 with quantization ([arXiv 2410.10819](https://arxiv.org/abs/2410.10819)).
- For us [derived]: at 32-64k, attention is roughly half of prefill FLOPs (section 0), so even a perfect sparse prefill is <= ~2x, with the same hybrid retrieval risk as R9 and only 20 KV heads in the whole model to classify.

---

## 3. Question-by-question findings

### 3.1 Decode attention at long context

See R3. Summary of split counts: FA-2 picks the smallest split count within 85% of the best wave efficiency (max 128, block_n 64 at head dim 256); llama.cpp Vulkan uses 2 x CU count / workgroups (40 splits for us on the 8060S) with GQA rows packed (Br 8, Bc 32); llama.cpp Metal uses a fixed 32 splits per head and no GQA packing in the vec kernel; SGLang Triton uses up to 8 splits by default with up to 16 heads packed; MLC's TIR path does not split. All combine with the LSE merge in a separate small kernel. Bandwidth: FlashInfer claims near 100% on GPUs for long single-request decode; Flash-Decoding reaches ~52-63% of A100-80GB [derived]; llama.cpp Vulkan reaches ~82-93% of measured Strix Halo bandwidth [derived]; we are at ~47%.

### 3.2 KV-cache quantization

See R2 and R8. For the Qwen3.5/3.6 hybrid family the measured impact is small down to q4_0 (PPL +0.25% on Qwen3.5-4B; 2/500 ARC flips on Qwen3.6-27B; mean KLD 0.0047 for q4_0 vs 0.0023 for q8_0 on Qwen3.6 [unverified article]). Non-hybrid Qwen models are fragile in K (Qwen2.5-7B collapses with 4-bit K; Qwen3-0.6B PPL 13.7 -> 62). Speed impact on Strix Halo: "same speed, half memory" for q8_0 in llama.cpp Vulkan; on A100 CUDA quantized KV was 17-38% slower (dequant bound). FP8: vLLM supports E4M3 with per-tensor scales and claims minimal degradation without long-context data ([vLLM docs](https://docs.vllm.ai/en/v0.6.2/quantization/fp8_e4m3_kvcache.html)); RDNA3.5 has no FP8 arithmetic, so it brings nothing over q8_0 here. No long-context retrieval benchmark (RULER/needle >= 64k) for q8_0/q4_0 KV on Qwen3.6 was found [unverified].

### 3.3 Prefill

See R1 and R7. Key fact: llama.cpp reaches ~1000 tok/s (pp512) and 322.7 tok/s at a 153k context on this exact chip with a *recurrent* fused GDN kernel, so our 34 tok/s is a batching problem, not a GDN-algorithm problem. Chunkwise GDN (chunk 64, WY/UT, FLA or FlashQLA) adds 3-21% on CUDA. No absolute Qwen3-Next prefill numbers from vLLM/SGLang were found; Qwen claims prefill ~7x vs Qwen3-32B at 4k and >10x beyond 32k, decode ~4x at 4k and >10x beyond 32k ([Qwen3-Next blog](https://www.alibabacloud.com/blog/qwen3-next-towards-ultimate-training-%26-inference-efficiency_602580)).

### 3.4 Sparse / approximate attention

| Method | Mechanism | Reported | Fit for our hybrid |
|---|---|---|---|
| Quest | top-k 16-token pages by min/max K bound | 7.03x attention / 2.23x e2e at 32k, 2048 budget | possible past 128k, validate first (R9) |
| DuoAttention | retrieval heads full, streaming heads windowed | GQA: 1.50x decode, 1.63x prefill | weak: only 20 KV heads, calibration needed |
| MInference | per-head sparse prefill patterns | up to 10x prefill at 1M (A100) | only for very long prompts |
| StreamingLLM | sinks + recent window | 22.2x vs recompute; needle collapses at 4x compression (22.3%) ([arXiv 2309.17453](https://arxiv.org/abs/2309.17453), [2407.01527](https://arxiv.org/abs/2407.01527)) | no: gated attention cut first-token attention from 46.7% to 4.8% in Qwen's study ([arXiv 2505.06708](https://arxiv.org/abs/2505.06708)), and evicted tokens are lost |
| RetrievalAttention | CPU ANN index over KV, 1-3% of tokens | 0.188 s/token at 128K, 8B model, RTX 4090 ([arXiv 2409.10516](https://arxiv.org/abs/2409.10516)) | no: built for small-VRAM GPUs |
| ShadowKV | low-rank K on GPU, V offloaded, sparse reconstruction | 6x batch, 3.04x throughput on A100 ([arXiv 2410.21465](https://arxiv.org/abs/2410.21465)) | no: throughput/capacity design |
| Trained sparse (DSA, QSA) | learned indexer, top 2048 tokens | QSA attention module 7.6x prefill / 4.9x decode at 1M, faster than dense from 64K; RULER equal to full attention | needs a model trained with it (Qwen3.8-Flash-Next); llama.cpp Vulkan sparse FA for such models: 64k q8_0 decode 27.9 -> 32.3 tok/s ([PR #29639](https://github.com/ggml-org/llama.cpp/pull/29639)) |

Hybrid-specific: with only 1/4 of layers holding KV, the absolute attention cost is already 4x smaller than in a dense model of the same width, so sparse attention buys less and risks the only exact-retrieval path. Qwen's own RULER for Qwen3-Next with dense attention: 97.6 at 64k, 96.0 at 128k, 93.5 at 256k ([Qwen3-Next card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)).

### 3.5 Speculative decoding

See R4 and R5. Best measured on Strix Halo for this exact model: MTP with 2 drafts plus ngram-mod, +73% on agent work (Q4_K_M); MTP alone 1.19x (Vulkan) to 2.26x (ROCm) at 76k depth (Q8). llama.cpp acceptance on Qwen3.6: 82.6% with 2 drafts, 72.2% with 3. Longer drafts hurt prose on MoE because verification loads more experts.

### 3.6 Qwen3-Next / 3.5 / 3.6 specifics

- Context: native 262,144; YaRN `factor` 4.0, `original_max_position_embeddings` 262144 for up to 1,010,000. Open-source frameworks implement static YaRN, which "can hurt performance on shorter texts"; enable only when needed and scale `factor` to the typical length; keep >= 128K context to preserve thinking ([Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)).
- Gated attention: output gate after SDPA, head dim 256, RoPE on the first 25% of dims; the gate is credited with removing attention sinks and massive activations ([Qwen3-Next blog](https://www.alibabacloud.com/blog/qwen3-next-towards-ultimate-training-%26-inference-efficiency_602580), [arXiv 2505.06708](https://arxiv.org/abs/2505.06708): mean max activation 1053 -> 94, RULER at 128k 31.65 -> 58.82 in their 15B MoE study).
- llama.cpp status: `qwen35moe` architecture supported; Vulkan GDN op merged ([#20334](https://github.com/ggml-org/llama.cpp/pull/20334): Qwen3.5-35B-A3B Q8_0 on 8060S tg 43.90 -> 53.33), subgroup sharding merged ([#20662](https://github.com/ggml-org/llama.cpp/pull/20662)), chunked Vulkan GDN open ([#30207](https://github.com/ggml-org/llama.cpp/pull/30207)); Metal GDN merged ([#20361](https://github.com/ggml-org/llama.cpp/pull/20361)); CUDA/HIP chunked GDN merged ([#29353](https://github.com/ggml-org/llama.cpp/pull/29353)); MTP merged ([#22673](https://github.com/ggml-org/llama.cpp/pull/22673)); GDN partial rollback closed ([#22400](https://github.com/ggml-org/llama.cpp/pull/22400)).
- Measured long-context tok/s on Strix Halo (Qwen3.6-35B-A3B, llama.cpp): Q8 Vulkan 32.7 -> 28.9 tok/s from empty to a 76k prompt (MTP off), ROCm 46.2 -> 16.6; Vulkan with q8_0 KV at a 153,562-token context: 24.1 tok/s decode, 322.7 tok/s prefill; llama-bench Q8_0 tg128 53.4 (Vulkan) / 44.6 (ROCm), pp512 ~1047 / ~1000 ([Level1Techs](https://forum.level1techs.com/t/strix-halo-llm-inference-notes/249466), self-reported, baselines between server and llama-bench runs are not consistent). UD-Q6_K with MTP at a 32k context: 59.5 tok/s ([Soothill](https://www.soothill.io/blog/2026/08/14/coding-model-benchmark-strix-halo/), backend not stated for that row).
- Apple: base M5 has 153 GB/s and a neural accelerator per GPU core; M5 Pro 307 GB/s, M5 Max 614 GB/s ([Apple M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/), [M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)). M5 Max 128 GB, llama.cpp Q8: 93/87 tok/s (short/8k) without MTP, 105/97 with; MLX 85/79 ([benchmark repo](https://github.com/stared/benching-local-llms-on-apple-silicon)). Base M5 (10-core GPU) with oMLX: ~59.5-61.6 tok/s at 1k, ~57 at 4k [unverified, omlx.ai pages seen only in search results]. With 153 GB/s on the base M5 the KV math of section 0 is ~1.4x worse than on Strix Halo [derived], so q8_0 KV and MTP matter even more there. No 64k+ measurements on Apple silicon were found [unverified].

---

## 4. Gaps and things I could not verify

- No RULER / needle results for q8_0 or q4_0 KV on Qwen3.5/3.6 at >= 64k; all Qwen3.6 KV-quant evidence is PPL, KLD (some at 65k context) and short-answer flips.
- No vLLM or SGLang absolute prefill/decode tok/s for Qwen3-Next at 32k+; no official MTP acceptance rates from Qwen.
- No llama-bench depth sweep (d32768 / d65536) for qwen35moe on the 8060S; the Strix Halo long-context numbers are server measurements from one forum poster.
- The Anbeeld long-context KV article was only seen as numbers quoted in a GitHub comment.
- The 175-198 GB/s llama.cpp attention bandwidth, all projected tok/s, the MTP estimate, the GDN snapshot cost and the prefill targets are my arithmetic from the sources above, not measurements.
- Quest, DuoAttention and MInference have no published evaluation on Qwen3-Next/3.5/3.6 or any GDN hybrid; the only hybrid evidence is Qwen's statement that untrained sparse selection drops quality.
- The 8060S DP4A FA results (PR #20797) and the Vulkan chunked-GDN 8060S results (PR #30207) exist only as images in the PRs.

---

## 5. Measured on our engine (2026-10-27)

Qwen3.6-35B-A3B Q4_K_S on the Radeon 8060S. **Protocol caveat:** the GPU was
shared the whole day with another session's render farm (~2 GPU-seconds of
engine time per second, `drm-engine-gfx`), and our timings swung by up to 15x
with its kernel mix (a decode step measured 13 ms and 210-300 ms within the hour).
Wall-clock numbers below are what we got under that load; kernel numbers are
minima of interleaved rounds.

**Prefill (batched chunks of 512).** What changed, in the order it mattered:
chunk-wide causal attention on cooperative matrices (K/V read once per chunk,
key split + merge for short chunks), a multi-token Gated DeltaNet kernel (state
in registers across the chunk), routed experts grouped by expert (each expert's
weights read once per 8 of its slots), batched prepare kernels, a tiled fp32 GEMM
for the dense router, the default submission budget (one 512-chunk at depth was a
multi-second submission that RADV cancelled: device lost).

| ingested segment | chunk 16 (before), q8_0 | chunk 512, q4_0 |
|---|---:|---:|
| 4k -> 16k | 202 tok/s | 805 tok/s |
| 16k -> 64k | 148 tok/s | 548 tok/s |
| 64k -> 128k | 98 tok/s | 339 tok/s |

GPU time of one 512-token chunk (per-pass timestamps): 0.5 s at depth 4k, 1.44 s
at depth 96k, of which attention 0.96 s (66%). The chunk attention runs at
~10-12 TFLOP/s (Float16 cache) of the 59 TFLOP/s WMMA peak. q8_0 staging first
ran at a fifth of Float16's speed: `Float16(scale * x)` narrows a float product,
which Lava spells in integer operations; building the half from the int8 bits
(0x6400 | (x+128), minus 1152) brought it to 0.7x of Float16.

**Decode (flash-decoding, 128 splits).** Plain decode at short depth: 12.8 ms
per token (846 passes; per-pass bandwidth 170-230 GB/s, the step is near its
floor). With q8_0 KV: 53 / 47 / 40 tok/s at 16k / 64k / 128k; with q4_0 KV:
53 / 46 / 49 tok/s (wall clock, contended).

**KV format quality**, teacher-forced over 64 real tokens after a 32k-token
prefix, against a Float16 cache: q8_0 mean KL 6.9e-4 (max 6.7e-3), argmax agrees
64/64; q4_0 mean KL 2.1e-3 (max 1.9e-2), 63/64. The needle (a code word stated at
position 0) is retrieved at 4k-128k with both.

**Prompt-lookup drafting**, offline on 160 saved tool-loop answers (358k
generated tokens): draft length following the matched n-gram (n >= 2, up to 8
tokens) gives 1.70 tokens a step; with the decode profile and independent
routing for the verify's experts that is an estimated 1.30x / 1.34x / 1.37x at
4k / 64k / 128k (perfect draft lengths: 1.55-1.61x). Implemented as a verify
chunk of up to 9 tokens with replay of the accepted tokens' recurrence after a
refusal; with decode's attention in the verify the output is plain greedy
decoding's, token for token.

**To the model's 262k limit** (q4_0 KV, chunk 512, a later and heavier phase of
the same shared-GPU load, so slower than the table above at equal depth): the
needle is retrieved at 4k, 64k, 128k, 196k and 261k; prefill 370 / 221 / 157 /
120 tok/s over the segments ending there, decode 30 / 25.5 / 22 / 20 tok/s. GTT
in use 27.2 GiB throughout (the q4_0 cache for 262k positions is 1.7 GB).

**Speculative decoding, measured** (prompt-lookup drafts, verify chunks of up to
9 tokens, interleaved rounds against plain decoding on the shared GPU, a prompt
that asks for its code back with a rename): at depth 0, plain 46.9 tok/s, verify
with decode's attention 87.5 (token-identical to plain), with the chunk attention
61.1; at depth 64k, plain 60.0, decode's attention 45.7, chunk attention 83.1.
The speculator switches to the chunk attention at a set depth (16k). In the
sampled tool loop (temperature 0.6) the output with drafts was identical to the
output without, but 2.3x slower under the heaviest load, because each 16-token
chunk then waits on the host several times instead of once.

**Prefix reuse.** `PrefixCache` (text_backbone.jl) copies a session's KV rows of
a prompt prefix and its recurrent state on the device; the tool loop takes it as
`prefix=` and ingests only the rest. Output is bit-identical to ingesting the
whole prompt: with bit-exact chunks for any prefix, with 512-token chunks for a
prefix that ends on a chunk boundary (the attention's key split and the GEMM
setup follow the chunk width, so differently cut chunks round differently; the
loop refuses a misaligned prefix). A 128k-token document is then ingested once
(~10 min on the shared GPU) and every question after it starts at once.

**Chunk attention with 64-row workgroups** (4 tiles × 4 subgroups; the 32-row
version was capped by every workgroup re-reading its KV head's cache, 64 FLOP a
Float16 byte): attention at depth 96k 96 → 68 ms a layer (q8_0, 512 tokens), and
the q8_0 profile, chunk 512, in a calmer phase of the same load:

| segment | prefill | decode at its end |
|---|---:|---:|
| 4k -> 16k | 838 tok/s | 51.6 tok/s |
| 16k -> 64k | 627 tok/s | 44.8 tok/s |
| 64k -> 128k | 431 tok/s | 45.7 tok/s |

128k tokens ingest in under 5 minutes (this morning, chunk 16: 22 minutes).

**Defaults.** `textsession` for Qwen3.6 keeps the receipts' Float32 cache and
grouped attention up to 32k positions and switches to q8_0 + flash-decoding
past that; the tool loop then ingests in 512-token chunks. The pipelined fetch of
quantized cache rows in the decode kernel took q4_0 decode attention at 128k
from 0.83 to 0.74 ms a layer (q8_0 0.88 → 0.87; both are bound by arithmetic
per position, not bytes). Status of section 1's ranking: R1 (batched prefill),
R2 (q8_0), R3 (split-K decode), R5 (n-gram drafts), R6 (prefix reuse) and R8
(q4_0) are implemented and tested; R4 (MTP) needs the MTP tensors, which this
GGUF does not have; R7 (chunkwise GDN) is a sequential multi-token kernel, 3% of
a deep chunk; R9/R10 (sparse attention) were not attempted, per section 3.4.

**K-quant GEMM staging** (DNNKernels `kq_stage_mma`): the wide-chunk matrix
products staged each weight to Float16 through Lava's exact narrowing of a
product (~20 integer ops). Staging through a select instead: Q4_K 8192x2048 at
512 columns 1.07 → 0.87 ms GPU time (15-19% less), Q6_K 8-16% less; the chunk's
qkv 42 → 32 ms, z/out ~19 → 14 ms; KL(chunk 16 || chunk 512) 6.9e-5.

**MTP drafts (R4), implemented** (`qwen36_mtp.jl`, weights from the native
checkpoint by `download_qwen36_mtp.py`, 1.6 GB BF16, quantized to Q4_K at load
in 45 s). The layer runs as vLLM's qwen3_5_mtp.py does, as layer 40 with its own
KV cache, filled by every prompt chunk (keys and values only). A round's drafts
and verify are one plan, one submission and one wait. With decode's attention in
the verify the output is plain greedy decoding's, token for token (k = 1, 2, 3;
Float32 and q8_0 caches); sampled tool loops verify column by column with the
draws of refused columns undone, as with n-gram drafts.

Acceptance (192 tokens, rename prompt / code written from scratch):

| drafts a round | accepted | tokens a round |
|---|---:|---:|
| k = 1 | 89-94% | 1.9 |
| k = 2 | 83-90% | 2.7-2.8 |
| k = 3 | 76-82% | 3.3-3.4 |

N-gram drafts on the from-scratch prompt: 20 of 60 accepted. Wall time on the
shared GPU (render farm at 86% of the gfx engine) swung 3x between runs of the
same variant, so the speed is each plan's own (minimum of 25 interleaved runs
with a host wait) times the acceptance:

| depth | plain decode | k=3 round | verify alone (4 tokens) | estimate, k=3 |
|---|---:|---:|---:|---:|
| 0 | 12.96 ms (77 tok/s) | 31.35 ms | 23.75 ms | ~104 tok/s |
| 64k | 17.59 ms (57 tok/s) | 39.27 ms | 30.72 ms | ~81 tok/s |

About 1.4x. A draft costs ~2.5 ms, most of it the 417 MB output head; the
verify of 4 tokens costs 1.8x a decode step because each token reads its own 8
experts (the grouped expert kernel reads each once, but differs by rounding).
Restricting the draft head to the 16384 most frequent tokens (FR-Spec; 99.5% of
held-out model output) cut a draft to 0.7-1.2 ms with rows bit-identical to the
full head, but acceptance fell (k=3 82% → 64%, k=2 90% → 72%), far more than
its coverage explains; not understood, parked on branch `sd/mtp-draft-vocabulary`.
