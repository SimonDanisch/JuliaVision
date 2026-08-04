# Whisper decoder: the port plan

Written 2026-08-04, after exporting the decoder and measuring the shapes it
actually produces. Every number below is measured on the RTX 4000 Ada, not
estimated. The encoder is done (`whisper-artifact-fp32`, rel rms 6.3e-5, shipped
as an artifact) and its mel front end now runs on GPU in 0.712 ms; this is the
other half.

## What the export settled

`tools/export_whisper_decoder.py` (new) exports **one decoder step** with the KV
cache passed in and out as plain tensors:

    (input_ids, self_k, self_v, cross_k, cross_v, cache_position)
        -> (logits (1, 51866), new_self_k, new_self_v)

`export_whisper.py` claims "the decoder is autoregressive with a KV cache, which
`torch.export` will not capture". That is true of the *generation loop* and false
of a *step*. A step is static-shaped and exports cleanly. This is the same
decomposition llama.cpp and whisper.cpp use: the graph is one token of work, and
the loop, the sampling and the cache rotation are host code.

**96 ops. `DNNKernels.coverage` reports exactly one missing: `index_put.default`,
now implemented.**

### The first export was 162 ops and quietly useless

The obvious way to write the step is to hand `WhisperDecoder` an
`EncoderDecoderCache` and let it do the cache work. That exports — 162 ops, one
missing op, everything looking healthy — and produces a graph that **cannot
work**, in three escalating ways:

1. `DynamicCache` seeds each layer with `torch.tensor([])`, which `torch.export`
   lifts into the signature as `lifted_tensor_0..15`: sixteen zero-element
   weights present in no `state_dict`, so the load fails naming tensors that
   exist in no module. *This is the failure that is loud.*
2. Passing the cache as a legacy tuple makes it `cat` that empty seed onto the
   real cache — 80 MB of copying per token for nothing.
3. **The updated cache never leaves the graph.** `DynamicCache` accumulates into
   a Python object, so the exported outputs are `[logits]` and nothing else. A
   host loop driving that graph attends to a *zeroed* cache at every step. It
   runs, it is fast, and it emits fluent nonsense.

So the step is now written out against the model's own submodules with the cache
write explicit, and the exporter asserts it against the real `WhisperDecoder`
over 6 tokens before writing anything: **rel rms 0.000e+00, argmax identical at
every token.** Bit-exact, not merely close. The check runs at export time
precisely because (3) is invisible downstream — a hand-written transformer with a
mask off by one still produces plausible logits and a plausible transcript.

Writing it out also removed a third of the graph (162 → 96 ops) and every one of
the sixteen lifted constants.

Two more things had to be right, and both are documented traps:

* **Export on CUDA, not CPU.** From CPU, every attention decomposes into
  `bmm + _softmax + logical_not + where + any.dim + full_like` — 531 ops, ~90 of
  them masking scaffolding. From CUDA it is 8 fused
  `_scaled_dot_product_efficient_attention`, which we already implement. The
  encoder exporter's own docstring says this; I hit it anyway.
* **`run_decompositions()` inside the precision context**, matching the encoder
  exporter, or the op names are not comparable to its histogram.

Do **not** measure coverage by grepping `runop!` registrations. `view`, `permute`
and `expand` never appear as ops — the graph converter folds them into buffer
metadata — so a grep-based gap analysis invents 250 missing ops that do not
exist. Use `DNNKernels.coverage(path)`.

## The regime is GEMV, and it is the whole problem

large-v3-turbo is distilled to **4 decoder layers** against the encoder's 32, so
the decoder is small — but every matmul has `M = 1`:

    x20  addmm  (1,1280) @ (1280,1280)   -> (1,1280)    attention projections
    x 4  addmm  (1,1280) @ (1280,5120)   -> (1,5120)    FFN up
    x 4  addmm  (1,5120) @ (5120,1280)   -> (1,1280)    FFN down
    x 4  mm     (1,1280) @ (1280,1280)   -> (1,1280)
    x 1  mm     (1,1280) @ (1280,51866)  -> (1,51866)   output projection

158M parameters read per token — 634 MB in fp32 — so at the measured 307 GB/s
DRAM roofline a token cannot cost less than **2.07 ms**. What our `mul!` does
today:

    shape                      ms       GB/s    % of roof
    (1,1280)@(1280,1280) x24   0.6146   10.7      3.5%
    (1,1280)@(1280,5120) x 4   0.5263   49.8     16.2%
    (1,5120)@(5120,1280) x 4   0.2473  106.0     34.5%
    (1,1280)@(1280,51866)x 1   2.5908  102.5     33.4%
    ------------------------------------------------------
    per token                 ~20.4 ms          ~10x the floor

**Three quarters of that is the twenty-four 1280x1280 projections at 3.5% of
roofline.** That is not a tuning gap. `coopmat_gemm!` needs `M >= 16` and we are
handing it `M = 1`, so fifteen sixteenths of every cooperative-matrix tile is
padding. A 100-token transcript would take 2 s of GEMV where the floor is 0.2 s.

This is why llama.cpp has a separate `mul_mat_vec.comp` from `mul_mm.comp`, and
it is the one kernel this port actually needs.

## References, per piece

Per `feedback-port-sota-kernels`: read the fastest open implementation for our
API first, port the STRUCTURE, re-measure every constant.

| piece | reference | why this one |
|---|---|---|
| GEMV | `dev/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp` | Vulkan GLSL, same target as our emitter; llama.cpp exists *for* batch-1 decode and this is its hot kernel |
| KV-cache attention | `flash_attn_cm1.comp` + `flash_attn_split_k_reduce.comp` | already the reference for our shipped split-KV work (task #42) |
| decode loop, beam search, timestamps | whisper.cpp (`whisper.cpp`, ggml) | the reference for Whisper's *loop* specifically — temperature fallback, compression-ratio and no-speech thresholds, timestamp rules |
| tokenizer | `gen/whisper/tokenizer.json` + `merges.txt` | already downloaded; BPE on the host, no kernel |

Not useful here, checked: the GPUMODE corpus has no GEMV board, and its
`amd-mla-decode` winners wrap `aiter`'s precompiled `.co` assembly with no source
([[kernelbot-corpus]]).

## Work items

Ordered so each one is verifiable on its own. Nothing here is speculative —
every item has a measurement or a reference that decides when it is done.

### D1 — the missing ops — **DONE**

`le.Tensor`/`le.Scalar` (the causal mask over the 448-slot cache), and after the
step was rewritten, `index_put.default` — the cache write itself, which
`slice_scatter` cannot express because the position is a tensor rather than a
literal. `coverage` reports nothing missing on either graph.

### D2 — the decoder graph runs, node by node, against PyTorch — **DONE**

`tools/dump_whisper_decoder_refs.py` and `tools/verify_whisper_decoder.jl`, both
new — separate scripts rather than a mode on the encoder's, because they record
two graphs, at a nonzero cache position, plus a token sequence.

Measured: `whispercross` 10/10 ops within tolerance (worst accumulated max|Δ|
2.6e-5), `whisperdec` 96 ops within tolerance (worst 1.9e-3 at
`native_layer_norm_12`). End to end: encoder hidden rel rms **6.30e-5** (the
encoder's own published number, reproduced through this path), cross-K **5.2e-5**,
cross-V **8.0e-5**.

The step is recorded at cache **position 3**, not 0. At position 0 a mask that
admits one slot too many, an `index_put` that writes the wrong slot, and an
attention that ignores the cache entirely all produce the same output.

This was deliberately before any performance work, and it paid immediately: it
is what made it safe to swap the whole matmul path underneath and still know the
answer was unchanged.

### D3 — the GEMV kernel — **BUILT, and it did not move the step**

Two findings, and the second is the one that matters.

**The ported llama.cpp kernel does not fit our data.** `mul_mat_vec.comp` reads
the matrix contiguous along the reduction axis. `hoistpermutes` materialises
every weight as torch `(K, N)` — Julia `(N, K)` — contiguous along the *output*
axis, because that is what `coopmat_gemm!` wants. So a second kernel
(`gemv_ncontig_kernel`) parallelises along the contiguous axis instead: one
thread per output row, `BLOCK/TM` threads splitting `K`, a shared-memory tree to
combine them. `gemv!` dispatches on `Transpose` to pick between the two. 1.9-3.0x
over `mul!` on the decoder's shapes, interleaved same-session, 88/88 tests.

**And the decoder step did not change: 13.58 ms -> 13.50 ms.** Because it is not
GPU-bound. A `(64, 64)` GEMV — 16 KiB — costs 0.0553 ms; a `(1280, 1280)` one —
6.4 MiB, four hundred times the data — costs 0.0555 ms. Timing the dispatch loop
without waiting for the device puts ~89% of that on the host, and a broadcast and
a `mul!` measure the same, so it is not the GEMV path: **Lava costs ~63 us of
host time per dispatch**, and 96 ops is a 6.1 ms floor under an 11.2 ms step.

So D3 is done as *arithmetic* and irrelevant as *latency*. The next item is not a
kernel — see D8.

**A measurement note that invalidated two of my own sweeps.** This card idles at
450 MHz of 3105 and a 50 us kernel never boosts it, so the first sweep ranked
`BLOCK = 512` as **2.7x** better than 256. Re-run after spinning the device for
0.25 s, the whole 24-configuration sweep spreads 1.04x-2.18x and the default is
within 8% of the best on all four shapes. The same shape measured 0.030, 0.048
and 0.122 ms in one session before this was understood. Every GEMV benchmark now
boosts first.

### D4 — the host loop: cache, sampling — **DONE** (tokenizer still out)

`WhisperRunner.decode.jl`: `KVCache`, `whisper`, `fillcross!`, `decodestep!`,
`greedy`, `transcribe`. **224 of 224 tokens match the reference exactly**, first
difference: none.

The reference is a *pure argmax loop* over the same PyTorch step, not
`generate`. That split was not in the original plan and is the right one:
`generate` applies Whisper's logits processors — suppression lists,
begin-suppression, timestamp ordering — which change which token wins the argmax.
Comparing against it here would report a missing policy (D6) as a broken cache.
`whisperdec/generate_tokens` holds that target for when D6 lands.

The cache is *not* written in place, as this item assumed: the graph is
functional, so it returns the updated cache and the host copies it back (~18 MB,
0.33 ms measured). In-place needs the graph to express output-aliases-input.

### D5 — the mel front end — **DONE, and it had a real bug**

`logmelspectrogram` handed `melfilters`' host `Matrix` straight to a device
`mul!`. That reaches `densify`, which broadcasts a host `Vector` inside a kernel
and fails with "passing non-bitstype argument … `Memory{Float32}` is not isbits"
— an error naming the broadcast machinery, not the array that should not have
been there. It had never run: every verification so far fed the *reference* mel
from PyTorch. Fixed by uploading inside the function.

Validated end to end rather than by comparing filter tables: our own mel, through
our own encoder and decoder, reproduces HuggingFace's transcript of `jfk.wav`
word for word.

### D6 — the real decoding policy — **DONE**

`policy.jl`, `window.jl` and `tokenizer.jl`, ported from whisper.cpp's
`whisper_process_logits` / `whisper_full_with_state` and llama.cpp's
`llm_tokenizer_bpe`. Both repositories are now cloned in full under `dev/`.

Shipped: the eight logit rules (blank/fixed suppression, prompt-token bans,
timestamp pairing, max-initial-timestamp, monotonic timestamps, timestamp-mass
vs text-mass), the temperature ladder with the entropy and log-probability
tests, language detection, the timestamp-driven window advance, context
carry-over, and both of whisper.cpp's tail rules.

**Measured on `jfk.wav`** — `tools/verify_whisper_decoder.jl` and a direct run:

    HF:   " And so, my fellow Americans, ask not what your country can do for
            you, ask what you can do for your country."
    ours: "And so, my fellow Americans, ask not what your country can do for
           you, ask what you can do for your country."   [0.00 -> 10.40]

Word for word. 6.05 s for 11 s of audio, **1.82x realtime**, warm.

Two things established by measurement rather than assumed:

  * **The tokenizer is real BPE, not whisper.cpp's.** whisper.cpp's `tokenize()`
    is greedy longest-match — a different function, which would silently change
    the context prompt. Ours follows llama.cpp and matches HuggingFace on
    **15/15** probe strings chosen to break merge order (leading spaces,
    contractions, digits, CJK, emoji, whitespace runs).
  * **`no_speech` is inert on turbo.** At the position openai/whisper reads it,
    this checkpoint puts 0.000000 on `<|nospeech|>` — on silence and on speech
    alike. The gate that actually works here is the log-probability threshold.
    Consequence, and it is whisper.cpp's behaviour too: `jfk.wav`'s trailing
    0.3 s of applause become a confident `"Yeah."`. HuggingFace avoids it only by
    never decoding a second window under 30 s.

### D8 — the dispatch floor — **the next item, and the biggest**

Not a kernel. The step is 96 dispatches at ~63 us of host time each: a 6.1 ms
floor under an 11.2 ms step, and no arithmetic change can reach it.
`Lava.capture` / `replay!` already removed exactly this cost from SAM 2's decode
(bit-exact, 90.4 steps/s), and the decoder step is a better fit than SAM 2's was
— the shapes are static, the graph is the same every token, and the only things
that change between steps are two one-element inputs and the cache buffers.

**Done when:** a replayed step is bit-identical to a recorded one and the token
sequence is still exact, with the per-step time measured against the 2.07 ms
weight-traffic floor.

Two things to watch, both already paid for once: `replay!` desynced the timeline
counter and died on the *next* submit, and a cached `===` check cannot invalidate
anything if the producing call returns the same tuple every time.

### D7 — fp16, once fp32 is right

The decoder is bandwidth-bound on weight reads, so fp16 halves the floor from
2.07 ms to ~1.03 ms per token — a bigger win here than anywhere in the encoder.
Note the encoder's fp16 is currently **5x worse than PyTorch's own fp16** and the
suspect is the cooperative-matrix path (task #58); the GEMV path does not use
cooperative matrices at all, so that defect may simply not apply. Worth checking
rather than assuming — in either direction.

## What this is not

No Rader/Bluestein, no multi-pass FFT, no new attention variant. The kernel
surface this port adds is **one GEMV and one elementwise compare**. Everything
else is graph plumbing, host code, and validation — which is the right shape for
a port whose model already runs 161 of its 162 ops.
