# Kernels to port

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


Every kernel in `dev/JuliaVision/DNNKernels`, matched against the GPUMODE
kernelbot corpus (`kernelbot.md`), with a verdict on whether a faster
implementation exists there that we could port.

The rule this follows is `feedback-port-sota-kernels`: read the fastest open
implementation for our shape before writing our own, port the **structure**, and
re-measure every constant on this device. Competition kernels are Blackwell /
MI300 / MI355X CUDA-HIP-Triton; we emit SPIR-V coopmat on an Ada-class card with
48 SMs. Instructions never port. Loop structure, split strategy, fusion
boundaries and memory layout do.

Corpus caveat that governs everything below: a top-10 slot is often a vendor
library wrapper, not a kernel. Before reading anything, filter — the
`aiter` / `flash_attn` / `torch.matmul` check in the queries at the bottom is why
several boards that look promising are marked *no source*.

## Which reference to read first

Four tiers, ranked by distance from what we emit. Read down, not up — a kernel in
our own API beats a faster kernel we cannot express.

1. **`dev/llama.cpp`** (checked out 2026-07-27) — GLSL compute shaders on Vulkan
   with coopmat, i.e. our API on our hardware class. Constants may not transfer;
   everything else does. This is the tier `feedback-port-sota-kernels` means.
2. **`dev/aiter`** (cloned 2026-08-02, `--depth 1`, 175 MB, MIT with five
   Apache-2.0 files) — AMD's ROCm kernel library. Its **Triton** tier
   (`aiter/ops/triton/`) is readable structure: 191 files covering attention,
   GEMM, normalisation, fusions, softmax, topk, conv, rope. Different hardware,
   but a fusion boundary that pays on MI355X usually pays here.
3. **The kernelbot corpus** — per-shape tuning evidence and, occasionally, a
   compact statement of an algorithm. Its real value is showing which choices
   people converged on independently across hundreds of attempts.
4. **AMD assembly** — does not port. Named here so nobody goes looking.

**aiter is not the source for MLA decode, despite being what the leaders call.**
Two things came out of the clone that matter:

- The path those competition kernels wrap (`aiter.mla.mla_decode_fwd`) ships as
  **precompiled code objects** — `hsa/gfx942/mla/*.co`, `hsa/gfx950/mla_v4/*.co`,
  launched through a jinja-generated C++ shim in `csrc/cpp_itfs/mla/`. There is
  no source in the repo to take structure from. It is the fastest thing on that
  board and the least readable artefact in the whole survey.
- aiter's *readable* MLA decode, `aiter/ops/triton/attention/mla_decode.py`, says
  in its own header that it is adapted from vLLM, "which was originally adapted
  from sgl-project/sglang and ModelTC/lightllm". Same `_fwd_kernel_stage1` /
  `_fwd_kernel_stage2` / `NUM_KV_SPLITS` / `Mid_O` skeleton as the competition
  kernel in item 1. It is a downstream copy of the structure we already read,
  not a better version of it.

So aiter earns its place for tier 2 breadth — fusion catalogue, GEMM block
ordering, norm variants — and specifically **not** for the item that motivated
cloning it.

Baselines this is ranked against (last measured, `sam2-encode-where-the-time-is`):
SAM 2 encode **100.4 ms** vs PyTorch 87.64 (87.0%); decode **3.30 ms** replayed
vs 2.10. Encode op families: addmm 44.38 (43.3%), attention 34.33 (33.5%),
convolution 6.91 (6.7%), layernorm 6.09 (5.9%), clone 3.28, add 3.15,
`_to_copy` 2.34, max_pool2d 2.05. Decode: attention 1.456 ms of it, and **95% of
that is the single `Lq=23, Lk=4096` shape** item 1 targets.

Two denominators worth keeping in view, because several items below are ranked
against them rather than against a family total: the device's *measured*
cooperative-matrix ceiling is **107.3 TF/s**, and `addmm` runs at 38.0 — **35%**
of it. Attention, by contrast, is at its shape's ceiling and is not a kernel
problem (item 11).

## Status board

`state`: **open** (worth doing, nothing built) · **verify** (cheap check first,
may be a no-op) · **deferred** (real but nothing exercises it yet) ·
**no source** (checked, the corpus has nothing better) · **done** (already ours).

| # | our kernel | file | board | best source | state | est. value |
|---|---|---|---|---|---|---|
| 1 | `attn_flash_cm!` — decode shape | `extern/flash.jl:389` | — | **llama.cpp `flash_attn_split_k_reduce.comp`** | **open** | ~1.2 ms of 3.30 decode |
| 2 | `coopmat_gemm_staged` block order | `Lava/src/array/gemm.jl:625` | amd-mxfp4-mm, amd-fp8-mm | **aiter `gemm_a16w16.py`** + `732442` | **open** | unknown, cheap |
| 3 | `layernorm_kernel!` → GEMM prologue | `kernels/layernorm.jl:55` | trimul | `407551` Zeyu Shen, −28.1% | **open** | ~1–2 ms of encode |
| 4 | `ndmap!` / `ndmap_flat!` elementwise | `launch.jl:77` | vectoradd_v2, grayscale_v2 | `779892` Kernel-Zhang | **verify** | ≤ 8.8 ms family total |
| 5 | flash softmax `exp` → `exp2` | `extern/flash.jl` | amd-mixed-mla | `754421` stage-1 | **verify** | small, one multiply/score |
| 6 | `conv2d_igemm!`, `im2col_kernel!` | `extern/conv_*.jl` | conv2d_v2 | `782583` ooousay, FFT | **deferred** | 0 for SAM 2 |
| 7 | `cumsum_body` | `kernels/resample.jl:109` | prefixsum_v2 | `500145` ağaç.mp4, DLB | **deferred** | 0 today |
| 8 | `softmax_kernel!` | `ops.jl:322` | amd-mla-decode | `31179` div_softmax | **verify** | small |
| 9 | reductions (`sum`/`mean`/`norm`) | via AcceleratedKernels | vectorsum_v2 | `779823` Kernel-Zhang | **verify** | not in top-8 families |
| 10 | `topk_softmax_kernel!` | `memory.jl:148` | amd-moe-mxfp4 | `745202` Danishlynx | **no source** | already fixed |
| 11 | `attn_flash_cm!` — encode shapes | `extern/flash.jl:389` | — | — | **done** | shape-limited, proven |
| 12 | `grid_sample2d`, `deform_conv2d` | `kernels/resample.jl` | — | — | **no source** | — |
| 13 | `avg_pool2d`, `maxpool`, `upsample_*`, `flip` | `kernels/resample.jl` | — | — | **no source** | — |
| 14 | `conv3d_kernel!` (Wan VAE) | `kernels/resample.jl:361` | causal_conv1d | 38 passing, 25 users | **deferred** | Wan not running yet |
| 15 | `transposeLE`, `padcols`, epilogues | `extern/{attention,matmul}.jl` | — | — | **no source** | — |
| 16 | `embedding_kernel!`, `index.Tensor` | `ops.jl:968` | — | — | **no source** | — |

Items **17–22** are a second class — Vulkan device features rather than ports, so
no corpus board applies. They are in their own table below, alongside a list of
what has closed since, so nothing here sends you down a path already measured out.

---

## 1. Flash-decoding: split the key axis — the one large item

**Ours.** `attn_flash_cm!` walks the whole key axis in one workgroup per query
block. SAM 2's decode is 95% one shape — `Lq=23, Lk=4096, E=16, H=8` — and
`Lq=23` is a single query block, so the launch is `1 × H·B` = **8 workgroups on
48 SMs**, running 0.242 GFLOP in 1.456 ms = 0.17 TF/s. `FLASHCM_MINGRID = 48`
already prefers a smaller tiling to raise the grid; that bought −12.8% and does
not change the shape of the problem.

**Read this first — it is in our own API.**
`dev/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_split_k_reduce.comp`
is the split-K merge, in GLSL, next to the `flash_attn_cm1/cm2` coopmat kernels
that produce its input. It merges differently from the Triton versions: two
passes over the splits (workgroup-reduce the max `m`, then workgroup-reduce
`L = Σ exp(m_k − m_max)·l_k`) instead of a sequential online update, and it
splits the `D` axis across workgroups in `y`. Fewer dependencies, more
parallelism in the merge itself.

The split count is decided host-side at `ggml-vulkan.cpp:10609-10639`, and the
rule is one line:

    split_k = shader_core_count * 2 / total_wgs_no_split       // aim at 2 wg/SM
    split_kv = ROUNDUP_POW2(max(1, KV / split_k), alignment)   // then re-derive
    split_k  = CEIL_DIV(KV, split_kv)                          // from the chunk

That is what our `FLASHCM_MINGRID = 48` is a one-sided approximation of, and it
means the per-shape split table I flagged as needed below does not have to be
written — it can be derived, the way llama.cpp derives it. Note the `gqa_ratio`
branch: with grouped heads it splits whenever `workgroups_x <= Br`, which is our
`Lq = 23` case exactly.

**Cross-check.** `amd-mixed-mla` (2,295 passing, 290 users) is the same problem
at `q_seq_len=1`, and `754421` (divc13, rank 11, 31.2 µs, 9 KB Triton, no
`aiter`) is the fastest fully hand-written entry — the top four are `aiter`
wrappers around precompiled assembly and teach nothing. Read it for the details
llama.cpp leaves implicit:

- **Two kernels.** Stage 1 splits the KV axis `NUM_KV_SPLITS` ways; each split
  runs a normal online-softmax flash loop over its slice and writes
  `(acc[BLOCK_H, 512], lse)` to a mid buffer. Stage 2 merges the splits with one
  log-sum-exp pass. Grid goes from `batch` to `batch × splits`.
- **A `DIRECT_OUT` compile-time flag.** When `splits == 1`, stage 1 writes the
  final output and stage 2 never launches — no mid buffer, no merge, no cost on
  shapes that do not need splitting. This is what makes the transformation free
  for our encode, and it is the detail to copy first.
- **A per-shape split table** (`_SHAPE_CONFIGS`) with a fallback heuristic keyed
  on `batch_size` and `kv_len`. Prefer llama.cpp's derivation over the table —
  the table is a competition artefact for eight known shapes.
- Stage 1 also carries `NUM_TILES` as a `constexpr` when the split length is an
  exact multiple of `BLOCK_N`, so the inner loop is statically bounded. We
  already do this kind of `Val`-parameterisation.

Ignore `_remap_xcd` (an 8-die MI355X concern) and the fp8 quantisation path.

**Where aiter fits: nowhere, for this item.** `aiter/ops/triton/attention/
mla_decode.py` is the same `_fwd_kernel_stage1/stage2` + `NUM_KV_SPLITS` +
`Mid_O` skeleton, and its header credits vLLM ← SGLang ← lightllm. The path the
leaderboard actually calls is `hsa/gfx942/mla/*.co` — precompiled assembly, no
source. Nothing to read there that the two references above do not give us in a
form we can use.

**Value.** 8 → 128+ workgroups on the decode's dominant shape. Prior estimate was
~1.2 ms of a 3.3 ms decode. Largest single item on the list, and the only large
one that needs **no** device feature — it is plain KHR and runs on AMD unchanged.

Two things have changed under it since this section was written, both in its
favour. The decode has now been attributed in context rather than from the
serialised table, and it confirms the premise: 36% of the decode is this one
shape. And `Lava.WORKGROUP_LIMIT` is no longer 256 — that cap was a pipeline-cache
hash collision, not the device — so the split kernel may use up to 1024 threads
per workgroup if that turns out to be the better shape for the merge.

## 2. GEMM: grouped block ordering

**Ours.** `Lava/src/array/gemm.jl:625` maps block index to tile as
`tm = (blk % nblk_m) * BM; tn = (blk ÷ nblk_m) * BN` — plain column-major. Blocks
launched together walk down one column of A while sweeping all of B, so the B
tiles they share are evicted from L2 before the next column reuses them.

**Reference.** `dev/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`
— this is what the clone was worth. Its every-GEMM entry path is

    pid = remap_xcd(pid, num_pid_m * num_pid_n * NUM_KSPLIT, NUM_XCDS=8)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

with `GROUP_SIZE_M` in the autotune key, i.e. AMD ships grouped ordering as the
default block map for a plain fp16 GEMM and tunes the grouping factor per shape.
`pid_grid` renumbers blocks so a group of `GROUP_M` consecutive blocks covers a
`GROUP_M × num_pid_n` rectangle — a column sweep becomes a square-ish tile, and
both operands' tiles stay resident in L2 across the group. Take `pid_grid`; drop
`remap_xcd`, which spreads blocks across eight dies we do not have.

**Corroboration.** 103 of the `amd-mxfp4-mm` best kernels and 31 of `amd-fp8-mm`
use the same `GROUP_SIZE_M` / `num_pid_in_group` idiom (best entry `732442`,
josusanmartin, 7.59 µs), plus 71 `nvfp4_gemm` and 96 `nvfp4_group_gemm` kernels
with explicit swizzle variants. Library default *and* independent competition
convergence, which is about as much prior evidence as this list ever gets.

**Why this and not double-buffering.** Double-buffering was already checked
against llama.cpp's `mul_mm.comp` and it does not double-buffer either — nothing
un-ported there. Grouped ordering is a handful of lines in the block→tile map,
does not touch the k-loop, cannot change results, and is untried here.

**Value.** Unknown — addmm runs at 38.03 TF/s against the device's measured
107.3 coopmat ceiling, and tiling, ILP and staging structure were each swept and
spent. This is the one lever on that list nobody pulled. Cheap enough to just
measure. A/B interleaved in one session (`lavadnn-benchmark-variance`).

## 3. LayerNorm into the GEMM prologue

**Ours.** `layernorm_kernel!` is fused internally (one read, one write, two
in-cache reduction passes) and lands at 6.09 ms for 96 ops against PyTorch's
5.64 — near parity, so the *kernel* is not the opportunity. The opportunity is
the boundary: it writes 4.7 MB and the `addmm` that follows immediately reads it
back.

**Corpus.** Zeyu Shen's trimul progression is a labelled optimisation log,
10.173 ms → 1.140 ms in 15 steps, and the two biggest steps are exactly this
boundary:

    fused_preprocess_kernel.py   −41.9%   407538
    ..._fp16.py                  −50.5%   407546
    fused_prologue_epilogue.py   −28.1%   407551

Read `407551`. Its `_fused_prologue_kernel` computes the LayerNorm statistics
over a tile, then in a second loop over the same C-axis re-loads the tile,
normalises it in registers, casts to fp16, and feeds it straight into **five**
`tl.dot` calls against five different weight matrices — the normalised tile is
staged once and consumed by every projection. The write of the normalised
activation to global memory never happens.

**How it maps.** Our staged GEMM already stages an A block into shared memory
per k-step. Normalising it there needs the group's `mean`/`rstd` available, which
means either a separate cheap statistics pass (one scalar pair per group, we
already write those) or computing them in the prologue as trimul does. The
`epi`/`bias` machinery from `lava-gemm-epilogue` is the mirror image of this and
shows the plumbing works.

**Value.** 96 layer norms × ~4.7 MB, write + read removed, at ~460 GB/s ≈ 1–2 ms
of the encode. Also removes 96 dispatches. Note SAM 2 does not have trimul's
five-projections-from-one-tile pattern, so the multi-consumer part of the win
does not transfer — only the write/read elimination does.

**aiter checked, does not have this.** `aiter/ops/triton/normalization/` fuses
things onto the *norm* — `layernorm2d_fwd_with_add`, `..._with_dynamicquant`,
`..._with_smoothquant`, `fused_add_rmsnorm_pad` — and `gemm/fused/` fuses onto
the *GEMM's epilogue* (`..._mul_add`, `..._split_cat`) or quantises `x` in a
separate pass first (`fused_gemm_a16w16_quant_x`). Nobody there normalises
inside the GEMM's staging loop. trimul `407551` stays the source.

## 4. Vectorized elementwise access

**Ours.** `ndmap!` / `ndmap_flat!` process one element per lane.
`LAUNCH_FLAT = true` (flat launch + `Lava.cart32` over `FastDiv32`) already fixed
the index-arithmetic half of this. The load width was never addressed. Combined
family cost: clone 3.28 + add 3.15 + `_to_copy` 2.34 = 8.77 ms, ~8.7% of encode.

**Corpus.** Every top vectoradd/grayscale entry on every GPU has `uses_vec_load`.
`779892` (Kernel-Zhang, A100, 891.9 µs) is the clean statement of it: reinterpret
as `uint4` (128 bit = 8 halves per lane), fixed trip count so the loop fully
unrolls, `__hadd2` on the four `half2` pairs, one vectorized store. The
grid-stride form with a compile-time iteration count is the part that ports;
`half2` SIMD arithmetic has no coopmat analogue but `Vec{8,Float16}` elementwise
does.

**Why *verify* and not *open*.** `lava-elementwise-negative-results` records two
plausible elementwise optimisations that both lost, and the dense-leaf broadcast
already measures 461 GB/s — possibly at the bandwidth ceiling, in which case
wider loads buy nothing. Measure a `Vec{8,Float16}` copy against the current
`ndmap_flat!` on the 4.5 MB clone shape *before* touching the launcher. If 461
GB/s is the wall, close this item.

## 5. `exp` → `exp2` with the scale folded

`extern/flash.jl` uses `exp` (19 call sites, no `exp2`, no `LOG2E`). Every fast
attention kernel in the corpus, `754421` included, folds
`sm_scale * log2(e)` into the QK scale constant and calls `exp2` on the scores,
because the hardware instruction is base-2 and `exp` costs one extra multiply per
element — per *score* element, of which our encode has `4096 × 4096 × 8`.

Whether Lava's `exp` already lowers to `OpExtInst Exp` → `v_exp_f32` with the
multiply folded by the driver is a five-minute check on the emitted SPIR-V. Do
that first; if the multiply is there, folding it into the scale is a two-line
change with bit-identical-or-better numerics.

## 6. Convolution — the corpus says FFT, which does not fit SAM 2

`convolution` is our worst-utilisation family, 13.4 GFLOP in 6.91 ms = 1.94
TF/s, and `conv2d_v2` looks like the obvious board. It is not.

The board's fast entries are **not** implicit-GEMM kernels. Rank 1 on A100
(`782583`, ooousay, 3.821 ms) is a tiled overlap-save **FFT convolution** —
`rfft2` on overlapping input tiles, a per-frequency-bin complex batched GEMM,
`irfft2`, crop — beating cuDNN 4.19x on the benchmark's `K = 8…32`, stride-1,
no-padding shapes. Rank 1 on B200 and H100 are cuDNN wrappers. The hand-written
Triton implicit-GEMM entries (`513373`, `732334`) sit at 39.9 ms and 44.2 ms,
**10x slower** than the FFT entry — those are the only ones structurally like
ours and they are not competitive on this board's shapes.

FFT conv wins only for large kernels. SAM 2 is 3×3 and 1×1 convs plus a 7×7
stride-4 patch embed, and stride 4 rules out overlap-save entirely. **Nothing to
port for SAM 2.** Revisit only if a model arrives with large stride-1 kernels;
BasicVSR++ and Wan should be checked against this when they run.

The stem is separately already handled — `CONV_CRS_PAD` padding `CRS` onto the
16-tile took it 2.800 → 1.147 ms (2.42 TF/s).

## 7. `cumsum` — deliberately quadratic, fix exists when needed

`cumsum_body` walks the axis from element 1 per output element: quadratic, one
thread per output. The docstring is explicit that this is a choice — the only
`cumsum` in any graph builds SAM 2's 64×64 positional encoding, 262k adds, once
per decoder call.

When a long axis needs scanning, the corpus has both tiers ready:
`500145` (ağaç.mp4, A100, 1.322 ms) is a single-pass **decoupled lookback** scan
with warp-parallel lookback over `volatile` tile descriptors, and `612455`
(dannywillowliu-uchi, B200, 480.1 µs) is the simpler three-pass
reduce → scan-block-sums → add-prefix, which is likely the right first port
because it needs no memory-ordering guarantees we would have to establish for
SPIR-V. Take the three-pass version first; DLB needs `volatile`/acquire-release
semantics that Lava has not been asked for yet.

## 8. `softmax_kernel!`

One workgroup per slice, `SOFTMAX_WG = 64`, fused max/sum/normalise. Correct
shape for our use (attention softmaxes 16–32 elements at a time). The corpus's
closest artefact is `div_softmax_kernel` in `31179` (amd-mla-decode rank 1),
which is a long-row softmax with `float4` staging and a three-level shared-memory
tree reduction — built for `sl` up to 8192, i.e. the opposite regime. The only
transferable detail is the vectorized load in the first pass, and that is the
same check as item 4. Low value.

## 9. Reductions

`sum` / `mean` / `linalg_vector_norm` go through AcceleratedKernels and do not
appear in the encode's top-8 families. `vectorsum_v2` winners
(`779823` Kernel-Zhang A100 135.3 µs, `755317` B200 40.8 µs) are the standard
`vec_load` + shuffle + shared-memory tree. If a reduction ever shows up hot,
that is the shape to copy — but it is not hot, and `softmax_kernel!` exists
precisely because AcceleratedKernels was wrong for the small-extent case.

## 10. `topk_softmax_kernel!`

MoE routing (`amd-moe-mxfp4`, `amd-mixture-of-experts`) is the only place in the
corpus that does a top-k, and it is a different problem: thousands of tokens,
k of 8 from 256 experts, and the interesting part is the scatter/gather of tokens
to experts, not the selection. Ours is K=30 of N≈360 over 120 queries, already
moved from a 3.2 ms register-spilling version to shared memory with one workgroup
per query, and the docstring records why the selection stays serial on lane 1
(K rounds of masked argmax or a full sort both cost more barriers than the serial
scan costs shared-memory ops). **No source. Closed.**

## 11–16. Checked, no source

- **Encode attention** (`attn_flash_cm!` at `Lq=Lk=4096, E=72`): its two inner
  products run *faster in-kernel* than the same shapes do as standalone
  `coopmat_gemm!` (12.92/8.75 TF/s standalone vs 13.5/11.9 in-kernel). The shape
  is the ceiling, not the kernel. Six hypotheses already died here. Do not
  reopen without a new standalone-GEMM measurement.
- **`grid_sample2d_kernel!`, `deform_conv2d_kernel!`** — zero matches for
  `grid_sample` or `deform` anywhere in 165k passing kernels. BasicVSR++ ops have
  no competition analogue.
- **`avg_pool2d_kernel!`, `maxpool`, `upsample_bilinear/nearest`, `flip_kernel!`** —
  zero matches for `max_pool`, `upsample`, `interpolate`. (max_pool2d is 2.05 ms
  / 2.0% of encode and has no external reference; it is ours to improve or leave.)
- **`im2col_kernel!`** — 2 matches, both in the slow tail of `conv2d_v2`.
- **`transposeLE`, `padcols_kernel!`, `mm_epilogue_kernel!`,
  `conv_epilogue_kernel!`** — layout adapters specific to our reversed-layout
  interchange. `transposeLE` is already 7.1x from the 32×33 shared tile.
- **`embedding_kernel!`, `index.Tensor`** — gather appears only inside
  linear-algebra boards where it is incidental.
- **`conv3d_kernel!`** — the Wan VAE's causal 3D conv. `causal_conv1d`
  (B200_Nebius, 38 passing, 25 users, best 10.049 µs) is depthwise causal conv in
  1D, so the causality/left-padding structure transfers but the 3D tiling does
  not. Worth a read *when Wan runs* — not before, per the model roadmap.

## 17–22. Device features, not ports

A second class of open work, added 2026-08-02. These are not corpus items — no
competition kernel can show us how to use a Vulkan extension — but they are open
kernel work against the same baselines, so they belong on the same page. Every
extension below is **enabled and queryable as of Lava `955e0b8`**; what is open
is using it.

`state` as above. Task numbers are this repo's list.

| # | feature | what it would change | state | est. value | task |
|---|---|---|---|---|---|
| 17 | coopmat2 **reductions** | flash softmax's row max/sum, currently a shared round-trip + 2 barriers | **open** | ≤ 8.3 ms of encode, fraction unknown | K3 |
| 18 | coopmat2 **flexible dimensions** | `E = 72` stops padding to `EP = 80`; reopens the tiling space | **open** | 10% of both attention products | K4 |
| 19 | coopmat2 **tensor addressing / block loads** | replaces flash's hand-coded root+stride staging and the GEMM's 2-barrier k-step | **open** | unknown; a route past the GEMM's 35% | K5 |
| 20 | **fp8 coopmat at K32** | twice the reduction depth per instruction on the 44.4 ms addmm bucket | **open** | unproven — see below, measure with int8 first | K6 |
| 21 | **maximal_reconvergence** + uniform control flow | makes a spin-wait well-defined, i.e. producer/consumer GEMM staging | **open** | the only identified route past 35% | K7 |
| 22 | **AMD fallback matrix** | every coopmat2-gated kernel needs a tested KHR path | **open** | correctness, not speed | — |

**What the device actually reports**, so nobody re-queries it: all seven coopmat2
sub-features (workgroup scope, flexible dimensions, reductions, conversions,
per-element operations, tensor addressing, block loads); `VK_NV_cooperative_vector`
with training; `maximal_reconvergence`; `subgroup_uniform_control_flow`;
`subgroup_rotate` with clustered. All 11 subgroup operation bits, `subgroupSize`
32. Fifteen cooperative-matrix shapes, **every one `M16`** — fp16 and bf16 at
K16, int8/uint8 at K32, and **fp8 `E4M3`/`E5M2` at K32** (into f16 or f32).
Device ceiling on cooperative matrices is a *measured* 107.3 TF/s (not the 153 on
the spec sheet, which assumes fp16 accumulate); `addmm` runs at 38.0, i.e. **35%**.

**fp8 is `VK_EXT_shader_float8`, not a vendor extension** — an earlier note here
said NV and was wrong; the component types are `COMPONENT_TYPE_FLOAT8_E4M3_EXT` /
`E5M2_EXT`. The extension is present with `shaderFloat8` and
`shaderFloat8CooperativeMatrix` both true, and Lava does **not** request it
(nor `VK_KHR_shader_bfloat16`, also present). The SPIR-V is expressible with our
current toolchain — an fp8 `OpCooperativeMatrixMulAddKHR` assembles and validates
against `vulkan1.3` — so the numbers are recorded here to save rediscovering
them: extension `SPV_EXT_float8`, capabilities `Float8EXT` = 4212 and
`Float8CooperativeMatrixEXT` = 4213, type `OpTypeFloat 8 <enc>` with
`Float8E4M3EXT` = 4214 and `Float8E5M2EXT` = 4215.

And the blocker that turns out not to be one: LLVM has no fp8 type, but a GEMM
never needs one. Load fp8 → `muladd` → accumulate fp32 → store fp32, so no fp8
arithmetic reaches LLVM IR; the coopmat handle is an `i32` and the load carries
only an address. The Julia side needs a 1-byte primitive type with host-side
conversion, not an arithmetic type. bf16 buys **no speed** — it is K16 like fp16
— so it is a range option only.

Notes that decide the order:

- **17 first, and measure before building.** The softmax stage is 1.622 ms of the
  4.930 ms global block and 0.108 of 0.443 windowed. That is the ceiling on the
  item, not the estimate.
- **20's premise is unmeasured, and can be tested for free.** The 107.3 TF/s
  ceiling was measured for **fp16→fp32**; whether `K32` buys anything depends on
  the device issuing those instructions at twice the rate, which nobody has
  checked. `int8`/`uint8` are *also* at K32 and are **already** supported
  component types in `coopmat_component_type!` — so run the muladd-only ceiling
  kernel at `i8→i32, K32` first. It needs no new code, and if it does not reach
  ~2x the fp16 number then item 20 is worth far less than it looks and should
  not start. Cost side the estimate must also carry: **activations convert per
  call** (weights are static and convert once, offline), and accuracy is gated on
  the current IoU 1.000 / 0.9996 / 1.000.
- **21 is structural.** Tiling, ILP and port utilisation are all measured out on
  the GEMM (a muladd-only kernel hits 107.3 TF/s at *one* independent chain per
  lane, so "more accumulators per warp" cannot help), and `mul_mm.comp` does not
  double-buffer either — so there is nothing left to port and this is the
  remaining idea.
- **22 is not optional.** coopmat2 and cooperative-vector are NVIDIA-only. RDNA3
  exposes WMMA through the portable `VK_KHR_cooperative_matrix` at 16×16×16
  fp16/bf16, and its subgroup is 32 **or** 64 depending on how the driver
  compiled the shader, so nothing may hard-code 32.

## Closed since this file was written

Recorded so nobody re-opens them from the board above.

- **Held `O` in accumulators** (was the obvious companion to item 1). Worth
  **−21%** on the encode's global block for as long as the register allocator
  stays under 128 (= 65536/(256·2), i.e. two resident workgroups), and +9% the
  moment it crosses. Six routes to keep three rescales under 128 were measured
  and all failed: a better rescale primitive (getcomp/setcomp, coopmat2
  per-element, and component-wise `OpFMul` all land within 1.5% — the primitive
  is irrelevant), hoisting one shared factor matrix (correct, 220 registers),
  removing the second rescale site (216), a barrier against software pipelining
  (no effect), register ballast (impossible — the driver caps *itself* at 128),
  and all 14 admissible tilings plus `NW = 16`. Vulkan has no `__launch_bounds__`,
  so the allocator's decision is not steerable. See `FLASHCM_HELD`.
- **`coopmat_mul` + a stride-0 broadcast load** — shipped, and portable. `OpFMul`
  on two cooperative matrices is component-wise in plain KHR; with a stride-0
  load every column reads the same vector, so a per-row factor costs one
  instruction, no shared memory and no component access. The GEMM's `gelu`
  epilogue is the obvious next user (255 registers via getcomp/setcomp today).
- **1×1 convolution → `matmul!`** — shipped, conv 6.06 → 4.90 ms, −59 MiB.
- **Decode attributed in context** — 36% of it is the one shape item 1 targets.
- **The 256-thread workgroup cap** was a pipeline-cache hash collision, not the
  device (`Base.hash` samples a large `Vector`; the 256- and 512-wide modules
  differ at one byte). Fixed; `WORKGROUP_LIMIT` is now 1024. Relevant here
  because a bigger workgroup is now a real tuning knob for any port below.

## Reproducing these queries

```bash
cd /sim/Programmieren/VideoEdit
.venv/bin/python tools/kernelbot.py show 754421          # the flash-decode source
.venv/bin/python tools/kernelbot.py progression trimul --user "Zeyu Shen"
.venv/bin/python tools/kernelbot.py dump amd-mixed-mla -n 5 --out reference/kernelbot-dumps
```

The wrapper filter that decides whether a board is worth reading at all:

```sql
select rank, round(score*1e6,1) us, user_name, submission_id, code_bytes,
       code ilike '%aiter%'      as aiter,
       code ilike '%flash_attn%' as fa,
       code like  '%triton.jit%' as tri,
       code like  '%__global__%' as raw
from best_kernels where problem_name = '<board>' order by score limit 20
```

`code_features` (no code column, instant) answers the same question by keyword
flags — `is_triton`, `uses_split_k`, `uses_tl_dot`, `uses_vec_load`. Both are
keyword heuristics over source text, good for "show me the split-K ones", not for
counting.

For the reference clones:

```bash
ls dev/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn*   # tier 1
ls dev/aiter/aiter/ops/triton/                                     # tier 2, readable
ls dev/aiter/hsa/gfx950/mla/                                       # tier 4, .co binaries
```

`dev/aiter` is `--depth 1` with no submodules — `3rdparty/` is empty on purpose,
composable_kernel is gigabytes and is C++ template metaprogramming we would not
port anyway. Refresh with `git -C dev/aiter fetch --depth 1 && git -C dev/aiter
reset --hard origin/main`. Licence: MIT, except five vLLM-derived files under
Apache-2.0 (`grep -rh SPDX-License-Identifier --include=*.py` gives 191 / 5).
Both permissive; attribute if we copy a file rather than an idea.

Attribution: `GPUMODE/kernelbot-data`, June 9 Researcher Reciprocity License.
Reading these to inform our own kernels is plain use and needs only attribution;
the reciprocity obligation attaches to Training Use, which this is not. See
`kernelbot.md`.
