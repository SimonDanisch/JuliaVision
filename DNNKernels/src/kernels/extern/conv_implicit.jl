"""
Implicit-GEMM convolution.

Ported from ggml's Vulkan `conv2d_mm.comp` (dev/ggml-vulkan/ggml/src/ggml-vulkan/
vulkan-shaders/conv2d_mm.comp), which is the fastest portable-Vulkan 2-D
convolution we have a reference for. The direct kernel in `conv.jl` reads every
operand from global memory once per multiply-add and shares nothing between
threads; at 3x3 256->256 that is 2304 uncached loads per output element, and it
measures 18 GFLOP/s.

The convolution is one GEMM:

    C[K, NPQ] = A[K, CRS] * B[CRS, NPQ]

    K   = Cout                 output channels   (rows)
    CRS = Cin * KH * KW        reduction extent
    NPQ = N * OH * OW          output pixels     (columns)

`B` is never materialised — that is the "implicit" part. Each element is fetched
straight from `x` with the im2col index computed on the fly, so we pay none of
the KH*KW memory blow-up that an explicit im2col costs.

Reuse comes from staging both operands in shared memory and giving each thread a
TS_K x TS_NPQ register tile: one pass of the inner loop does TS_K + TS_NPQ shared
loads and TS_K * TS_NPQ multiply-adds.

Layout is the reversed one used throughout (`x` is `(W,H,Cin,N)`, `w` is
`(KW,KH,Cin,Cout)`, `out` is `(OW,OH,Cout,N)`), which happens to match ggml's
own memory order exactly, so the index arithmetic carries over unchanged apart
from 1-based offsets.

Tile sizes are `Val` parameters rather than constants because the right choice is
shape-dependent: ggml's 128x128 default would put this model's dominant
convolution (NPQ=120, K=256) on two workgroups. `convtiles` picks them.
"""

"""
    convtiles(K, NPQ) -> (BS_K, BS_NPQ, TS_K, TS_NPQ, WG)

Block and thread tile for a `K x NPQ` output. Chosen to keep enough workgroups in
flight to fill the device: a 128x128 block is only worth it when the output is
big enough to still produce many blocks, and this model's feature maps are small
(NPQ as low as 120), so we step down rather than launch two workgroups.

`WG == (BS_K / TS_K) * (BS_NPQ / TS_NPQ)` is required — one thread per output
element of the block tile.
"""
# The four block shapes ggml ships, as (BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ);
# see vk_conv_block_sizes and the WG_SIZE/TS_K rules at ggml-vulkan.cpp:511,5668.
# TS_NPQ is not free — it is BS_K*BS_NPQ/WG/TS_K, so every row here satisfies
# WG == (BS_K/TS_K) * (BS_NPQ/TS_NPQ). Note 64x32 is the odd one out: 128 threads
# and a deeper CRS block, not 256.
const CONV_SHAPE_128x128 = (128, 128, 16, 256, 8, 8)
const CONV_SHAPE_64x32   = ( 64,  32, 32, 128, 4, 4)
const CONV_SHAPE_32x256  = ( 32, 256, 16, 256, 8, 4)
const CONV_SHAPE_64x128  = ( 64, 128, 16, 256, 8, 4)

"""
    convtiles(K, NPQ; cores) -> (BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ)

Block shape for a `K x NPQ` output, ported from `ggml_vk_conv_select_shape`
(ggml-vulkan.cpp:10768). Take the largest tile that still yields at least two
tiles per shader core, gated on `K` so a narrow output does not get a block wider
than it is; otherwise fall back to 64x32.

`cores` is the shader-core count — 48 on the RTX 4000 Ada this was tuned against,
and `ctx.dev.cores` at the call site. ggml uses 32 as its placeholder when the
count cannot be queried, and so does the default here: `M.DeviceCaps` reports 0 when
the device will not say, and 0 would make every `>= 2cores` test trivially true
and always pick the widest block.
"""
function convtiles(K::Int, NPQ::Int; cores::Int = 32)
    cores <= 0 && (cores = 32)
    ntiles(s) = cld(K, s[1]) * cld(NPQ, s[2])
    if K > 64 && ntiles(CONV_SHAPE_128x128) >= 2cores
        CONV_SHAPE_128x128
    elseif K <= 32 && ntiles(CONV_SHAPE_32x256) >= 2cores
        CONV_SHAPE_32x256
    elseif K <= 64 && ntiles(CONV_SHAPE_64x128) >= 2cores
        CONV_SHAPE_64x128
    else
        CONV_SHAPE_64x32
    end
end

# `ACC` is the accumulator and shared-memory element type, and it is passed as a
# static parameter rather than computed in the body as `accum(eltype(x))`.
# `@localmem` with a type that is a *local variable* rather than a static
# parameter compiles without complaint and then silently writes nothing — the
# kernel runs and every output stays untouched. Worth remembering: it fails
# quietly, so it looks like an indexing bug.
# ── The kernel that was here ──────────────────────────────────────────────────
#
# `conv2d_igemm!`, 188 lines of `@kernel`, DELETED. It was the same kernel as
# `kernels/conv_igemm.jl`'s `conv2d_igemm_ki!` — same arithmetic, same tiling,
# same split-K, same argument list down to the order of the `Val`s — written
# once against KernelAbstractions and once against `KernelInterface`. The port
# made the second, and keeping both meant two places to fix a tiling bug and two
# kernels to tune.
#
# `convolution_igemm!` below launches the `KernelInterface` one. What stays here
# is the part that was never duplicated: the shape table, `convtiles` and
# `convsplit`.

"""
    convsplit(nbk, nbn, nblk; cores) -> SPLITK

How many ways to split the reduction. Output tiling alone gives `nbk*nbn`
workgroups; split enough to reach roughly two per shader core, but never finer
than one CRS block per split (below that the splits have nothing to do) and
capped at 8, past which the atomic traffic on the output outweighs the extra
parallelism.
"""
function convsplit(nbk::Int, nbn::Int, nblk::Int; cores::Int = 48)
    # Already enough workgroups to fill the device: splitting then only buys the
    # cost of pre-filling the output and the atomic traffic. Measured on
    # 3x3 64->64 at 60x32 (60 workgroups), a 2-way split was 2.8x *slower*, while
    # 3x3 256->256 at 15x8 (16 workgroups) gained 4.3x from a 6-way split.
    nbk * nbn >= cores && return 1
    want = cld(2cores, max(nbk * nbn, 1))
    clamp(min(want, nblk), 1, 8)
end

"""
    convolution_igemm!(ctx, out, x, w, bias, stride, padding, dilation) -> out

Implicit-GEMM path. Dense (`groups == 1`) convolutions only; the grouped case
still goes through the direct kernel, and this model has none.
"""
function convolution_igemm!(ctx, out, x, w, bias, stride, padding, dilation; act::Symbol=:none)
    KWk, KHk, Cin, Cout = size(w)
    Wid, Hei = size(x, 1), size(x, 2)
    OW, OH, _, N = size(out)
    CRS = Cin * KHk * KWk
    NPQ = N * OH * OW
    # The shader-core count comes off the context, not from the literal 48 this
    # was tuned against: the whole rule is "at least two tiles per core", so on a
    # part with a different count it picks the wrong block outright.
    BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ = convtiles(Cout, NPQ; cores = ctx.dev.cores)
    nbk = cld(Cout, BS_K)
    nbn = cld(NPQ, BS_NPQ)
    splitk = convsplit(nbk, nbn, cld(CRS, BS_CRS))
    backend = ctx.backend
    # Split-K accumulates with `Atomix.@atomic +=` on the destination, and
    # Vulkan 1.3 has no fp16 atomic add (`AtomicFloat16AddEXT` is not in the
    # allowed capability set), so an fp16 output accumulates into an fp32 scratch
    # and is converted once at the end. Two extra dispatches against a 4-8x win
    # from the split, so it stays worth it.
    #
    # A fused activation forces the same detour even for an fp32 output: each
    # split contributes a *partial* sum, so applying `relu` inside the write-back
    # would clamp partial sums independently and silently give the wrong answer.
    # It has to be applied once, after the splits have been summed.
    #
    # **`splitk > 1` makes this convolution non-reproducible**, because the order
    # the atomics land in is not fixed and float addition is not associative.
    # Measured on `segment`'s `convolution_21` (fp32, 3x3, 512 -> 768, NPQ = 64,
    # `splitk = 4`): two runs on identical inputs differ by 3.05e-5 on 9% of
    # elements, which carries to ~5e-7 at the graph's outputs. The same
    # convolution at NPQ = 4096 takes `splitk = 1` and is bit-reproducible.
    #
    # Not a bug — it is what buys the 4-8x — but it is the floor for anything
    # measured on a graph containing one, and it is the first thing to rule out
    # when a change appears to move a small-NPQ result by ~1e-7.
    acc = (splitk > 1 && (eltype(out) !== Float32 || act !== :none)) ?
          KernelAbstractions.allocate(backend, Float32, size(out)...) : out
    if splitk > 1
        # every split accumulates atomically, so the destination has to start at
        # the bias (or zero) rather than being overwritten
        bias === nothing ? fill!(acc, zero(eltype(acc))) :
            (acc .= reshape(bias, ntuple(i -> i == 3 ? length(bias) : 1, ndims(acc))))
    end
    # Only fold the activation into the write-back when there is a single split;
    # otherwise it is applied in the conversion pass below.
    kact = (splitk == 1 && act === :relu) ? :relu : :none
    # `conv2d_igemm_ki!` from `kernels/conv_igemm.jl`, which is this kernel: the
    # declared path in `emit.jl` dispatches the same function.
    harnesslaunch!(backend, conv2d_igemm_ki!,
        acc, x, w, bias, Val(accum(eltype(x))), Val(splitk), Val(kact),
        Val(BS_K), Val(BS_CRS), Val(BS_NPQ), Val(TS_K), Val(TS_NPQ),
        Val(KWk), Val(KHk),
        Val(stride[1]), Val(stride[2]), Val(padding[1]), Val(padding[2]),
        Val(dilation[1]), Val(dilation[2]),
        Cin, Cout, Wid, Hei, OW, OH, NPQ, CRS, nbn;
        ndrange = (nbk * WG, nbn * splitk), workgroupsize = (WG, 1))
    if acc !== out
        act === :relu ? (out .= max.(acc, zero(eltype(acc)))) : (out .= acc)
    elseif splitk > 1 && act === :relu
        out .= max.(out, zero(eltype(out)))
    end
    out
end
