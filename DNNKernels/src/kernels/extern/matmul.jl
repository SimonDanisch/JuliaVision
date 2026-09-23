"""
Matmul: hand-written, type-parameterized, shared by every model.

`aten::mm`, `aten::addmm` and `aten::bmm` in the reversed layout. A torch
`(m, k) x (k, n) -> (m, n)` becomes Julia `(k, m)` and `(n, k)` giving `(n, m)`,
so the product is written with the operands swapped:

    out[i, j] = Σ_k A[i, k] * B[k, j]      A = rhs_j, B = lhs_j

Accumulation is in `accum(T)` - fp32 for half operands, matching what the
reference gets from tensor cores.

This is the scalar instantiation. Capability dispatch to a cooperative-matrix
variant happens at instantiate from a device query, not here.
"""

@inline function mm2(I, A, B, bias)
    i, j = I
    @inbounds begin
        T = accum(eltype(A))
        acc = bias === nothing ? zero(T) : T(bias[bidx(bias, CartesianIndex(i, j))])
        for k in axes(A, 2)
            acc = muladd(T(A[i, k]), T(B[k, j]), acc)
        end
        acc
    end
end

@inline function mm3(I, A, B)
    i, j, b = I
    @inbounds begin
        T = accum(eltype(A))
        acc = zero(T)
        for k in axes(A, 2)
            acc = muladd(T(A[i, k, b]), T(B[k, j, b]), acc)
        end
        acc
    end
end

# A 2-D permutation *is* a transpose, so say so. `Transpose` is what LinearAlgebra
# and cuBLAS dispatch on; a `PermutedDimsArray` carries the same memory but matches
# no BLAS method, so it falls through to the generic matmul, which scalar-indexes a
# GPU array from the host. The exported graph produces these constantly — every
# `addmm` operand arrives permuted.

"""
    densematrix(T, X) -> Bool

Whether an operand of type `X` is a dense rank-2 matrix of `T` — which is what
the cooperative-matrix and GEMV kernels require, and the whole of what their
plans ask about the operand itself.

`A isa Mantle.LavaArray{Float16,2}` was the test. It names ONE backend's array
type from a consumer, which is a portability smell on its own, and it cannot be
asked of a declared operand at all: at emit time an operand is a graph resource
with no storage yet. Both families answer this, so plan selection is one
predicate and an immediate call and a declared one cannot disagree — which
matters because disagreeing is silent. Measured on SAM 2's encoder: 195 fp16
products whose extents are all multiples of 16 declined the tensor-core path and
took `strided_gemm_kernel!`, one invocation per output element.

A `BufferRange` is deliberately absent: it is rank 1 by construction.
"""
densematrix(::Type{T}, ::Type) where {T} = false
# A concrete backend array is dense storage regardless of whether its owner is
# Lava, AMDGPU, CUDA, or another GPUArrays backend.  Views are wrapper types and
# do not match this method; they still have to be materialised (or represented
# by one of the explicit graph-resource forms below) before a kernel that assumes
# column-major dense addressing may accept them.
densematrix(::Type{T}, ::Type{<:GPUArrays.AbstractGPUArray{T,2}}) where {T} = true
densematrix(::Type{T}, ::Type{<:M.Buffer{T,2}}) where {T} = true
densematrix(::Type{T}, ::Type{<:M.TransientBuffer{T,2}}) where {T} = true
densematrix(::Type{T}, ::Type{<:M.ResourceView{T,2}}) where {T} = true

astranspose(a) = a
astranspose(a::PermutedDimsArray{T,2,(2, 1)}) where {T} = transpose(parent(a))

"""
    matmul!(ctx, out, A, B, bias=nothing)

`out[i, j] = Σ_k A[i, k] B[k, j]`, plus `bias` broadcast over the output.

Takes the cooperative-matrix path when `mm_coopmat_plan` returns one, and
`LinearAlgebra.mul!` otherwise.

This function *does* know tensor cores exist: the predicate below checks
`Mantle.coopmat_gemm_available(ctx)` and `Mantle.GEMM_TILE`, and gates on
`Mantle.LavaArray{Float16,2}`, so the fast path is reachable only on Lava. An
earlier docstring here claimed the opposite; it described a design that was
replaced.

What *is* device-independent is which device takes it. The tile and the
availability are queried per device, so the same source picks cooperative
matrices on Ada (subgroup 32) and on RDNA 3.5 (subgroup 64, `16x16x16` Float16)
with no vendor branch anywhere.

`gemm` is forwarded to `Mantle.coopmat_gemm!` as keywords — `staged`, `vec2`,
`narrow_ok`, `tiling`. It exists so a benchmark can pick a kernel **for one
call** instead of mutating a process-wide `Ref`; the defaults are the measured
winners and no shipping path passes it. Ignored by the paths that have no
cooperative-matrix kernel to select, which take it and drop it rather than
erroring, so a caller sweeping shapes does not have to know which path each one
lands on.
"""
function matmul!(ctx, out, A, B, bias=nothing; epi=identity, gemm=NamedTuple())
    # A backend with a native dense GEMM owns this product.  The immediate path
    # reaches it through `mul!`; the declared path asks
    # `native_gemm_dispatch!` before consulting the portable planner.  Without
    # this guard a native backend whose capability record also says
    # `coopmat=true` enters the Vulkan staged-GEMM planner below.  Besides making
    # the two execution modes disagree, that planner's padding policy is not a
    # generic cooperative-matrix primitive (`gemm_padn` belongs to Mantle's
    # Vulkan implementation), so interpreted Metal failed before launching.
    plan = M.native_gemm_available(ctx.backend, eltype(A), eltype(B), eltype(out)) ?
           Decline(:native) : mmplan(ctx.dev, out, A, B, bias)
    matmul!(ctx, plan, out, A, B, bias, epi; gemm)
end

"""
    mmplan(dev, out, A, B, bias) -> MMCoopMatPlan | MMGemvPlan | Decline

Which path this shape takes. Tensor cores first, then the batch-1 GEMV, then the
scalar kernel — most specific to least, and each predicate says why it declined.

The order is not a preference between the first two: they are disjoint.
`mm_coopmat_plan` requires fp16 operands and `M` on the tile, and declines a
single column outright; `mm_gemv_plan` requires fp32 and exactly one column of
`B`. A one-column fp16 product therefore reaches `Decline` and `mul!`, whose
split-K GEMV is the bandwidth-bound kernel that shape wants.
"""
function mmplan(dev, out, A, B, bias)
    # First, and not a preference either: an int8 weight is not an operand any
    # of the float paths can read at all.
    A isa ConvRotQInt8Matrix && return MMConvRotInt8Plan()
    A isa QInt8Matrix && return MMInt8Plan()
    p = mm_coopmat_plan(dev, out, A, B)
    p isa Decline || return p
    mm_gemv_plan(dev, out, A, B, bias)
end

"""
    mmplan(caps, Tout, Ta, Tb, sout, sa, sb, hasbias) -> plan

The same decision from TYPES and SHAPES, which is all a declaration has.

`emitop!` holds graph resources, not arrays: a transient has no storage until
`Place` has run, so `typeof` cannot be asked and `Mantle.devicetype` answers
instead. The array form above forwards to this one, so which path a shape takes
is decided in one place and an immediate call and a declared one cannot pick
differently -- and which path a shape takes is measured, so two copies drifting
is a performance regression nothing would report.
"""
function mmplan(caps, ::Type{Tout}, ::Type{Ta}, ::Type{Tb},
                sout::Dims, sa::Dims, sb::Dims, hasbias::Bool) where {Tout,Ta,Tb}
    Ta <: ConvRotQInt8Matrix && return MMConvRotInt8Plan()
    Ta <: QInt8Matrix && return MMInt8Plan()
    p = mm_coopmat_plan(caps, Tout, Ta, Tb, sa, sb)
    p isa Decline || return p
    mm_gemv_plan(caps, Tout, Ta, Tb, sa, sb, hasbias)
end

# A single policy shared by immediate and declared execution. The immediate
# context records the answer obtained from the explicitly passed device; it
# does not retain or rediscover that device. The declared path still has the
# device in hand and asks the same question directly.
library_gemm_preferred(canrun::Bool, epi) = canrun && epi === identity
library_gemm_preferred(dev::M.Device, epi) =
    library_gemm_preferred(M.runscalls(dev), epi)

# One method per plan type (review finding 1): a new GEMM path is a new plan type
# and a new method here, not another branch in the function above.
function matmul!(ctx, plan::MMCoopMatPlan, out, A, B, bias, epi; gemm=NamedTuple())
    # When a graph can record ecosystem calls, an identity epilogue does not
    # recover enough memory traffic to pay for replacing the tuned library
    # GEMM.  Keep the portable cooperative-matrix kernel for a real fused
    # epilogue; this is a graph capability decision, not a vendor decision.
    if library_gemm_preferred(ctx.runscalls, epi)
        return matmul!(ctx, Decline(:library_faster), out, A, B, bias, epi; gemm)
    end
    matmul_coopmat!(ctx, out, plan, A, B, bias, epi; gemm)
end

"""
    mm_gemv_plan(dev, out, A, B, bias) -> MMGemvPlan | Decline

Whether this is a matrix-*vector* product: one column of `B`, fp32, dense
operands, and a bias this kernel's store can apply.

`size(B, 2) == 1` is what "batch 1" means in the reversed layout — torch's `M`,
the token count, is `B`'s trailing extent here. Every matmul in an autoregressive
decoder step has it, and none in an encoder does.

Dense, because `Mantle.gemv!`'s addressing is `W[m + M * (k - 1)]`: a strided view
would read the wrong elements rather than fail, so the layout is required, not
adapted to. `A` is `(M, K)` contiguous along `m` — what `hoistpermutes` produces
— and reaches `gemv!` as `transpose(A)`, which is the dispatch that picks the
kernel for that layout.

The bias is checked here rather than asserted in the kernel so that an unexpected
shape *declines* to the scalar path, which broadcasts anything, instead of
throwing. Whisper's decoder biases are all `(M,)`; the check is for the next
model.
"""
mm_gemv_plan(dev, out, A, B, bias) =
    mm_gemv_plan(dev, typeof(out), typeof(A), typeof(B), size(A), size(B),
                 biasfoldable(bias, size(A, 1)))

"""
    biasfoldable(bias, M) -> Bool

Whether `bias` is one the GEMV's store can apply: one element per output row.

Named because BOTH callers of the plan have to agree on it, and they did not.
The array form above stated the rule inline while `emitop!`'s `gemm!` passed
`bias !== nothing`, so a declared product could choose `MMGemvPlan` for a bias
the kernel cannot index — the same "two non-equivalent rules for one decision"
`mmplan`'s own docstring promises cannot happen.

`ndims` rather than `isa AbstractVector`: a declared operand is a graph resource
and not an `AbstractArray` at all, so the `isa` test answered `false` for every
bias on the path that has no arrays in it.
"""
biasfoldable(::Nothing, ::Int) = false
biasfoldable(bias, M::Int) = ndims(bias) == 1 && length(bias) == M

function mm_gemv_plan(dev, ::Type{Tout}, ::Type{Ta}, ::Type{Tb},
                      sa::Dims, sb::Dims, biasok::Bool) where {Tout,Ta,Tb}
    densematrix(Float32, Ta) || return Decline(:operands)
    densematrix(Float32, Tb) || return Decline(:operands)
    densematrix(Float32, Tout) || return Decline(:operands)
    sb[2] == 1 || return Decline(:notvector)
    biasok || return Decline(:bias)
    MMGemvPlan()
end

"""Int8 weight: the GEMV dequantises in its inner loop, wider products
dequantise once into the workspace and reuse the fp16 GEMM."""
function matmul!(ctx, ::MMInt8Plan, out, A, B, bias, epi; gemm=NamedTuple())
    if size(B, 2) == 1
        q8gemv!(ctx, out, A, B, bias)
        epi === identity || (out .= epi.(out))
        return out
    end
    tiling = q8gemm_tiling(ctx.dev, A, B, out)
    tiling === nothing || return q8gemm!(out, A, B; tiling, bias, epilogue=epi)
    # Wider than one column: hand the fp16 GEMM a dense operand and let it plan
    # for that operand like any other. Re-planning rather than assuming a path,
    # because which one applies depends on the dequantised extents.
    W = q8dequant(ctx, A)
    matmul!(ctx, mmplan(ctx.dev, out, W, B, bias), out, W, B, bias, epi; gemm)
end

"""ConvRot is an orthogonal, group-wise activation transform preceding INT8 GEMM."""
function matmul!(ctx, ::MMConvRotInt8Plan, out, A, B, bias, epi; gemm=NamedTuple())
    Brot = convrot(ctx, B, A.group_size)
    matmul!(ctx, MMInt8Plan(), out, QInt8Matrix(A.q, A.scale, A.m), Brot,
            bias, epi; gemm)
end

"""`Mantle.gemv!`: the M = 1 path, with the bias and activation in its store."""
function matmul!(ctx, ::MMGemvPlan, out, A, B, bias, epi; gemm=NamedTuple())
    Mantle.gemv!(out, B, transpose(A); bias, epilogue = epi)
    return out
end

"""`LinearAlgebra.mul!`, which is Lava's scalar kernel — always available."""
function matmul!(ctx, ::Decline, out, A, B, bias, epi; gemm=NamedTuple())
    mul!(out, astranspose(A), astranspose(B))
    bias === nothing || (out .= out .+ bias)
    # The scalar path has no epilogue to fold into, so the activation is a second
    # pass here — the same one the graph would have run as its own op. Folding is
    # an optimisation on the tensor-core path, never a correctness requirement.
    epi === identity || (out .= epi.(out))
    out
end


"""
    mm_coopmat_plan(dev, out, A, B) -> MMCoopMatPlan | Decline

Whether `matmul!` can take the tensor-core path.

`Lava.mul!` has its own cooperative-matrix path but only for an fp32
destination, and under autocast every `addmm` in this model writes fp16 — so
without this the graph's 48 matmuls all landed on the scalar kernel. The
accumulate-and-convert is done here rather than there because the bias add
belongs in the same epilogue and `mul!` has no bias.

`N` is padded internally; `M` and `K` are the operands' own extents and are
required to land on the tile.

**The padding target is the staged kernel's block, not the tile** — see
`Mantle.gemm_padn`. Rounding to `dev.tile` is enough to make the cooperative-matrix
*instruction* legal and not enough to make the fast kernel applicable, and the
difference is a factor of several: Whisper's 1500 tokens round to 1504, which no
tiling's 64- or 128-wide block divides, so every one of its 160 matmuls ran on
the register-blocked kernel. Rounding to 1536 costs 2.4% more arithmetic.

**Which block, though, and `gemm_padn` takes the smallest.** Measured on this
device at `M = 65536, K = 2592`, sweeping the column count alone:

    N        16    32    48    64    96   128   144   160   192   256   288   384
    TFLOP/s 0.83  2.28  0.70  8.09  1.24 16.40  0.47  1.26  9.32 16.15  1.27 17.29

The 128-wide tiling is about 1.85x the 64-wide one PER COLUMN, so the least
padding is not the fastest padding: `N = 288` pads to 320 at 9.32 where 384
would run at 17.29, which is 12.97 against 8.39 for the columns that are real.
`conv_coopmat_plan` scores by tile width for exactly this reason — see
[`convcoutpad`](@ref) — and this path does not, because reaching 384 needs more
slack than `GEMM_PAD_SLACK` allows and that constant is tuned across models this
repository cannot measure in one session. What is here is the measurement, not
the change.
"""
mm_coopmat_plan(dev::M.DeviceCaps, out, A, B) =
    mm_coopmat_plan(dev, typeof(out), typeof(A), typeof(B), size(A), size(B))

function mm_coopmat_plan(dev::M.DeviceCaps, ::Type{Tout}, ::Type{Ta}, ::Type{Tb},
                         sa::Dims, sb::Dims) where {Tout,Ta,Tb}
    densematrix(Float16, Ta) && densematrix(Float16, Tb) || return Decline(:operands)
    # A MATRIX-VECTOR PRODUCT IS NOT A TENSOR-CORE SHAPE. One column of `B` has
    # no reuse to amortise a 16-wide tile over, and `gemm_padn` rounds `N = 1` up
    # to the staged kernel's block, so the cooperative-matrix path does the whole
    # product at the block's width and throws away all but one column.
    #
    # This is the entire autoregressive decode path: under fp16 the operands
    # satisfy every test above, and `mm_gemv_plan` below asks for fp32, so
    # nothing else catches them. On K2 Horizon
    # 32B's 449 decode GEMVs it cost **733 ms per token against 380** for
    # `mul!`'s split-K GEMV, which is bandwidth-bound and the right kernel here.
    sb[2] == 1 && return Decline(:notmatrix)
    # BEFORE the extent test, which divides by `dev.tile`. A device with no
    # matrix hardware reports no tile, and the extent test would then throw a
    # DivideError instead of declining.
    #
    # `coopmatkernels` and not `dev.coopmat`, for the reason given there: this plan
    # is a promise to call `coopmat_gemm_shape`, which exists only where the
    # staged kernels do.
    coopmatkernels(dev) || return Decline(:nocoopmat)
    sa[1] % dev.tile == 0 && sa[2] % dev.tile == 0 || return Decline(:extent)
    MMCoopMatPlan(Mantle.gemm_padn(sa[1], sb[2], sa[2]; tile = dev.tile), dev.tile)
end

"""Copy `B` into the leading `N` columns of a `K x NP` scratch, zeroing the rest."""
function padcols_kernel!(dst, B, ::Val{K}, N) where {K}
    i, j = Tuple(KI.get_global_id())
    # The bounds check `@kernel` used to insert. The launch names no workgroup,
    # so `KernelInterface` fills one up to the device limit and it need not
    # divide `(K, NP)`.
    (i <= K && i + K * (j - 1) <= length(dst)) || return nothing
    @inbounds dst[i + K * (j - 1)] = j <= N ? B[i, j] : zero(eltype(dst))
    return nothing
end

"""
`out[i, j] = T(C[i, j] + bias[i])`, dropping the padded columns of `C` and
summing the split-K planes on the way — this pass already reads every element,
so a separate reduction kernel would be a second full traversal for nothing.
"""
function mm_epilogue_kernel!(out, C, bias, epi, ::Val{M},
                                     ::Val{SPLITK}, plane, ntot) where {M,SPLITK}
    # Flat launch: a 2-D `ndrange` is partitioned into 2-D workgroups, so a warp
    # covers only a few consecutive `i` and neither the read of `C` nor the write
    # of `out` (both `i`-major) coalesces. Doing the same to the convolution
    # epilogue and im2col was worth 2.3 ms of a 31.7 ms step.
    lin = KI.get_global_id().x
    if lin <= ntot
        @inbounds begin
            q = Int32(lin) - Int32(1)
            i = q % Int32(M) + Int32(1)
            j = q ÷ Int32(M) + Int32(1)
            k = Int32(i) + Int32(M) * (j - Int32(1))
            v = C[k]
            for s in Int32(1):Int32(SPLITK - 1)
                v += C[k + s * Int32(plane)]
            end
            bias === nothing || (v += Float32(bias[i]))
            # The activation goes on the *converted* value, exactly as it does in
            # the fused path and for the same reason: the graph rounds between
            # the matmul and the activation, so applying it to the fp32
            # accumulator would compute a different function.
            out[i, j] = epi(eltype(out)(v))
        end
    end
    return nothing
end

# ── Why the GEMM writes `out` directly ───────────────────────────────────────
#
# Bias in the accumulator's initial value, fp32→fp16 conversion in registers,
# instead of an fp32 scratch plus `mm_epilogue_kernel!`. Re-measuring that has
# to A/B the two **inside one session**: cross-session encode numbers on this
# machine have disagreed by 40 ms in both directions.
#
# The two are not bit-identical and are not meant to be: the bias joins the fp32
# accumulation chain rather than being added after it, so 31 elements of a 2.36 M
# output differ by at most 0.5 ulp of fp16. Maximum error against a Float64
# reference is the same to four significant figures on every shape.

function matmul_coopmat!(ctx, out, plan::MMCoopMatPlan, A, B, bias, epi;
                         gemm=NamedTuple())
    M, K = size(A)
    N = size(B, 2)
    NP = plan.NP
    backend = ctx.backend

    Bp = B
    if NP != N
        Bp = scratch!(ctx, Float16, K, NP)
        KI.Kernel(backend, padcols_kernel!)(Bp, B, Val(K), N; ndrange = (K, NP))
    end
    blk_split = Mantle.coopmat_gemm_shape(M, NP, K)
    splitk = blk_split[2]
    # Nothing to reduce: the GEMM can start its accumulators from the bias and
    # convert to `out`'s type as it stores, so there is no fp32 scratch and no
    # second pass. `mm_epilogue_kernel!` was 23% of matmul time on exactly these
    # shapes — `splitk == 1` for every one an encoder runs — and all it did was
    # read `M x N` fp32 back and write fp16.
    #
    # A padded `N` does not force the epilogue back. The destination is
    # column-major, so **columns 1..N of an `M x NP` buffer are its first `M*N`
    # elements, contiguously** — the GEMM can write the padded width into scratch
    # of `out`'s own type and the discard is a linear copy, not a gather. Against
    # the fp32 route that is 11.6 MB of traffic instead of 19.6 on Whisper's
    # attention shape and 46 instead of 78 on its `fc1`, and the bias and the
    # activation stay fused in the GEMM's store where the unpadded path has them.
    if splitk == 1
        dst = NP == N ? out : scratch!(ctx, eltype(out), M, NP)
        Mantle.coopmat_gemm!(dst, A, Bp, M, NP, K; blk_split, bias, epilogue = epi, gemm...)
        NP == N || copyto!(out, 1, dst, 1, M * N)
        return out
    end
    C = scratch!(ctx, Float32, M, NP, max(splitk, 1))
    Mantle.coopmat_gemm!(C, A, Bp, M, NP, K; blk_split, partials = C, reduce = false, gemm...)
    KI.Kernel(backend, mm_epilogue_kernel!)(out, C, bias, epi, Val(M), Val(splitk), M * NP, M * N;
                                 ndrange = M * N)
    out
end

"""
    batchedmatmul!(ctx, out, A, B)

`out[i, j, b] = Σ_k A[i, k, b] B[k, j, b]`.

One `matmul!` per batch plane, so a batched product gets the same capability
dispatch a 2-D one does: cooperative matrices where the operands qualify, Lava's
`mul!` where they do not. `mm3` stays for anything not shaped as three
dimensions with a common batch extent.

On a dense operand the slice is also free in the strongest sense:
`view(A, :, :, b)` of a `LavaArray{T,3}` *is* a `LavaArray{T,2}` with unit
leading stride, so `mm_coopmat_plan` still sees the operand type and can take
the tensor-core path. Non-dense operands slice to a nested `SubArray` instead
and land on `mul!`, which is where the 2-D path already sends them.
"""
# ── Why this is not `launch!(ctx, mm3, ...)` any more ────────────────────────
#
# It was, and on Depth Anything V2 Small that single line was **79.6% of the
# forward pass**. `mm3` gives one thread one output element and walks K in global
# memory: no tile, no shared memory, no reuse. The graph's 24 `bmm` cost 826 ms
# of a 972 ms frame.
#
# The damage is very unevenly distributed, which is what made it hard to see. The
# two attention products do *identical* arithmetic, 720.7 M MACs each, and
# measured 10.1x apart (`opdouble` on `bmm.default`, filtered by operand shape,
# in situ on a real step):
#
#     attn*V  out (64, 1370, 6)  K=1370    64.47 ms/call   x12 = 773.6 ms
#     Q*K'    out (1370, 1370, 6) K=64      6.39 ms/call   x12 =  76.7 ms
#
# so 94% of the cost sat in one of the two, and profiling by kernel *name* hid it
# completely: both launch as `gpu_ndmap_flat!`, which reads like an elementwise
# kernel.
#
# Routing the planes through `matmul!` instead:
#
#     bmm.default    826 ms -> 76.5 ms      (10.8x)
#     whole frame    972 ms -> 233 ms        (4.2x)
#
# Output is unchanged to fp32 reassociation: max |diff| 1.9e-6 on a value range
# of 4.79, i.e. 4.0e-7 relative, against a model verified to 4.2e-5 of PyTorch.
#
# Note this model is fp32 throughout, so every one of its 50 `matmul!` calls
# still declines the tensor-core path on `:operands` and lands on `mul!`. The
# 4.2x here is what a *tiled* GEMM buys over a naive one; the cooperative-matrix
# path is a further win this export cannot take.
#
# ── Why the guard is `ndims`, and not a density check ────────────────────────
#
# Half of this model's `bmm` do not get a dense operand. Twelve arrive with `A`
# as a `ReshapedArray` of a `SubArray` of a `PermutedDimsArray` of a
# `LavaArray{T,5}`, so the plane is a deeply nested `SubArray` rather than a
# `LavaArray{T,2}`, and those twelve are the *expensive* ones. A `DenseArray`
# guard was tried and sent exactly them back to `mm3`: 4.2x collapsed to 1.02x.
#
# Passing them to `matmul!` anyway is not a new risk. `matmul!`'s own decline
# path already hands arbitrary 2-D operands to `mul!`, which is what
# `astranspose` above exists for, so a batched plane is no worse placed than the
# 2-D operands that path takes today. Measured: correct to 4.0e-7 of range, and
# 10.8x faster on the op.
"""
Is one dispatch per batch plane worth it, against one flat launch for all of them?

Routing per plane costs `nbatch - 1` extra dispatches, so it has to buy more than
that back. Three clauses, each of which a measurement put there:

  * `nbatch == 1` pays nothing extra, so it always wins: it gets a real GEMM
    kernel instead of `mm3` for free. SAM 2's decoder `bmm` is this shape.
  * the plane must **fill at least one workgroup tile**. This is the clause that
    matters, and "the plane can use a good kernel" is NOT a sufficient test for
    it: MatAnyone's planes are 16x32 fp16, which `mm_coopmat_plan` accepts
    happily, and eight dispatches of a single cooperative-matrix tile lose badly
    to one flat launch.
  * above that, the plane still has to qualify for one of the real kernels, or
    per-plane just buys `strided_gemm_kernel!` at the price of N dispatches.

Measured end to end, interleaved in one session (`mm3` for every plane against
this predicate):

    MatAnyone 512x512   unguarded 173.8 ms vs 151.8   0.87x, a 14.5% REGRESSION
    MatAnyone 512x512   with this clause                within noise of `mm3`
    Depth Anything      with this clause                7.8x, undiminished

Without the clause MatAnyone costs 14.5% more. A static count of dispatch
overhead puts that at ~0.5%, wrong by a factor of thirty, so this is a question
for a model run and not for the dispatch count.
"""
@inline function planewise_worth(ctx, out, A, B)
    size(out, 3) == 1 && return true
    M, N = size(out, 1), size(out, 2)
    (M >= Mantle.SGEMM_BM && N >= Mantle.SGEMM_BN) || return false
    t = ctx.dev.tile
    # A plane of a DENSE fp16 operand is a `LavaArray{T,2}`, so the parent type
    # settles this without building the view — the views are not free at this
    # call rate.
    if ctx.dev.coopmat && eltype(A) === Float16 && eltype(B) === Float16 &&
       islavaarray(A) && islavaarray(B) &&
       size(A, 1) % t == 0 && size(A, 2) % t == 0
        return true
    end
    # ── The tile-count clause applies to BOTH dtypes.
    #
    # NOT `eltype(out) === Float32 || return false`: an fp16 plane that misses
    # the branch above still has somewhere to go per-plane, because `matmul!`
    # pads a plane onto a real kernel, where the flat `mm3` it falls back to
    # instead is roughly a 1 TF/s path.
    #
    # It cost Depth Anything's autocast export a factor of **3.75 end to end**,
    # invisibly, because every extent it has is 1370 (37x37 patches + cls) and
    # 1370 % 16 = 10. Measured on its own `QK^T`, (1370,64,6) x (64,1370,6):
    #
    #     fp16 via mm3        1.440 ms   1.00 TF/s     <- what this line caused
    #     fp32 planewise      0.400      3.60
    #     fp16 planewise      0.209      7.29
    #
    # So the fp16 arm was 3.6x SLOWER than the fp32 one for a dtype that should
    # win, which is what made the whole export look like a dead end.
    #
    # MatAnyone — the model the clause below was added to protect, whose 16x32
    # fp16 planes lose 14.5% when routed per plane — is unaffected: those planes
    # are already rejected by the `M >= SGEMM_BM && N >= SGEMM_BN` test above,
    # which is dtype-independent and fires first.
    tm, tn = cld(M, Mantle.SGEMM_BM), cld(N, Mantle.SGEMM_BN)
    return tm * tn >= Mantle.SGEMM_MINTILES &&
           (tm * Mantle.SGEMM_BM) * (tn * Mantle.SGEMM_BN) <= Mantle.SGEMM_MAXWASTE * M * N
end

"""
    BMM_PADSTEP
    bmmpad(n) -> Int

Round a batched-GEMM plane extent up to something a tiling block divides.

`Mantle.gemm_tiling` takes the first tiling whose block **divides the shape
exactly**, so an extent that divides nothing gets the register-blocked kernel
however healthy it looks. Depth Anything's every extent is 1370 (37x37 patches
plus a cls token); 1370 is `2 * 5 * 137` and divides none of 16/32/64/96/128.

64, not `GEMM_BLOCK`. `GEMM_BLOCK` is `lcm(96, 64, 32) = 192` — it makes *every*
tiling applicable, which is what the im2col row pad wants, but it takes 1370 to
1536 and 25.7% more arithmetic. Measured on the real QK^T shape,
`(1370,64,6) x (64,1370,6)` fp16:

    L = 1370   1.289 ms   1.12 TF/s   (+0.0% work)  <- divides nothing
    L = 1408   0.212      7.19        (+5.6%)       <- 64, and the fastest
    L = 1536   0.227      8.00       (+25.7%)       <- GEMM_BLOCK, better rate
                                                       and worse wall time

The higher rate at 1536 with the worse time is the trap in picking this constant
off TF/s instead of milliseconds.
"""
const BMM_PADSTEP = 64
bmmpad(n::Int) = cld(n, BMM_PADSTEP) * BMM_PADSTEP

"""Would padding this plane's `M`/`N` onto a block buy a real kernel?"""
@inline function bmmpad_worth(ctx, out, A, B)
    eltype(out) === Float16 && eltype(A) === Float16 && eltype(B) === Float16 || return false
    islavaarray(A) && islavaarray(B) && islavaarray(out) || return false
    M, N = size(out, 1), size(out, 2)
    Mp, Np = bmmpad(M), bmmpad(N)
    # **`M` is the gate, not `M` or `N`.** The pad is not free — the operands have
    # to be copied into the padded scratch and the result copied back out — so it
    # only pays where the tiling is actually stuck, and that is the block-row axis.
    #
    # Both of Depth Anything's attention products have a bad `N`, and they
    # disagree completely, measured fp16 against the unpadded planewise path:
    #
    #     QK^T  (1370,64,6)x(64,1370,6)   M 1370 bad   1.283 -> 0.563 ms   2.3x
    #     P.V   (64,1370,6)x(1370,1370,6) M   64 fine  1.079 -> 1.623      0.7x
    #
    # P.V regresses because fixing its `N` means copying `B`, which is the
    # 1370x1370 operand — 22.5 MB moved to pad an axis that was not what held it
    # back. Gating on `M` leaves it alone.
    Mp == M && return false
    # The pad is also paid for in arithmetic, so cap it. 1.25 matches `crspad`
    # next door; 1370 -> 1408 is 1.056, and a 100-row plane padded to 128 is 1.28
    # and correctly refused.
    (Mp * Np) <= 1.25 * M * N
end

"""
Batched GEMM on planes padded up to a tiling block, then the sub-block copied out.

**Nothing is zeroed, and that is not an oversight.** Only `M` and `N` are padded,
never `K`, so a padded row of `A` only ever produces a padded row of `C` and a
padded column of `B` only a padded column of `C` — both discarded. No garbage in
the scratch can reach `out[1:M, 1:N, :]`, which is what makes this cheap enough
to be worth doing.

That is a statement about *contamination*, not about bits. The padded shape can
pick a different tiling, and a different tiling sums `K` in a different order, so
the result is not bit-identical — measured on the QK^T shape against the unpadded
planewise product: **rel rms 1.0e-05, max 2 ULPs, 0.074% of elements differ**.
(The `(64,1370,6)` shape, where only `N` moves, IS exact.) Floating-point
addition is not associative; this is that and nothing more.
"""
function batchedmatmul_padded!(ctx, out, A, B)
    T = eltype(out)
    M, N, nb = size(out, 1), size(out, 2), size(out, 3)
    K = size(A, 2)
    Mp, Np = bmmpad(M), bmmpad(N)
    Ap = scratch!(ctx, T, Mp, K, nb)
    Bp = scratch!(ctx, T, K, Np, nb)
    Cp = scratch!(ctx, T, Mp, Np, nb)
    copyto!(view(Ap, 1:M, :, :), A)
    copyto!(view(Bp, :, 1:N, :), B)
    for b in 1:nb
        matmul!(ctx, view(Cp, :, :, b), view(Ap, :, :, b), view(Bp, :, :, b))
    end
    copyto!(out, view(Cp, 1:M, 1:N, :))
    out
end

# ── N-blocked batched matmul ──────────────────────────────────────────────────
#
# `mm3` is one invocation per output element, so `A[m, :, b]` is re-read once for
# every column `n`. Attention is exactly the shape that punishes: grouped-query
# decode gives `N = n_rep = 8`, so both `bmm`s read their large operand EIGHT
# times. On K2 Horizon 32B that is 2.1 GB per token against 268 MB of actual
# data, and 17.5 ms of a 211 ms step.
#
# Here each thread keeps `N` accumulators and reads `A[m, k, b]` ONCE, reusing it
# across all of them — the same trick the int8 GEMV uses along rows, applied
# along columns. Split over K as well, because `M * nbatch` alone is 8192 threads
# for the first `bmm` and 1024 for the second, nowhere near enough to fill the
# device.
#
# One kernel per N, generated with N as a literal: `@nexprs` needs that, and a
# `Val`-gated loop would put a tuple in the inner loop and not compile.
for NB in (2, 4, 8, 16)
    kname = Symbol("bmm_n", NB, "!")
    @eval function $kname(P, A, Bm, M::Int32, K::Int32,
                                            KC::Int32, NBATCH::Int32, ntot::Int32)
        lin = KI.get_global_id().x
        if lin <= ntot
            @inbounds begin
                l = Int32(lin) - Int32(1)
                m = l % M + Int32(1)        # consecutive lanes -> consecutive rows
                r = l ÷ M
                b = r % NBATCH + Int32(1)
                sp = r ÷ NBATCH             # 0-based K split
                k1 = min(sp * KC + KC, K)
                Base.Cartesian.@nexprs $NB n -> acc_n = 0f0
                k = sp * KC
                while k < k1
                    kk = k + Int32(1)
                    av = Float32(A[m, kk, b])
                    Base.Cartesian.@nexprs $NB n -> begin
                        acc_n = muladd(av, Float32(Bm[kk, n, b]), acc_n)
                    end
                    k += Int32(1)
                end
                Base.Cartesian.@nexprs $NB n -> (P[m, n, b, sp + Int32(1)] = acc_n)
            end
        end
        return nothing
    end
end

const BMM_N_KERNELS = Dict(2 => bmm_n2!, 4 => bmm_n4!, 8 => bmm_n8!, 16 => bmm_n16!)

# Flat launch for the same reason `indexput_kernel!` has one: a 3-D `ndrange` is
# partitioned into 3-D workgroups and costs several times what a flat one does.
function bmm_nsplit_reduce!(C, P, S::Int32,
                                              ::Val{SZ}, n::Int64) where {SZ}
    i = KI.get_global_id().x
    if i <= n
        @inbounds begin
            I = CartesianIndices(SZ)[i]
            acc = 0f0
            for s in Int32(1):S
                acc += P[I[1], I[2], I[3], s]
            end
            C[I] = eltype(C)(acc)
        end
    end
    return nothing
end

# Threads to aim for, as in `Mantle.gemv_split`. A `Ref` because the split's cost is
# not only the launch width: each extra split writes another `M x N x batch` plane
# of fp32 partials, and on the decode attention those planes are as large as the
# operand being read, so the best value has to be measured rather than reasoned.
const BMM_TARGET_THREADS = Ref(1 << 16)

"""
    bmm_nblocked!(ctx, out, A, B) -> out | nothing

`nothing` when the shape is not one this path covers, so the caller falls
through to `mm3` rather than this having to know about every case.
"""
function bmm_nblocked!(ctx, out, A, B)
    ndims(out) == 3 && ndims(A) == 3 && ndims(B) == 3 || return nothing
    M, N, NBATCH = size(out)
    haskey(BMM_N_KERNELS, N) || return nothing
    size(A, 1) == M && size(A, 3) == NBATCH || return nothing
    size(B, 2) == N && size(B, 3) == NBATCH || return nothing
    K = size(A, 2)
    K == size(B, 1) || return nothing
    S = 1
    while S * M * NBATCH < BMM_TARGET_THREADS[] && cld(K, 2S) >= 16
        S *= 2
    end
    KC = cld(K, S)
    P = scratch!(ctx.ws, ctx.backend, Float32, M, N, NBATCH, S)
    KI.Kernel(ctx.backend, BMM_N_KERNELS[N])(P, A, B, Int32(M), Int32(K), Int32(KC),
                                       Int32(NBATCH), Int32(M * NBATCH * S);
                                       ndrange = M * NBATCH * S, workgroupsize = 256)
    KI.Kernel(ctx.backend, bmm_nsplit_reduce!)(out, P, Int32(S), Val((M, N, NBATCH)),
                                         Int64(M * N * NBATCH); ndrange = M * N * NBATCH, workgroupsize = 256)
    out
end

function batchedmatmul!(ctx, out, A, B)
    # Before the planewise test: a narrow `N` wants its own kernel, and the
    # planewise path would hand each plane to a GEMM that pads `N` up to a tile.
    r = bmm_nblocked!(ctx, out, A, B)
    r === nothing || return r
    if ndims(out) == 3 && ndims(A) == 3 && ndims(B) == 3 &&
       size(A, 3) == size(B, 3) == size(out, 3) && planewise_worth(ctx, out, A, B)
        bmmpad_worth(ctx, out, A, B) && return batchedmatmul_padded!(ctx, out, A, B)
        for b in axes(out, 3)
            matmul!(ctx, view(out, :, :, b), view(A, :, :, b), view(B, :, :, b))
        end
        return out
    end
    launch!(ctx, mm3, out, A, B)
end
