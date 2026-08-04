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

"""
A 2-D permutation *is* a transpose, so say so. `Transpose` is what LinearAlgebra
and cuBLAS dispatch on; a `PermutedDimsArray` carries the same memory but matches
no BLAS method, so it falls through to the generic matmul, which scalar-indexes a
GPU array from the host. The exported graph produces these constantly — every
`addmm` operand arrives permuted.
"""
astranspose(a) = a
astranspose(a::PermutedDimsArray{T,2,(2, 1)}) where {T} = transpose(parent(a))

"""
    matmul!(ctx, out, A, B, bias=nothing)

`out[i, j] = Σ_k A[i, k] B[k, j]`, plus `bias` broadcast over the output.

Takes the cooperative-matrix path when `mm_coopmat_plan` returns one, and
`LinearAlgebra.mul!` otherwise.

This function *does* know tensor cores exist: the predicate below checks
`Lava.coopmat_gemm_available()` and `Lava.GEMM_TILE`, and gates on
`Lava.LavaArray{Float16,2}`, so the fast path is reachable only on Lava. An
earlier docstring here claimed the opposite; it described a design that was
replaced.

What *is* device-independent is which device takes it. The tile and the
availability are queried per device, so the same source picks cooperative
matrices on Ada (subgroup 32) and on RDNA 3.5 (subgroup 64, `16x16x16` Float16)
with no vendor branch anywhere.
"""
# `gemm` forwards Lava's kernel-selection keywords — `staged`, `vec2`,
# `narrow_ok`, `tiling`. They were module-level `Ref`s in Lava and are now
# arguments, so a benchmark or a test that wants a non-default kernel passes it
# here instead of mutating a process-wide setting and restoring it in a `finally`.
# Empty by default, so the shipped path is exactly the measured one.
matmul!(ctx, out, A, B, bias=nothing; epi=identity, gemm=NamedTuple()) =
    matmul!(ctx, mm_coopmat_plan(ctx.dev, out, A, B), out, A, B, bias, epi; gemm)

# One method per plan type (review finding 1): a new GEMM path is a new plan type
# and a new method here, not another branch in the function above.
matmul!(ctx, plan::MMCoopMatPlan, out, A, B, bias, epi; gemm=NamedTuple()) =
    matmul_coopmat!(ctx, out, plan, A, B, bias, epi; gemm)

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
"""
function mm_coopmat_plan(dev::Device, out, A, B)
    A isa Lava.LavaArray{Float16,2} && B isa Lava.LavaArray{Float16,2} ||
        return Decline(:operands)
    size(A, 1) % dev.tile == 0 && size(A, 2) % dev.tile == 0 || return Decline(:extent)
    dev.coopmat || return Decline(:nocoopmat)
    MMCoopMatPlan(cld(size(B, 2), dev.tile) * dev.tile, dev.tile)
end

"""Copy `B` into the leading `N` columns of a `K x NP` scratch, zeroing the rest."""
@kernel function padcols_kernel!(dst, @Const(B), ::Val{K}, N) where {K}
    i, j = @index(Global, NTuple)
    @inbounds dst[i + K * (j - 1)] = j <= N ? B[i, j] : zero(eltype(dst))
end

"""
`out[i, j] = T(C[i, j] + bias[i])`, dropping the padded columns of `C` and
summing the split-K planes on the way — this pass already reads every element,
so a separate reduction kernel would be a second full traversal for nothing.
"""
@kernel function mm_epilogue_kernel!(out, @Const(C), @Const(bias), epi, ::Val{M},
                                     ::Val{SPLITK}, plane, ntot) where {M,SPLITK}
    # Flat launch: a 2-D `ndrange` is partitioned into 2-D workgroups, so a warp
    # covers only a few consecutive `i` and neither the read of `C` nor the write
    # of `out` (both `i`-major) coalesces. Doing the same to the convolution
    # epilogue and im2col was worth 2.3 ms of a 31.7 ms step.
    lin = @index(Global, Linear)
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
end

# ── Why the GEMM writes `out` directly ───────────────────────────────────────
#
# Bias in the accumulator's initial value, fp32→fp16 conversion in registers,
# instead of an fp32 scratch plus `mm_epilogue_kernel!`. This was a switch
# (`MATMUL_FUSED`), on by default; the winner is inlined below and the switch is
# gone (`kernel-library-review.md` finding 3, tier two). It was a switch because
# that is the only way to compare the two **inside one session**, and
# cross-session encode numbers on this machine have disagreed by 40 ms in both
# directions — so any re-measurement has to A/B the two branches in one process,
# not two runs.
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
        padcols_kernel!(backend)(Bp, B, Val(K), N; ndrange = (K, NP))
    end
    blk_split = Lava.coopmat_gemm_shape(M, NP, K)
    splitk = blk_split[2]
    # Nothing to reduce, and the destination is not padded: the GEMM can start
    # its accumulators from the bias and convert to `out`'s type as it stores, so
    # there is no fp32 scratch and no second pass. `mm_epilogue_kernel!` was 23%
    # of matmul time on exactly these shapes — `splitk == 1` for every one the
    # encoder runs — and all it did was read `M x N` fp32 back and write fp16.
    #
    # `NP != N` still needs the epilogue: the GEMM writes the padded width and
    # the padding columns must not reach `out`.
    if splitk == 1 && NP == N
        Lava.coopmat_gemm!(out, A, Bp, M, N, K; blk_split, bias, epilogue = epi, gemm...)
        return out
    end
    C = scratch!(ctx, Float32, M, NP, max(splitk, 1))
    Lava.coopmat_gemm!(C, A, Bp, M, NP, K; blk_split, partials = C, reduce = false, gemm...)
    mm_epilogue_kernel!(backend)(out, C, bias, epi, Val(M), Val(splitk), M * NP, M * N;
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
# kernel and is how this was once written up as "83.2% elementwise".
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

The unguarded version shipped in an earlier revision of this file and cost
MatAnyone 14.5%. It was found by benchmarking the model rather than by reading
the code: a static count of dispatch overhead had put it at ~0.5%, which was
wrong by a factor of thirty.
"""
@inline function planewise_worth(ctx, out, A, B)
    size(out, 3) == 1 && return true
    M, N = size(out, 1), size(out, 2)
    (M >= Lava.SGEMM_BM && N >= Lava.SGEMM_BN) || return false
    t = ctx.dev.tile
    # A plane of a DENSE fp16 operand is a `LavaArray{T,2}`, so the parent type
    # settles this without building the view — the views are not free at this
    # call rate.
    if ctx.dev.coopmat && eltype(A) === Float16 && eltype(B) === Float16 &&
       A isa Lava.LavaArray && B isa Lava.LavaArray &&
       size(A, 1) % t == 0 && size(A, 2) % t == 0
        return true
    end
    eltype(out) === Float32 || return false
    tm, tn = cld(M, Lava.SGEMM_BM), cld(N, Lava.SGEMM_BN)
    return tm * tn >= Lava.SGEMM_MINTILES &&
           (tm * Lava.SGEMM_BM) * (tn * Lava.SGEMM_BN) <= Lava.SGEMM_MAXWASTE * M * N
end

function batchedmatmul!(ctx, out, A, B)
    if ndims(out) == 3 && ndims(A) == 3 && ndims(B) == 3 &&
       size(A, 3) == size(B, 3) == size(out, 3) && planewise_worth(ctx, out, A, B)
        for b in axes(out, 3)
            matmul!(ctx, view(out, :, :, b), view(A, :, :, b), view(B, :, :, b))
        end
        return out
    end
    launch!(ctx, mm3, out, A, B)
end
