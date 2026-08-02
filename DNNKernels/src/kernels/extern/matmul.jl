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
    matmul!(out, A, B, bias=nothing)

`out[i, j] = Σ_k A[i, k] B[k, j]`, plus `bias` broadcast over the output.

Takes the cooperative-matrix path when `mm_coopmat_applicable` says so, and
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
function matmul!(out, A, B, bias=nothing; ws=nothing, epi=identity)
    mm_coopmat_applicable(out, A, B) && return matmul_coopmat!(out, A, B, bias, ws, epi)
    mul!(out, astranspose(A), astranspose(B))
    bias === nothing || (out .= out .+ bias)
    # The scalar path has no epilogue to fold into, so the activation is a second
    # pass here — the same one the graph would have run as its own op. Folding is
    # an optimisation on the tensor-core path, never a correctness requirement.
    epi === identity || (out .= epi.(out))
    out
end

"""
    mm_coopmat_applicable(out, A, B) -> Bool

Whether `matmul!` can take the tensor-core path.

`Lava.mul!` has its own cooperative-matrix path but only for an fp32
destination, and under autocast every `addmm` in this model writes fp16 — so
without this the graph's 48 matmuls all landed on the scalar kernel. The
accumulate-and-convert is done here rather than there because the bias add
belongs in the same epilogue and `mul!` has no bias.

`N` is padded internally; `M` and `K` are the operands' own extents and are
required to land on the tile.
"""
function mm_coopmat_applicable(out, A, B)
    A isa Lava.LavaArray{Float16,2} && B isa Lava.LavaArray{Float16,2} || return false
    size(A, 1) % Lava.GEMM_TILE == 0 && size(A, 2) % Lava.GEMM_TILE == 0 || return false
    Lava.coopmat_gemm_available()
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

"""
    MATMUL_FUSED[] :: Bool

Whether the GEMM writes `out` directly — bias in the accumulator's initial value,
fp32→fp16 conversion in registers — instead of an fp32 scratch plus
`mm_epilogue_kernel!`.

A switch rather than a constant for the same reason as `LAUNCH_FLAT`: it is the
only way to compare the two **inside one session**, and cross-session encode
numbers on this machine have disagreed by 40 ms in both directions.

The two are not bit-identical and are not meant to be: the bias joins the fp32
accumulation chain rather than being added after it, so 31 elements of a 2.36 M
output differ by at most 0.5 ulp of fp16. Maximum error against a Float64
reference is the same to four significant figures on every shape.
"""
const MATMUL_FUSED = Ref(true)

function matmul_coopmat!(out, A, B, bias, ws, epi)
    M, K = size(A)
    N = size(B, 2)
    NP = padtile(N)
    backend = KernelAbstractions.get_backend(out)

    Bp = B
    if NP != N
        Bp = scratch!(ws, backend, Float16, K, NP)
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
    if MATMUL_FUSED[] && splitk == 1 && NP == N
        Lava.coopmat_gemm!(out, A, Bp, M, N, K; blk_split, bias, epilogue = epi)
        return out
    end
    C = scratch!(ws, backend, Float32, M, NP, max(splitk, 1))
    Lava.coopmat_gemm!(C, A, Bp, M, NP, K; blk_split, partials = C, reduce = false)
    mm_epilogue_kernel!(backend)(out, C, bias, epi, Val(M), Val(splitk), M * NP, M * N;
                                 ndrange = M * N)
    out
end

"""
    batchedmatmul!(out, A, B)

`out[i, j, b] = Σ_k A[i, k, b] B[k, j, b]`.
"""
batchedmatmul!(out, A, B) = launch!(mm3, out, A, B)
