using Test, DNNKernels, Lava, KernelAbstractions
using Mantle: LavaBackend
using Adapt: adapt
using KernelAbstractions: @kernel, @index
const KA = KernelAbstractions

# `@index(Global, NTuple)` used to be recovered by dividing a flattened block
# index back apart — `2(N-1)` integer divisions by runtime extents — even though
# `pad_to_3d` had dispatched the block grid on the three hardware axes and the
# workgroup builtins already held the answer. `Lava.directdispatch` now takes it
# straight off the builtins when the dispatch permits, which on a plain 70.8 MB
# elementwise copy moved rank 3 from 100 GB/s to 247, level with what the same
# copy reaches through `@index(Global, Linear)`.
#
# The risk it introduces is a *wrong index*, silently, in whichever kernels the
# guard misjudges — so this stamps a value that is a unique function of the full
# index tuple. A transposed or truncated coordinate cannot coincidentally
# reproduce it, which a copy-shaped test would let through.
#
# Every shape below has failed at some point during development. The rank-4
# entries are why `directdispatch` declines rank 4 outright: there the *device*
# evaluates the guard differently from the host and admits launches it must not.

@kernel function idxstamp1!(o)
    i, = @index(Global, NTuple)
    @inbounds o[i] = i
end
@kernel function idxstamp2!(o)
    i, j = @index(Global, NTuple)
    @inbounds o[i, j] = i + 1000j
end
@kernel function idxstamp3!(o)
    i, j, k = @index(Global, NTuple)
    @inbounds o[i, j, k] = i + 1000j + 1000_000k
end
@kernel function idxstamp4!(o)
    i, j, k, l = @index(Global, NTuple)
    @inbounds o[i, j, k, l] = i + 1000j + 1000_000k + 1000_000_000l
end
@kernel function idxstamp5!(o)
    i, j, k, l, m = @index(Global, NTuple)
    @inbounds o[i, j, k, l, m] = i + 1000j + 1000_000k + 1000_000_000l + 7m
end
@kernel function idxlinear!(o)
    i = @index(Global, Linear)
    @inbounds o[i] = i
end

stampkernel(::Val{1}) = idxstamp1!
stampkernel(::Val{2}) = idxstamp2!
stampkernel(::Val{3}) = idxstamp3!
stampkernel(::Val{4}) = idxstamp4!
stampkernel(::Val{5}) = idxstamp5!

stampvalue(t::NTuple{1,Int}) = float(t[1])
stampvalue(t::NTuple{2,Int}) = t[1] + 1000t[2]
stampvalue(t::NTuple{3,Int}) = t[1] + 1000t[2] + 1000_000t[3]
stampvalue(t::NTuple{4,Int}) = t[1] + 1000t[2] + 1000_000t[3] + 1000_000_000t[4]
stampvalue(t::NTuple{5,Int}) = t[1] + 1000t[2] + 1000_000t[3] + 1000_000_000t[4] + 7t[5]

function stamped(backend, dims; wg = DNNKernels.launchgroup(dims))
    d = adapt(backend, zeros(Float64, dims))
    stampkernel(Val(length(dims)))(backend)(d; ndrange = dims, workgroupsize = wg)
    KA.synchronize(backend)
    Array(d)
end

reference(dims) = [stampvalue(Tuple(I)) for I in CartesianIndices(dims)]

@testset "global index recovery" begin
    backend = LavaBackend()

    @testset "NTuple, launchgroup workgroup" begin
        for dims in [(1024,), (100003,), (65537,),
                     (1920, 1152), (255, 257), (1, 1),
                     (1920, 1152, 4), (2048, 1024, 4), (33, 17, 9), (1, 1, 1),
                     # rank 4: (W, H, C, 1) is the shape RIFE and MatAnyone run
                     (1920, 1152, 4, 1), (1920, 1152, 3, 1), (256, 1, 1, 1),
                     (1, 1, 1, 1), (3, 1, 1, 1),
                     # rank 4 with a non-unit axis past the third, and the three
                     # unit-axis shapes whose guard disagreed between host and device
                     (16, 8, 4, 2), (7, 5, 3, 2), (72, 256, 8, 16),
                     (1, 7, 1, 1), (1, 1, 4, 1),
                     (8, 4, 2, 2, 2), (5, 5, 5, 1, 1), (2, 3, 5, 7, 11)]
            @test stamped(backend, dims) == reference(dims)
        end
    end

    # A workgroup with more than one non-unit extent cannot use the direct path:
    # the workgroup is dispatched 1-D, so the linear thread id is only the
    # fastest coordinate when every other extent is 1. These must divide.
    @testset "multi-dimensional workgroup falls back" begin
        for (dims, wg) in [((64, 64, 4), (8, 8, 4)),
                           ((128, 128), (16, 16)),
                           ((33, 17, 9), (33, 7, 1)),
                           ((1920, 1152, 4, 1), (64, 2, 2, 1)),
                           ((7, 5, 3, 2), (7, 5, 3, 2))]
            @test stamped(backend, dims; wg) == reference(dims)
        end
    end

    @testset "Linear" begin
        # 20e6 exceeds max_wg_dims * 256, so `pad_to_3d` re-splits the grid and
        # the dispatch is no longer 1:1 with the blocks — the guard must decline.
        for dims in [(1024,), (1920, 1152), (1920, 1152, 4), (1920, 1152, 4, 1),
                     (16, 8, 4, 2), (20_000_000,), (2, 3, 5, 7, 11)]
            d = adapt(backend, zeros(Float64, dims))
            idxlinear!(backend)(d; ndrange = dims,
                                workgroupsize = DNNKernels.launchgroup(dims))
            KA.synchronize(backend)
            @test Array(d) == reshape(Float64.(1:prod(dims)), dims)
        end
    end
end
