"""
`transposeLE` through a shared-memory tile, reading the operand by strides.

`toLE` was a batched 2-D transpose written as an elementwise gather: consecutive
threads vary `L` and read with stride `E`, so every warp issued 32 transactions —
37.7 GB/s against ~300 for a copy. And the operand is never dense: attention's
`q` arrives as `PermutedDimsArray -> ReshapedArray -> SubArray -> LavaArray`,
whose `SignedMultiplicativeInverse`s add four integer divisions per element.

The tiled kernel fixes both by staging a 32x32 tile and taking the operand as a
base array plus `strides()`. Measured on SAM 2's windowed shape 0.250 -> 0.035 ms
(7.1x), and -19.3 ms of the encoder's `ndmap` total.

**The `SubArray` is the whole test.** Refusing views made every shape that
matters fall back silently — correct, and none of the win — while accepting one
without accounting for its offset would read the wrong elements just as silently.
"""

using Test, Lava, DNNKernels, KernelAbstractions
using DNNKernels: transposeLE, stridedroot, Workspace, Ctx
const KA = KernelAbstractions

"`transposeLE(a)` against `permutedims(host, (2,1,3,4))`."
function checkLE(a, host)
    back = LavaBackend()
    d = transposeLE(Ctx(back; ws = Workspace(back)), a)
    KA.synchronize(back)
    Array(d) == permutedims(host, (2, 1, 3, 4))
end

@testset "transposeLE" begin
    back = LavaBackend()
    E, L, H, B = 72, 64, 4, 3          # E = 72 is not a multiple of the 32 tile

    @testset "a bare device array" begin
        h = Float16.(reshape(1:(E*L*H*B), E, L, H, B))
        a = KA.allocate(back, Float16, E, L, H, B); copyto!(a, h)
        @test stridedroot(a) !== nothing
        @test checkLE(a, h)
        a = nothing; GC.gc()
    end

    @testset "permute over reshape over VIEW — the stack attention actually gets" begin
        # Lava specialises `view(A, :, :, :, :, i)` to return a `LavaArray` that
        # already carries the offset, so a trailing scalar index does NOT produce
        # a `SubArray`. Slicing a middle axis does, and that is the case the
        # offset arithmetic exists for.
        hbig = Float16.(reshape(1:(E * H * 3 * L * B), E, H, 3, L, B))
        big = KA.allocate(back, Float16, E, H, 3, L, B)
        copyto!(big, hbig)
        for slice in (1, 2, 3)
            v = view(big, :, :, slice, :, :)
            @test v isa SubArray                       # a real view, not folded away
            r = reshape(v, E, H, L, B)
            a = PermutedDimsArray(r, (1, 3, 2, 4))     # -> (E, L, H, B)
            root = stridedroot(a)
            @test root !== nothing
            @test root[2] == (slice - 1) * E * H       # LinearIndices offset
            # Reference built on the HOST: `Array(a)` on this stack falls back to
            # scalar indexing, which the GPU array guard rejects.
            ha = permutedims(reshape(hbig[:, :, slice, :, :], E, H, L, B), (1, 3, 2, 4))
            @test checkLE(a, ha)
        end
        big = nothing; GC.gc()
    end

    @testset "Float32 as well as Float16" begin
        h = Float32.(reshape(1:(E*L*H*B), E, L, H, B))
        a = KA.allocate(back, Float32, E, L, H, B); copyto!(a, h)
        @test checkLE(a, h)
        a = nothing; GC.gc()
    end

    @testset "an unsupported wrapper falls back rather than reading garbage" begin
        a = KA.allocate(back, Float16, E, L, H, B); fill!(a, Float16(1))
        # A broadcast is not a strided view of anything.
        @test stridedroot(Base.Broadcast.broadcasted(identity, a)) === nothing
        a = nothing; GC.gc()
    end
end
