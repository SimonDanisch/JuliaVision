# What a declared view costs, decided by arithmetic rather than by a copy.
#
# `reshapestrides` is the one step of that arithmetic with a wrong answer
# available: a reshape of strided memory is free only when every output axis is a
# run of the parent's, and a stride returned where it is not would read the right
# number of the wrong elements. Every case below is a shape a graph in this tree
# actually produces, plus the refusals.

using Test
import DNNKernels
const DK = DNNKernels

colst(d) = ntuple(k -> prod(d[1:k-1]; init = 1), length(d))

"""Read `ps`-shaped `a` at `pst` strides and `od` coordinates, as the kernel does."""
function strided_read(a, ps, pst, od, st)
    out = Array{eltype(a)}(undef, od)
    for I in CartesianIndices(od)
        off = sum((Tuple(I)[k] - 1) * st[k] for k in eachindex(od); init = 0)
        out[I] = a[1 + off]
    end
    return out
end

@testset "a reshape of DENSE memory is always free" begin
    ps = (4, 3, 2)
    for od in ((24,), (4, 6), (12, 2), (2, 2, 3, 2), (4, 3, 2, 1))
        st = DK.reshapestrides(ps, colst(ps), od)
        # A run, which is the property `viewfor` asks about. Not `colst(od)`
        # itself: an axis of extent 1 is never indexed past 0, so its stride is
        # free and `(4, 3, 2, 1)` is entitled to end in anything.
        @test st !== nothing
        @test DK.isdenserun(od, st)
        @test all(od[k] == 1 || st[k] == colst(od)[k] for k in eachindex(od))
    end
end

@testset "isdenserun asks about the run, not the tuple" begin
    @test DK.isdenserun((4, 3), (1, 4))
    @test DK.isdenserun((4, 3, 1), (1, 4, 99))      # the 1-axis stride is free
    @test DK.isdenserun((1, 4, 3), (7, 1, 4))
    @test !DK.isdenserun((4, 3), (3, 1))            # permuted
    @test !DK.isdenserun((4, 3), (1, 5))            # a window inside a wider parent
end

@testset "an extent-1 axis costs nothing, dropped or inserted" begin
    ps, pst = (4, 1, 3), (1, 99, 4)          # 99 is unreachable: the axis is 1 wide
    @test DK.reshapestrides(ps, pst, (4, 3)) == (1, 4)
    @test DK.reshapestrides((4, 3), (1, 4), (4, 1, 3)) == (1, 4, 4)
    @test DK.reshapestrides((4, 3), (1, 4), (1, 4, 3)) == (1, 1, 4)
end

@testset "merging needs the axes to be contiguous" begin
    # Contiguous: the second axis picks up exactly where the first ends.
    @test DK.reshapestrides((4, 3), (1, 4), (12,)) == (1,)
    # Not contiguous: a permuted parent, where 12 elements are not one run.
    @test DK.reshapestrides((4, 3), (3, 1), (12,)) === nothing
    # Nor is a slice's parent, whose axis is 4 wide inside a 5-wide one.
    @test DK.reshapestrides((4, 3), (1, 5), (12,)) === nothing
end

@testset "splitting an axis is free, and the extents have to divide" begin
    @test DK.reshapestrides((12,), (2,), (4, 3)) == (2, 8)
    @test DK.reshapestrides((12,), (2,), (3, 4)) == (2, 6)
    @test DK.reshapestrides((12,), (2,), (5, 2)) === nothing   # 5 does not divide 12
end

@testset "the strides it answers read the elements the reshape names" begin
    # The permuted-then-reshaped shape SAM 2's attention output takes: a rank-4
    # parent read at rank 2, where the merge IS contiguous.
    a = collect(1:(4 * 3 * 2))
    ps, pst = (4, 3, 2), colst((4, 3, 2))
    for od in ((12, 2), (4, 6), (24,))
        st = DK.reshapestrides(ps, pst, od)
        @test st !== nothing
        @test vec(strided_read(a, ps, pst, od, st)) == vec(reshape(a, od))
    end
end

@testset "a strided parent reads its own elements, not the dense ones" begin
    # A 4x3 window inside a 5x3 parent: the rows are 5 apart, so a reshape that
    # merges them is refused and one that keeps them is not.
    a = collect(1:15)
    ps, pst = (4, 3), (1, 5)
    @test DK.reshapestrides(ps, pst, (4, 3)) == (1, 5)
    got = strided_read(a, ps, pst, (4, 3), (1, 5))
    @test got == [a[1 + (i - 1) + 5 * (j - 1)] for i in 1:4, j in 1:3]
    @test DK.reshapestrides(ps, pst, (2, 2, 3)) == (1, 2, 5)
end
