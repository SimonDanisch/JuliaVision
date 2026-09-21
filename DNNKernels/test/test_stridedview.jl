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

# `stridedwindow` is the other half: what `ew!` can address without a copy. The
# strides it answers have to mean the same thing `bcstrides` means for a dense
# operand, because `ew!` cannot tell a window from a buffer.
@testset "a strided operand broadcasts the way a dense one does" begin
    # A descriptor over a host vector: `stridedwindow` only reads `parent`'s
    # length and hands back a view of it.
    root = collect(Float32, 1:600)
    s = DK.StridedOperand(root, (1, 1, 64), (0, 0, 1), 0)
    w = DK.stridedwindow(s, (128, 128, 64, 1))
    @test w !== nothing
    # Leading-aligned, zero on a broadcast axis and past the operand's rank,
    # which is `bcstrides`' rule to the letter.
    @test w[2] == DK.bcstrides((128, 128, 64, 1), (1, 1, 64))
    # Refusals: an operand that outranks its output has no window, and an extent
    # that neither matches nor is 1 is not a broadcast.
    @test DK.stridedwindow(s, (128, 128)) === nothing
    @test DK.stridedwindow(DK.StridedOperand(root, (3, 4), (1, 3), 0), (5, 4)) === nothing
    # And a window that would read past the root is refused rather than clamped.
    @test DK.stridedwindow(DK.StridedOperand(root, (4, 4), (1, 200), 0), (4, 4)) === nothing
end

@testset "transpose-cast recognises only dense batched matrix transposes" begin
    # `(144, 65536, 1)` source planes read as dense `(65536, 144, 1)` output:
    # SAM's spatial dimensions remain split here, which must give the same M.
    @test DK.transposecastshape((256, 256, 144, 1),
                                (144, 144 * 256, 1, 144 * 256 * 256), 0) ==
          (256 * 256, 144, 1)
    @test DK.transposecastshape((64, 32, 7, 3),
                                (7, 7 * 64, 1, 7 * 64 * 32), 0) ==
          (64 * 32, 7, 3)
    @test DK.transposecastshape((64, 32, 7), (7, 7 * 64, 1), 1) === nothing
    @test DK.transposecastshape((64, 32, 7), (7, 999, 1), 0) === nothing
    @test DK.transposecastshape((64, 32, 7), (1, 64, 64 * 32), 0) === nothing
end

@testset "a contiguous innermost axis gets the vectorised copy" begin
    # `ast[1] == 1` is the whole condition, and then the widest `V` that divides the
    # run. 98 of SAM 2.1's 104 strided copies qualify; their innermost extents are
    # 72, 144, 288, 576 and 1152.
    @test DK.stridedrunwidth((144, 8, 32, 8, 32, 1), (1, 144, 9216, 1152, 294912, 1)) == 8
    @test DK.stridedrunwidth((72, 2, 64, 1024), (1, 4608, 72, 9216)) == 8
    @test DK.stridedrunwidth((4, 3, 2), (1, 4, 12)) == 4
    # Not a run in the SOURCE, so there is nothing to widen — this is the case the
    # kernel would read wrong rather than slowly.
    @test DK.stridedrunwidth((144, 256, 256, 1), (65536, 1, 256, 9437184)) === nothing
    # A run no vector width divides.
    @test DK.stridedrunwidth((6, 4), (1, 6)) === nothing
    @test DK.stridedrunwidth((), ()) === nothing
end

# …and that it copies the same bytes. Bit-for-bit, because a copy has no tolerance:
# the two kernels differ only in how many elements one thread addresses, so any
# difference at all is the vectorised one reading outside its run.
import Mantle
const MMs = Mantle

"""Both kernels over one descriptor, through a declared plan, read back."""
function bothcopies(dev, od, ast, off, T)
    n = prod(od)
    hi = off + sum((od[k] - 1) * ast[k] for k in eachindex(od); init = 0)
    # Modulo in INTEGERS and then converted: `Float16(70000)` is `Inf`, and `Inf % x`
    # is `NaN`, so the obvious spelling makes every comparison below false — including
    # the scalar kernel against its own definition.
    host = T.((1:(hi + 1)) .% 997)
    src = MMs.Buffer(dev, host)
    got = map((:scalar, :run)) do which
        out = MMs.Buffer(dev, T, od)
        g = MMs.Graph(dev)
        if which === :scalar
            MMs.dispatch!(g, DK.stridedcopy32!,
                          (out, MMs.broadcastextents(od), src, map(Int32, ast),
                           Int32(off), Int32(n)), n; group = 256, name = "c")
        else
            V = DK.stridedrunwidth(od, ast)
            MMs.dispatch!(g, DK.stridedcopyrun32!,
                          (out, MMs.broadcastextents(od), src, map(Int32, ast),
                           Int32(off), Int32(n ÷ V), Val(V)), n ÷ V;
                          group = 256, name = "c")
        end
        p = MMs.Plan(g); MMs.record!(p); MMs.run!(p); MMs.waitidle(dev)
        a = Array(MMs.storage(out)); MMs.free!(p); a
    end
    # …and against the definition, so "both agree" cannot mean "both wrong".
    want = Array{T}(undef, od)
    for I in CartesianIndices(od)
        want[I] = host[1 + off + sum((Tuple(I)[k] - 1) * ast[k]
                                     for k in eachindex(od); init = 0)]
    end
    (got[1], got[2], want)
    end

for be in MMs.eachbackend()
    copydev = MMs.todevice(be)
    @testset "the vectorised copy moves the same bytes — $(nameof(typeof(be)))" begin
        # SAM 2.1's two shapes, plus an offset one and a `V = 4` run.
        for (od, ast, off) in
                (((144, 8, 32, 8, 32, 1), (1, 144, 9216, 1152, 294912, 1), 0),
                 ((72, 2, 64, 1024), (1, 4608, 72, 9216), 0),
                 ((72, 16, 8, 16), (1, 1728, 72, 27648), 576),
                 ((12, 5, 3), (1, 12 * 7, 12 * 7 * 5), 0))
            scalar, run, want = bothcopies(copydev, od, ast, off, Float16)
            @test scalar == want
            @test run == want
            @test run == scalar
        end

# What `toback` hands to `copyto!`, which is the same question one level up.
#
# A compact Qwen-Image 2.1 checkpoint arrives as 224 column slices of mmap'd
# `Matrix{Int8}`, 6.5 GiB of them, and a `SubArray` is not a `DenseArray` — so
# they were all `collect`ed into anonymous memory first, which is exactly the
# mmap defeat `toback`'s own comment is about. The test is that contiguity is
# decided by the STRIDES and not by the index types: `view(A, 1:3, 1:5)` of a
# 10x10 has a `UnitRange` first index and its columns are still ten apart.
@testset "a contiguous view uploads without being collected" begin
    A = reshape(Int8.(1:(8 * 6)), 8, 6)
    wrapped(x) = (s = DK.uploadsource(x); s isa Array && !(x isa Array) && pointer(s) == pointer(x))

    @test DK.uploadsource(A) === A                      # already dense
    let v = vec(A); @test DK.uploadsource(v) === v; end
    @test wrapped(view(A, :, 2:4))                      # whole columns
    @test wrapped(view(A, :, :))
    @test !wrapped(view(A, 2:5, 2:4))                   # a window, not a run
    @test !wrapped(view(A, 1:2:8, :))                   # strided rows

    # And whichever branch it takes, the elements are the view's own.
    for v in (A, view(A, :, 2:4), view(A, :, :), view(A, 2:5, 2:4),
              view(A, 1:2:8, :), vec(A))
        @test collect(DK.uploadsource(v)) == collect(v)
        @test size(DK.uploadsource(v)) == size(v)
    end
end
