"""
`index.Tensor` with more than one index tensor: which cases are an outer product
and which are genuinely paired.

## The regression this exists to stop

Torch pairs index tensors by broadcasting them against each other; Julia's
`x[i, j]` crosses them. Those are different functions, and `runop!` used to spell
the torch one as the Julia one — right for a single index, wrong for two, and it
surfaced as a `BoundsError` in Kokoro's ALBERT mask rather than as a wrong
answer.

The fix required **every** axis to be indexed, which is true of Kokoro's mask and
false of SAM 2's position-embedding interpolation: sixteen gathers of a
`(7, 7, 144, 1)` table by a `(256,)` and a `(1, 256)`, two axes indexed and two
sliced. `Model` threw while constant-folding them, so **SAM 2 did not load at
all** between `6ddff7a` and this. Its own suite caught it and nothing else did,
because the only DNNKernels tests that build a real graph use MatAnyone's.

Both are now handled, and the point of this file is that the two are told apart
by a property rather than by a model name.

## What is actually being asserted

That for a *separable* index set — each index tensor varying along its own
broadcast axis, size 1 in the others — the outer product Julia computes **is**
the broadcast pairing torch computes. That is the licence for keeping the case on
the device as a `view` instead of gathering it on the host, so it is worth
checking against an explicit per-element loop rather than against either
implementation.
"""

using Test, DNNKernels, SAM2Runner, KokoroRunner
const DK = DNNKernels

"""
Torch's semantics, written out: broadcast the index tensors, then take one
element per broadcast position, leaving the sliced axes whole. Deliberately
naive — it is the definition, not an implementation.
"""
function torchindex(x, dims, arrs, bs)
    sliced = [d for d in 1:ndims(x) if !(d in dims)]
    out = Array{eltype(x)}(undef, bs..., size(x)[sliced]...)
    for J in CartesianIndices(bs), S in CartesianIndices(Tuple(size(x)[sliced]))
        I = Vector{Int}(undef, ndims(x))
        for (i, d) in enumerate(dims)
            # Broadcasting: an index tensor of extent 1 on an axis repeats along it.
            K = CartesianIndex(ntuple(k -> size(arrs[i], k) == 1 ? 1 : J[k], length(bs)))
            I[d] = arrs[i][K]
        end
        for (i, d) in enumerate(sliced)
            I[d] = S[i]
        end
        out[J, S] = x[I...]
    end
    out
end

@testset "index.Tensor, several index tensors" begin
    @testset "indexseparable tells the two apart" begin
        # SAM 2's position-embedding interpolation, in Julia's reversed layout:
        # a (7,7,144,1) table, dims 1 and 2 indexed by a (256,) and a (1,256).
        @test DK.indexseparable([1, 2], [rand(1:7, 256), rand(1:7, 1, 256)])
        # Kokoro's ALBERT mask: a 2-d input, both axes indexed, but the broadcast
        # has FOUR axes — the result outranks the input and no cross can give it.
        @test !DK.indexseparable([1, 2], [ones(Int, 1, 1, 1, 1), ones(Int, 30, 1, 1, 1)])
        # Equal-shaped 1-d indices are the paired case proper: one element per
        # position, not `n^2` of them.
        @test !DK.indexseparable([1, 2], [rand(1:4, 8), rand(1:4, 8)])
        # Two indices varying along the SAME broadcast axis pair, they do not cross.
        @test !DK.indexseparable([1, 2], [rand(1:4, 8, 1), rand(1:4, 8, 1)])
        # Non-adjacent advanced indices: torch moves the gathered axes to the
        # front and `view` leaves them in place, so this must decline even though
        # the index shapes themselves are separable.
        @test !DK.indexseparable([1, 3], [rand(1:4, 5), rand(1:4, 1, 6)])
        # Adjacent but not leading is fine — `view` and torch agree there.
        @test DK.indexseparable([2, 3], [rand(1:4, 5), rand(1:4, 1, 6)])
        # `dims` MUST arrive ascending. Reversing the axes of a torch spec walks
        # the Julia dims backwards, so `runop!` sorts before calling this — and
        # unsorted the pairing is transposed and this reads false, which is
        # exactly how SAM 2 failed on the first attempt at the fix.
        @test !DK.indexseparable([2, 1], [rand(1:7, 1, 256), rand(1:7, 256)])
    end

    @testset "the outer product IS the broadcast, when separable" begin
        # This is the claim the device path rests on.
        x = reshape(collect(1:(7 * 7 * 5 * 2)), 7, 7, 5, 2)
        r = rand(1:7, 256)
        c = rand(1:7, 1, 256)
        want = torchindex(x, [1, 2], [r, c], (256, 256))
        got = x[vec(r), vec(c), :, :]
        @test size(got) == size(want) == (256, 256, 5, 2)
        @test got == want

        # ...and it is NOT the broadcast when the indices are not separable, which
        # is what makes the predicate load-bearing rather than decorative.
        a, b = rand(1:7, 8), rand(1:7, 8)
        @test !DK.indexseparable([1, 2], [a, b])
        @test size(torchindex(x, [1, 2], [a, b], (8,))) == (8, 5, 2)
        @test size(x[a, b, :, :]) == (8, 8, 5, 2)
    end

    # Against the REAL graphs, because the regression was not that the predicate
    # was wrong — it was that nobody asked whether a shipped model produced the
    # form being refused. This is the assertion that would have caught it, and it
    # costs no device: every shape it needs is declared in the graph.
    @testset "every index.Tensor in a shipped graph is handled" begin
        checked = 0
        for (pkg, ready, graphs) in
            (("SAM 2", SAM2Runner.ready(),
              () -> [SAM2Runner.sam2graph(n) for n in ("sam2_encoder", "sam2_decoder")]),
             ("Kokoro", KokoroRunner.ready(),
              () -> [DK.loadgraph(joinpath(KokoroRunner.assetdir(), "$n.json"))
                     for n in ("kokorotext", "kokorovoc")]))
            if !ready
                @info "$pkg's artifact is not installed; skipping"
                @test_skip ready
                continue
            end
            for g in graphs(), op in g.ops
                op.aten == "index.Tensor" || continue
                x = g.buffers[op.ins[1]]
                n = length(x.shape)
                dims, shapes = Int[], Vector{Int}[]
                for (k, e) in enumerate(op.attrs["arg1"])
                    e === nothing && continue
                    push!(dims, n - k + 1)
                    # Reversed, like the runtime sees it. Symbolic extents are
                    # strings; any concrete value keeps the RANK right, which is
                    # all `indexseparable` reads besides which axes are size 1.
                    b = g.buffers[String(e)[2:end]]
                    push!(shapes, [v isa Integer ? Int(v) : 7 for v in reverse(b.shape)])
                end
                checked += 1
                length(dims) == 1 && continue
                # Ascending, exactly as `runop!` sorts them before asking.
                p = sortperm(dims)
                dims, shapes = dims[p], shapes[p]
                arrs = [ones(Int, s...) for s in shapes]
                # Handled means one of the two paths takes it: the separable
                # outer product on the device, or the paired host gather, which
                # needs every axis indexed.
                @test DK.indexseparable(dims, arrs) || length(dims) == n
            end
        end
        @test checked > 0        # a walk that found nothing is not a pass
    end

    @testset "a size-1 index axis still broadcasts" begin
        # The degenerate separable case: one index selects a single row, the other
        # a range. Torch gives (1, m, ...) and so must the cross.
        x = reshape(collect(1:(4 * 4 * 3)), 4, 4, 3)
        r = [2]
        c = reshape([1, 3, 4], 1, 3)
        @test DK.indexseparable([1, 2], [r, c])
        @test x[vec(r), vec(c), :] == torchindex(x, [1, 2], [r, c], (1, 3))
    end
end
