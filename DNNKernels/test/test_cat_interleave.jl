"""
`cat` along an axis that is innermost in the output.

The general path writes one part per dispatch through `blockcopy!`. Where the
joined axis is the innermost one and each part owns a single slot of it, that
makes every store touch every OTHER element: half of a cache line per
dispatch, and the other half on the next one. It is how a rotary embedding
exports -- `stack((-x2, x1), -1)` -- and on Qwen-Image 2.1's `(1, 4118, 32,
64)` fp32 pair the four dispatches were 12.6 ms of a 222 ms layer against 1.2
ms for `interleave2!`.

Checked here: the values (against the interleave written out by hand), that the
fast path is the one that ran, and that the shapes it must NOT take are still
the general path's.
"""

using Test, DNNKernels, Mantle

const DKI = DNNKernels

catbuf(id, kind, shape, T) =
    DKI.Buffer(id, kind, Any[shape...], T, "", (0, 3), "", "", Dict{String,Any}())

"""`cat(parts, dim)` as a graph, with `nparts` inputs of `pshape`."""
function catgraph(pshape, dim, nparts; T = Float32)
    oshape = collect(pshape)
    oshape[end + 1 + dim] = pshape[end + 1 + dim] * nparts   # dim is torch, negative
    ins = ["p$j" for j in 1:nparts]
    bufs = Dict{String,DKI.Buffer}(id => catbuf(id, :external, pshape, T) for id in ins)
    bufs["y"] = catbuf("y", :transient, Tuple(oshape), T)
    op = DKI.Op("cat", "cat.default", ins, "y",
                Dict{String,Any}("arg1" => dim, "arg0" => ["\$$id" for id in ins]))
    DKI.Graph("cat", String[], ins, ["y"], bufs, [ins; "y"], [op])
end

@testset "cat on the innermost axis is one interleaving dispatch" begin
    backend = Mantle.LavaBackend()
    rows, k = 5, 3            # torch (rows, k, 1) x2 -> (rows, k, 2)
    g = catgraph((rows, k, 1), -1, 2)
    model = DKI.Model(Dict("cat" => g), Dict{String,Any}(), backend, 1, 1, 1)
    # Julia order is reversed: each part is (1, k, rows), the output (2, k, rows).
    ah = reshape(Float32[i for i in 1:(rows * k)], 1, k, rows)
    bh = reshape(Float32[-i for i in 1:(rows * k)], 1, k, rows)
    got = Array(only(DKI.call(model, "cat", DKI.toback(backend, ah),
                              DKI.toback(backend, bh); dims = (;))))
    @test size(got) == (2, k, rows)
    want = similar(got)
    want[1, :, :] .= ah[1, :, :]
    want[2, :, :] .= bh[1, :, :]
    @test got == want

    # It is the interleaving dispatch and not two block copies.
    plan = DKI.planfor(model.device, model.graphs["cat"], model.weights, (;);
                       profile = true)
    DKI.replay!(plan, "cat", (DKI.toback(backend, ah), DKI.toback(backend, bh)))
    names = [String(t.name) for t in Mantle.timings(plan.plan)]
    @test "cat.ilv" in names
    @test !("cat.1" in names)
end

@testset "the shapes it must not take" begin
    backend = Mantle.LavaBackend()
    # Three parts is not this kernel, and neither is a join on any other axis.
    for (pshape, dim, nparts) in (((4, 3, 1), -1, 3), ((4, 3, 2), -2, 2))
        g = catgraph(pshape, dim, nparts)
        model = DKI.Model(Dict("cat" => g), Dict{String,Any}(), backend, 1, 1, 1)
        parts = [DKI.toback(backend, reshape(Float32[j * 100 + i for i in 1:prod(pshape)],
                                             reverse(pshape)...))
                 for j in 1:nparts]
        plan = DKI.planfor(model.device, model.graphs["cat"], model.weights, (;);
                           profile = true)
        out = only(DKI.replay!(plan, "cat", Tuple(parts)))
        names = [String(t.name) for t in Mantle.timings(plan.plan)]
        @test !("cat.ilv" in names)
        @test "cat.1" in names
        # And it still concatenates: every part's elements are all in there.
        got = vec(Array(out))
        for p in parts
            @test all(v -> v in got, vec(Array(p)))
        end
    end
end

# Which parts land on ONE run of the destination, which is what lets `cat`,
# `slice_scatter` and `constant_pad_nd` skip `blockcopy!`'s coordinate
# arithmetic. That arithmetic cost 9x what moving the bytes did, so the
# predicate is worth being exactly right about, and the case it must refuse
# looks almost identical to the case it must take.
@testset "a part is one run of the destination, or it is not" begin
    cs = DKI.contiguousslab

    # The `cat` this exists for: joined on the outermost non-singleton axis.
    @test cs((128, 32, 4118, 1), (128, 32, 4096, 1), (0, 0, 22, 0)) == 22 * 128 * 32
    @test cs((128, 32, 4118, 1), (128, 32, 22, 1), (0, 0, 0, 0)) == 0
    # The whole destination, which is a copy.
    @test cs((4, 5), (4, 5), (0, 0)) == 0
    # A 1-D run, and one that starts partway in.
    @test cs((4, 1, 5), (4, 1, 2), (0, 0, 3)) == 12

    # Narrow on a LOW axis is a stride, not a run, however small the gap: this
    # is `out[1:2, :]`, which skips two elements every two.
    @test cs((4, 4), (2, 4), (0, 0)) === nothing
    # Same shape as the rotary interleave, which really is every other element.
    @test cs((2, 64, 32), (1, 64, 32), (0, 0, 0)) === nothing
    # Narrow on two axes at once is never one run.
    @test cs((4, 4, 4), (4, 2, 2), (0, 0, 0)) === nothing
    # An offset on an axis the part spans in full moves every run, not one.
    @test cs((4, 4), (4, 2), (1, 0)) === nothing
    # The conv weight pad: rows padded, columns not.
    @test cs((72, 128), (64, 128), (0, 0)) === nothing
end
