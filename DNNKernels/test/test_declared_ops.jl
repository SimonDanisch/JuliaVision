# The declared ops, against the definitions they replace.
#
# Each of these was a `runop!` that launched immediately and allocated its own
# destination; declared, the destination is the planned buffer and the launch is
# a `dispatch!` the graph holds. So what has to be checked is the ARITHMETIC,
# because the index math moved: `runop!`'s `repeat` indexed a Cartesian index
# modulo the source extent and its `sum` was `Base.sum(a; dims)`, and both are
# now a linear id decomposed by hand inside a macro-free kernel.
#
# Against `Base` and not against a golden array, on purpose. `repeat(ah, 2, 3)`
# and `sum(abs2, xh; dims = 2)` are the definitions; a recorded expectation
# would pin whatever the kernel did the first time it ran.
#
# Over every backend rather than `LavaBackend()`, which is what most of this
# suite still hardcodes. `sumdims!`'s loop bound and `bnstats!`'s stride
# arithmetic are the kind of thing that works on one device and not another, and
# the point of writing a kernel macro-free is that one source runs on all of
# them.

using Test
import DNNKernels
import Mantle

const DK = DNNKernels
const MM = Mantle

function declaredops(dev, label)
    @testset "declared shape and norm ops — $label" begin
        @testset "repeat tiles its source" begin
            ah = reshape(collect(Float32, 1:12), 4, 3)
            od = (8, 9)
            g = MM.Graph(dev)
            out = MM.Transient.Buffer(g, Float32, od)
            a = MM.Buffer(dev, ah)
            MM.compute!(g, "repeat") do p
                MM.dispatch!(p, DK.tilecopy!,
                             (MM.use(p, out; write = true), od,
                              MM.use(p, a; read = true), size(ah)), prod(od))
            end
            pl = MM.Plan(g)
            MM.record!(pl); MM.run!(pl); MM.waitidle(dev)
            @test Array(MM.storage(out)) == repeat(ah, 2, 3)
            MM.free!(pl)
        end

        # Two axes, because the bug a single reduced axis cannot show is a
        # reduced axis that is not the last one: the kept coordinates and the
        # reduced ones are interleaved in the linear index, and only a middle
        # axis makes `base` and the inner loop disagree if either is wrong.
        @testset "sum reduces the axes the output holds as one" begin
            xh = reshape(collect(Float32, 1:(4 * 3 * 2 * 5 * 3)), 4, 3, 2, 5, 3)
            id = size(xh)
            for (kd, dims, f, ref) in
                (((4, 3, 2, 5, 1), 5, identity, sum(xh; dims = 5)),
                 ((4, 1, 2, 5, 3), 2, abs2, sum(abs2, xh; dims = 2)),
                 ((1, 3, 1, 5, 3), (1, 3), identity, sum(xh; dims = (1, 3))))
                g = MM.Graph(dev)
                out = MM.Transient.Buffer(g, Float32, dropdims(ref; dims = dims) |> size)
                x = MM.Buffer(dev, xh)
                MM.compute!(g, "sum") do p
                    MM.dispatch!(p, DK.sumdims!,
                                 (MM.use(p, out; write = true), kd,
                                  MM.use(p, x; read = true), id, f), prod(kd))
                end
                pl = MM.Plan(g)
                MM.record!(pl); MM.run!(pl); MM.waitidle(dev)
                @test Array(MM.storage(out)) == dropdims(ref; dims = dims)
                MM.free!(pl)
            end
        end

        # Training mode, so the statistics are the batch's. The channel axis is
        # torch's dim 1, which in the reversed shape is `ndims - 1` — here the
        # third of four, so `cstride` and `nouter` are both greater than one and
        # a kernel that confused them would still pass on a 3-D input.
        @testset "batch norm computes its own statistics" begin
            W, H, C, N = 5, 4, 3, 2
            bh = reshape(collect(Float32, 1:(W * H * C * N)) ./ 7, W, H, C, N)
            gh = Float32[2, 3, 4]
            beh = Float32[-1, 0, 1]
            eps = 1.0f-5
            cstride, nouter = W * H, N

            g = MM.Graph(dev)
            o = MM.Transient.Buffer(g, Float32, (W, H, C, N))
            mu = MM.Transient.Buffer(g, Float32, C)
            ivs = MM.Transient.Buffer(g, Float32, C)
            xb, gb, bb = MM.Buffer(dev, bh), MM.Buffer(dev, gh), MM.Buffer(dev, beh)
            MM.compute!(g, "bn.stats") do p
                MM.dispatch!(p, DK.bnstats!,
                             (MM.use(p, mu; write = true), MM.use(p, ivs; write = true),
                              MM.use(p, xb; read = true), cstride, C, nouter, eps), C)
            end
            MM.compute!(g, "bn") do p
                MM.dispatch!(p, DK.bnapply!,
                             (MM.use(p, o; write = true), MM.use(p, xb; read = true),
                              MM.use(p, mu; read = true), MM.use(p, ivs; read = true),
                              MM.use(p, gb; read = true), MM.use(p, bb; read = true),
                              length(bh), cstride, C), length(bh))
            end
            pl = MM.Plan(g)
            MM.record!(pl); MM.run!(pl); MM.waitidle(dev)

            rd = (1, 2, 4)
            n = length(bh) ÷ C
            mu_ref = sum(bh; dims = rd) ./ n
            d = bh .- mu_ref
            iv_ref = 1 ./ sqrt.(sum(abs2, d; dims = rd) ./ n .+ eps)
            want = d .* (iv_ref .* reshape(gh, 1, 1, C, 1)) .+ reshape(beh, 1, 1, C, 1)
            @test Array(MM.storage(mu)) ≈ vec(mu_ref)
            @test Array(MM.storage(ivs)) ≈ vec(iv_ref)
            @test Array(MM.storage(o)) ≈ want

            # The second pass reads what the first wrote, which is why they are
            # two passes: `Barriers` derives the wait from the uses. A replay
            # has to hold that too, so run it twice.
            fill!(MM.storage(o), 0.0f0)
            MM.run!(pl); MM.waitidle(dev)
            @test Array(MM.storage(o)) ≈ want
            MM.free!(pl)
        end
    end
end

let bes = MM.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the declared ops"
    for be in bes
        declaredops(MM.Device(be), string(nameof(typeof(be))))
    end
end
