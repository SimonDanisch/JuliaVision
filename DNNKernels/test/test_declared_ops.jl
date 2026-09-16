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
            MM.dispatch!(g, DK.tilecopy!, (out, od, a, size(ah)), prod(od);
                         name = "repeat")
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
                MM.dispatch!(g, DK.sumdims!, (out, kd, x, id, f), prod(kd);
                             name = "sum")
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
            MM.dispatch!(g, DK.bnstats!,
                         (mu, ivs, xb, cstride, C, nouter, eps), C;
                         name = "bn.stats")
            MM.dispatch!(g, DK.bnapply!,
                         (o, xb, mu, ivs, gb, bb, length(bh), cstride, C),
                         length(bh); name = "bn")
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

        # ── one `ew!` at every arity ─────────────────────────────────────────
        #
        # It was `ew1!`/`ew2!`/`ew3!`, one kernel per operand count, and the
        # fourth was refused by name. One variadic kernel over a tuple of
        # operands needed three walks in Mantle to agree that a tuple argument
        # is the argument list grouped rather than a value of its own — see
        # `Mantle.nestinglevels` and `Mantle/test/test_argument_packing.jl`.
        #
        # Three arities in one loop, which is the point: one compiled kernel per
        # arity from one source, and the middle operand is broadcast on two axes
        # so a wrong stride shows up as a wrong number rather than a crash.
        #
        # The `gather` walk is also pinned here. Its base case first read
        # `gather(::Tuple{}, ::Tuple{}, lin, od)`, with `od` untyped, which is
        # AMBIGUOUS against the recursive method rather than more specific than
        # it: more specific in the operands, less specific in `od`. An ambiguity
        # inside a kernel is not a `MethodError` at the call site, it is a
        # `jl_f_throw_methoderror` in the IR that the backend reports as "method
        # lookup failure" in `gather`, with nothing about dispatch anywhere in
        # the message.
        @testset "one ew! covers every arity" begin
            od = (4, 3, 2)
            ah = reshape(collect(Float32, 1:prod(od)), od)
            bh = reshape(Float32[10, 20, 30], 1, 3, 1)   # broadcast on axes 1, 3
            ch = fill(0.5f0, od)
            a = MM.Buffer(dev, ah); b = MM.Buffer(dev, bh); c = MM.Buffer(dev, ch)
            for (ops, hostops, f) in (((a,), (ah,), x -> 2x),
                                      ((a, b), (ah, bh), (x, y) -> x + y),
                                      ((a, b, c), (ah, bh, ch),
                                       (x, y, z) -> (x + y) * z))
                g = MM.Graph(dev)
                out = MM.Transient.Buffer(g, Float32, od)
                o, st = DK.operandtuples(od, ops)
                MM.dispatch!(g, DK.ew!, (out, od, o, st, f), prod(od);
                             name = "ew")
                pl = MM.Plan(g)
                MM.record!(pl); MM.run!(pl); MM.waitidle(dev)
                @test reshape(Array(MM.storage(out)), od) == f.(hostops...)
                MM.free!(pl)
            end
            # `gather`'s base case, asked directly: ambiguous methods answer
            # here too, and the message names dispatch when the kernel's does
            # not.
            @test DK.gather((), (), 0, od) === ()
        end
    end
end

let bes = MM.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the declared ops"
    for be in bes
        declaredops(MM.Device(be), string(nameof(typeof(be))))
    end
end
