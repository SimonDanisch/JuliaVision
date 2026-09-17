"""
The non-overlapping transposed convolution, against the gather it replaces, and
the predicates that route every convolution.

`ConvTranspose2d` with stride equal to the kernel and no padding is not really a
convolution: the receptive fields do not overlap, so each output pixel comes from
exactly one input pixel and one weight slice, and the whole op is a GEMM plus a
depth-to-space interleave. SAM 2's mask decoder upsamples with two of them, and
they were 3.73 ms of an 8.44 ms decode on the gather.

Checked against `convtranspose2d` — the gather — rather than against a host
reference. What is at stake is that a *different route to the same answer*
agrees, and the gather is the answer.

**Declared, because there is no other way to launch these any more.** This file
drove `DK.Ctx(backend; ws = nothing)` and the immediate entry
points (`convolutiontranspose!`, `convolution!`, `convolution_coopmat!`), and
`Workspace` went with the interpreted run on 2026-09-15 — every one of those
takes its scratch from `ctx.ws`, so none of them can be called. The declared form
takes scratch from the graph (`scratch(emitctx, …)`, a transient the placer
aliases), which is the whole point: an op's working buffer is planned against the
rest of the graph rather than bump-allocated from an arena that resets per op.

Three numerical comparisons went with those entry points and are named at the
bottom, with what replaced each one. The PREDICATES are all still here and all
still live — `shufflecase`, `phasecase`, `onebyone` and `conv_coopmat_plan` are
what the emits route on, so a wrong answer from one of them sends a real graph
down the wrong path.
"""

using Test, DNNKernels, Lava, KernelAbstractions
using Mantle: LavaBackend
import Mantle
const KA = KernelAbstractions
const DK = DNNKernels
const MC = Mantle

"""
Run one transposed convolution as a declared plan, `route` deciding which form.

`:shuffle` is `emitconvtransposeshuffle!` — the GEMM and the interleave — and
`:gather` is the one dispatch of `convtranspose2d` that covers every case. Two
routes to the same answer, which is what this file compares; the emit picks
between them with `shufflecase` and a caller cannot ask for one, so the test
calls the two halves rather than the `emitop!` above them.
"""
function transposedplan(dev, route::Symbol, x, w, bias, od, stride)
    g = MC.Graph(dev)
    T = eltype(x)
    xr = MC.Buffer(dev, x)
    wr = MC.Buffer(dev, w)
    br = bias === nothing ? nothing : MC.Buffer(dev, bias)
    out = MC.Buffer(dev, T, od)
    # An `EmitCtx` with no ATen graph behind it: `scratch` and `dispatch!` are
    # all these two take from it, and neither reads `aten`. The op id names the
    # passes.
    op = DK.Op("t", "convolution.default", String[], "t", Dict{String,Any}())
    emitctx = DK.EmitCtx(DK.Graph("t", String[], String[], String[],
                                  Dict{String,DK.Buffer}(), String[], DK.Op[],
                                  Vector{Vector{String}}()),
                         g, dev, NamedTuple(), Dict{String,Any}("t" => out),
                         Set{String}(), Ref("t"),
                         # `make` is never reached here, so nothing is owned;
                         # the buffers above are this function's own.
                         Any[])
    if route === :shuffle
        DK.emitconvtransposeshuffle!(emitctx, op, xr, wr, br, out, stride)
    else
        DK.mapbody!(emitctx, op, DK.convtranspose2d, out, xr, wr, br,
                    Val(stride[1]), Val(stride[2]), Val(0), Val(0),
                    Val(1), Val(1), Val(1))
    end
    plan = MC.Plan(g)
    MC.record!(plan)
    MC.run!(plan)
    MC.waitidle(dev)
    got = Array(MC.storage(out))
    MC.free!(plan)
    return got
end

@testset "transposed convolution via GEMM" begin
    back = LavaBackend()
    dev = MC.Device(back)

    @testset "agrees with the gather" begin
        # The decoder's two shapes, plus small ones whose channel counts are not
        # multiples of the GEMM tile.
        for (Ci, Hi, Co, T) in ((256, 64, 64, Float16), (64, 128, 32, Float16),
                                (256, 64, 64, Float32), (64, 128, 32, Float32),
                                (32, 16, 8, Float16), (24, 8, 12, Float32))
            hx = T.(randn(Float32, Hi, Hi, Ci, 1) .* 0.1f0)
            hw = T.(randn(Float32, 2, 2, Co, Ci) .* 0.1f0)
            hb = T.(randn(Float32, Co) .* 0.1f0)
            od = (2Hi, 2Hi, Co, 1)
            r = transposedplan(dev, :gather, hx, hw, hb, od, (2, 2))
            g = transposedplan(dev, :shuffle, hx, hw, hb, od, (2, 2))
            rf, gf = Float32.(r), Float32.(g)
            # A shuffle that drops a sub-pixel phase leaves an exact lattice of
            # zeros, and a relative-error check averages straight over it. So
            # assert per phase — not "everything is nonzero", which fails
            # honestly: with random data a handful of fp16 outputs round to zero
            # (109 of 1 048 576 on the first shape here).
            for dx in 1:2, dy in 1:2
                @test any(!iszero, @view gf[dx:2:end, dy:2:end, :, :])
            end
            @test maximum(abs, gf .- rf) / maximum(abs, rf) <
                  (T === Float16 ? 5e-3 : 1e-5)
            GC.gc()
        end
    end

    @testset "only the non-overlapping case is taken" begin
        w22 = KA.allocate(back, Float32, 2, 2, 8, 8)
        w33 = KA.allocate(back, Float32, 3, 3, 8, 8)
        # stride == kernel, nothing else set: the case this exists for.
        @test DK.shufflecase(w22, (2,2), (0,0), (1,1), (0,0), 1)
        # A kernel wider than the stride overlaps, so the identity fails.
        @test !DK.shufflecase(w33, (2,2), (0,0), (1,1), (0,0), 1)
        # Padding, dilation, output padding and groups each break it.
        @test !DK.shufflecase(w22, (2,2), (1,1), (1,1), (0,0), 1)
        @test !DK.shufflecase(w22, (2,2), (0,0), (2,2), (0,0), 1)
        @test !DK.shufflecase(w22, (2,2), (0,0), (1,1), (1,1), 1)
        @test !DK.shufflecase(w22, (2,2), (0,0), (1,1), (0,0), 2)
        w22 = w33 = nothing; GC.gc()
    end

    @testset "an overlapping kernel is not the shuffle's" begin
        # 3x3 stride 2: the fields overlap, so `shufflecase`'s GEMM route must
        # not fire, and `emitconvtranspose!` sends it to the gather. `phasecase`
        # accepts it — it is the more general decomposition — and the emit does
        # NOT take that route, which is stated at the bottom of this file.
        w = KA.allocate(back, Float16, 3, 3, 8, 16)
        @test !DK.shufflecase(w, (2,2), (0,0), (1,1), (0,0), 1)
        @test DK.phasecase(w, (2,2), (0,0), (1,1), (0,0), 1)
        w = nothing; GC.gc()
    end
end

@testset "the convolution routing predicates" begin
    back = LavaBackend()

    @testset "only the 1x1 case is taken" begin
        w11 = KA.allocate(back, Float16, 1, 1, 8, 8)
        w33 = KA.allocate(back, Float16, 3, 3, 8, 8)
        @test DK.onebyone(w11, (1,1), (0,0), (1,1), 1)
        @test !DK.onebyone(w33, (1,1), (0,0), (1,1), 1)   # a real receptive field
        @test !DK.onebyone(w11, (2,2), (0,0), (1,1), 1)   # stride skips pixels
        @test !DK.onebyone(w11, (1,1), (1,1), (1,1), 1)   # padding adds rows
        @test !DK.onebyone(w11, (1,1), (0,0), (1,1), 2)   # grouped
        # Dilation is meaningless at 1x1 but an exported graph may still carry
        # it; refuse rather than assume.
        @test !DK.onebyone(w11, (1,1), (0,0), (2,2), 1)
        w11 = w33 = nothing; GC.gc()
    end

    @testset "the reduction-axis pad is refused when the waste is large" begin
        # `CRS = Cin*KH*KW` is the weight's own extent. A concatenated scalar
        # channel gives MatAnyone `Cin = 17`, which would round to 32 and pay 88%
        # waste to reach the tensor cores; `crspad` is where that line sits.
        dev = DK.caps(back)
        x = KA.allocate(back, Float16, 32, 32, 17, 1); fill!(x, Float16(0.1))
        w = KA.allocate(back, Float16, 1, 1, 17, 32); fill!(w, Float16(0.1))
        o = KA.allocate(back, Float16, 32, 32, 32, 1); fill!(o, Float16(0))
        # A keyword, not a global flipped and restored: the second call simply
        # asks a different question, and a failing `@test` between them can no
        # longer leave the policy changed for everything after.
        @test DK.conv_coopmat_plan(dev, o, x, w).reason === :crswaste   # 17 -> 32
        @test DK.conv_coopmat_plan(dev, o, x, w; crspad = 2.0) isa
              DK.ConvCoopMatPlan                            # ...only by policy
        # 1.0 admits nothing off the tile at all. SAM 2's
        # stem is `7x7x3`, `CRS = 147`, and padding it to 160 took it from 2.800
        # to 1.147 ms.
        w2 = KA.allocate(back, Float16, 7, 7, 3, 144); fill!(w2, Float16(0.1))
        x2 = KA.allocate(back, Float16, 256, 256, 3, 1); fill!(x2, Float16(0.1))
        o2 = KA.allocate(back, Float16, 64, 64, 144, 1); fill!(o2, Float16(0))
        @test DK.conv_coopmat_plan(dev, o2, x2, w2) isa DK.ConvCoopMatPlan
        @test DK.conv_coopmat_plan(dev, o2, x2, w2; crspad = 1.0).reason ===
              :crswaste
        x = w = o = w2 = x2 = o2 = nothing; GC.gc()
    end

    @testset "the overlapping transposed decomposition, as a predicate" begin
        # Every HiFi-GAN-family upsampler — Kokoro's iSTFTNet — uses `K == 2S`
        # with `P == S/2`, which `shufflecase` refuses, so those calls take the
        # gather: one thread per output element, no reuse, 137 ms of a 217 ms
        # utterance at 0.02 TF/s. `convolutiontranspose_phase!` is the answer and
        # `emitconvtranspose!` does not declare it yet — see its docstring, and
        # the note at the bottom of this file.
        for (K, S, P, cin, cout) in ((20, 10, 5, 512, 256), (12, 6, 3, 256, 128),
                                     (8, 4, 2, 32, 16), (6, 3, 1, 8, 8),
                                     (5, 2, 1, 16, 32))
            w = KA.allocate(back, Float16, K, 1, cout, cin)
            @test DK.phasecase(w, [S, 1], [P, 0], [1, 1], [0, 0], 1)
            w = nothing
        end
        GC.gc()

        # SAM 2's decoder is `K == S` and belongs to `shufflecase`. `phasecase`
        # would also accept it — it is the more general test — so what keeps the
        # decoder on its own GEMM is the ORDER the emit asks in. Asserting
        # `!phasecase` here was true only while the phase path was 1-D, and
        # became a false statement about the dispatch the moment it covered 2-D.
        w = KA.allocate(back, Float16, 2, 2, 32, 64)
        @test DK.shufflecase(w, [2, 2], [0, 0], [1, 1], [0, 0], 1)
        @test DK.phasecase(w, [2, 2], [0, 0], [1, 1], [0, 0], 1)
        # Grouped stays on the gather: the phase weight flattens all input
        # channels into one convolution, which is what groups forbid.
        wg = KA.allocate(back, Float16, 12, 1, 128, 256)
        @test !DK.phasecase(wg, [6, 1], [3, 0], [1, 1], [0, 0], 4)
        w = wg = nothing; GC.gc()
    end
end

# ── Three numerical comparisons that went with the immediate entry points ────
#
# Each drove a `DK.Ctx(backend; ws = DK.Workspace(backend))`, and that arena is
# gone. They are named here rather than deleted quietly, because each says what
# is now unchecked and by what:
#
#   * **1x1 convolution as a plain GEMM** compared `convolution_coopmat!`
#     against `convolution!`'s im2col route on four of SAM 2's shapes. The
#     declared convolution has ONE route — `conv2d_igemm_ki!`, with the tiling
#     and the split factor as `Val` parameters — so there is no second route to
#     disagree with it. `onebyone` and `conv_coopmat_plan` are still asserted
#     above; what the numbers are checked against now is the reference dump,
#     through `verifygraph`, which walks every op of the encoder.
#
#   * **A reduction axis off the tile** compared the padded cooperative-matrix
#     convolution against `convolution_igemm!`. Same reason, same replacement;
#     the `crspad` policy it exists to pin is asserted above as a plan query.
#
#   * **An overlapping transposed convolution** compared
#     `convolutiontranspose_phase!` against the gather, elementwise, on Kokoro's
#     and RIFE's shapes. That one is NOT covered by anything else, because
#     `emitconvtranspose!` does not declare the phase route: it sends an
#     overlapping kernel to the gather, which is slow and right. The comparison
#     belongs with the port, and `phasecase`'s own answers are pinned above so
#     the router cannot drift in the meantime.
