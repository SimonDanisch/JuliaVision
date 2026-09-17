# A whole model, declared, against a host reference.
#
# NeuralLUT is 26 ATen ops over 7 distinct kinds — `convolution`, `leaky_relu`,
# `repeat`, `_native_batch_norm_legit.no_stats`, `clone`, `mul` and
# `sum.dim_IntList` — which makes it the smallest export that exercises a
# declared graph end to end: a 5-D broadcast, a reduction, a multi-output op read
# through `getitem`, a chain of reshape views, and an implicit-GEMM convolution
# with split-K scratch.
#
# Against `test/hostref.jl` and not against the interpreted path, which shares
# the kernels. The reference is the DEFINITION — four nested loops for the
# convolution, `sum(a; dims)` for the reduction — so it cannot share an index or
# tiling bug with them.
#
# Per-op and not only end to end, and `alias = false` for it. A transient's bytes
# are reused the moment it dies, so reading one after the plan has run compares
# against whatever took its place: with aliasing on, that reads as the first
# convolution off by 37 and four `repeat`s of a WEIGHT off by 1.5, all of which
# were the placer doing its job. With aliasing off every buffer keeps its own
# bytes and the first divergence is the real one.
#
# On every backend `eachbackend()` reports. The ROCm run is the reason the
# per-op table exists: the extension flattened a dispatch's launch geometry to
# `prod(gridsize)`, which is correct for a `@kernel` — its index comes out of
# the `CompilerMetadata` — and wrong for a macro-free kernel that asks the
# hardware, so the convolution's second grid axis collapsed and every workgroup
# read block 1. Lava was exact throughout the same graph.

using Test
import DNNKernels
import Mantle
import NeuralLUTRunner
using ColorTypes: RGB

const DK = DNNKernels
const MM = Mantle

# The reference lives with the ops it is a reference FOR; the graph and the
# weights live with the model that binds their artifact. This file is where the
# two meet, which is why it is here and not in `DNNKernels/test`: that suite
# would need NeuralLUT's weights, and a kernel library's tests depending on a
# model package is the dependency backwards.
include(joinpath(pkgdir(DNNKernels), "test", "hostref.jl"))

"""The op ids whose output is a tensor this test can compare: everything except
a multi-output op's statistics, which `opparity` reads through element 0."""
function opparity(ec, graph, refs; atol)
    bad = Tuple{String,Float64}[]
    for op in graph.ops
        r = get(ec.res, op.out, get(ec.res, "$(op.out)#0", nothing))
        r === nothing && continue
        got = Array(MM.storage(r))
        want = refs[op.out]
        want isa Tuple && (want = want[1])
        size(got) == size(want) || (push!(bad, (op.id, NaN)); continue)
        d = maximum(abs.(got .- want))
        d <= atol || push!(bad, (op.id, Float64(d)))
    end
    return bad
end

@testset "NeuralLUT, declared" begin
    aten = NeuralLUTRunner.neurallutgraph()
    weights = NeuralLUTRunner.neurallutweights()
    img = Float32.(reshape(collect(1:(256 * 256 * 3)), 256, 256, 3, 1) ./ (256 * 256 * 3))

    for be in MM.eachbackend()
        label = string(nameof(typeof(be)))
        @testset "$label" begin
            model = DK.Model(Dict("neurallut" => aten), weights; backend = be)
            graph = model.graphs["neurallut"]
            dev = MM.Device(be)
            hostw = Dict{String,Any}(k => Array(v) for (k, v) in model.weights)
            refs = HostRef.forward(graph, hostw, Dict("img" => img))
            out = only(graph.outputs)

            g, ec = DK.emitgraph(dev, graph, model.weights, (;))
            # Every op declared: the fallback `emitop!` names the aten it has no
            # method for, so reaching here means all 7 are ported.
            @test length(g.passes) >= length(graph.ops)

            # ── per op, with the placer's aliasing off
            pl = MM.Plan(g; alias = false)
            copyto!(MM.storage(ec.res["img"]), img)
            MM.record!(pl); MM.run!(pl); MM.waitidle(dev)
            # 1e-3 on activations that reach ~200 mid-graph: the tiled GEMM sums
            # in a different order than four nested loops do, and `splitk > 1`
            # makes the convolution non-reproducible outright — the atomics land
            # in no fixed order and float addition is not associative.
            bad = opparity(ec, graph, refs; atol = 1e-3)
            @test isempty(bad)
            isempty(bad) || @info "NeuralLUT: ops past tolerance on $label" bad
            MM.free!(pl)

            # ── and with it on, which is what actually runs
            g2, ec2 = DK.emitgraph(dev, graph, model.weights, (;))
            pl2 = MM.Plan(g2)
            copyto!(MM.storage(ec2.res["img"]), img)
            MM.record!(pl2); MM.run!(pl2); MM.waitidle(dev)
            # `maximum(abs.(...))` and not `≈ atol`: for ARRAYS `isapprox`
            # compares NORMS, and when `atol` is given `rtol` defaults to zero,
            # so the bound has to cover `norm(x - y)` over all 107,811 elements
            # — which an rms of 1.6e-6 exceeds while every element is within
            # 5e-6. It also prints both arrays on failure, which is 30 KB of
            # output saying nothing.
            @test maximum(abs.(Array(MM.storage(ec2.res[out])) .- refs[out])) < 1e-4
            # Aliasing has to SAVE something, or the placer is not doing
            # anything and the per-op run above would be the same plan.
            @test MM.peakbytes(pl2) < MM.peakbytes(pl)

            # A recorded plan replays: the second run submits the same recording
            # against the same bytes and must land on the same answer.
            fill!(MM.storage(ec2.res[out]), 0.0f0)
            MM.run!(pl2); MM.waitidle(dev)
            @test maximum(abs.(Array(MM.storage(ec2.res[out])) .- refs[out])) < 1e-4
            MM.free!(pl2)
        end
    end

    # And through the entry point callers use, which is the thing that had to
    # change: `neurallut` declares the graph and records the plan once, and
    # `predictlut` writes the input, replays and returns the output buffer. What
    # it replaced held a slab, a `planslab` layout, a `Workspace` arena and a
    # lazy-broadcast set, and re-ran the interpreter per call.
    @testset "predictlut" begin
        model = NeuralLUTRunner.neurallut()
        # A frame, as the API takes it: a matrix of colours at any resolution.
        frame = [RGB{Float32}((i + j) % 97 / 97, i % 61 / 61, j % 53 / 53)
                 for i in 1:180, j in 1:320]
        lut = NeuralLUTRunner.predictlut(model, frame)
        @test size(lut) == (33, 33, 33, 3)
        a = Array(lut)
        @test all(isfinite, a)
        # Called twice, the second replay has to land on the same table to
        # within the convolution's own reproducibility floor. NOT exactly: a
        # `splitk > 1` convolution accumulates with atomics, the order they
        # land in is not fixed, and float addition is not associative, so two
        # runs on identical inputs differ by ~1e-7 here and ~3e-5 inside the
        # graph. That is the price of the 4-8x the split buys, and it is why
        # this bound exists rather than `==`.
        @test maximum(abs.(Array(NeuralLUTRunner.predictlut(model, frame)) .- a)) < 1e-5
    end
end
