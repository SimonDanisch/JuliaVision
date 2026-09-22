"""
Which decomposed RMS norms `fusegroupedrms` recognises, and that the fused op
computes what the chain it replaced computed.

Two exports of the same idea reach the runtime, and they differ in where the
narrowing cast sits and how far the gain reaches:

    K2 Horizon 32B          xf = fp32(x); sq = xf^2; ms = mean(sq)
    (grouped)               r  = rsqrt(ms + eps)
                            y  = xf * r
                            z  = y * gamma          # gamma spans C * ng
                            out = fp16(z)

    Qwen-Image 2.1          xf = fp32(x); sq = xf^2; ms = mean(sq)
    (per-head q/k norm)     r  = rsqrt(ms + eps)
                            y  = x * r              # the PRE-cast value
                            yh = fp16(y)            # rounds here
                            out = yh * gamma        # gamma spans C, repeated

The second used to decline on all three counts, and its `mean.dim` alone was
5.35 ms of a 223 ms Qwen layer: one thread per row of 128 contiguous floats is
a cache line per lane. The kernel already ran the shape -- a gain spanning one
group is `ng = 1`, where `gof = (g % 1) * C` is zero -- so what was missing was
the recognition and the rounding.

The rounding is the part that has to be exact rather than close: `fp16(x*r) *
gamma` and `fp16(x * r * gamma)` are different numbers, and a fusion that
silently picks the second changes the model.
"""

using Test, DNNKernels, Mantle
import KernelAbstractions as KAR

const DKR = DNNKernels

buf(id, kind, shape, T; key = "") =
    DKR.Buffer(id, kind, Any[shape...], T, key, (0, 9), "", "", Dict{String,Any}())

"""The chain above, as a graph. `order = :qwen` or `:k2`."""
function rmsgraph(; order::Symbol, rows = 3, NG = 2, C = 4, eps = 1.0f-5,
                  gainspan = order === :qwen ? :group : :row,
                  tailcast = order === :qwen ? :none : :narrow)
    T = Float16
    sh = (rows, NG, C)
    gl = gainspan === :group ? C : gainspan === :row ? C * NG : C + 1
    bufs = Dict{String,DKR.Buffer}(
        "x"     => buf("x", :external, sh, T),
        "gamma" => buf("gamma", :weight, (gl,), T; key = "gamma"),
        "xf"    => buf("xf", :transient, sh, Float32),
        "sq"    => buf("sq", :transient, sh, Float32),
        "ms"    => buf("ms", :transient, (rows, NG, 1), Float32),
        "e"     => buf("e", :transient, (rows, NG, 1), Float32),
        "r"     => buf("r", :transient, (rows, NG, 1), Float32))
    A(p...) = Dict{String,Any}(p...)
    ops = DKR.Op[
        DKR.Op("xf", "_to_copy.default", ["x"], "xf", A()),
        DKR.Op("sq", "pow.Tensor_Scalar", ["xf"], "sq", A("arg1" => 2)),
        DKR.Op("ms", "mean.dim", ["sq"], "ms", A("arg1" => [-1], "arg2" => true)),
        DKR.Op("e", "add.Tensor", ["ms"], "e", A("arg1" => eps)),
        DKR.Op("r", "rsqrt.default", ["e"], "r", A())]
    if order === :qwen
        bufs["y"]  = buf("y", :transient, sh, Float32)
        bufs["yh"] = buf("yh", :transient, sh, T)
        bufs["z"]  = buf("z", :transient, sh, T)
        append!(ops, [DKR.Op("y", "mul.Tensor", ["x", "r"], "y", A()),
                      DKR.Op("yh", "_to_copy.default", ["y"], "yh", A()),
                      DKR.Op("z", "mul.Tensor", ["yh", "gamma"], "z", A())])
    else
        bufs["y"] = buf("y", :transient, sh, Float32)
        bufs["z"] = buf("z", :transient, sh, Float32)
        append!(ops, [DKR.Op("y", "mul.Tensor", ["xf", "r"], "y", A()),
                      DKR.Op("z", "mul.Tensor", ["y", "gamma"], "z", A())])
    end
    out = "z"
    if tailcast === :narrow
        bufs["zc"] = buf("zc", :transient, sh, T)
        push!(ops, DKR.Op("zc", "_to_copy.default", ["z"], "zc", A()))
        out = "zc"
    elseif tailcast === :widen
        bufs["zc"] = buf("zc", :transient, sh, Float32)
        push!(ops, DKR.Op("zc", "_to_copy.default", ["z"], "zc", A()))
        out = "zc"
    end
    DKR.Graph("rms", String[], ["x"], [out], bufs, collect(keys(bufs)), ops)
end

@testset "grouped RMS: both exported orders are recognised" begin
    @testset "the Qwen order" begin
        g, n = DKR.fusegroupedrms(rmsgraph(order = :qwen))
        @test n == 1
        op = only(o for o in g.ops if o.aten == "fused.groupedrms")
        @test op.attrs["C"] == 4
        # A gain covering one group repeats over the groups, which is `ng = 1`.
        @test op.attrs["ng"] == 1
        # The cast between the scale and the gain is a rounding the kernel owes.
        @test op.attrs["midround"] === true
        # It reads the pre-cast fp16 value, not the widened copy.
        @test op.ins == ["x", "gamma"]
        @test !any(o -> o.aten in ("pow.Tensor_Scalar", "mean.dim", "rsqrt.default"), g.ops)
    end

    @testset "the K2 order" begin
        g, n = DKR.fusegroupedrms(rmsgraph(order = :k2))
        @test n == 1
        op = only(o for o in g.ops if o.aten == "fused.groupedrms")
        @test op.attrs["ng"] == 2          # the gain spans every group
        @test op.attrs["midround"] === false
        @test op.ins == ["x", "gamma"]
        # The narrowing tail cast is absorbed: the kernel narrows in its store.
        @test op.out == "zc"
    end

    @testset "a widening tail cast is not absorbed" begin
        # `fp32(fp16(y) * gamma)` is not `fp32(y * gamma)`, so the cast stays an
        # op and the fused norm keeps writing the type the chain wrote.
        g, n = DKR.fusegroupedrms(rmsgraph(order = :qwen, tailcast = :widen))
        @test n == 1
        op = only(o for o in g.ops if o.aten == "fused.groupedrms")
        @test op.out == "z"
        @test any(o -> o.id == "zc" && o.aten == "_to_copy.default", g.ops)
    end

    @testset "a gain that spans neither one group nor all of them declines" begin
        @test last(DKR.fusegroupedrms(rmsgraph(order = :qwen, rows = 3, NG = 2,
                                               C = 4, gainspan = :neither))) == 0
    end
end

# Only the Qwen order runs here. The K2 one multiplies a `(rows, ng, C)` value
# by a `C * ng` gain, which is a multiply on the FLAT view of the row, and
# building that view by hand tests the test rather than the pass; its runtime
# path is `test_recorded_call.jl`'s "grouped RMS norm is declared
# backend-independently", which drives the same kernel with `ng = 3` and no
# `midround` attribute, i.e. through the new argument's default.
@testset "grouped RMS: the fused op computes the chain it replaced" begin
    # Whatever backend is loaded, not a named one: `Mantle.LavaBackend` exists
    # only where Lava does, so naming it made this file error out rather than
    # run on a machine with a different GPU.
    backend = first(Mantle.eachbackend())
    rows, NG, C, eps = 3, 2, 4, 1.0f-5
    for order in (:qwen,)
        g0 = rmsgraph(; order, rows, NG, C, eps)
        g1, n = DKR.fusegroupedrms(g0)
        @test n == 1
        # As the driver does: the widening cast in front of the square has no
        # consumer once the norm is one op, and `dropdead` is what removes it.
        g1, _ = DKR.dropdead(g1)
        gl = order === :qwen ? C : C * NG
        γh = Float16[0.5 + i / 20 for i in 1:gl]
        xh = Float16.(reshape([sin(i) for i in 1:(rows * NG * C)], C, NG, rows))

        m0 = DKR.Model(Dict("rms" => g0), Dict{String,Any}("gamma" => γh), backend, 1, 1, 1)
        m1 = DKR.Model(Dict("rms" => g1), Dict{String,Any}("gamma" => γh), backend, 1, 1, 1)
        chain = Array(only(DKR.call(m0, "rms", DKR.toback(backend, xh); dims = (;))))
        fused = Array(only(DKR.call(m1, "rms", DKR.toback(backend, xh); dims = (;))))
        # Not `approx`: the point of `midround` is that this is the same number.
        @test fused == chain

        # And against the definition, so a chain that is itself wrong cannot
        # make this pass.
        want = similar(chain)
        for row in 1:rows, grp in 1:NG
            v = Float32.(@view xh[:, grp, row])
            r = 1f0 / sqrt(sum(abs2, v) / C + eps)
            gain = order === :qwen ? γh : @view γh[(grp - 1) * C + 1:grp * C]
            y = order === :qwen ? Float32.(Float16.(v .* r)) : v .* r
            want[:, grp, row] .= eltype(chain).(y .* Float32.(gain))
        end
        @test fused == want
    end
end

# How wide a workgroup reduces a row, and that a narrower one is still RIGHT.
#
# The kernel sized its shared array and its first tree stride from the `LN_WG`
# constant rather than from the group it was launched with, so launching it
# narrower read `red` past what the live lanes wrote. That produces NaN, and it
# produces them fast: a group-size sweep taken before this was fixed reported a
# 4.9x that was entirely the wrong answer arriving sooner. Both halves are
# pinned here — the width choice, and the values at every width.
@testset "a narrower reduction group is still the same norm" begin
    @test DKR.rmsgroup(64) == 32
    @test DKR.rmsgroup(128) == 32
    @test DKR.rmsgroup(256) == 32
    @test DKR.rmsgroup(512) == 64
    @test DKR.rmsgroup(4096) == 256
    # Clamped at both ends, and always a power of two, because the tree halves.
    @test DKR.rmsgroup(1) == 32
    @test DKR.rmsgroup(1 << 20) == 256
    @test all(ispow2(DKR.rmsgroup(c)) for c in (1, 3, 64, 100, 128, 333, 512, 4096))

    backend = first(Mantle.eachbackend())
    for C in (64, 128, 256)
        R = 32
        ah = Float16.(reshape(sin.(range(0.3, 9.0; length = C * R)), C * R))
        gh = Float16.(range(0.5, 1.5; length = C))
        av = DKR.toback(backend, ah); gv = DKR.toback(backend, gh)
        want = Array{Float32}(undef, C * R)
        for r in 0:(R - 1)
            row = Float32.(ah[r*C+1 : r*C+C])
            rr = 1f0 / sqrt(sum(abs2, row) / C + 1f-6)
            for i in 1:C
                want[r*C+i] = Float32(Float16(row[i] * rr)) * Float32(gh[i])
            end
        end
        # Every width the rule can pick, not just the one it does pick.
        for wg in (32, 64, 128, 256)
            out = KAR.allocate(backend, Float16, C * R)
            # NaN, so a lane that never wrote is not read as an answer.
            fill!(out, Float16(NaN))
            DKR.groupedrms_kernel!(backend, wg)(out, av, gv, Int32(C), Int32(1), 1f-6,
                                                Val(true), Val(wg); ndrange = R * wg)
            KAR.synchronize(backend)
            got = Float32.(Array(out))
            @test all(isfinite, got)
            @test maximum(abs, got .- want) / maximum(abs, want) < 3e-3
        end
    end
end
