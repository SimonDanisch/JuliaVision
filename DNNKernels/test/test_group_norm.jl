"""
`native_group_norm` and `add.Scalar`, both of which arrived with FLUX.2-klein:
the VAE decoder normalises 32 groups over its channels, and the transformer's
modulation is `(1 + scale) * norm(x)` with the 1 in the schema.

Group norm is the layer-norm kernel over the contiguous `(W, H, C ÷ groups)`
runs of the reversed layout plus one per-channel affine pass; what is checked
is the number against a host reference, read through the `getitem` view a real
export puts in front of a multi-result op, with the scalar add consuming it.
"""

using Test, DNNKernels, Mantle

const DKG = DNNKernels

gnbuf(id, kind, shape, T; attrs = Dict{String,Any}(), of = "", op = "", live = (0, 0), key = "") =
    DKG.Buffer(id, kind, Any[shape...], T, key, live, of, op, attrs)

"""`y = group_norm(x, γ, β) + 1.5` over torch `(N, C, H, W)` with `groups`."""
function groupnormgraph(N, C, H, W, groups; eps = 1e-6)
    bufs = Dict{String,DKG.Buffer}(
        "x" => gnbuf("x", :external, (N, C, H, W), Float32),
        "gamma" => gnbuf("gamma", :weight, (C,), Float32; key = "gamma"),
        "beta" => gnbuf("beta", :weight, (C,), Float32; key = "beta"),
        # A multi-result op declares `shapes`/`dtypes`: the output, the means and
        # the reciprocal standard deviations, torch's `(N, groups)` each.
        "gn" => gnbuf("gn", :transient, (), Float32; live = (0, 1), attrs = Dict{String,Any}(
            "shapes" => Any[Any[N, C, H, W], Any[N, groups], Any[N, groups]],
            "dtypes" => Any[Float32, Float32, Float32])),
        "getitem" => gnbuf("getitem", :view, (N, C, H, W), Float32; of = "gn",
                           op = "<built-in function getitem>", attrs = Dict{String,Any}("arg1" => 0)),
        "y" => gnbuf("y", :transient, (N, C, H, W), Float32; live = (1, 1)),
    )
    ops = [DKG.Op("gn", "native_group_norm.default", ["x", "gamma", "beta"], "gn",
                  Dict{String,Any}("arg3" => N, "arg4" => C, "arg5" => H * W, "arg6" => groups, "arg7" => eps)),
           DKG.Op("y", "add.Scalar", ["getitem"], "y", Dict{String,Any}("arg1" => 1.5))]
    return DKG.Graph("gn", String[], ["x"], ["y"], bufs, ["x", "gamma", "beta", "gn", "getitem", "y"], ops)
end

"The same on the host: `x` is `(W, H, C, N)`, the reversed layout."
function groupnormref(x, γ, β, groups; eps = 1e-6)
    W, H, C, N = size(x)
    cg = C ÷ groups
    y = similar(x)
    for n in 1:N, g in 1:groups
        cs = ((g - 1) * cg + 1):(g * cg)
        block = @view x[:, :, cs, n]
        μ = sum(block) / length(block)
        v = sum(abs2, block .- μ) / length(block)
        r = 1 / sqrt(v + eps)
        for c in cs
            @views y[:, :, c, n] .= (x[:, :, c, n] .- μ) .* (r * γ[c]) .+ β[c] .+ 1.5f0
        end
    end
    return y
end

@testset "native_group_norm, then add.Scalar" begin
    backend = first(Mantle.eachbackend())
    # 4 groups of 4 channels on a 6x5 plane, batch 2: runs of 120 elements, 8 of them.
    N, C, H, W, groups = 2, 16, 5, 6, 4
    g = groupnormgraph(N, C, H, W, groups)
    γ = Float32[0.5 + 0.1k for k in 1:C]
    β = Float32[-0.2 + 0.05k for k in 1:C]
    model = DKG.Model(Dict("gn" => g), Dict{String,Any}("gamma" => γ, "beta" => β), backend, 1, 1, 1)
    x = Float32.(reshape(sin.(1:(W * H * C * N)) .* 3 .+ 1, W, H, C, N))
    got = Array(only(DKG.call(model, "gn", DKG.toback(backend, x); dims = (;))))
    want = groupnormref(x, γ, β, groups)
    @test size(got) == size(want)
    @test maximum(abs.(got .- want)) < 1e-4
end

@testset "native_group_norm over groups too long for one workgroup" begin
    # 102400 values a group, past the 32k where the statistics split into 8192-value
    # chunks: 12.5 of them, so the last chunk is ragged. The values sit on a mean of
    # 100 with a spread of 3, which is where taking the variance as E[x²] - E[x]²
    # would lose it; against a Float64 host reference.
    backend = first(Mantle.eachbackend())
    N, C, H, W, groups = 1, 8, 128, 200, 2
    g = groupnormgraph(N, C, H, W, groups)
    γ = Float32[0.5 + 0.1k for k in 1:C]
    β = Float32[-0.2 + 0.05k for k in 1:C]
    model = DKG.Model(Dict("gn" => g), Dict{String,Any}("gamma" => γ, "beta" => β), backend, 1, 1, 1)
    x = Float32.(reshape(sin.(1:(W * H * C * N)) .* 3 .+ 100, W, H, C, N))
    got = Array(only(DKG.call(model, "gn", DKG.toback(backend, x); dims = (;))))
    want = groupnormref(Float64.(x), Float64.(γ), Float64.(β), groups)
    @test size(got) == size(want)
    @test maximum(abs.(got .- want)) < 2e-4
end
