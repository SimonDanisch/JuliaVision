using Test, OIDNRunner, Mantle
using .OIDNRunner.DNNKernels
const exports = normpath(joinpath(@__DIR__, "..", "..", "gen", "graphs", "oidn"))
@testset "OIDN transforms" begin
    for x in Float32[0, 1.0f-7, 0.001, 0.1, 1, 100, 65504]
        @test isapprox(OIDNRunner.pu_inverse(OIDNRunner.pu_forward(x)), x; rtol = 1.0f-5, atol = 1.0f-8)
    end
    @test OIDNRunner.inputscale(ones(Float32, 4, 4, 3)) ≈ 0.18f0 rtol = 1.0f-5
    @test OIDNRunner.inputscale(zeros(Float32, 4, 4, 3)) == 1.0f0
end

@testset "accelerated OIDN preserves HDR and reference quality" begin
    @test_throws ArgumentError OIDNDenoiser(; precision = Float64)
    @test_throws ArgumentError OIDNDenoiser(; tilesize = 257)
    d = OIDNDenoiser(; precision = Float16, tilesize = 768)
    try
        for (kind, name) in OIDNRunner.MODEL_NAMES
            refs = DNNKernels.readsafetensors(joinpath(exports, name * "-reference.safetensors"))
            out = OIDNRunner.network(d, kind, refs["input"][:, :, :, 1])
            expected = refs["output"][:, :, :, 1]
            @test eltype(out) === Float32
            @test all(isfinite, out)
            @test sqrt(sum(abs2, out .- expected) / sum(abs2, expected)) < 0.005
        end
        for file in ("pipeline-reference.safetensors", "tiled-reference.safetensors")
            ref = DNNKernels.readsafetensors(joinpath(exports, file))
            color = ref["color"][:, :, :, 1]
            albedo = ref["albedo"][:, :, :, 1]
            normal = ref["normal"][:, :, :, 1]
            before = (copy(color), copy(albedo), copy(normal))
            out = denoise(d, color; albedo, normal)
            expected = ref["output"][:, :, :, 1]
            @test sqrt(sum(abs2, out .- expected) / sum(abs2, expected)) < 0.005
            @test all(isfinite, out)
            @test maximum(out) > 1
            @test (color, albedo, normal) == before
            saved = copy(out)
            guides = prefilter(d, albedo, normal)
            savedguides = (copy(guides.albedo), copy(guides.normal))
            denoise(d, color; albedo, normal)
            @test out == saved
            @test (guides.albedo, guides.normal) == savedguides
            @test_throws ErrorException denoise(d, color; albedo = zeros(Float32, 1, 1, 1), normal)
        end
    finally
        OIDNRunner.release!(d)
    end
    @test isempty(d.models)
    @test isempty(d.workspace)
end

@testset "fused convolution write-back matches the unfused kernels" begin
    fused = OIDNDenoiser(; precision = Float16, fuseconv = true)
    plain = OIDNDenoiser(; precision = Float16, fuseconv = false)
    try
        for (kind, name) in OIDNRunner.MODEL_NAMES
            refs = DNNKernels.readsafetensors(joinpath(exports, name * "-reference.safetensors"))
            input = refs["input"][:, :, :, 1]
            a = OIDNRunner.network(fused, kind, input)
            b = OIDNRunner.network(plain, kind, input)
            # Small layers can use the existing atomic split-K fallback. Two
            # repeats of even the UNFUSED network can differ by one fp16 ULP.
            @test sqrt(sum(abs2, a .- b) / sum(abs2, b)) < 0.001
            @test maximum(abs, a .- b) < 0.002
        end
    finally
        OIDNRunner.release!(fused)
        OIDNRunner.release!(plain)
    end
end
@testset "native OIDN matches upstream" begin
    d = OIDNDenoiser(; precision = Float32, tilesize = 768)
    try
        for (kind, name) in OIDNRunner.MODEL_NAMES
            refs = DNNKernels.readsafetensors(joinpath(exports, name * "-reference.safetensors"))
            out = OIDNRunner.network(d, kind, refs["input"][:, :, :, 1])
            @test maximum(abs.(out .- refs["output"][:, :, :, 1])) < 5.0f-6
        end
        for file in ("pipeline-reference.safetensors", "tiled-reference.safetensors")
            ref = DNNKernels.readsafetensors(joinpath(exports, file))
            color = ref["color"][:, :, :, 1];alb = ref["albedo"][:, :, :, 1];normal = ref["normal"][:, :, :, 1]
            original = copy(color)
            guides = OIDNRunner.prefilter(d, alb, normal)
            @test maximum(abs.(guides.albedo .- ref["filtered_albedo"][:, :, :, 1])) < 5.0f-6
            @test maximum(abs.(guides.normal .- ref["filtered_normal"][:, :, :, 1])) < 5.0f-6
            out = denoise(d, color; albedo = alb, normal)
            expected = ref["output"][:, :, :, 1]
            rms = sqrt(sum(abs2, out .- expected) / sum(abs2, expected))
            @test rms < 1.0f-5
            @test all(isfinite, out)
            @test color == original
            @test maximum(out) > 1
        end
    finally
        OIDNRunner.release!(d)
    end
    @test isempty(d.models)
end

# CPU reference for one layer: Y[o,x,y] = relu(b[o] + Σ w[kx,ky,c,o] * [up?(X0); X1][c, x+kx-1, y+ky-1])
function refconv(X0, X1, w, b, W, H, up)
    C0 = size(X0, 1); C1 = size(X1, 1); CO = size(w, 4)
    Y = zeros(Float32, CO, W, H)
    for y in 1:H, x in 1:W
        acc = Float32.(b)
        for ky in 1:3, kx in 1:3
            sx = x + kx - 2; sy = y + ky - 2
            (1 <= sx <= W && 1 <= sy <= H) || continue
            for c in 1:C0
                v = Float32(up ? X0[c, (sx + 1) ÷ 2, (sy + 1) ÷ 2] : X0[c, sx, sy])
                @views acc .+= w[kx, ky, c, :] .* v
            end
            for c in 1:C1
                @views acc .+= w[kx, ky, C0 + c, :] .* Float32(X1[c, sx, sy])
            end
        end
        Y[:, x, y] .= max.(acc, 0.0f0)
    end
    return Y
end

@testset "every convolution tiling matches a CPU reference" begin
    be = OIDNRunner.preferredbackend()
    dev(a) = (d = OIDNRunner.KA.allocate(be, eltype(a), length(a)); copyto!(d, vec(a)); d)
    # 30 x 20 leaves a partial pixel tile; the concatenated, upsampled source
    # covers the decoder's gather.
    W, H, C0, C1 = 30, 20, 64, 32
    X0 = Float16.(randn(Float32, C0, W ÷ 2, H ÷ 2) .* 0.5f0)
    X1 = Float16.(randn(Float32, C1, W, H) .* 0.5f0)
    for tiling in OIDNRunner.CONV_TILINGS
        CO = max(tiling[1], 96 ÷ tiling[1] * tiling[1])
        w = Float32.(Float16.(randn(Float32, 3, 3, C0 + C1, CO) ./ sqrt(9.0f0 * (C0 + C1))))
        b = randn(Float32, CO) .* 0.1f0
        Wt, bias, K = OIDNRunner.packconv(w, b, C0, C0, C1, C1, CO)
        L = OIDNRunner.ConvLayer(C0, C1, true, CO, K, dev(Wt), dev(bias))
        t = OIDNRunner.TileBuffers(be, W, H)
        Y = OIDNRunner.KA.allocate(be, Float16, CO * OIDNRunner.capacity(W, H, 0))
        OIDNRunner.conv!(be, t, L, Y, dev(X0), dev(X1), 0; tiling)
        got = reshape(Float32.(Array(Y))[1:(CO * W * H)], CO, W, H)
        ref = refconv(X0, X1, w, b, W, H, true)
        @test maximum(abs.(got .- ref)) / maximum(abs.(ref)) < 2.0f-3
        OIDNRunner.free!(t)
    end
end

@testset "RGB matrices and planar arrays denoise identically" begin
    ref = DNNKernels.readsafetensors(joinpath(exports, "pipeline-reference.safetensors"))
    color = ref["color"][:, :, :, 1]; albedo = ref["albedo"][:, :, :, 1]; normal = ref["normal"][:, :, :, 1]
    rgb(a) = [OIDNRunner.RGB{Float32}(a[x, y, 1], a[x, y, 2], a[x, y, 3]) for x in axes(a, 1), y in axes(a, 2)]
    d = OIDNDenoiser()
    try
        @test denoise(d, rgb(color); albedo = rgb(albedo), normal = rgb(normal)) == denoise(d, color; albedo, normal)
    finally
        OIDNRunner.release!(d)
    end
end
