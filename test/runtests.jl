using GPUFiltering
using Test
using ColorTypes
using FixedPointNumbers
using GeometryBasics
import ImageFiltering
import KernelAbstractions as KA

maxcomponentdiff(a, b) = maximum(map((x, y) -> max(abs(Float32(red(x)) - Float32(red(y))),
                                                   abs(Float32(green(x)) - Float32(green(y))),
                                                   abs(Float32(blue(x)) - Float32(blue(y)))), a, b))

@testset "coloradjust!" begin
    img = rand(RGB{Float32}, 64, 48)
    # neutral is a no-op
    ref = copy(img)
    coloradjust!(img)
    @test img == ref

    # brightness against manual reference
    img = rand(RGB{Float32}, 64, 48)
    ref = map(c -> RGB{Float32}(clamp(c.r + 0.25f0, 0, 1), clamp(c.g + 0.25f0, 0, 1),
                                clamp(c.b + 0.25f0, 0, 1)), img)
    coloradjust!(img; brightness = 0.25f0)
    KA.synchronize(KA.get_backend(img))
    @test maxcomponentdiff(img, ref) < 1.0f-6

    # saturation 0 → grayscale
    img = rand(RGB{N0f8}, 32, 32)
    coloradjust!(img; saturation = 0)
    KA.synchronize(KA.get_backend(img))
    @test all(c -> abs(Float32(c.r) - Float32(c.g)) < 2 / 255 && abs(Float32(c.g) - Float32(c.b)) < 2 / 255, img)
end

@testset "gaussianblur! vs ImageFiltering" begin
    for σ in (1.0, 2.5)
        img = rand(RGB{Float32}, 96, 80)
        out = similar(img)
        gaussianblur!(out, img, σ)
        KA.synchronize(KA.get_backend(img))
        weights, radius = GPUFiltering.gaussianweights(σ)
        kernelfac = ImageFiltering.kernelfactors((ImageFiltering.centered(weights),
                                                  ImageFiltering.centered(weights)))
        ref = ImageFiltering.imfilter(img, kernelfac, "replicate")
        @test maxcomponentdiff(out, ref) < 2.0f-3
    end
end

@testset "unsharpmask!" begin
    img = rand(RGB{Float32}, 64, 64)
    out = similar(img)
    unsharpmask!(out, img, 1.5, 0.0)  # amount 0 = identity
    @test out == img
    unsharpmask!(out, img, 1.5, 0.8)
    KA.synchronize(KA.get_backend(img))
    @test out != img
end

@testset "opticalflow!" begin
    using Statistics: median
    base = zeros(Float32, 320, 180)
    GPUFiltering.smooth!(base, rand(Float32, 320, 180), 3.0)
    shifted = similar(base)
    u = similar(base)
    v = similar(base)
    for (du, dv) in ((3.2f0, -1.7f0), (-0.6f0, 0.4f0))
        flowwarp!(shifted, base, fill(du, size(base)), fill(dv, size(base)))
        opticalflow!(u, v, base, shifted; levels = 3, iterations = 5)
        # flowwarp by (du, dv) displaces content by -(du, dv); flow = content motion
        @test abs(median(vec(u)) + du) < 0.05
        @test abs(median(vec(v)) + dv) < 0.05
    end
end

@testset "fitaffine" begin
    # exact flow field of a known affine (1.5° rotation, 1% scale, translation)
    θ = deg2rad(1.5)
    s = 1.01
    A = Float32[s*cos(θ) -s*sin(θ); s*sin(θ) s*cos(θ)]
    t = Float32[3.5, -2.0]
    u = Float32[(A[1, 1] - 1) * i + A[1, 2] * j + t[1] for i in 1:320, j in 1:180]
    v = Float32[A[2, 1] * i + (A[2, 2] - 1) * j + t[2] for i in 1:320, j in 1:180]
    M = fitaffine(u, v)
    @test isapprox(M[1, 1], A[1, 1]; atol = 1e-4)
    @test isapprox(M[1, 2], A[1, 2]; atol = 1e-4)
    @test isapprox(M[2, 1], A[2, 1]; atol = 1e-4)
    @test isapprox(M[2, 2], A[2, 2]; atol = 1e-4)
    @test isapprox(M[1, 3], t[1]; atol = 0.02)
    @test isapprox(M[2, 3], t[2]; atol = 0.02)

    # robustness: a "moving object" corrupts 25% of the field
    u2 = copy(u); v2 = copy(v)
    u2[1:160, 1:90] .+= 25
    v2[1:160, 1:90] .-= 15
    M2 = fitaffine(u2, v2)
    @test isapprox(M2[1, 3], t[1]; atol = 0.5)
    @test isapprox(M2[2, 3], t[2]; atol = 0.5)
    @test isapprox(M2[1, 1], A[1, 1]; atol = 2e-3)

    # end-to-end: warp a real image by a known affine, recover it via
    # opticalflow! + fitaffine. Sampling by T displaces content by ≈T⁻¹, so
    # the fitted "align-back" transform must equal inv(T) — warping the
    # warped image by the fit must restore the original (the stabilization
    # use-case).
    grayf = zeros(Float32, 320, 180)
    GPUFiltering.smooth!(grayf, rand(Float32, 320, 180), 3.0)
    base = map(x -> RGB{Float32}(x, x, x), grayf)
    T = Mat3f(cosd(1.0), sind(1.0), 0, -sind(1.0), cosd(1.0), 0, 4.0, -1.5, 1)
    Ti = inv(T)
    warped = similar(base)
    warp!(warped, base, T)
    g1 = similar(grayf); g2 = similar(grayf)
    grayscale!(g1, base); grayscale!(g2, warped)
    uf = similar(grayf); vf = similar(grayf)
    opticalflow!(uf, vf, g1, g2; levels = 3, iterations = 6)
    Mr = fitaffine(uf, vf)
    @test isapprox(Mr[1, 1], Ti[1, 1]; atol = 3e-3)
    @test isapprox(Mr[2, 1], Ti[2, 1]; atol = 3e-3)
    @test isapprox(Mr[1, 3], Ti[1, 3]; atol = 0.3)
    @test isapprox(Mr[2, 3], Ti[2, 3]; atol = 0.3)

    # ...and applying the fit really restores the original pixels
    restored = similar(base)
    warp!(restored, warped, Mr)
    inner = (40:280, 30:150)
    d = maximum(abs(restored[i, j].r - base[i, j].r) for i in inner[1], j in inner[2])
    @test d < 0.04
end

@testset "fithomography" begin
    proj(T, x, y) = begin
        d = T[3, 1] * x + T[3, 2] * y + T[3, 3]
        ((T[1, 1] * x + T[1, 2] * y + T[1, 3]) / d,
         (T[2, 1] * x + T[2, 2] * y + T[2, 3]) / d)
    end
    corners = ((1, 1), (320, 1), (1, 180), (320, 180))
    cornererr(Ta, Tb) = maximum(c -> hypot((proj(Ta, c...) .- proj(Tb, c...))...), corners)
    flowof(T) = (Float32[proj(T, i, j)[1] - i for i in 1:320, j in 1:180],
                 Float32[proj(T, i, j)[2] - j for i in 1:320, j in 1:180])

    # exact flow of a known homography: rotation + translation + keystone
    θ = deg2rad(1.0)
    T = Mat3f(cos(θ), sin(θ), 2.0f-5, -sin(θ), cos(θ), -1.0f-5, 3.0, -2.0, 1.0)
    u, v = flowof(T)
    M = fithomography(u, v)
    @test cornererr(M, T) < 1e-3

    # robustness: a "moving object" corrupts 25% of the field
    u2 = copy(u); v2 = copy(v)
    u2[1:160, 1:90] .+= 12
    v2[1:160, 1:90] .-= 9
    @test cornererr(fithomography(u2, v2), T) < 0.01

    # affine flow degenerates gracefully: perspective terms vanish
    Taff = Mat3f(1.01, 0.005, 0, -0.005, 0.99, 0, 2.0, -1.5, 1.0)
    ua, va = flowof(Taff)
    Ma = fithomography(ua, va)
    @test abs(Ma[3, 1]) < 1e-8 && abs(Ma[3, 2]) < 1e-8
    @test cornererr(Ma, Taff) < 1e-3

    # end-to-end: warp a real image by a known homography, recover it via
    # opticalflow! + fithomography — same align-back convention as fitaffine
    # (fit ≈ inv(T)), and applying the fit restores the original pixels
    grayf = zeros(Float32, 320, 180)
    GPUFiltering.smooth!(grayf, rand(Float32, 320, 180), 3.0)
    base = map(x -> RGB{Float32}(x, x, x), grayf)
    S = Mat3f(cosd(0.6), sind(0.6), 5.0f-5, -sind(0.6), cosd(0.6), -3.0f-5, 2.5, -1.5, 1.0)
    Si3 = inv(S)
    Si = Mat3f(Si3 ./ Si3[3, 3])
    warped = similar(base)
    warp!(warped, base, S)
    g1 = similar(grayf); g2 = similar(grayf)
    grayscale!(g1, base); grayscale!(g2, warped)
    uf = similar(grayf); vf = similar(grayf)
    opticalflow!(uf, vf, g1, g2; levels = 3, iterations = 6)
    Mr = fithomography(uf, vf)
    @test cornererr(Mr, Si) < 0.3
    restored = similar(base)
    warp!(restored, warped, Mr)
    inner = (40:280, 30:150)
    @test maximum(abs(restored[i, j].r - base[i, j].r) for i in inner[1], j in inner[2]) < 0.04
end

@testset "warp!" begin
    img = rand(RGB{Float32}, 64, 64)
    out = similar(img)
    warp!(out, img, Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1))  # identity
    KA.synchronize(KA.get_backend(img))
    @test maxcomponentdiff(out, img) < 1.0f-5

    # full-frame crop = identity
    warp!(out, img, (0.0, 0.0, 1.0, 1.0))
    KA.synchronize(KA.get_backend(img))
    @test maxcomponentdiff(out, img) < 1.0f-5

    # integer translation by (8, 4): out[i,j] = img[i+8, j+4]
    M = Mat3f(1, 0, 0, 0, 1, 0, 8, 4, 1)
    warp!(out, img, M)
    KA.synchronize(KA.get_backend(img))
    @test maxcomponentdiff(out[1:50, 1:50], img[9:58, 5:54]) < 1.0f-5

    # quarter crop of 64² into 32² output = exact 1:1 sampling of that region
    small = similar(img, 32, 32)
    warp!(small, img, (0.5, 0.5, 0.5, 0.5))
    KA.synchronize(KA.get_backend(img))
    @test maxcomponentdiff(small, img[33:64, 33:64]) < 1.0f-5
end

@testset "samplewindow! / nccpeak" begin
    using Statistics: mean
    img = rand(Float32, 128, 128)

    # samplewindow! with a pure integer translation is an exact shifted view
    dst = Matrix{Float32}(undef, 32, 32)
    M = Mat3f(1, 0, 0, 0, 1, 0, 8, 4, 1)
    samplewindow!(dst, img, M, 20:51, 30:61)
    @test dst ≈ img[28:59, 34:65]

    # identity sampling reproduces the window
    samplewindow!(dst, img, Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1), 20:51, 30:61)
    @test dst ≈ img[20:51, 30:61]

    # nccpeak: template cut at a known integer offset is recovered exactly,
    # with a perfect score
    w, R = 32, 12
    tmpl = img[40:(40 + w - 1), 50:(50 + w - 1)]
    region = img[(40 - R - 5):(40 + w - 1 + R - 5), (50 - R + 3):(50 + w - 1 + R + 3)]
    dx, dy, score = nccpeak(region, tmpl)
    @test round(Int, dx) == 5 && round(Int, dy) == -3
    @test score > 0.999

    # sub-pixel: a smooth pattern shifted by 0.3 px (bilinear) is recovered
    # to better than a tenth of a pixel
    smoothimg = zeros(Float32, 128, 128)
    GPUFiltering.smooth!(smoothimg, rand(Float32, 128, 128), 2.0)
    tmpl2 = smoothimg[40:(40 + w - 1), 50:(50 + w - 1)]
    shifted = Matrix{Float32}(undef, w + 2R, w + 2R)
    samplewindow!(shifted, smoothimg, Mat3f(1, 0, 0, 0, 1, 0, 0.3, 0, 1),
                  (40 - R):(40 + w - 1 + R), (50 - R):(50 + w - 1 + R))
    dx2, dy2, score2 = nccpeak(shifted, tmpl2)
    @test abs(dx2 - (-0.3)) < 0.1
    @test abs(dy2) < 0.1

    # lighting invariance: gain and offset don't move the peak
    dx3, dy3, _ = nccpeak(0.5f0 .* region .+ 0.2f0, tmpl)
    @test round(Int, dx3) == 5 && round(Int, dy3) == -3
end

@testset "fitsimilarity / PatchTracker" begin
    # similarity point fit: exact recovery of rotation + uniform scale + shift
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    θ, s, tx, ty = 3.0, 1.02, 4.0, -2.5
    a, b = s * cosd(θ), s * sind(θ)
    for x in 20:40:200, y in 20:40:200
        X = a * x - b * y + tx
        Y = b * x + a * y + ty
        push!(xs, x); push!(ys, y); push!(us, X - x); push!(vs, Y - y)
    end
    M = fitsimilarity(xs, ys, us, vs)
    @test isapprox(atand(M[2, 1], M[1, 1]), θ; atol = 1e-6)
    @test isapprox(hypot(M[1, 1], M[2, 1]), s; atol = 1e-6)
    @test isapprox(M[1, 3], tx; atol = 1e-6)
    @test isapprox(M[2, 3], ty; atol = 1e-6)
    # trimmed: gross outliers don't move the fit
    us[1] += 50; vs[2] -= 80
    Mt = fitsimilarity(xs, ys, us, vs)
    @test isapprox(atand(Mt[2, 1], Mt[1, 1]), θ; atol = 1e-3)
    @test isapprox(Mt[1, 3], tx; atol = 0.05)

    # PatchTracker (CPU backend): a shifted frame is recovered by every patch
    img = zeros(Float32, 256, 256)
    GPUFiltering.smooth!(img, rand(Float32, 256, 256), 2.0)
    shifted = similar(img)
    samplewindow!(shifted, img, Mat3f(1, 0, 0, 0, 1, 0, 6.5, -3.25, 1), 1:256, 1:256)
    tr = PatchTracker(KA.CPU(), img, [(64, 64), (192, 64), (64, 192), (192, 192)];
                      window = 48, maxradius = 16)
    ms = matchpatches!(tr, shifted, Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1); radius = 12)
    @test length(ms) == 4
    for m in ms
        @test m.score > 0.9
        @test m.margin > 0.05
        # sampling through T(t) moves content by -t
        @test isapprox(m.dx, -6.5; atol = 0.2)
        @test isapprox(m.dy, 3.25; atol = 0.2)
    end
end
