using Test, UVDocRunner, ColorTypes, DNNKernels
using UVDocRunner: plan, pageinput!, upload!, unwarp!
import Mantle

rgbarray(a) = [RGB{Float32}(a[x,y,1,1], a[x,y,2,1], a[x,y,3,1]) for x in axes(a,1), y in axes(a,2)]
maxerror(a,b) = maximum(max(abs(red(x)-red(y)), abs(green(x)-green(y)), abs(blue(x)-blue(y))) for (x,y) in zip(a,b))

# The warp alone, along `grid` written where the network writes its own.
function warped(m, img, grid)
    page = upload!(m, img)
    out = only(plan(m).outputs)
    copyto!(out, reshape(grid, size(out)))
    Mantle.run!(page.plans[:warp])
    result = Matrix{RGB{Float32}}(undef, size(img))
    copyto!(result, page.output)
    return result
end

# A grid of the network's size: each point the normalised source coordinate
# `f(u, v)` of where it sits, `u` and `v` running from -1 to 1.
function grid(m, f)
    gw, gh = size(only(plan(m).outputs))[1:2]
    g = Array{Float32}(undef, gw, gh, 2, 1)
    for j in 1:gh, i in 1:gw
        u, v = -1 + 2(i - 1) / (gw - 1), -1 + 2(j - 1) / (gh - 1)
        g[i, j, 1, 1], g[i, j, 2, 1] = f(u, v)
    end
    return g
end

@testset "UVDocRunner" begin
    if !UVDocRunner.ready()
        @test_skip "UVDoc artifact not installed"
        return
    end
    m = uvdoc()
    try
        @testset "grid coordinates and padding" begin
            # Unequal sides and distinct channels catch transposition and
            # channel swaps.
            img = [RGB{Float32}(x/90, y/70, (x+y)/160) for x in 1:90, y in 1:70]
            @test maxerror(warped(m, img, grid(m, (u, v) -> (u, v))), img) < 2e-6
            @test maxerror(warped(m, img, grid(m, (u, v) -> (-u, v))), img[end:-1:1, :]) < 2e-6
            # Outside the page is zero, as torch's zero padding.
            @test all(iszero ∘ red, warped(m, img, grid(m, (u, v) -> (3f0, 3f0))))
            # Half a pixel left of the first column: half of it, half padding.
            half = -1 - 1 / (size(img, 1) - 1)
            @test red(warped(m, img, grid(m, (u, v) -> (half, -1f0)))[1, 1]) ≈ red(img[1, 1]) / 2
            @test_throws ArgumentError pageinput!(m, (0, 3))
        end

        @testset "every page type is converted on the device" begin
            gray = [Gray{Float32}((x + y) / 160) for x in 1:90, y in 1:70]
            identity = grid(m, (u, v) -> (u, v))
            @test maxerror(warped(m, gray, identity), RGB{Float32}.(gray)) < 2e-6
            @test upload!(m, gray) === pageinput!(m, size(gray))
        end

        refpath = joinpath(@__DIR__, "..", "..", "gen", "graphs", "uvdoc", "reference.safetensors")
        if isfile(refpath)
            ref = readsafetensors(refpath)
            img = rgbarray(ref["original"])
            @testset "against PyTorch" begin
                page = upload!(m, img)
                Mantle.run!(page.plans[:input])
                @test maximum(abs.(Array(only(plan(m).inputs)) .- ref["input"])) < 2e-5
                unwarp!(m, page)
                @test maximum(abs.(Array(only(plan(m).outputs)) .- ref["grid"])) < 2e-4
                @test maxerror(unwarp(m, img), rgbarray(ref["unwarped"])) < 5e-3
            end
        else
            @test_skip "Export references with tools/export_uvdoc.py to test against PyTorch"
        end
    finally
        close(m)
    end
end
