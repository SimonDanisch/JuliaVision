# Checks for `tools/project_texture.jl`, on synthetic meshes, no model needed:
#     julia --project=<root env> tools/check_project_texture.jl
# Each testset pins a bug the Qwen-Image apple in `docs/examples/hunyuan3d.jl`
# ran into: a noisy matte framing the object off centre, a decimated mesh
# splatting a silhouette full of holes, and a hole fill quadratic in the
# number of unpainted vertices.
using Test, LinearAlgebra
include(joinpath(@__DIR__, "project_texture.jl"))

"An octahedron with half-axes `r`: six vertices, eight large faces."
function octahedron(r = (1f0, 1f0, 1f0))
    P = Float32[r[1] -r[1] 0 0 0 0; 0 0 r[2] -r[2] 0 0; 0 0 0 0 r[3] -r[3]]
    faces = [(1, 3, 5), (3, 2, 5), (2, 4, 5), (4, 1, 5), (3, 1, 6), (2, 3, 6), (4, 2, 6), (1, 4, 6)]
    return P, faces
end

"A `w x h` vertex grid in the plane, two triangles per cell."
function gridmesh(w, h)
    P = Float32[mod1(k, w) for _ in 1:1, k in 1:w*h]
    P = vcat(P, Float32[cld(k, w) for _ in 1:1, k in 1:w*h], zeros(Float32, 1, w * h))
    id(i, j) = (j - 1) * w + i
    faces = NTuple{3,Int}[]
    for j in 1:h-1, i in 1:w-1
        push!(faces, (id(i, j), id(i + 1, j), id(i, j + 1)), (id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)))
    end
    return P, faces
end

@testset "objectframe ignores a matte's noise floor" begin
    alpha = fill(0x03, 100, 100, 1)          # Qwen-Image's background is 1..7, not 0
    alpha[21:60, 31:50, 1] .= 0xff
    @test all(objectframe(alpha) .≈ (0.4f0, 0.4f0, 0.4f0))
end

@testset "silhouette! covers large triangles" begin
    P, faces = octahedron()
    g = falses(64, 64)
    silhouette!(g, P, faces, Matrix{Float32}(I, 3, 3), (0.5f0, 0.5f0, 1f0))
    # The diamond |x| + |y| <= 1 filling the grid covers half of it. Splatting the
    # six vertices instead set six cells.
    @test isapprox(count(g) / 64^2, 0.5; atol = 0.03)
end

@testset "orientfit finds an off-centre view" begin
    P, faces = octahedron((1f0, 0.55f0, 0.3f0))
    S = 128
    θ = 30
    h = Float32[cosd(θ), 0, sind(θ)]; u = Float32[0, 1, 0]
    Rtrue = Matrix{Float32}(vcat(h', -u', cross(h, u)'))
    g = falses(S, S)
    silhouette!(g, P, faces, Rtrue, (0.4f0, 0.6f0, 0.6f0))
    alpha = [g[y, x] ? 0xff : 0x03 for x in 1:S, y in 1:S, _ in 1:1]
    iou, R = orientfit(P, faces, alpha; n = S)
    @test iou > 0.95
end

@testset "fillholes! is linear in the vertices it fills" begin
    P, faces = gridmesh(100, 100)
    col = zeros(Float32, 3, size(P, 2)); seen = falses(size(P, 2))
    col[1, 1] = 1; seen[1] = true                # one painted corner
    ptr, adj = adjacency(size(P, 2), faces)
    fillholes!(col, seen, ptr, adj)              # compile
    col .= 0; col[1, 1] = 1; seen .= false; seen[1] = true
    # 199 rounds over a front of up to 9999 vertices: the old `k in still`
    # marking made each round quadratic in the front, tens of seconds here.
    t = @elapsed fillholes!(col, seen, ptr, adj)
    @test t < 2
    @test all(==(1f0), @view col[1, :])
end
