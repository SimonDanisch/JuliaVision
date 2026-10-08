"""
What GPUMeshing promises, on the host and on the device.

The case table is derived rather than transcribed, so its tests are properties:
every case closes, the decider agrees with the bilinear interpolant it stands in
for, and meshes come out closed and consistently wound on fields with every kind
of ambiguous face. The host path is then checked against an independent loop walk
(`reference.jl`), and the device path against the host path, face for face.
"""

using Test, GPUMeshing, Lava, Random, LinearAlgebra
import Mantle
using GeometryBasics: GeometryBasics, Vec3f, Point3f, GLTriangleFace, coordinates, faces, normals
const GM = GPUMeshing

include("reference.jl")

"""Directed edges `(a, b)` with how many faces have them."""
function directededges(F)
    d = Dict{Tuple{Int32,Int32},Int}()
    for j in axes(F, 2), r in 1:3
        k = (Int32(F[r, j]), Int32(F[mod1(r + 1, 3), j]))
        d[k] = get(d, k, 0) + 1
    end
    return d
end

"""Closed and consistently wound: every directed edge as often as its reverse."""
balanced(F) = (d = directededges(F); all(k -> d[k] == get(d, (k[2], k[1]), 0), keys(d)))

"""Every edge on exactly two faces, once in each direction, and the Euler characteristic."""
function manifold(V, F)
    d = directededges(F)
    ok = all(==(1), values(d)) && all(k -> haskey(d, (k[2], k[1])), keys(d))
    return ok, size(V, 2) - length(d) ÷ 2 + size(F, 2)
end

"""Area and signed volume; the volume is positive when the faces point outward."""
function areavolume(V, F)
    area = vol = 0.0
    for j in axes(F, 2)
        p1, p2, p3 = V[:, F[1, j]], V[:, F[2, j]], V[:, F[3, j]]
        a, b = p2 .- p1, p3 .- p1
        cr = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
        area += 0.5 * sqrt(sum(abs2, cr))
        vol += sum(p1 .* cr) / 6
    end
    return area, vol
end

"""Noise with a positive border, so every surface in it closes inside the grid."""
function closednoise(rng, dims)
    f = randn(rng, Float32, dims)
    f[1, :, :] .= 1; f[end, :, :] .= 1
    f[:, 1, :] .= 1; f[:, end, :] .= 1
    f[:, :, 1] .= 1; f[:, :, end] .= 1
    return f
end

function sphere(n, R)
    xs = range(-1, 1; length = n)
    return Float32[R^2 - (x^2 + y^2 + z^2) for x in xs, y in xs, z in xs]
end

download(b::Mantle.Buffer) = Array(Mantle.storage(b))

"""A mesh's positions `(3, V)` and one-based faces `(3, F)`, on the host."""
matrices(m::GeometryBasics.Mesh) = (collect(reinterpret(reshape, Float32, Array(coordinates(m)))),
                                    Int32.(reinterpret(reshape, UInt32, Array(faces(m)))) .+ Int32(1))

@testset "GPUMeshing" begin
    @testset "the edge table and its arithmetic agree" begin
        for e in 1:12
            ox, oy, oz, a = GM.edgeowner(Int32(e))
            lo, _, axis = GM.EDGE[e]
            @test (ox, oy, oz) == GM.CORNER[lo]
            @test a == axis
        end
    end

    @testset "every case closes" begin
        # Every crossed edge of the cube is on exactly one loop, for all 256 sign
        # patterns under both answers on every ambiguous face.
        bad = Tuple{Int,Int}[]
        for cube in 0:255, joined in 0:63
            tris = GM.celltriangles(cube, joined)
            crossed = [e for e in 1:12 if ((cube >> (GM.EDGE[e][1] - 1)) & 1) != ((cube >> (GM.EDGE[e][2] - 1)) & 1)]
            used = sort(unique(e for t in tris for e in t))
            used == crossed || push!(bad, (cube, joined))
        end
        @test isempty(bad)
        @test size(GM.MCTABLE) == (11, 256 * 64)
    end

    @testset "the decider joins what the bilinear interpolant joins" begin
        # Sample the interpolant over the face and flood fill from corner 0: the
        # decider says corners 0 and 2 are connected exactly when the fill
        # reaches corner 2. Faces whose saddle is within 0.02 of the level are
        # skipped, the sampling cannot resolve them.
        function connected(f0, f1, f2, f3; m = 201)
            B(s, t) = f0 * (1 - s) * (1 - t) + f1 * s * (1 - t) + f2 * s * t + f3 * (1 - s) * t
            same = [signbit(B((i - 1) / (m - 1), (j - 1) / (m - 1))) == signbit(f0) for i in 1:m, j in 1:m]
            seen = falses(m, m)
            seen[1, 1] = true
            stack = [(1, 1)]
            while !isempty(stack)
                i, j = pop!(stack)
                for (a, b) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                    (1 <= a <= m && 1 <= b <= m && same[a, b] && !seen[a, b]) || continue
                    seen[a, b] = true
                    push!(stack, (a, b))
                end
            end
            return seen[m, m]
        end
        rng = Xoshiro(1)
        tested = disagree = 0
        for _ in 1:300
            a, c = rand(rng, Float32, 2) .+ 0.05f0
            b, d = -(rand(rng, Float32, 2) .+ 0.05f0)
            f = rand(rng, Bool) ? (a, b, c, d) : (-a, -b, -c, -d)
            abs((f[1] * f[3] - f[2] * f[4]) / (f[1] + f[3] - f[2] - f[4])) < 0.02 && continue
            tested += 1
            (GM.facebit(f..., 1) == 1) == connected(f...) || (disagree += 1)
        end
        @test tested > 200
        @test disagree == 0
    end

    @testset "a sphere: watertight, outward, the right size" begin
        n, R = 48, 0.7
        V, F = matrices(marchingcubes(sphere(n, R)))
        ok, euler = manifold(V, F)
        @test ok
        @test euler == 2
        area, vol = areavolume(V .* Float32(2 / (n - 1)) .- 1f0, F)
        @test isapprox(area, 4π * R^2; rtol = 0.01)
        @test isapprox(vol, 4 / 3 * π * R^3; rtol = 0.01)
        @test vol > 0                      # toward decreasing values: outward here
    end

    @testset "normals: unit, outward, and the sphere's own" begin
        n, R = 48, 0.7
        m = marchingcubes(sphere(n, R); origin = Vec3f(-1), spacing = Vec3f(2 / (n - 1)))
        p, nrm = coordinates(m), normals(m)
        @test all(x -> isapprox(norm(x), 1; atol = 1f-5), nrm)
        # the field decreases outward, and the normals face that way
        @test all(dot.(nrm, p) .> 0)
        @test maximum(acosd.(clamp.(dot.(nrm, normalize.(p)), -1, 1))) < 0.5
        # placement scales the positions, and the normals by the inverse
        stretched = marchingcubes(sphere(n, R); origin = Vec3f(-1), spacing = Vec3f(2 / (n - 1), 4 / (n - 1), 2 / (n - 1)))
        q = coordinates(stretched)
        @test all(isapprox.(getindex.(q, 2), 2 .* getindex.(p, 2) .+ 1; atol = 1f-5))
        ellipsoid = map(x -> normalize(Vec3f(x[1], (x[2] - 1) / 4, x[3])), q)   # the gradient of x² + ((y - 1)/2)² + z²
        @test maximum(acosd.(clamp.(dot.(normals(stretched), ellipsoid), -1, 1))) < 1
    end

    @testset "noise: closed and consistently wound" begin
        # Thousands of ambiguous faces. A fan diagonal can coincide with a
        # neighbour's, so an edge may have four faces; it still has as many in
        # each direction, which is what closed and consistently wound means.
        rng = Xoshiro(2)
        @test all(1:20) do _
            balanced(matrices(marchingcubes(closednoise(rng, (14, 11, 9))))[2])
        end
    end

    @testset "the host path is the loop walk" begin
        rng = Xoshiro(3)
        for f in (sphere(24, 0.6), randn(rng, Float32, 9, 12, 7), closednoise(rng, (10, 10, 10)))
            V, F = matrices(marchingcubes(f; level = 0.05))
            rv, rt = Reference.marchingcubes(f; level = 0.05)
            pts = [Tuple(V[:, i]) for i in axes(V, 2)]
            @test sort(pts) == sort(rv)
            @test sort([sort([pts[F[r, j]] for r in 1:3]) for j in axes(F, 2)]) == sort(rt)
        end
    end

    @testset "fields that do not cross, and the smallest one that does" begin
        V, F = matrices(marchingcubes(ones(Float32, 4, 5, 6)))
        @test size(V) == (3, 0) && size(F) == (3, 0)
        V, F = matrices(marchingcubes(Float32[-1 1; 1 1;;; 1 1; 1 1]))
        @test size(V, 2) == 3 && size(F, 2) == 1
        @test_throws ArgumentError marchingcubes(zeros(Float32, 1, 4, 4))
    end

    dev = Mantle.Device()

    @testset "the device path is the host path" begin
        rng = Xoshiro(4)
        for f in (sphere(48, 0.7), randn(rng, Float32, 17, 23, 31), randn(rng, Float32, 2, 2, 2), ones(Float32, 5, 6, 7))
            place = (; origin = Vec3f(-1, 2, 0.5), spacing = Vec3f(0.5, 1, 2))
            host = marchingcubes(f; level = 0.1, place...)
            fb = Mantle.Buffer(dev, f)
            onboard = marchingcubes(dev, fb; level = 0.1, place...)
            Mantle.free!(fb)
            @test coordinates(onboard) isa Mantle.LavaArray{Point3f,1}
            @test faces(onboard) isa Mantle.LavaArray{GLTriangleFace,1}
            @test normals(onboard) isa Mantle.LavaArray{Vec3f,1}
            @test Array(faces(onboard)) == faces(host)
            # the device's division is not correctly rounded
            dp, hp = Array(coordinates(onboard)), coordinates(host)
            @test length(dp) == length(hp) && all(norm.(dp .- hp) .<= 2f-5)
            @test all(norm.(Array(normals(onboard)) .- normals(host)) .<= 1f-5)
        end
    end

    @testset "a workspace follows a changing field" begin
        # One buffer, rewritten between frames: a surface, nothing, noise that
        # outgrows the buffers, a smaller surface again. Every frame against the
        # host path, and the buffers only grow.
        rng = Xoshiro(7)
        dims = (33, 29, 21)
        fb = Mantle.Buffer(dev, Float32, dims)
        mc = MarchingCubes(dev, fb; vertices = 16, triangles = 16)
        sph = Float32[0.3 - (x^2 + y^2 + z^2) for x in range(-1, 1; length = dims[1]),
                      y in range(-1, 1; length = dims[2]), z in range(-1, 1; length = dims[3])]
        frames = (sph, ones(Float32, dims), randn(rng, Float32, dims), 0.5f0 .* sph, randn(rng, Float32, dims))
        caps = Int[]
        for (n, f) in enumerate(frames)
            copyto!(fb, vec(f))
            level = n == 4 ? 0.02 : 0.0
            m = marchingcubes!(mc; level)
            hv, hf = matrices(marchingcubes(f; level))
            dv, df = matrices(m)
            @test size(df) == size(hf) && df == hf
            @test size(dv) == size(hv) && all(abs.(dv .- hv) .<= 4f-6)
            push!(caps, Mantle.capacity(mc.V))
        end
        @test issorted(caps)
        Mantle.free!(mc)
        Mantle.free!(fb)
    end

    @testset "scan and compaction" begin
        rng = Xoshiro(5)
        n = 1000
        flags = Int32.(rand(rng, Bool, n))
        fb = Mantle.Buffer(dev, flags)
        pos = GM.prefixsum(dev, fb)
        @test download(pos) == cumsum(flags)
        @test GM.lastentry(pos) == sum(flags)
        idx, m = GM.select(dev, fb, n)
        @test download(idx)[1:m] == findall(==(1), flags)
        src = rand(rng, Int32(0):Int32(99), 3, n)
        dst, m = GM.compactcolumns(dev, Mantle.Buffer(dev, src), fb, n)
        @test download(dst)[:, 1:m] == src[:, flags .== 1]
        foreach(Mantle.free!, (fb, pos, idx, dst))
    end

    @testset "sparse grid: corners and quads" begin
        rng = Xoshiro(6)
        res = 12
        coords = unique(eachcol(rand(rng, Int32(0):Int32(res - 1), 3, 600)))
        coords = reduce(hcat, coords)
        n = size(coords, 2)
        grid = SparseGrid(dev, Mantle.Buffer(dev, coords), n, res)
        @test length(grid) == n
        corners, m = voxelcorners(dev, grid)
        listed = Set(Tuple(c) for c in eachcol(download(corners)))
        want = Set((x + i, y + j, z + k) for (x, y, z) in eachcol(coords), i in 0:1, j in 0:1, k in 0:1)
        @test listed == want
        # quads against a host lookup of the four voxels around every far edge
        crossing = rand(rng, Int32(-1):Int32(1), 3, n)
        quads, flags = edgequads(dev, grid, Mantle.Buffer(dev, crossing))
        q, fl = download(quads), download(flags)
        column = Dict(Tuple(c) => Int32(i) for (i, c) in enumerate(eachcol(coords)))
        expected = map(1:3n) do s
            i, a = (s - 1) ÷ 3 + 1, (s - 1) % 3 + 1
            crossing[a, i] == 0 && return nothing
            ids = map(1:4) do k
                o = GM.aroundoffset(Int32(a), Int32(k))
                get(column, (coords[1, i] + o[1], coords[2, i] + o[2], coords[3, i] + o[3]), nothing)
            end
            any(isnothing, ids) ? nothing : ids
        end
        @test fl == Int32.(.!isnothing.(expected))
        @test all(s -> isnothing(expected[s]) || q[:, s] == expected[s], 1:3n)
        foreach(Mantle.free!, (grid, corners, quads, flags))
    end

    @testset "dual contouring a sphere's distance on a narrow band" begin
        res, R = 48, 15.3
        c = res / 2
        sdf(x, y, z) = Float32(sqrt((x - c)^2 + (y - c)^2 + (z - c)^2) - R)
        coords = reduce(hcat, [Int32[x, y, z] for z in 0:(res - 1), y in 0:(res - 1), x in 0:(res - 1)
                               if abs(sdf(x + 0.5, y + 0.5, z + 0.5)) < 1.5])
        n = size(coords, 2)
        grid = SparseGrid(dev, Mantle.Buffer(dev, coords), n, res)
        corners, m = voxelcorners(dev, grid)
        ch = download(corners)
        values = Mantle.Buffer(dev, [sdf(ch[1, i], ch[2, i], ch[3, i]) for i in 1:m])
        V, F = dualcontour(dev, grid, corners, values, m; origin = Vec3f(-c), cell = 1)
        v, f = download(V), download(F)
        foreach(Mantle.free!, (grid, corners, values, V, F))
        ok, euler = manifold(v, f)
        @test ok
        @test euler == 2
        area, vol = areavolume(v, f)
        @test isapprox(area, 4π * R^2; rtol = 0.01)
        @test isapprox(vol, 4 / 3 * π * R^3; rtol = 0.01)
        @test vol > 0                      # toward the positive side: outward
    end
end
