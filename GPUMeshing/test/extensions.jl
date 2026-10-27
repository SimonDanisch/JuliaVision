"""
Closest points, Hermite dual contouring, error-bounded simplification and the device mesh
checks, against known answers: distances to a cube and its winding numbers, the solutions of
corner, edge and face planes, a rotated box contoured with its sharp edges and corners, a
finely meshed cube simplified to its corners, and a sphere simplified within a tolerance.
`extensiontests(dev)` runs them on the device `dev`.
"""

using Test, GPUMeshing, LinearAlgebra
import Mantle, KernelInterface as KI
using GeometryBasics: Vec3f
const GMX = GPUMeshing

upload(dev, x) = Mantle.Buffer(dev, x)
fetch(b::Mantle.Buffer) = Array(Mantle.storage(b))

"""The unit cube's corners and its twelve outward faces."""
function unitcube()
    V = Float32[0 1 1 0 0 1 1 0; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1]
    F = Int32[1 1 5 5 1 1 2 2 3 3 4 4; 3 4 6 7 2 6 3 7 4 8 1 5; 2 3 7 8 6 5 7 6 8 7 5 8]
    return V, F
end

"""The unit cube with every face split into `n`×`n` squares of two triangles each, closed."""
function finecube(n)
    index = Dict{NTuple{3,Int},Int32}()
    V = Vector{Float32}[]
    vid(p) = get!(index, p) do
        push!(V, Float32.(collect(p)) ./ n)
        Int32(length(V))
    end
    F = Vector{Int32}[]
    for axis in 1:3, side in (0, n), i in 0:(n - 1), j in 0:(n - 1)
        u, v = mod1(axis + 1, 3), mod1(axis + 2, 3)
        corner(a, b) = (q = zeros(Int, 3); q[axis] = side; q[u] = a; q[v] = b; Tuple(q))
        p00, p10, p11, p01 = vid(corner(i, j)), vid(corner(i + 1, j)), vid(corner(i + 1, j + 1)), vid(corner(i, j + 1))
        # Outward: (u, v, axis) is right-handed, so counterclockwise in (u, v) faces +axis.
        if side == n
            push!(F, [p00, p10, p11], [p00, p11, p01])
        else
            push!(F, [p00, p11, p10], [p00, p01, p11])
        end
    end
    return reduce(hcat, V), reduce(hcat, F)
end

"""A box of half sides `h`, turned by `R`: its signed distance and outward normal at `p`."""
function boxsdf(p, R, h)
    q = R' * p
    d = abs.(q) .- h
    outside = norm(max.(d, 0))
    inside = min(maximum(d), 0)
    # The normal of the face whose plane is nearest (sharp: one face at a time).
    k = argmax(d)
    n = zeros(3); n[k] = sign(q[k])
    return outside + inside, R * n
end

"""
A box of half sides `h`, turned by `R` and moved by `shift`, contoured by [`dualmc`](@ref) on
a 32³ grid over [-1, 1]³ from Hermite data computed here (crossings by bisection, the face
normal at each): `(V, F, grid)`.
"""
function boxmesh(dev, R, h, shift)
    res, lo, cell = 32, -1.0, 2.0 / 32
    corners(i, j, k) = [lo + cell * i, lo + cell * j, lo + cell * k]
    sdf(p) = boxsdf(p - shift, R, h)
    val(i, j, k) = sdf(corners(i, j, k))[1]
    coords = reduce(hcat, [Int32[i, j, k] for k in 0:(res - 1), j in 0:(res - 1), i in 0:(res - 1)
                           if length(unique(val(i + a, j + b, k + c) < 0 for a in 0:1, b in 0:1, c in 0:1)) == 2])
    n = size(coords, 2)
    keys = zeros(Int32, n)
    hermite = zeros(Float32, 6, 12, n)
    boxes = zeros(Float32, 6, n)
    for v in 1:n
        i, j, k = coords[:, v]
        vals = [Float32(val(i + GMX.CORNER[c][1], j + GMX.CORNER[c][2], k + GMX.CORNER[c][3])) for c in 1:8]
        keys[v] = GMX.cellkey(vals...)
        boxes[:, v] = Float32[corners(i, j, k); corners(i + 1, j + 1, k + 1)]
        for (e, (c1, c2, _)) in enumerate(GMX.EDGE)
            f1, f2 = vals[c1], vals[c2]
            (f1 < 0) == (f2 < 0) && continue
            p1 = corners((i, j, k) .+ GMX.CORNER[c1]...)
            p2 = corners((i, j, k) .+ GMX.CORNER[c2]...)
            t0, t1 = 0.0, 1.0
            for _ in 1:40
                t = (t0 + t1) / 2
                (sdf(p1 + t * (p2 - p1))[1] < 0) == (f1 < 0) ? (t0 = t) : (t1 = t)
            end
            p = p1 + (t0 + t1) / 2 * (p2 - p1)
            hermite[:, e, v] = Float32[p; sdf(p)[2]]
        end
    end
    grid = SparseGrid(dev, upload(dev, coords), n, res)
    kb, hb, bb = upload(dev, keys), upload(dev, hermite), upload(dev, boxes)
    V, F = dualmc(dev, grid, kb, hb, bb)
    foreach(Mantle.free!, (kb, hb, bb))
    return V, F, grid
end

function extensiontests(dev::Mantle.Device)
    @testset "closest points on a cube" begin
        V, F = unitcube()
        Vb, Fb = upload(dev, V), upload(dev, F)
        grid = SurfaceGrid(dev, Vb, Fb; cell = 0.5)
        P = Float32[0.5 1.3 0.5 1.2; 0.5 0.5 0.5 1.2; 1.2 0.5 0.5 1.2]
        d = fetch(distances(dev, grid, upload(dev, P), 4))
        @test d ≈ Float32[0.2, 0.3, 0.5, sqrt(3) * 0.2] atol = 1f-6
        a, v = areavolume(dev, Vb, Fb)
        @test a ≈ 6 && v ≈ 1
        @test closedmanifold(dev, Fb)
        @test !closedmanifold(dev, upload(dev, F[:, 1:11]))
        # Inside, outside, and a hair inside and outside a face.
        Q = Float32[0.5 0.1 1.5 -0.2 0.3 0.3; 0.5 0.9 0.5 0.5 0.6 0.6; 0.5 0.2 0.5 0.5 0.999 1.001]
        @test fetch(windingnumbers(dev, Vb, Fb, upload(dev, Q), 6)) ≈ Float32[1, 1, 0, 0, 1, 0] atol = 1f-4
        Mantle.free!(grid)
    end

    @testset "quadratic error functions: corner, edge, face" begin
        R = Matrix{Float32}(qr(Float32[1 2 0.5; -0.3 1 2; 0.7 -1 1]).Q)
        corner = Float32[1, 2, 3]
        # Three planes through `corner` (the turned axes), two, and one.
        for k in 1:3
            ns = [R[:, j] for j in 1:k]
            mass = corner + R[:, 3] * (k < 3 ? 0.4f0 : 0f0) + R[:, 2] * (k < 2 ? -0.2f0 : 0f0)
            ata = zeros(Float32, 3, 3)
            atb = zeros(Float32, 3)
            for n in ns
                ata .+= n * n'
                atb .+= n * dot(n, corner)
            end
            x = qefsolve((ata[1, 1], ata[1, 2], ata[1, 3], ata[2, 2], ata[2, 3], ata[3, 3]), Vec3f(atb), Vec3f(mass))
            # The mass point projected onto the planes' intersection: corner, line or plane.
            expected = mass - sum(n * dot(n, mass - corner) for n in ns)
            @test x ≈ expected atol = 1f-5
        end
    end

    @testset "dual marching cubes with Hermite data: an axis-aligned box exactly" begin
        h = [0.61, 0.43, 0.37]
        V, F, grid = boxmesh(dev, Matrix(1.0I, 3, 3), h, [0.013, -0.021, 0.008])
        @test closedmanifold(dev, F)
        a, vol = areavolume(dev, V, F)
        @test vol ≈ 8 * prod(h) rtol = 1e-5
        @test a ≈ 8 * (h[1] * h[2] + h[2] * h[3] + h[1] * h[3]) rtol = 1e-5
        Vh = fetch(V)
        sh = [0.013, -0.021, 0.008]
        @test maximum(abs(boxsdf(Float64.(Vh[:, i]) - sh, Matrix(1.0I, 3, 3), h)[1]) for i in axes(Vh, 2)) < 1e-5
        boxcorners = [sh + h .* [sx, sy, sz] for sx in (-1, 1), sy in (-1, 1), sz in (-1, 1)]
        @test all(c -> minimum(norm(Vh[:, i] - c) for i in axes(Vh, 2)) < 1e-5, boxcorners)
        foreach(Mantle.free!, (grid, V, F))
    end

    @testset "dual marching cubes with Hermite data: a turned box, closed and within its bounds" begin
        R = qr([1.0 0.3 -0.2; -0.4 1 0.1; 0.2 0.5 1]).Q |> Matrix
        h = [0.61, 0.43, 0.37]
        V, F, grid = boxmesh(dev, R, h, zeros(3))
        @test closedmanifold(dev, F)
        a, vol = areavolume(dev, V, F)
        # A corner or edge whose face crosses none of its voxel's edges is cut off: the
        # known limit of one vertex per patch, within a fraction of a cell (2 / 32 here).
        @test vol ≈ 8 * prod(h) rtol = 3e-3
        @test a ≈ 8 * (h[1] * h[2] + h[2] * h[3] + h[1] * h[3]) rtol = 5e-3
        Vh = fetch(V)
        @test maximum(abs(boxsdf(Float64.(Vh[:, i]), R, h)[1]) for i in axes(Vh, 2)) < 0.15 * 2 / 32
        foreach(Mantle.free!, (grid, V, F))
    end

    @testset "simplifying a fine cube down to its corners" begin
        V, F = finecube(8)
        Vb, Fb = upload(dev, V), upload(dev, F)
        @test closedmanifold(dev, Fb)
        V2, F2 = simplify(dev, Vb, Fb; tolerance = 1e-4)
        @test closedmanifold(dev, F2)
        a, vol = areavolume(dev, V2, F2)
        @test a ≈ 6 rtol = 1e-5
        @test vol ≈ 1 rtol = 1e-5
        @test size(F2, 2) <= 24
        foreach(Mantle.free!, (V2, F2))
    end

    @testset "simplifying a sphere within a tolerance" begin
        res, R = 64, 20.3
        c = res / 2
        sdf(x, y, z) = Float32(sqrt((x - c)^2 + (y - c)^2 + (z - c)^2) - R)
        coords = reduce(hcat, [Int32[x, y, z] for z in 0:(res - 1), y in 0:(res - 1), x in 0:(res - 1)
                               if abs(sdf(x + 0.5, y + 0.5, z + 0.5)) < 1.5])
        n = size(coords, 2)
        grid = SparseGrid(dev, upload(dev, coords), n, res)
        cs, m, cornerof = voxelcorners(dev, grid)
        ch = fetch(cs)
        values = upload(dev, [sdf(ch[1, i], ch[2, i], ch[3, i]) for i in 1:m])
        V, F = dualcontour(dev, grid, cornerof, values; origin = Vec3f(-c), cell = 1)
        foreach(Mantle.free!, (grid, cs, cornerof, values))
        @test closedmanifold(dev, F)
        nf = size(F, 2)
        # The original kept for the distance check (`simplify` consumes its input).
        V0 = Mantle.Buffer(dev, copy(Mantle.storage(V)))
        F0 = Mantle.Buffer(dev, copy(Mantle.storage(F)))
        tol = 0.05
        V2, F2 = simplify(dev, V, F; tolerance = tol)
        @test closedmanifold(dev, F2)
        @test size(F2, 2) < nf ÷ 3
        # Within the tolerance both ways.
        g2 = SurfaceGrid(dev, V2, F2; cell = 2.0)
        @test maximum(fetch(distances(dev, g2, V0, size(V0, 2)))) <= tol * 1.01
        g0 = SurfaceGrid(dev, V0, F0; cell = 2.0)
        @test maximum(fetch(distances(dev, g0, V2, size(V2, 2)))) <= tol * 1.01
        foreach(Mantle.free!, (g2, g0, V2, F2, V0, F0))
    end
end
