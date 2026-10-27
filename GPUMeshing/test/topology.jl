"""
Connectivity, clean-up, tetrahedral boundaries, volume moments and parts, against known
answers: the unit cube's 18 edges and one component, two cubes as two components, duplicates
and degenerate faces taken out of a cube, a missing face filled, flipped faces turned back, two
cubes sharing an edge split apart, a speck removed, the Kuhn tetrahedra of a cube with their 12
boundary faces and 19 edges, and the moments of boxes in closed form. Needs `extensions.jl`
(`unitcube`, `upload`, `fetch`). `topologytests(dev)` runs them on the device `dev`.
"""

"""The cube `[lo, lo + s]^3` as `unitcube` winds it, its vertex ids starting after `offset`."""
function cubemesh(lo, s; offset = 0)
    V, F = unitcube()
    return V .* Float32(s) .+ Float32.(lo), F .+ Int32(offset)
end

"""Closed-form moments of the box `[a, b]` (three ranges): `∫1`, `∫x`, `∫x xᵀ`."""
function boxmoments(a, b)
    L = b .- a
    m0 = prod(L)
    m1 = [m0 * (a[i] + b[i]) / 2 for i in 1:3]
    m2 = zeros(3, 3)
    for i in 1:3, j in 1:3
        if i == j
            m2[i, i] = (b[i]^3 - a[i]^3) / 3 * m0 / L[i]
        else
            k = 6 - i - j
            m2[i, j] = (b[i]^2 - a[i]^2) / 2 * (b[j]^2 - a[j]^2) / 2 * L[k]
        end
    end
    return m0, m1, m2
end

"""The six Kuhn tetrahedra of the unit cube around its diagonal 1–7, positively oriented."""
kuhntets() = Int32[1 1 1 1 1 1; 2 3 4 8 5 6; 3 4 8 5 6 2; 7 7 7 7 7 7]

"""Edges on more than two faces (host check of a downloaded face list)."""
function overfull(F)
    d = Dict{Tuple{Int32,Int32},Int}()
    for f in axes(F, 2), k in 1:3
        a, b = F[k, f], F[mod1(k + 1, 3), f]
        e = a < b ? (a, b) : (b, a)
        d[e] = get(d, e, 0) + 1
    end
    return count(>(2), values(d))
end

function topologytests(dev::Mantle.Device)
    @testset "the unit cube's edges and component" begin
        V, F = unitcube()
        Fd = upload(dev, F)
        e = Edges(dev, Fd, 12)
        @test e.ne == 18
        _, m2 = edgeswithfaces(dev, e, 2)
        @test m2 == 18
        E, ukeys = edgelist(dev, e)
        Eh = fetch(E)
        @test all(Eh[1, :] .< Eh[2, :])
        @test issorted(fetch(ukeys))
        adj, m = manifoldfaceadjacency(dev, e)
        @test m == 18
        ids = fetch(components(dev, adj, m, 12))
        @test all(==(1), ids)
    end

    @testset "two cubes are two components" begin
        V1, F1 = cubemesh((0, 0, 0), 1)
        V2, F2 = cubemesh((3, 0, 0), 1; offset = 8)
        F = hcat(F1, F2)
        e = Edges(dev, upload(dev, F), 24)
        adj, m = manifoldfaceadjacency(dev, e)
        ids = fetch(components(dev, adj, m, 24))
        @test sort(unique(ids)) == [1, 13]
        @test all(==(1), ids[1:12]) && all(==(13), ids[13:24])
    end

    @testset "duplicate faces out, order kept" begin
        V, F = unitcube()
        dup = hcat(F, F[[2, 3, 1], 4:4], F[[3, 2, 1], 9:9], F[:, 1:1])
        @test removeduplicates(dev, dup) == F
    end

    @testset "degenerate faces and their lone vertices out" begin
        V, F = unitcube()
        mid = (V[:, 1] .+ V[:, 2]) ./ 2
        V2 = hcat(V, mid)
        F2 = hcat(F, Int32[1, 2, 9], Int32[3, 3, 4])
        Vr, Fr = removedegenerate(dev, V2, F2)
        @test size(Vr) == (3, 8) && Vr == V
        @test Fr == F
    end

    @testset "a missing face filled" begin
        V, F = unitcube()
        top = [all(V[3, F[:, f]] .== 1) for f in axes(F, 2)]
        Fo = F[:, .!top]
        Vf, Ff = fillholes(dev, V, Fo; maxperimeter = 5)
        @test size(Vf, 2) == 9
        @test Vf[:, 9] ≈ Float32[0.5, 0.5, 1]
        @test size(Ff, 2) == 10 + 4
        # closed; cumesh winds a fan face by its edge's key, so it is oriented after
        _, nb = edgeswithfaces(dev, Edges(dev, upload(dev, Ff), 14), 1)
        @test nb == 0
        Fo = orientfaces(dev, Ff)
        @test closedmanifold(dev, upload(dev, Fo))
        _, vol = areavolume(dev, upload(dev, Vf), upload(dev, Fo))
        @test vol ≈ 1
        # below the perimeter nothing changes
        Vn, Fn = fillholes(dev, V, Fo; maxperimeter = 3)
        @test Vn == V && Fn == Fo
    end

    @testset "flipped faces turned back" begin
        V, F = unitcube()
        Fx = copy(F)
        for f in (2, 5, 7, 12)
            Fx[2, f], Fx[3, f] = Fx[3, f], Fx[2, f]
        end
        @test !closedmanifold(dev, upload(dev, Fx))
        Fo = orientfaces(dev, Fx)
        @test closedmanifold(dev, upload(dev, Fo))
        _, vol = areavolume(dev, upload(dev, V), upload(dev, Fo))
        @test vol ≈ 1                  # face 1, which stays, points outward
    end

    @testset "two cubes sharing an edge come apart" begin
        V1, F1 = unitcube()
        V2, F2 = cubemesh((1, 1, 0), 1)
        # cube 2's vertices 1 and 5 are cube 1's 3 and 7
        map2 = Int32[3, 9, 10, 11, 7, 12, 13, 14]
        V = hcat(V1, V2[:, [2, 3, 4, 6, 7, 8]])
        F = hcat(F1, map2[F2])
        @test overfull(F) == 1
        Vs, Fs = splitnonmanifold(dev, V, F)
        @test size(Vs, 2) == 16
        @test overfull(Fs) == 0
        @test closedmanifold(dev, upload(dev, Fs))
        @test Vs[:, 15:16] == V[:, [3, 7]]
        _, vol = areavolume(dev, upload(dev, Vs), upload(dev, Fs))
        @test vol ≈ 2
    end

    @testset "a speck removed" begin
        V1, F1 = unitcube()
        V2, F2 = cubemesh((2, 0, 0), 1f-3; offset = 8)
        Vr, Fr = removesmallcomponents(dev, hcat(V1, V2), hcat(F1, F2); minarea = 1e-5)
        @test Vr == V1 && Fr == F1
    end

    @testset "the Kuhn tetrahedra of a cube: faces, edges, boundary" begin
        V, _ = unitcube()
        T = kuhntets()
        b = tetboundary(dev, upload(dev, T), 6, 8)
        @test b.nf == 18 && b.ne == 19
        Fb, cnt = fetch(b.F), fetch(b.facecount)
        @test count(==(1), cnt) == 12 && count(==(2), cnt) == 6
        @test all(Fb[1, :] .< Fb[2, :] .< Fb[3, :])
        @test issorted([Tuple(Fb[:, f]) for f in 1:18])
        @test all(==(1), fetch(b.tetflag))
        @test all(==(1), fetch(b.vertexflag))
        eflag, Eb = fetch(b.edgeflag), fetch(b.E)
        @test count(==(1), eflag) == 18
        inner = findfirst(==(0), eflag)
        @test Eb[:, inner] == Int32[1, 7]
        parent = fetch(b.parent)
        @test all(f -> Fb[:, f] ⊆ T[:, parent[f]], 1:18)
        orient = faceorientation(dev, upload(dev, V), upload(dev, T), b.F, b.parent, 18)
        o = fetch(orient)
        @test all(o .!= 0)
        # the boundary faces, each turned outward, enclose the unit volume
        bnd = findall(==(1), cnt)
        Fo = Fb[:, bnd]
        mo = volumemoments(dev, upload(dev, V), upload(dev, Fo); orient = upload(dev, o[bnd]))
        @test mo.volume ≈ 1
    end

    @testset "volume moments of boxes, from the surface and from tetrahedra" begin
        for (lo, s) in (((0.0, 0.0, 0.0), 1.0), ((1.0, -2.0, 0.5), 2.0))
            V, F = cubemesh(lo, s)
            m0, m1, m2 = boxmoments(collect(lo), collect(lo) .+ s)
            for Vd in (upload(dev, V), Mantle.Buffer(dev, Float64.(V)))
                ms = volumemoments(dev, Vd, upload(dev, F))
                mt = tetvolumemoments(dev, Vd, upload(dev, kuhntets()))
                for m in (ms, mt)
                    @test m.volume ≈ m0 rtol = 1e-6
                    @test collect(m.first) ≈ m1 rtol = 1e-6
                    @test Matrix(m.second) ≈ m2 rtol = 1e-6
                end
            end
        end
        # a face marked -1 counts turned over: the inward-wound cube is the same solid
        V, F = unitcube()
        Fi = F[[1, 3, 2], :]
        m = volumemoments(dev, upload(dev, V), upload(dev, Fi); orient = Mantle.Buffer(dev, fill(Int32(-1), 12)))
        @test m.volume ≈ 1
    end

    @testset "parts joined and split again" begin
        V1, F1 = cubemesh((0, 0, 0), 1)
        V2, F2 = cubemesh((5, 0, 0), 2)
        V, F, P = joinparts(dev, [(upload(dev, V1), upload(dev, F1)), (upload(dev, V2), upload(dev, F2))])
        @test fetch(P) == Int32[fill(0, 8); fill(1, 8)]
        @test fetch(F) == hcat(F1, F2 .+ Int32(8))
        parts = splitparts(dev, V, F, P, 2)
        @test fetch(parts[1][1]) == V1 && fetch(parts[1][2]) == F1
        @test fetch(parts[2][1]) == V2 && fetch(parts[2][2]) == F2
    end
end
