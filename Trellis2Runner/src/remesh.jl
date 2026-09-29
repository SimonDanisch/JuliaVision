"""
    remesh(dev, vertices, faces; resolution, scale, center = Vec3f(0), band = 1) -> (V, F)

cumesh's `remesh_narrow_band_dc`, what `to_glb(remesh = True)` rebuilds the
mesh with, on the device: the surface `band` voxels away from the mesh, the zero
set of its unsigned distance minus `band` voxels, extracted by dual contouring
on a `resolution`^3 grid over the cube of side `scale` around `center`. Only
voxels near that surface are visited, those whose centre is within 0.87 voxels
of it. `Mantle.Buffer`s out, `(3, V)` Float32 and `(3, F)` Int32.

Holes and slits narrower than about two voxels close, and the result is a
closed shell wound outward whatever the input was: the dual-grid mesh's open
sheets and edges of three or more faces are gone. A thin part becomes a shell
two voxels thick.

The active voxels: cumesh refines an octree from 32^3, keeping at every level
the cells whose centre is within 0.87 cells of the band surface by its BVH's
exact distance. [`SurfaceGrid`](@ref) answers only within one of its cells, so
the coarse levels are one step here: the cells of four voxels whose triangle
boxes, grown by the band and both half-diagonals, hold a triangle
([`markband!`](@ref)) expand into their voxels, and those within 0.87 voxels
of the band surface are the active set — cumesh's, since the box test takes a
superset of what its coarse levels keep. Then as cumesh: the active voxels and
their corners in a [`HashMap`](@ref) each, one vertex per voxel at the mean of
its edges' crossings ([`dualcontour!`](@ref)), a quad of the four voxels around
every crossed edge, split as cumesh splits it (see [`remeshtriangles!`](@ref)).
"""
function remesh(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                resolution::Int, scale::Real, center::Vec3f = Vec3f(0), band::Real = 1)
    res = resolution
    ratio = 4
    res % ratio == 0 || throw(ArgumentError("remesh: the resolution $res is not a multiple of $ratio"))
    cell = Float32(scale / res)
    eps = Float32(band) * cell
    lo = center .- Float32(scale) / 2
    # Distances are exact out to a cell of this grid, four voxels; the farthest
    # asked for is a band voxel's corner, under three.
    grid = SurfaceGrid(vertices, faces; res = cld(res, 4))
    sv, sf = Mantle.Buffer(dev, grid.vertices), Mantle.Buffer(dev, grid.faces)
    ss, st = Mantle.Buffer(dev, grid.start), Mantle.Buffer(dev, grid.tris)
    nf = size(faces, 2)
    # Coarse cells that may hold a band voxel, and their voxels.
    cres = res ÷ ratio
    ccell = cell * ratio
    ncell = cres^3
    mask = fill!(Mantle.Buffer(dev, Int32, ncell), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markband!, (mask, sv, sf, cres, lo[1], lo[2], lo[3], ccell,
                                    eps + 0.87f0 * cell + 0.87f0 * ccell, nf), nf; name = "remesh/band")
    runonce!(g)
    cpos = prefixsum(dev, mask)
    ncand = lastentry(cpos) * ratio^3
    cand = Mantle.Buffer(dev, Int32, (3, max(ncand, 1)))
    d = Mantle.Buffer(dev, Float32, max(ncand, 1))
    flags = Mantle.Buffer(dev, Int32, max(ncand, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, expandcells!, (cand, mask, cpos, Int32(cres), Int32(ratio), ncell), ncell; name = "remesh/expand")
    Mantle.dispatch!(g, coorddistance!, (d, cand, 0.5f0, 0f0, cell, lo[1], lo[2], lo[3], sv, sf, ss, st, grid.res, ncand),
                     ncand; name = "remesh/centre distance")
    Mantle.dispatch!(g, nearband!, (flags, d, eps, 0.87f0 * cell, ncand), ncand; name = "remesh/active")
    runonce!(g)
    coords, n = compactcolumns(dev, cand, flags, ncand)
    foreach(Mantle.free!, (mask, cpos, cand, d, flags))
    # The voxels, and the corners they share.
    vox = insert!(dev, HashMap(dev, n), coords, n, res)
    cnt = Mantle.Buffer(dev, Int32, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, cornercount!, (cnt, coords, vox.keys, vox.vals, vox.capacity, Int32(res), n), n;
                     name = "remesh/corner count")
    runonce!(g)
    cornerpos = prefixsum(dev, cnt)
    ncorner = lastentry(cornerpos)
    corners = Mantle.Buffer(dev, Int32, (3, ncorner))
    value = Mantle.Buffer(dev, Float32, ncorner)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, cornerfill!, (corners, cornerpos, cnt, coords, vox.keys, vox.vals, vox.capacity, Int32(res), n), n;
                     name = "remesh/corners")
    Mantle.dispatch!(g, coorddistance!, (value, corners, 0f0, eps, cell, lo[1], lo[2], lo[3], sv, sf, ss, st, grid.res, ncorner),
                     ncorner; name = "remesh/corner distance")
    runonce!(g)
    foreach(Mantle.free!, (cnt, cornerpos, sv, sf, ss, st))
    R = res + 1
    cmap = insert!(dev, HashMap(dev, ncorner), corners, ncorner, R)
    Mantle.free!(corners)
    dual = Mantle.Buffer(dev, Float32, (3, n))
    inter = Mantle.Buffer(dev, Int32, (3, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualcontour!, (dual, inter, coords, cmap.keys, cmap.vals, cmap.capacity, value, Int32(R), n), n;
                     name = "remesh/dual contour")
    runonce!(g)
    foreach(Mantle.free!, (cmap, value))
    # A quad of the four voxels around every crossed edge, when all four exist.
    ns = 3n
    quads = Mantle.Buffer(dev, Int32, (4, ns))
    dir = Mantle.Buffer(dev, Int32, ns)
    qflag = Mantle.Buffer(dev, Int32, ns)
    used = fill!(Mantle.Buffer(dev, Int32, n), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, quadcandidates!, (quads, dir, qflag, coords, inter, vox.keys, vox.vals, vox.capacity, Int32(res), n),
                     ns; name = "remesh/quads")
    Mantle.dispatch!(g, markusedquads!, (used, quads, qflag, ns), ns; name = "remesh/used")
    runonce!(g)
    foreach(Mantle.free!, (vox, inter, coords))
    # Voxels no quad reaches drop out.
    qpos = prefixsum(dev, qflag)
    nq = lastentry(qpos)
    vpos = prefixsum(dev, used)
    nvert = lastentry(vpos)
    V = Mantle.Buffer(dev, Float32, (3, max(nvert, 1)))
    F = Mantle.Buffer(dev, Int32, (3, 2 * max(nq, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactdual!, (V, dual, used, vpos, cell, lo[1], lo[2], lo[3], n), n; name = "remesh/vertices")
    Mantle.dispatch!(g, remeshtriangles!, (F, quads, dir, qflag, qpos, vpos, V, ns), ns; name = "remesh/triangles")
    runonce!(g)
    foreach(Mantle.free!, (dual, quads, dir, qflag, qpos, used, vpos))
    return V, F
end

"""Mark, in the `res`^3 `mask`, every cell whose centre lies in a triangle's
bounding box grown by `reach`; the grid's corner is at `lo`, cells `cell` wide."""
function markband!(mask, vertices, faces, res, lox, loy, loz, cell, reach, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        ia, ib, ic = faces[1, f], faces[2, f], faces[3, f]
        # Cell units, cell centres at the integers.
        gx(v) = (vertices[1, v] - lox) / cell - 0.5f0
        gy(v) = (vertices[2, v] - loy) / cell - 0.5f0
        gz(v) = (vertices[3, v] - loz) / cell - 0.5f0
        r = reach / cell
        x0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gx(ia), gx(ib), gx(ic)) - r)))
        x1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gx(ia), gx(ib), gx(ic)) + r)))
        y0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gy(ia), gy(ib), gy(ic)) - r)))
        y1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gy(ia), gy(ib), gy(ic)) + r)))
        z0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gz(ia), gz(ib), gz(ic)) - r)))
        z1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gz(ia), gz(ib), gz(ic)) + r)))
        for z in z0:z1, y in y0:y1, x in x0:x1
            mask[x + Int32(res) * (y + Int32(res) * z) + Int32(1)] = Int32(1)
        end
    end
    return nothing
end

"""The `ratio`^3 voxels of every marked cell, cell by cell; cell `c` covers
voxels `ratio (cx, cy, cz)` onwards."""
function expandcells!(cand, mask, pos, cres, ratio, ncell)
    c = Int32(KI.get_global_id().x)
    c <= ncell || return nothing
    @inbounds begin
        mask[c] == Int32(1) || return nothing
        c0 = c - Int32(1)
        cx = c0 % cres
        cy = (c0 ÷ cres) % cres
        cz = c0 ÷ (cres * cres)
        j = (pos[c] - Int32(1)) * ratio * ratio * ratio
        for z in Int32(0):(ratio - Int32(1)), y in Int32(0):(ratio - Int32(1)), x in Int32(0):(ratio - Int32(1))
            j += Int32(1)
            cand[1, j] = cx * ratio + x
            cand[2, j] = cy * ratio + y
            cand[3, j] = cz * ratio + z
        end
    end
    return nothing
end

"""The unsigned distance from grid point `coords[:, i] + offset` (in voxels of
`cell`, from `lo`) to the surface of the grid, minus `minus`; `Inf` beyond the
grid's reach (see [`nearest`](@ref))."""
function coorddistance!(out, coords, offset, minus, cell, lox, loy, loz, sv, sf, ss, st, sres, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        p = Vec3f(lox + (Float32(coords[1, i]) + offset) * cell,
                  loy + (Float32(coords[2, i]) + offset) * cell,
                  loz + (Float32(coords[3, i]) + offset) * cell)
        _, d2 = nearest(p, sv, sf, ss, st, sres)
        out[i] = sqrt(d2) - minus
    end
    return nothing
end

function nearband!(flags, d, eps, tol, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds flags[i] = abs(d[i] - eps) < tol ? Int32(1) : Int32(0)
    return nothing
end

"""
Whether voxel `(x, y, z)` owns the corner at `(x + i, y + j, z + k)`: the
voxel at that coordinate would, when it exists (cumesh's `get_vertex_num`).
"""
@inline function ownscorner(keys, vals, cap, res, x, y, z)
    (x >= res || y >= res || z >= res) && return true
    return hashlookup(keys, vals, gridkey(x, y, z, res), cap) == Int32(0)
end

function cornercount!(cnt, coords, keys, vals, cap, res, n)
    v = Int32(KI.get_global_id().x)
    v <= n || return nothing
    @inbounds begin
        x, y, z = coords[1, v], coords[2, v], coords[3, v]
        c = Int32(1)
        for i in Int32(0):Int32(1), j in Int32(0):Int32(1), k in Int32(0):Int32(1)
            (i == Int32(0) && j == Int32(0) && k == Int32(0)) && continue
            ownscorner(keys, vals, cap, res, x + i, y + j, z + k) && (c += Int32(1))
        end
        cnt[v] = c
    end
    return nothing
end

function cornerfill!(corners, pos, cnt, coords, keys, vals, cap, res, n)
    v = Int32(KI.get_global_id().x)
    v <= n || return nothing
    @inbounds begin
        x, y, z = coords[1, v], coords[2, v], coords[3, v]
        j = pos[v] - cnt[v] + Int32(1)
        corners[1, j] = x; corners[2, j] = y; corners[3, j] = z
        for i in Int32(0):Int32(1), jj in Int32(0):Int32(1), k in Int32(0):Int32(1)
            (i == Int32(0) && jj == Int32(0) && k == Int32(0)) && continue
            ownscorner(keys, vals, cap, res, x + i, y + jj, z + k) || continue
            j += Int32(1)
            corners[1, j] = x + i; corners[2, j] = y + jj; corners[3, j] = z + k
        end
    end
    return nothing
end

"""The band distance at corner `(x, y, z)` of the `R`^3 corner grid."""
@inline cornervalue(keys, vals, cap, value, R, x, y, z) =
    @inbounds value[hashlookup(keys, vals, gridkey(x, y, z, R), cap)]

"""
cumesh's `simple_dual_contour_kernel`: per voxel, the mean of the points where
the band distance changes sign along its twelve edges (the voxel's centre when
none does), in voxel units, and on its three far edges the sign change's
direction: 1 from inside to outside, -1 the other way, 0 for none.
"""
function dualcontour!(dual, inter, coords, keys, vals, cap, value, R, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        x, y, z = coords[1, i], coords[2, i], coords[3, i]
        sx = sy = sz = 0f0
        m = Int32(0)
        for a in Int32(1):Int32(3), u in Int32(0):Int32(1), v in Int32(0):Int32(1)
            # The edge along axis `a` at offsets `(u, v)` in the other two.
            px = a == Int32(1) ? x : a == Int32(2) ? x + u : x + u
            py = a == Int32(1) ? y + u : a == Int32(2) ? y : y + v
            pz = a == Int32(1) ? z + v : a == Int32(2) ? z + v : z
            val1 = cornervalue(keys, vals, cap, value, R, px, py, pz)
            val2 = a == Int32(1) ? cornervalue(keys, vals, cap, value, R, px + Int32(1), py, pz) :
                   a == Int32(2) ? cornervalue(keys, vals, cap, value, R, px, py + Int32(1), pz) :
                                   cornervalue(keys, vals, cap, value, R, px, py, pz + Int32(1))
            crossed = (val1 < 0f0) != (val2 < 0f0)
            if crossed
                t = -val1 / (val2 - val1)
                sx += Float32(px) + (a == Int32(1) ? t : 0f0)
                sy += Float32(py) + (a == Int32(2) ? t : 0f0)
                sz += Float32(pz) + (a == Int32(3) ? t : 0f0)
                m += Int32(1)
            end
            if u == Int32(1) && v == Int32(1)
                inter[a, i] = crossed ? (val1 < 0f0 ? Int32(1) : Int32(-1)) : Int32(0)
            end
        end
        if m > Int32(0)
            dual[1, i] = sx / m; dual[2, i] = sy / m; dual[3, i] = sz / m
        else
            dual[1, i] = Float32(x) + 0.5f0; dual[2, i] = Float32(y) + 0.5f0; dual[3, i] = Float32(z) + 0.5f0
        end
    end
    return nothing
end

"""Slot `s` is voxel `(s - 1) ÷ 3 + 1`, axis `(s - 1) % 3 + 1`: the quad of
the four voxels around its far edge when that edge is crossed and all four
exist, and the crossing's direction."""
function quadcandidates!(quads, dir, qflag, coords, inter, keys, vals, cap, res, n)
    s = Int32(KI.get_global_id().x)
    s <= Int32(3) * n || return nothing
    @inbounds begin
        i = (s - Int32(1)) ÷ Int32(3) + Int32(1)
        a = (s - Int32(1)) % Int32(3) + Int32(1)
        qflag[s] = Int32(0)
        d = inter[a, i]
        d == Int32(0) && return nothing
        x, y, z = coords[1, i], coords[2, i], coords[3, i]
        for k in Int32(1):Int32(4)
            o = aroundoffset(a, k)
            id = lookupcoord(keys, vals, x + o[1], y + o[2], z + o[3], res, cap)
            id == Int32(0) && return nothing
            quads[k, s] = id
        end
        dir[s] = d
        qflag[s] = Int32(1)
    end
    return nothing
end

function markusedquads!(used, quads, qflag, ns)
    s = Int32(KI.get_global_id().x)
    s <= ns || return nothing
    @inbounds if qflag[s] == Int32(1)
        used[quads[1, s]] = Int32(1)
        used[quads[2, s]] = Int32(1)
        used[quads[3, s]] = Int32(1)
        used[quads[4, s]] = Int32(1)
    end
    return nothing
end

"""The used dual vertices, renumbered, in world units."""
function compactdual!(V, dual, used, vpos, cell, lox, loy, loz, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds if used[i] == Int32(1)
        j = vpos[i]
        V[1, j] = lox + dual[1, i] * cell
        V[2, j] = loy + dual[2, i] * cell
        V[3, j] = loz + dual[3, i] * cell
    end
    return nothing
end

"""
Two triangles per quad, wound by the crossing's direction so the quad faces
out, split as cumesh chooses: `|n(v1, v2, v3) ⋅ n(v2, v3, v4)|` over each
split's six indices, the first split where that is larger. Meant to compare a
split's two triangles, the second cross product takes vertices 2-4, which for
the first split is its first triangle again and for the second a zero vector:
the first split unless its first triangle is degenerate. Choosing the flatter
split instead changed 37% of the triangles on the 1024 example and nothing
measurable after simplification (the final mesh's distance to the surface
agreed to 0.004 voxels at the 99th percentile), so this does what upstream does.
"""
function remeshtriangles!(F, quads, dir, qflag, qpos, vpos, V, ns)
    s = Int32(KI.get_global_id().x)
    s <= ns || return nothing
    @inbounds begin
        qflag[s] == Int32(1) || return nothing
        q = qpos[s]
        i1, i2, i3, i4 = vpos[quads[1, s]], vpos[quads[2, s]], vpos[quads[3, s]], vpos[quads[4, s]]
        positive = dir[s] == Int32(1)
        # cumesh's two splits, each wound so the quad faces out.
        s1 = positive ? (i1, i3, i2, i1, i4, i3) : (i1, i2, i3, i1, i3, i4)
        s2 = positive ? (i1, i4, i2, i4, i3, i2) : (i1, i2, i4, i4, i2, i3)
        a1 = splitalign(V, s1)
        a2 = splitalign(V, s2)
        t = a1 > a2 ? s1 : s2
        j1, j2 = Int32(2) * q - Int32(1), Int32(2) * q
        F[1, j1] = t[1]; F[2, j1] = t[2]; F[3, j1] = t[3]
        F[1, j2] = t[4]; F[2, j2] = t[5]; F[3, j2] = t[6]
    end
    return nothing
end

@inline function splitalign(V, s)
    p1, p2, p3, p4 = vertex(V, s[1]), vertex(V, s[2]), vertex(V, s[3]), vertex(V, s[4])
    return abs(dot(cross(p2 - p1, p3 - p1), cross(p3 - p2, p4 - p2)))
end
