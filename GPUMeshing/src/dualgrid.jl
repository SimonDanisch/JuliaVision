"""
Sparse dual-grid meshing: one vertex per active voxel, one quad of the four
voxels around every crossed voxel edge. [`dualcontour`](@ref) places the vertices
from a field sampled at the voxels' corners (cumesh's
`simple_dual_contour_kernel`); a caller with its own vertex placement, like
TRELLIS.2's learned dual grid, uses [`edgequads`](@ref) alone.
"""

"""
    SparseGrid(dev, coords, n, res)

The `n` active voxels of a `res`^3 grid: `coords` `(3, ≥ n)`, zero-based Int32
on the device, and a [`HashMap`](@ref) from each voxel's coordinate to its
column. The grid owns `coords`, and `Mantle.free!(grid)` frees both.
"""
struct SparseGrid
    coords::Mantle.Buffer{Int32,2}
    map::HashMap
    n::Int
    res::Int
end

SparseGrid(dev::Mantle.Device, coords::Mantle.Buffer{Int32,2}, n::Int, res::Int) =
    SparseGrid(coords, insert!(dev, HashMap(dev, n), coords, n, res), n, res)

Base.length(grid::SparseGrid) = grid.n

function Mantle.free!(grid::SparseGrid)
    Mantle.free!(grid.coords)
    Mantle.free!(grid.map)
    return nothing
end

"""
    voxelcorners(dev, grid) -> (corners, m)

The corners of the grid's voxels, `(3, m)` zero-based Int32 in the `(res + 1)`^3
corner grid, as cumesh's `get_vertex_num` lists them: each voxel its own low
corner, and each of its other seven whose own voxel, the one that would list it
as its low corner, is not in the grid. A corner none of whose voxels own it is
listed once by every voxel around it, so some appear more than once.
[`dualcontour`](@ref) reads one copy of each, which is the same answer for any
field that is a function of position.
"""
function voxelcorners(dev::Mantle.Device, grid::SparseGrid)
    n, res, vox = grid.n, grid.res, grid.map
    cnt = Mantle.Buffer(dev, Int32, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, cornercount!, (cnt, grid.coords, vox.keys, vox.vals, vox.capacity, Int32(res), n), n;
                     name = "voxelcorners/count")
    runonce!(g)
    pos = prefixsum(dev, cnt)
    m = lastentry(pos)
    corners = Mantle.Buffer(dev, Int32, (3, m))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, cornerfill!, (corners, pos, cnt, grid.coords, vox.keys, vox.vals, vox.capacity, Int32(res), n), n;
                     name = "voxelcorners/fill")
    runonce!(g)
    foreach(Mantle.free!, (cnt, pos))
    return corners, m
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

"""The `k`th of the four voxels around the edge along axis `a` from a voxel,
as an offset; o-voxel's `edge_neighbor_voxel_offset`."""
@inline function aroundoffset(a, k)
    if a == Int32(1)
        return k == Int32(1) ? (Int32(0), Int32(0), Int32(0)) : k == Int32(2) ? (Int32(0), Int32(0), Int32(1)) :
               k == Int32(3) ? (Int32(0), Int32(1), Int32(1)) : (Int32(0), Int32(1), Int32(0))
    elseif a == Int32(2)
        return k == Int32(1) ? (Int32(0), Int32(0), Int32(0)) : k == Int32(2) ? (Int32(1), Int32(0), Int32(0)) :
               k == Int32(3) ? (Int32(1), Int32(0), Int32(1)) : (Int32(0), Int32(0), Int32(1))
    else
        return k == Int32(1) ? (Int32(0), Int32(0), Int32(0)) : k == Int32(2) ? (Int32(0), Int32(1), Int32(0)) :
               k == Int32(3) ? (Int32(1), Int32(1), Int32(0)) : (Int32(1), Int32(0), Int32(0))
    end
end

"""
    edgequads(dev, grid, crossing) -> (quads, flags)

The quads of a dual grid. Slot `s` is voxel `i = (s - 1) ÷ 3 + 1` and axis
`a = (s - 1) % 3 + 1`: when `crossing[a, i]` is nonzero, the edge along `a` at
the voxel's far side in the other two axes is crossed, and when all four voxels
around that edge are in the grid, `quads[:, s]` are their columns in
[`aroundoffset`](@ref) order and `flags[s]` is 1. `crossing` is `(3, n)` Int32,
and its sign is the caller's to read: [`dualcontour`](@ref) winds by it.
"""
function edgequads(dev::Mantle.Device, grid::SparseGrid, crossing::Mantle.Buffer)
    ns = 3 * grid.n
    quads = Mantle.Buffer(dev, Int32, (4, ns))
    flags = Mantle.Buffer(dev, Int32, ns)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, quadcandidates!, (quads, flags, grid.coords, crossing, grid.map.keys, grid.map.vals,
                                          grid.map.capacity, Int32(grid.res), grid.n), ns; name = "edgequads")
    runonce!(g)
    return quads, flags
end

function quadcandidates!(quads, qflag, coords, crossing, keys, vals, cap, res, n)
    s = Int32(KI.get_global_id().x)
    s <= Int32(3) * n || return nothing
    @inbounds begin
        i = (s - Int32(1)) ÷ Int32(3) + Int32(1)
        a = (s - Int32(1)) % Int32(3) + Int32(1)
        qflag[s] = Int32(0)
        crossing[a, i] == Int32(0) && return nothing
        x, y, z = coords[1, i], coords[2, i], coords[3, i]
        for k in Int32(1):Int32(4)
            o = aroundoffset(a, k)
            id = lookupcoord(keys, vals, x + o[1], y + o[2], z + o[3], res, cap)
            id == Int32(0) && return nothing
            quads[k, s] = id
        end
        qflag[s] = Int32(1)
    end
    return nothing
end

"""
    dualcontour(dev, grid, corners, values, m; origin = Vec3f(0), cell = 1) -> (V, F)

The zero set of a field sampled at the `m` corners of a sparse voxel grid, by
dual contouring: one vertex per voxel at the mean of the points where the field
changes sign along its twelve edges, a quad of the four voxels around every
crossed edge, wound so it faces the field's positive side. `corners` are what
[`voxelcorners`](@ref) returns and `values` `(m,)` Float32 the field at each;
the caller computes them however its field is defined. Voxels no quad reaches
drop out. Vertices are `origin + cell * p` for `p` in voxel units, as
`Mantle.Buffer`s `(3, V)` Float32 and `(3, F)` Int32.

Each quad is split as cumesh's `remesh_narrow_band_dc` splits it, see
[`splitquads!`](@ref).
"""
function dualcontour(dev::Mantle.Device, grid::SparseGrid, corners::Mantle.Buffer, values::Mantle.Buffer, m::Int;
                     origin::Vec3f = Vec3f(0), cell::Real = 1)
    n = grid.n
    R = grid.res + 1
    cmap = insert!(dev, HashMap(dev, m), corners, m, R)
    dual = Mantle.Buffer(dev, Float32, (3, n))
    crossing = Mantle.Buffer(dev, Int32, (3, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualpoints!, (dual, crossing, grid.coords, cmap.keys, cmap.vals, cmap.capacity, values, Int32(R), n), n;
                     name = "dualcontour/points")
    runonce!(g)
    Mantle.free!(cmap)
    quads, qflag = edgequads(dev, grid, crossing)
    ns = 3n
    used = fill!(Mantle.Buffer(dev, Int32, n), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markusedquads!, (used, quads, qflag, ns), ns; name = "dualcontour/used")
    runonce!(g)
    qpos = prefixsum(dev, qflag)
    nq = lastentry(qpos)
    vpos = prefixsum(dev, used)
    nvert = lastentry(vpos)
    V = Mantle.Buffer(dev, Float32, (3, max(nvert, 1)))
    F = Mantle.Buffer(dev, Int32, (3, 2 * max(nq, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactdual!, (V, dual, used, vpos, Float32(cell), origin[1], origin[2], origin[3], n), n;
                     name = "dualcontour/vertices")
    Mantle.dispatch!(g, splitquads!, (F, quads, crossing, qflag, qpos, vpos, V, ns), ns; name = "dualcontour/triangles")
    runonce!(g)
    foreach(Mantle.free!, (dual, crossing, quads, qflag, qpos, used, vpos))
    return V, F
end

"""The field at corner `(x, y, z)` of the `R`^3 corner grid."""
@inline cornervalue(keys, vals, cap, value, R, x, y, z) =
    @inbounds value[hashlookup(keys, vals, gridkey(x, y, z, R), cap)]

"""
cumesh's `simple_dual_contour_kernel`: per voxel, the mean of the points where
the field changes sign along its twelve edges (the voxel's centre when none
does), in voxel units, and on its three far edges the sign change's direction:
1 from negative to positive, -1 the other way, 0 for none.
"""
function dualpoints!(dual, crossing, coords, keys, vals, cap, value, R, n)
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
                crossing[a, i] = crossed ? (val1 < 0f0 ? Int32(1) : Int32(-1)) : Int32(0)
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

"""The used dual vertices, renumbered, at `origin + cell * p`."""
function compactdual!(V, dual, used, vpos, cell, ox, oy, oz, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds if used[i] == Int32(1)
        j = vpos[i]
        V[1, j] = ox + dual[1, i] * cell
        V[2, j] = oy + dual[2, i] * cell
        V[3, j] = oz + dual[3, i] * cell
    end
    return nothing
end

"""
Two triangles per quad, wound by the crossing's direction so the quad faces
the positive side, split as cumesh chooses: `|n(v1, v2, v3) ⋅ n(v2, v3, v4)|`
over each split's six indices, the first split where that is larger. Meant to
compare a split's two triangles, the second cross product takes vertices 2-4,
which for the first split is its first triangle again and for the second a zero
vector: the first split unless its first triangle is degenerate. Choosing the
flatter split instead changed 37% of the triangles on TRELLIS.2's 1024 example
and nothing measurable after simplification (the final mesh's distance to the
surface agreed to 0.004 voxels at the 99th percentile), so this does what
upstream does.
"""
function splitquads!(F, quads, crossing, qflag, qpos, vpos, V, ns)
    s = Int32(KI.get_global_id().x)
    s <= ns || return nothing
    @inbounds begin
        qflag[s] == Int32(1) || return nothing
        q = qpos[s]
        i1, i2, i3, i4 = vpos[quads[1, s]], vpos[quads[2, s]], vpos[quads[3, s]], vpos[quads[4, s]]
        positive = crossing[(s - Int32(1)) % Int32(3) + Int32(1), (s - Int32(1)) ÷ Int32(3) + Int32(1)] == Int32(1)
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
