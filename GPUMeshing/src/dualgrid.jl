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
    voxelcorners(dev, grid) -> (corners, m, cornerof)

The corners of the grid's voxels, `(3, m)` zero-based Int32 in the `(res + 1)`^3
corner grid, as cumesh's `get_vertex_num` lists them: each voxel its own low
corner, and each of its other seven whose own voxel, the one that would list it
as its low corner, is not in the grid. A corner none of whose voxels own it is
listed once by every voxel around it, so some appear more than once.

`cornerof` `(8, n)` Int32 is, for each voxel, the column of its corner
`(x + i, y + j, z + k)` in row `1 + 4i + 2j + k`: the owner's copy where the
corner has an owner, the voxel's own copy where it does not. It is what
[`dualcontour`](@ref) reads a voxel's corners through, which is the same answer
as any other copy for a field that is a function of position.
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
    cornerof = Mantle.Buffer(dev, Int32, (8, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, cornerfill!, (corners, cornerof, pos, cnt, grid.coords, vox.keys, vox.vals, vox.capacity,
                                      Int32(res), n), n; name = "voxelcorners/fill")
    runonce!(g)
    foreach(Mantle.free!, (cnt, pos))
    return corners, m, cornerof
end

"""
The column of the voxel whose low corner is the grid point `(x, y, z)`, 0 when
there is none (cumesh's `get_vertex_num`): that voxel owns the corner, and a
voxel only lists a corner other than its low one when nothing owns it.
"""
@inline function cornerowner(keys, vals, cap, res, x, y, z)
    (x >= res || y >= res || z >= res) && return Int32(0)
    return hashlookup(keys, vals, gridkey(x, y, z, res), cap)
end

"""Whether the corner at grid point `(x, y, z)` has no owner, see [`cornerowner`](@ref)."""
@inline ownscorner(keys, vals, cap, res, x, y, z) = cornerowner(keys, vals, cap, res, x, y, z) == Int32(0)

"""The row of `cornerof` holding a voxel's corner at offset `(i, j, k)`."""
@inline cornerrow(i, j, k) = Int32(1) + Int32(4) * i + Int32(2) * j + k

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

function cornerfill!(corners, cornerof, pos, cnt, coords, keys, vals, cap, res, n)
    v = Int32(KI.get_global_id().x)
    v <= n || return nothing
    @inbounds begin
        x, y, z = coords[1, v], coords[2, v], coords[3, v]
        j = pos[v] - cnt[v] + Int32(1)
        corners[1, j] = x; corners[2, j] = y; corners[3, j] = z
        cornerof[1, v] = j
        for i in Int32(0):Int32(1), jj in Int32(0):Int32(1), k in Int32(0):Int32(1)
            (i == Int32(0) && jj == Int32(0) && k == Int32(0)) && continue
            u = cornerowner(keys, vals, cap, res, x + i, y + jj, z + k)
            if u == Int32(0)
                j += Int32(1)
                corners[1, j] = x + i; corners[2, j] = y + jj; corners[3, j] = z + k
                cornerof[cornerrow(i, jj, k), v] = j
            else
                cornerof[cornerrow(i, jj, k), v] = pos[u] - cnt[u] + Int32(1)
            end
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
    dualcontour(dev, grid, cornerof, values; origin = Vec3f(0), cell = 1) -> (V, F)

The zero set of a field sampled at the corners of a sparse voxel grid, by dual
contouring: one vertex per voxel at the mean of the points where the field
changes sign along its twelve edges, a quad of the four voxels around every
crossed edge, wound so it faces the field's positive side. `values` `(m,)`
Float32 is the field at each of the `m` corners [`voxelcorners`](@ref) lists,
computed however the caller's field is defined, and `cornerof` the voxels'
corners among them that it returns with them. Voxels no quad reaches
drop out. Vertices are `origin + cell * p` for `p` in voxel units, as
`Mantle.Buffer`s `(3, V)` Float32 and `(3, F)` Int32.

Each quad is split as cumesh's `remesh_narrow_band_dc` splits it, see
[`splitquads!`](@ref).
"""
function dualcontour(dev::Mantle.Device, grid::SparseGrid, cornerof::Mantle.Buffer, values::Mantle.Buffer;
                     origin::Vec3f = Vec3f(0), cell::Real = 1)
    n = grid.n
    dual = Mantle.Buffer(dev, Float32, (3, n))
    crossing = Mantle.Buffer(dev, Int32, (3, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualpoints!, (dual, crossing, grid.coords, cornerof, values, n), n;
                     name = "dualcontour/points")
    runonce!(g)
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

"""The field at voxel `v`'s corner at offset `o`, through [`voxelcorners`](@ref)' `cornerof`."""
@inline cornervalue(value, cornerof, v, o) = @inbounds value[cornerof[cornerrow(o[1], o[2], o[3]), v]]

"""
cumesh's `simple_dual_contour_kernel`: per voxel, the mean of the points where
the field changes sign along its twelve edges (the voxel's centre when none
does), in voxel units, and on its three far edges the sign change's direction:
1 from negative to positive, -1 the other way, 0 for none.
"""
function dualpoints!(dual, crossing, coords, cornerof, value, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        x, y, z = coords[1, i], coords[2, i], coords[3, i]
        sx = sy = sz = 0f0
        m = Int32(0)
        for a in Int32(1):Int32(3), u in Int32(0):Int32(1), v in Int32(0):Int32(1)
            # The edge along axis `a` at offsets `(u, v)` in the other two: its
            # two ends as offsets from the voxel's low corner.
            o1 = a == Int32(1) ? (Int32(0), u, v) : a == Int32(2) ? (u, Int32(0), v) : (u, v, Int32(0))
            o2 = a == Int32(1) ? (Int32(1), u, v) : a == Int32(2) ? (u, Int32(1), v) : (u, v, Int32(1))
            px, py, pz = x + o1[1], y + o1[2], z + o1[3]
            val1 = cornervalue(value, cornerof, i, o1)
            val2 = cornervalue(value, cornerof, i, o2)
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
