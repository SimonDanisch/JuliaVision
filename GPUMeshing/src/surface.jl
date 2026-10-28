"""
Closest-point queries against triangle meshes on the device: the triangles bucketed into the
cubic cells of a sparse grid, each listed in every cell its bounding box touches. The cells
are sorted keys, `(part, x, y, z)` at 16 bits each, built by a count, a scan, a fill and a
sort by key, so only cells that hold triangles take memory however far the meshes reach; a
hash table over them finds a cell in a probe or two. Its slots are claimed with 32-bit
compare-and-swap on the cell's index (Metal has no 64-bit one), the keys compared after.
Several meshes can share one grid as parts that never see each other: a query names its
part. A query searches the 27 cells around its point, so it finds the closest point
whenever that lies within one cell; farther out it reports `Inf32`.
"""

"""
    SurfaceGrid(dev, vertices, faces; cell, part = nothing)

`vertices` `(3, nv)` Float32 and one-based `faces` `(3, nf)` Int32 (`Mantle.Buffer`s, kept,
not copied) bucketed into cubic cells of side `cell` from the corner `lo` (the meshes'
lowest corner less a cell). `part` `(nv,)` Int32, from 0 to 65535, splits the triangles into
parts by their first vertex (all one part without it). Cell `c` holds triangles
`tris[start[c] + 1:start[c + 1]]`; `keys` are the `ncells` cells, sorted, and `slots` (a
power of two of them, `cap`) the hash table holding their indices.
"""
struct SurfaceGrid
    vertices::Mantle.Buffer{Float32,2}
    faces::Mantle.Buffer{Int32,2}
    keys::Mantle.Buffer{UInt64,1}
    start::Mantle.Buffer{Int32,1}
    tris::Mantle.Buffer{Int32,1}
    slots::Mantle.Buffer{Int32,1}
    cap::Int32
    ncells::Int32
    lo::Vec3f
    cell::Float32
end

function SurfaceGrid(dev::Mantle.Device, V::Mantle.Buffer{Float32,2}, F::Mantle.Buffer{Int32,2}; cell::Real,
                     part::Union{Nothing,Mantle.Buffer{Int32,1}} = nothing)
    c = Float32(cell)
    s = Mantle.storage(V)
    lower = Array(minimum(s; dims = 2))
    lo = Vec3f(lower[1] - c, lower[2] - c, lower[3] - c)
    nf = size(F, 2)
    parts = part === nothing ? fill!(Mantle.Buffer(dev, Int32, size(V, 2)), 0) : part
    cnt = Mantle.Buffer(dev, Int32, max(nf, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, countcells!, (cnt, V, F, lo[1], lo[2], lo[3], c, nf), nf; name = "surfacegrid/count")
    runonce!(g)
    off = prefixsum(dev, cnt)
    m = lastentry(off)
    allkeys = Mantle.Buffer(dev, UInt64, max(m, 1))
    tris = Mantle.Buffer(dev, Int32, max(m, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, fillcells!, (allkeys, tris, off, V, F, parts, lo[1], lo[2], lo[3], c, nf), nf; name = "surfacegrid/fill")
    runonce!(g)
    AK.sort_by_key!(Mantle.storage(allkeys), Mantle.storage(tris))
    heads = Mantle.Buffer(dev, Int32, max(m, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, runstarts!, (heads, allkeys, m), m; name = "surfacegrid/runs")
    runonce!(g)
    idx, ncells = select(dev, heads, m)
    keys = Mantle.Buffer(dev, UInt64, max(ncells, 1))
    start = Mantle.Buffer(dev, Int32, ncells + 1)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, cellstarts!, (keys, start, allkeys, idx, ncells, m), ncells + 1; name = "surfacegrid/cells")
    runonce!(g)
    cap = Int32(nextpow(2, max(2ncells, 16)))
    slots = fill!(Mantle.Buffer(dev, Int32, cap), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, hashcells!, (slots, keys, cap, ncells), max(ncells, 1); name = "surfacegrid/hash")
    runonce!(g)
    foreach(Mantle.free!, (cnt, off, allkeys, heads, idx))
    part === nothing && Mantle.free!(parts)
    return SurfaceGrid(V, F, keys, start, tris, slots, cap, Int32(ncells), lo, c)
end

"""Frees the grid's own buffers; the mesh's stay with their owner."""
function Mantle.free!(s::SurfaceGrid)
    foreach(Mantle.free!, (s.keys, s.start, s.tris, s.slots))
    return nothing
end

"""What a kernel needs to query the grid, after the mesh: cells, triangles, hash table and placement."""
gridargs(s::SurfaceGrid) = (s.vertices, s.faces, s.keys, s.start, s.tris, s.slots, s.cap, s.lo[1], s.lo[2], s.lo[3], s.cell)

"""Murmur3's 64-bit finaliser."""
@inline function fmix64(k::UInt64)
    k ⊻= k >> 33
    k *= 0xff51afd7ed558ccd
    k ⊻= k >> 33
    k *= 0xc4ceb9fe1a85ec53
    k ⊻= k >> 33
    return k
end

@inline hashslot64(key::UInt64, cap) = Int32(fmix64(key) & UInt64(cap - Int32(1))) + Int32(1)

"""Each cell's index into the first free slot from its key's hash on (linear probing)."""
function hashcells!(slots, keys, cap, ncells)
    c = Int32(KI.get_global_id().x)
    c <= ncells || return nothing
    h = hashslot64(@inbounds(keys[c]), cap)
    while true
        old = (@inbounds Atomix.@atomicreplace :monotonic slots[h] Int32(0) => c).old
        old == Int32(0) && return nothing
        h = h == cap ? Int32(1) : h + Int32(1)
    end
end

@inline cellof(x, lo, cell) = unsafe_trunc(Int32, floor((x - lo) / cell))

"""The key of cell `(x, y, z)` of part `part`; coordinates outside 0 to 65535 have none."""
@inline cellkey64(part, x, y, z) = (UInt64(part % UInt16) << 48) | (UInt64(x % UInt16) << 32) | (UInt64(y % UInt16) << 16) |
                                   UInt64(z % UInt16)

@inline incells(x, y, z) = Int32(0) <= x < Int32(65536) && Int32(0) <= y < Int32(65536) && Int32(0) <= z < Int32(65536)

"""The cells triangle `f`'s bounding box touches, as index ranges."""
@inline function trianglebox(V, F, f, lox, loy, loz, cell)
    @inbounds a, b, c = vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f])
    lo, hi = min.(a, b, c), max.(a, b, c)
    return (cellof(lo[1], lox, cell), cellof(hi[1], lox, cell), cellof(lo[2], loy, cell), cellof(hi[2], loy, cell),
            cellof(lo[3], loz, cell), cellof(hi[3], loz, cell))
end

function countcells!(cnt, V, F, lox, loy, loz, cell, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    x0, x1, y0, y1, z0, z1 = trianglebox(V, F, f, lox, loy, loz, cell)
    @inbounds cnt[f] = (x1 - x0 + Int32(1)) * (y1 - y0 + Int32(1)) * (z1 - z0 + Int32(1))
    return nothing
end

"""Triangle `f`, keyed by every cell its box touches, into its run of `keys` and `tris`."""
function fillcells!(keys, tris, off, V, F, part, lox, loy, loz, cell, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    x0, x1, y0, y1, z0, z1 = trianglebox(V, F, f, lox, loy, loz, cell)
    p = @inbounds part[F[1, f]]
    k = f == Int32(1) ? Int32(0) : @inbounds(off[f - Int32(1)])
    for z in z0:z1, y in y0:y1, x in x0:x1
        k += Int32(1)
        @inbounds keys[k] = cellkey64(p, x, y, z)
        @inbounds tris[k] = f
    end
    return nothing
end

function cellstarts!(cells, start, keys, idx, ncells, m)
    c = Int32(KI.get_global_id().x)
    c <= ncells + Int32(1) || return nothing
    if c <= ncells
        @inbounds cells[c] = keys[idx[c]]
        @inbounds start[c] = idx[c] - Int32(1)
    else
        @inbounds start[c] = m
    end
    return nothing
end

"""The triangles of the cell with key `key`, as a range into `tris` (empty for a cell that has none)."""
@inline function cellrange(keys, start, slots, cap, key)
    h = hashslot64(key, cap)
    @inbounds while true
        c = slots[h]
        c == Int32(0) && return Int32(1):Int32(0)
        keys[c] == key && return (start[c] + Int32(1)):start[c + Int32(1)]
        h = h == cap ? Int32(1) : h + Int32(1)
    end
end

"""
The point of triangle `a b c` closest to `p` (Ericson, *Real-Time Collision Detection*,
5.1.5): by the Voronoi region of the triangle `p` falls in.
"""
@inline function closestpoint(p::Vec3f, a::Vec3f, b::Vec3f, c::Vec3f)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    (d1 <= 0f0 && d2 <= 0f0) && return a
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    (d3 >= 0f0 && d4 <= d3) && return b
    vc = d1 * d4 - d3 * d2
    (vc <= 0f0 && d1 >= 0f0 && d3 <= 0f0) && return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    (d6 >= 0f0 && d5 <= d6) && return c
    vb = d5 * d2 - d1 * d6
    (vb <= 0f0 && d2 >= 0f0 && d6 <= 0f0) && return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    (va <= 0f0 && d4 - d3 >= 0f0 && d5 - d6 >= 0f0) &&
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    s = 1f0 / (va + vb + vc)
    return a + ab * (vb * s) + ac * (vc * s)
end

"""
`(q, d2)`: the point `q` of part `part` of the grid's surface closest to `p` and the squared
distance to it, `(p, Inf32)` when no triangle lies within the 27 cells around `p`. A
degenerate triangle's `NaN` loses every comparison and drops out.
"""
@inline function nearest(p::Vec3f, part, V, F, keys, start, tris, slots, cap, lox, loy, loz, cell)
    cx, cy, cz = cellof(p[1], lox, cell), cellof(p[2], loy, cell), cellof(p[3], loz, cell)
    best = Inf32
    q = p
    @inbounds for z in (cz - Int32(1)):(cz + Int32(1)), y in (cy - Int32(1)):(cy + Int32(1)), x in (cx - Int32(1)):(cx + Int32(1))
        incells(x, y, z) || continue
        for i in cellrange(keys, start, slots, cap, cellkey64(part, x, y, z))
            f = tris[i]
            r = closestpoint(p, vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f]))
            e = r - p
            d = dot(e, e)
            if d < best
                best = d
                q = r
            end
        end
    end
    return q, best
end

"""
Whether a triangle of part `part` of the grid's surface lies within `sqrt(r2)` of `p`: only
the cells the ball reaches, up to eight while its radius stays below the cell, and done at
the first such triangle.
"""
@inline function nearwithin(p::Vec3f, r2::Float32, part, V, F, keys, start, tris, slots, cap, lox, loy, loz, cell)
    r = sqrt(r2)
    x0, x1 = cellof(p[1] - r, lox, cell), cellof(p[1] + r, lox, cell)
    y0, y1 = cellof(p[2] - r, loy, cell), cellof(p[2] + r, loy, cell)
    z0, z1 = cellof(p[3] - r, loz, cell), cellof(p[3] + r, loz, cell)
    @inbounds for z in z0:z1, y in y0:y1, x in x0:x1
        incells(x, y, z) || continue
        for i in cellrange(keys, start, slots, cap, cellkey64(part, x, y, z))
            f = tris[i]
            e = closestpoint(p, vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f])) - p
            dot(e, e) <= r2 && return true
        end
    end
    return false
end

"""
    distances(dev, grid, points, n; part = nothing) -> Mantle.Buffer{Float32}

The distance from each of the first `n` columns of `points` `(3, ≥ n)` Float32 to the
grid's surface, to the part `part[i]` names for point `i` (`(≥ n,)` Int32; part 0 without
it); `Inf32` beyond one cell.
"""
function distances(dev::Mantle.Device, s::SurfaceGrid, P::Mantle.Buffer, n::Int; part::Union{Nothing,Mantle.Buffer} = nothing)
    out = Mantle.Buffer(dev, Float32, max(n, 1))
    parts = part === nothing ? fill!(Mantle.Buffer(dev, Int32, max(n, 1)), 0) : part
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, pointdistances!, (out, P, parts, gridargs(s)..., n), n; name = "surfacegrid/distances")
    runonce!(g)
    part === nothing && Mantle.free!(parts)
    return out
end

function pointdistances!(out, P, part, V, F, keys, start, tris, slots, cap, lox, loy, loz, cell, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    _, d2 = nearest(vertex(P, i), @inbounds(part[i]), V, F, keys, start, tris, slots, cap, lox, loy, loz, cell)
    @inbounds out[i] = sqrt(d2)
    return nothing
end
