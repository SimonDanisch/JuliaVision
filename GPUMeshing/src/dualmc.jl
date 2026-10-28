"""
Dual marching cubes with Hermite data on a sparse set of voxels (Nielson's dual marching
cubes, vertices placed as in Ju et al.'s Hermite dual contouring). One-vertex-per-voxel dual
contouring ([`dualcontour`](@ref), [`meshdual`](@ref)) joins every surface piece in a voxel
into one vertex, so where a voxel's face has its corners inside and outside crosswise, four
quads meet at one edge and the mesh is no manifold. Here a voxel gets one vertex per patch of
the surface that marching cubes would put in it ([`MCTABLE`](@ref)'s loops, ambiguous faces
decided by the asymptotic decider from the face's own four values, so both voxels sharing it
agree), placed by [`qefsolve`](@ref) on that patch's crossings alone. Every crossed grid edge
gets a quad of the four patch vertices around it: a closed, consistently wound two-manifold.
"""

"""
    cellloops(cube, joined) -> Vector{Vector{Int}}

The loops [`celltriangles`](@ref) fans, as lists of edge indices into [`EDGE`](@ref): one per
patch of the surface in the cell.
"""
function cellloops(cube::Int, joined::Int)
    below(c) = (cube >> (c - 1)) & 1 == 1
    adj = zeros(Int, 2, 12)
    for f in 1:6
        cs, es = FACE[f], FACEEDGE[f]
        crossed = ntuple(k -> below(cs[k]) != below(cs[mod1(k + 1, 4)]), 4)
        for (p, q) in facepairs(crossed, (joined >> (f - 1)) & 1 == 1)
            p == 0 && continue
            a, b = es[p], es[q]
            adj[adj[1, a] == 0 ? 1 : 2, a] = b
            adj[adj[1, b] == 0 ? 1 : 2, b] = a
        end
    end
    loops = Vector{Int}[]
    seen = falses(12)
    for start in 1:12
        (adj[1, start] == 0 || seen[start]) && continue
        loop = Int[]
        e, prev = start, 0
        while true
            push!(loop, e)
            seen[e] = true
            next = adj[1, e] == prev ? adj[2, e] : adj[1, e]
            prev, e = e, next
            e == start && break
        end
        push!(loops, loop)
    end
    return loops
end

"""
    dmctable() -> Vector{UInt64}

For every [`MCTABLE`](@ref) case, `cube + 256 joined + 1`: the number of patches in bits
0–2, and for edge `e` the patch it lies on (1–4, 0 for none) in bits `3e .. 3e + 2`.
"""
function dmctable()
    table = zeros(UInt64, 256 * 64)
    for key in 0:(256 * 64 - 1)
        loops = cellloops(key & 255, key >> 8)
        code = UInt64(length(loops))
        for (p, loop) in enumerate(loops), e in loop
            code |= UInt64(p) << (3e)
        end
        table[key + 1] = code
    end
    return table
end

"""[`dmctable`](@ref), evaluated once when the package is compiled."""
const DMCTABLE = dmctable()

@inline patchcount(code::UInt64) = Int32(code & UInt64(7))
@inline patchof(code::UInt64, e) = Int32((code >> (UInt64(3) * UInt64(e))) & UInt64(7))

"""
The local index of the edge along axis `a` whose low corner sits at offset `(x, y, z)` in a
cell: [`EDGE`](@ref) numbering, `edgeowner`'s inverse.
"""
@inline localedge(a, x, y, z) = a == Int32(1) ? Int32(1) + y + Int32(2) * z :
                                a == Int32(2) ? Int32(5) + x + Int32(2) * z : Int32(9) + x + Int32(2) * y

"""
    DualPatches(dev, keys)

The patches of a sparse set of voxels: `keys` `(n,)` Int32 is each voxel's
[`MCTABLE`](@ref) case, `cube + 256 joined`, as [`cellkey`](@ref) computes it from the
voxel's eight corner values (bit `c - 1` of `cube` set where corner `c` is below the level,
inside). Voxel `i`'s patches are vertices `pos[i] - np_i + 1 .. pos[i]` of the `count` in all;
`table` is [`DMCTABLE`](@ref) on the device, where [`patchof`](@ref) finds an edge's patch.
A caller places each patch's vertex (its own field's crossings, [`qefsolve`](@ref)) and
hands them to [`dualmc`](@ref).
"""
struct DualPatches
    keys::Mantle.Buffer{Int32,1}
    table::Mantle.Buffer{UInt64,1}
    pos::Mantle.Buffer{Int32,1}
    count::Int
end

function DualPatches(dev::Mantle.Device, keys::Mantle.Buffer{Int32,1})
    n = length(keys)
    table = Mantle.Buffer(dev, DMCTABLE)
    cnt = Mantle.Buffer(dev, Int32, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, patchcounts!, (cnt, keys, table, n), n; name = "dualmc/patches")
    runonce!(g)
    pos = prefixsum(dev, cnt)
    Mantle.free!(cnt)
    return DualPatches(keys, table, pos, lastentry(pos))
end

"""Frees the table and offsets; the keys stay the caller's."""
Mantle.free!(dp::DualPatches) = (Mantle.free!(dp.table); Mantle.free!(dp.pos); nothing)

"""
    dualmc(dev, grid, patches, P) -> (V, F)

The surface through the voxels of `grid`: `P` `(3, patches.count)` Float32 holds each
patch's vertex (see [`DualPatches`](@ref)), and every crossed grid edge gets a quad of the
four patch vertices around it, faces toward the outside (the field's positive side).
`Mantle.Buffer`s `(3, V)` Float32 and `(3, F)` one-based Int32 out, the vertices no quad
uses dropped.
"""
function dualmc(dev::Mantle.Device, grid::SparseGrid, dp::DualPatches, P::Mantle.Buffer)
    n, res, vox = grid.n, grid.res, grid.map
    np = dp.count
    keys, table, pos = dp.keys, dp.table, dp.pos
    ns = 3n
    quads = Mantle.Buffer(dev, Int32, (4, ns))
    qflag = Mantle.Buffer(dev, Int32, ns)
    crossing = Mantle.Buffer(dev, Int32, (3, n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, patchquads!, (quads, qflag, crossing, pos, keys, table, grid.coords, vox.keys, vox.vals,
                                      vox.capacity, Int32(res), n), ns; name = "dualmc/quads")
    runonce!(g)
    used = fill!(Mantle.Buffer(dev, Int32, max(np, 1)), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markusedquads!, (used, quads, qflag, ns), ns; name = "dualmc/used")
    runonce!(g)
    qpos = prefixsum(dev, qflag)
    nq = lastentry(qpos)
    vpos = prefixsum(dev, used)
    nvert = lastentry(vpos)
    V = Mantle.Buffer(dev, Float32, (3, max(nvert, 1)))
    F = Mantle.Buffer(dev, Int32, (3, 2 * max(nq, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactdual!, (V, P, used, vpos, 1f0, 0f0, 0f0, 0f0, np), max(np, 1); name = "dualmc/compact")
    Mantle.dispatch!(g, splitquads!, (F, quads, crossing, qflag, qpos, vpos, V, ns), ns; name = "dualmc/triangles")
    runonce!(g)
    foreach(Mantle.free!, (quads, qflag, crossing, used, qpos, vpos))
    return V, F
end

"""
    dualmc(dev, grid, keys, hermite, boxes) -> (V, F)

[`dualmc`](@ref) from Hermite data stored per edge, for small sets of voxels: `hermite`
`(6, 12, n)` Float32 holds, for each of a voxel's twelve edges ([`EDGE`](@ref) order) that
crosses, the crossing point and the field's unit outward gradient there, and `boxes` `(6, n)`
each voxel's low and high corner, which its vertices stay in.
"""
function dualmc(dev::Mantle.Device, grid::SparseGrid, keys::Mantle.Buffer, hermite::Mantle.Buffer, boxes::Mantle.Buffer)
    n = grid.n
    dp = DualPatches(dev, keys)
    P = Mantle.Buffer(dev, Float32, (3, max(dp.count, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, patchvertices!, (P, dp.pos, keys, dp.table, hermite, boxes, n), n; name = "dualmc/vertices")
    runonce!(g)
    V, F = dualmc(dev, grid, dp, P)
    Mantle.free!(dp); Mantle.free!(P)
    return V, F
end

function patchcounts!(cnt, keys, table, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds cnt[i] = patchcount(table[keys[i] + Int32(1)])
    return nothing
end

"""Each patch's vertex: the QEF of its crossings, kept in its voxel's box."""
function patchvertices!(P, pos, keys, table, hermite, boxes, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        code = table[keys[i] + Int32(1)]
        np = patchcount(code)
        first = pos[i] - np
        lo = Vec3f(boxes[1, i], boxes[2, i], boxes[3, i])
        hi = Vec3f(boxes[4, i], boxes[5, i], boxes[6, i])
        for p in Int32(1):np
            a11 = a12 = a13 = a22 = a23 = a33 = 0f0
            b = Vec3f(0f0)
            m = Vec3f(0f0)
            c = 0f0
            for e in Int32(1):Int32(12)
                patchof(code, e) == p || continue
                x = Vec3f(hermite[1, e, i], hermite[2, e, i], hermite[3, e, i])
                nn = Vec3f(hermite[4, e, i], hermite[5, e, i], hermite[6, e, i])
                a11 += nn[1] * nn[1]; a12 += nn[1] * nn[2]; a13 += nn[1] * nn[3]
                a22 += nn[2] * nn[2]; a23 += nn[2] * nn[3]; a33 += nn[3] * nn[3]
                b += nn * dot(nn, x)
                m += x
                c += 1f0
            end
            v = qefsolve((a11, a12, a13, a22, a23, a33), b, m / c)
            v = Vec3f(clamp(v[1], lo[1], hi[1]), clamp(v[2], lo[2], hi[2]), clamp(v[3], lo[3], hi[3]))
            P[1, first + p], P[2, first + p], P[3, first + p] = v
        end
    end
    return nothing
end

"""
The quad around voxel `i`'s far edge along axis `a` (slot `s`, as [`edgequads`](@ref) numbers
them): the patch vertex of each of the four voxels around it, wound by the sign change.
"""
function patchquads!(quads, qflag, crossing, pos, keys, table, coords, hkeys, hvals, cap, res, n)
    s = Int32(KI.get_global_id().x)
    s <= Int32(3) * n || return nothing
    @inbounds begin
        i = (s - Int32(1)) ÷ Int32(3) + Int32(1)
        a = (s - Int32(1)) % Int32(3) + Int32(1)
        qflag[s] = Int32(0)
        crossing[a, i] = Int32(0)
        cube = keys[i] & Int32(255)
        # The far edge: low corner at 1 in the other two axes.
        fx, fy, fz = a == Int32(1) ? (Int32(0), Int32(1), Int32(1)) : a == Int32(2) ? (Int32(1), Int32(0), Int32(1)) :
                                                                       (Int32(1), Int32(1), Int32(0))
        lo = Int32(1) + fx + Int32(2) * fy + Int32(4) * fz
        hi = lo + (a == Int32(1) ? Int32(1) : a == Int32(2) ? Int32(2) : Int32(4))
        blo = (cube >> (lo - Int32(1))) & Int32(1)
        bhi = (cube >> (hi - Int32(1))) & Int32(1)
        blo == bhi && return nothing
        x, y, z = coords[1, i], coords[2, i], coords[3, i]
        for k in Int32(1):Int32(4)
            o = aroundoffset(a, k)
            id = lookupcoord(hkeys, hvals, x + o[1], y + o[2], z + o[3], res, cap)
            id == Int32(0) && return nothing
            e = localedge(a, fx - o[1], fy - o[2], fz - o[3])
            code = table[keys[id] + Int32(1)]
            quads[k, s] = pos[id] - patchcount(code) + patchof(code, e)
        end
        qflag[s] = Int32(1)
        # Inside below the level at the low end: negative to positive along the axis.
        crossing[a, i] = blo == Int32(1) ? Int32(1) : Int32(-1)
    end
    return nothing
end
