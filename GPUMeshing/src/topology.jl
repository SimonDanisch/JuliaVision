"""
Mesh connectivity on the device, cumesh's `CuMesh` caches computed on demand: the edges of a
triangle mesh with their faces, the boundary of each vertex, face adjacency across manifold
edges, and connected components. A mesh is `(3, nv)` vertices and `(3, nf)` one-based Int32
faces, both `Mantle.Buffer`s.
"""

@inline edgekey(a, b) = a < b ? (UInt64(a % UInt32) << 32) | UInt64(b % UInt32) : (UInt64(b % UInt32) << 32) | UInt64(a % UInt32)
@inline edgelow(key::UInt64) = (key >> 32) % Int32
@inline edgehigh(key::UInt64) = (key & 0xffffffff) % Int32

# ── edges ─────────────────────────────────────────────────────────────────────

"""
    Edges(dev, F, nf)

Every face edge as the key `(min << 32) | max` of its vertex ids, sorted, and
where each came from: `keys[i]` is an edge of face [`edgeface`](@ref)`(edges, i)`.
The runs of equal keys are the mesh's `ne` distinct edges, edge `e` being
`keys[starts[e]:starts[e + 1] - 1]`: the run's length is how many faces the edge
is on, and the run's faces are its faces. cumesh sorts the same keys and
run-length-encodes them, then finds an edge's faces again by walking one end's
faces; carrying the face through the sort gives them for nothing.
"""
struct Edges
    keys::Mantle.Buffer{UInt64,1}
    perm::Mantle.Buffer{Int32,1}
    starts::Mantle.Buffer{Int32,1}
    ne::Int
end

function Edges(dev::Mantle.Device, F::Mantle.Buffer, nf::Int)
    n = 3nf
    keys = Mantle.Buffer(dev, UInt64, max(n, 1))
    perm = Mantle.Buffer(dev, Int32, max(n, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, expandedges!, (keys, F, nf), nf; name = "edges/expand")
    Mantle.dispatch!(g, iota!, (perm, n, 1), n; name = "edges/iota")
    runonce!(g)
    n > 1 && AK.sort_by_key!(view(Mantle.storage(keys), 1:n), view(Mantle.storage(perm), 1:n))
    starts, ne = runs(dev, keys, n)
    return Edges(keys, perm, starts, ne)
end

function Mantle.free!(e::Edges)
    Mantle.free!(e.keys)
    Mantle.free!(e.perm)
    Mantle.free!(e.starts)
    return nothing
end

"""The face sorted position `i` of `edges.keys` belongs to."""
@inline edgeface(perm, i) = @inbounds (perm[i] - Int32(1)) ÷ Int32(3) + Int32(1)

function expandedges!(keys, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds for k in Int32(1):Int32(3)
        keys[Int32(3) * (f - Int32(1)) + k] = edgekey(F[k, f], F[k % Int32(3) + Int32(1), f])
    end
    return nothing
end

"""
    edgelist(dev, edges) -> (E, ukeys)

The `ne` distinct edges `(2, ne)` Int32, low vertex first, in key order, and their keys
`ukeys` (sorted, for [`edgeindex`](@ref)).
"""
function edgelist(dev::Mantle.Device, edges::Edges)
    E = Mantle.Buffer(dev, Int32, (2, max(edges.ne, 1)))
    ukeys = Mantle.Buffer(dev, UInt64, max(edges.ne, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, edgerows!, (E, ukeys, edges.keys, edges.starts, edges.ne), edges.ne; name = "edges/list")
    runonce!(g)
    return E, ukeys
end

function edgerows!(E, ukeys, keys, starts, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds begin
        key = keys[starts[e]]
        ukeys[e] = key
        E[1, e] = edgelow(key)
        E[2, e] = edgehigh(key)
    end
    return nothing
end

"""The index of the edge `a b` in the sorted distinct keys `ukeys[1:ne]` (binary search), 0 if absent."""
@inline function edgeindex(ukeys, ne, a, b)
    key = edgekey(a, b)
    lo, hi = Int32(1), Int32(ne)
    @inbounds while lo < hi
        mid = (lo + hi) >> 1
        ukeys[mid] < key ? (lo = mid + Int32(1)) : (hi = mid)
    end
    return (ne > 0 && @inbounds ukeys[lo] == key) ? lo : Int32(0)
end

"""Flag the edges on exactly `c` faces."""
function flagfacecount!(flags, starts, ne, c)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds flags[e] = (starts[e + 1] - starts[e] == Int32(c)) ? Int32(1) : Int32(0)
    return nothing
end

"""Edges on `c` faces, as indices into the edge list, and how many."""
function edgeswithfaces(dev::Mantle.Device, edges::Edges, c::Int)
    flags = Mantle.Buffer(dev, Int32, max(edges.ne, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, flagfacecount!, (flags, edges.starts, edges.ne, c), edges.ne; name = "edges/flag $c")
    runonce!(g)
    idx, m = select(dev, flags, edges.ne)
    Mantle.free!(flags)
    return idx, m
end

"""
    manifoldfaceadjacency(dev, edges) -> (adj, m)

cumesh's `manifold_face_adj`: the two faces across every edge of exactly two,
`(2, m)`.
"""
function manifoldfaceadjacency(dev::Mantle.Device, edges::Edges)
    eids, m = edgeswithfaces(dev, edges, 2)
    adj = Mantle.Buffer(dev, Int32, (2, max(m, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, facepairs!, (adj, eids, edges.starts, edges.perm, m), max(m, 1); name = "edges/face pairs")
    runonce!(g)
    Mantle.free!(eids)
    return adj, m
end

function facepairs!(adj, eids, starts, perm, m)
    i = Int32(KI.get_global_id().x)
    i <= m || return nothing
    @inbounds begin
        s = starts[eids[i]]
        adj[1, i] = edgeface(perm, s)
        adj[2, i] = edgeface(perm, s + Int32(1))
    end
    return nothing
end

# ── boundaries ────────────────────────────────────────────────────────────────

"""
    VertexBoundaries(dev, edges, bnd, nb, nv)

cumesh's `vert2bound`: vertex `v`'s boundary edges (indices into `bnd`) are
`v2b[off[v] + 1:off[v + 1]]`; `nbound[v]` is how many, and `pinched[v]` is set
when an edge of three or more faces ends at `v`. A vertex is manifold when it
has at most two boundary edges and is not pinched, which is what cumesh's
`vert_is_manifold` asks of the vertex's edges.
"""
struct VertexBoundaries
    off::Mantle.Buffer{Int32,1}
    v2b::Mantle.Buffer{Int32,1}
    nbound::Mantle.Buffer{Int32,1}
    pinched::Mantle.Buffer{Int32,1}
end

function VertexBoundaries(dev::Mantle.Device, edges::Edges, bnd::Mantle.Buffer, nb::Int, nv::Int)
    cnt = fill!(Mantle.Buffer(dev, Int32, nv + 1), 0)
    pinched = fill!(Mantle.Buffer(dev, Int32, nv), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, boundarycount!, (cnt, edges.keys, edges.starts, bnd, nb), nb; name = "boundaries/count")
    Mantle.dispatch!(g, pinch!, (pinched, edges.keys, edges.starts, edges.ne), edges.ne; name = "boundaries/pinched")
    runonce!(g)
    off = prefixsum(dev, cnt)
    nbound = Mantle.Buffer(dev, Int32, nv)
    v2b = Mantle.Buffer(dev, Int32, 2nb)
    fill!(cnt, 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, boundaryfill!, (v2b, off, cnt, edges.keys, edges.starts, bnd, nb), nb; name = "boundaries/fill")
    Mantle.dispatch!(g, boundarycounts!, (nbound, off, nv), nv; name = "boundaries/nbound")
    runonce!(g)
    Mantle.free!(cnt)
    return VertexBoundaries(off, v2b, nbound, pinched)
end

function Mantle.free!(vb::VertexBoundaries)
    Mantle.free!(vb.off)
    Mantle.free!(vb.v2b)
    Mantle.free!(vb.nbound)
    Mantle.free!(vb.pinched)
    return nothing
end

function boundarycount!(cnt, keys, starts, bnd, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds begin
        key = keys[starts[bnd[b]]]
        Atomix.@atomic cnt[edgelow(key) + Int32(1)] += Int32(1)
        Atomix.@atomic cnt[edgehigh(key) + Int32(1)] += Int32(1)
    end
    return nothing
end

function boundaryfill!(v2b, off, cnt, keys, starts, bnd, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds begin
        key = keys[starts[bnd[b]]]
        lo, hi = edgelow(key), edgehigh(key)
        s = Atomix.@atomic cnt[lo] += Int32(1)
        v2b[off[lo] + s] = b
        s = Atomix.@atomic cnt[hi] += Int32(1)
        v2b[off[hi] + s] = b
    end
    return nothing
end

function boundarycounts!(nbound, off, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    @inbounds nbound[v] = off[v + 1] - off[v]
    return nothing
end

function pinch!(pinched, keys, starts, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds if starts[e + 1] - starts[e] > Int32(2)
        key = keys[starts[e]]
        pinched[edgelow(key)] = Int32(1)
        pinched[edgehigh(key)] = Int32(1)
    end
    return nothing
end

"""
    manifoldboundaryadjacency(dev, vb, nv) -> (adj, m)

cumesh's `manifold_bound_adj`: at every manifold vertex with two boundary edges,
the pair of them `(2, m)`, as indices into the boundary list.
"""
function manifoldboundaryadjacency(dev::Mantle.Device, vb::VertexBoundaries, nv::Int)
    flags = Mantle.Buffer(dev, Int32, nv)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, flagmanifoldboundary!, (flags, vb.nbound, vb.pinched, nv), nv; name = "boundaries/manifold")
    runonce!(g)
    vids, m = select(dev, flags, nv)
    adj = Mantle.Buffer(dev, Int32, (2, max(m, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, boundarypairs!, (adj, vids, vb.off, vb.v2b, m), max(m, 1); name = "boundaries/pairs")
    runonce!(g)
    Mantle.free!(flags)
    Mantle.free!(vids)
    return adj, m
end

function flagmanifoldboundary!(flags, nbound, pinched, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    @inbounds flags[v] = (nbound[v] == Int32(2) && pinched[v] == Int32(0)) ? Int32(1) : Int32(0)
    return nothing
end

function boundarypairs!(adj, vids, off, v2b, m)
    i = Int32(KI.get_global_id().x)
    i <= m || return nothing
    @inbounds begin
        o = off[vids[i]]
        adj[1, i] = v2b[o + Int32(1)]
        adj[2, i] = v2b[o + Int32(2)]
    end
    return nothing
end

# ── connected components ──────────────────────────────────────────────────────

"""
    components(dev, adj, m, n) -> ids

cumesh's hook-and-compress union-find over `n` items joined by the pairs `adj`
`(2, m)`: `ids[i]` is the root of `i`'s component, the least member index. Hook
every pair whose roots differ (the higher root points at the lower, by atomic
min), compress every path, repeat until a round hooks nothing.
"""
function components(dev::Mantle.Device, adj::Mantle.Buffer, m::Int, n::Int)
    ids = Mantle.Buffer(dev, Int32, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, iota!, (ids, n, 1), n; name = "components/init")
    runonce!(g)
    m == 0 && return ids
    flag = Mantle.Buffer(dev, Int32, 1)
    while true
        fill!(flag, 1)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, hook!, (adj, m, ids, flag), m; name = "components/hook")
        Mantle.dispatch!(g, compress!, (ids, n), n; name = "components/compress")
        runonce!(g)
        lastentry(flag) == 1 && break
    end
    Mantle.free!(flag)
    return ids
end

@inline function findroot(ids, i)
    @inbounds begin
        r = ids[i]
        while r != ids[r]
            r = ids[r]
        end
        return r
    end
end

function hook!(adj, m, ids, flag)
    i = Int32(KI.get_global_id().x)
    i <= m || return nothing
    @inbounds begin
        r0 = findroot(ids, adj[1, i])
        r1 = findroot(ids, adj[2, i])
        r0 == r1 && return nothing
        hi, lo = r0 > r1 ? (r0, r1) : (r1, r0)
        Atomix.@atomic min(ids[hi], lo)
        flag[1] = Int32(0)
    end
    return nothing
end

function compress!(ids, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds ids[i] = findroot(ids, i)
    return nothing
end

"""
    orientedcomponents(dev, adj, flipped, m, n) -> ids

[`components`](@ref) with a parity: `ids[i] = (root << 1) | flip`, `flip` set
when `i` has to turn over to agree with its root, given that the pair `k` of
`adj` disagrees when `flipped[k]` is set (cumesh's `unify_face_orientations`).
"""
function orientedcomponents(dev::Mantle.Device, adj::Mantle.Buffer, flipped::Mantle.Buffer, m::Int, n::Int)
    ids = Mantle.Buffer(dev, Int32, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, iota!, (ids, n, 2), n; name = "components/init")
    runonce!(g)
    m == 0 && return ids
    flag = Mantle.Buffer(dev, Int32, 1)
    while true
        fill!(flag, 1)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, hookoriented!, (adj, flipped, m, ids, flag), m; name = "components/hook")
        Mantle.dispatch!(g, compressoriented!, (ids, n), n; name = "components/compress")
        runonce!(g)
        lastentry(flag) == 1 && break
    end
    Mantle.free!(flag)
    return ids
end

"""The root of `i` and the parity of the path to it."""
@inline function findorientedroot(ids, i)
    @inbounds begin
        r = ids[i] >> 1
        f = ids[i] & Int32(1)
        while r != ids[r] >> 1
            f ⊻= ids[r] & Int32(1)
            r = ids[r] >> 1
        end
        return r, f
    end
end

function hookoriented!(adj, flipped, m, ids, flag)
    i = Int32(KI.get_global_id().x)
    i <= m || return nothing
    @inbounds begin
        r0, f0 = findorientedroot(ids, adj[1, i])
        r1, f1 = findorientedroot(ids, adj[2, i])
        r0 == r1 && return nothing
        hi, lo = r0 > r1 ? (r0, r1) : (r1, r0)
        Atomix.@atomic min(ids[hi], (lo << 1) | (flipped[i] ⊻ f0 ⊻ f1))
        flag[1] = Int32(0)
    end
    return nothing
end

function compressoriented!(ids, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        r, f = findorientedroot(ids, i)
        ids[i] = (r << 1) | f
    end
    return nothing
end
