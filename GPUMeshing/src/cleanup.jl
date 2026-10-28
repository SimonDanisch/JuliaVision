"""
cumesh's mesh clean-up on the device: `fill_holes`, `remove_degenerate_faces`,
`remove_duplicate_faces`, `remove_unreferenced_vertices`, `unify_face_orientations`,
`repair_non_manifold_edges` and `remove_small_connected_components`, on the connectivity of
`topology.jl`. Every function takes `Mantle.Buffer`s, consumes the ones it replaces, and
returns the mesh it made; the host-matrix methods upload, run and download for callers that
hold arrays.
"""

upload(dev::Mantle.Device, x::AbstractMatrix{Float32}) = Mantle.Buffer(dev, Matrix{Float32}(x))
upload(dev::Mantle.Device, x::AbstractMatrix{<:Integer}) = Mantle.Buffer(dev, Matrix{Int32}(x))
download(b::Mantle.Buffer) = Array(Mantle.storage(b))

"""
Fixed point with 32 fractional bits, rounded half away from zero, what the
fill's sums accumulate in: integer atomics are exact and order-free where float
ones are neither. Exact in the argument's own precision (the scale is a power
of two and `y - floor(y)` is exact), so a Float32 needs no Float64, which Metal
does not have.
"""
@inline function fixed(x::AbstractFloat)
    y = abs(x) * oftype(x, 4294967296)
    f = floor(y)
    a = unsafe_trunc(Int64, f) + Int64(y - f >= oftype(x, 0.5))
    return signbit(x) ? -a : a
end
fixed(x::Real) = fixed(Float64(x))

"""
Add the fixed-point `v` to the sum held in `words[2j-1]` (low) and `words[2j]`
(high). Two 32-bit atomics with the carry, since a 64-bit atomic add is not
portable (Metal has 64-bit atomic min and max only); in any order the words end
as the exact Int64 sum modulo 2^64.
"""
@inline function addfixed!(words, j, v::Int64)
    u = reinterpret(UInt64, v)
    lo = u % UInt32
    @inbounds newlo = Atomix.@atomic :monotonic words[Int32(2) * j - Int32(1)] += lo
    @inbounds Atomix.@atomic :monotonic words[Int32(2) * j] += (u >> 32) % UInt32 + UInt32(newlo < lo)
    return nothing
end

"""The sum [`addfixed!`](@ref) accumulated at `j`."""
@inline fixedsum(words, j) =
    @inbounds reinterpret(Int64, UInt64(words[Int32(2) * j]) << 32 | UInt64(words[Int32(2) * j - Int32(1)]))

# ── fill holes ────────────────────────────────────────────────────────────────

"""
    fillholes(dev, V, F; maxperimeter = 3e-2) -> (V, F)

cumesh's `fill_holes`: every hole whose perimeter is below `maxperimeter` (in
the unit cube's units) is closed by a fan, one face per boundary edge, around
the mean of its edges' midpoints. The dual grid leaves a hole wherever a voxel
of an edge's four is missing.

A hole is cumesh's boundary loop: the boundary edges (on one face) joined at
every vertex that has exactly two of them and no edge of three or more faces
([`manifoldboundaryadjacency`](@ref)), kept when every edge meets another of its
own component at both ends. So two holes that touch at a vertex are two holes,
each filled. A fan face is `(hi, lo, centre)` over the edge's vertex ids, as
cumesh winds it from its sorted key.
"""
function fillholes(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer; maxperimeter::Real = 3e-2)
    nv, nf = size(V, 2), size(F, 2)
    edges = Edges(dev, F, nf)
    bnd, nb = edgeswithfaces(dev, edges, 1)
    if nb == 0
        Mantle.free!(edges)
        Mantle.free!(bnd)
        return V, F
    end
    vb = VertexBoundaries(dev, edges, bnd, nb, nv)
    adj, m = manifoldboundaryadjacency(dev, vb, nv)
    comp = components(dev, adj, m, nb)
    Mantle.free!(adj)
    isloop = fill!(Mantle.Buffer(dev, Int32, nb), 1)
    perimeter = fill!(Mantle.Buffer(dev, UInt32, 2nb), 0)
    centre = fill!(Mantle.Buffer(dev, UInt32, 6nb), 0)
    count = fill!(Mantle.Buffer(dev, Int32, nb), 0)
    keep = Mantle.Buffer(dev, Int32, nb)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, loopcheck!, (isloop, comp, edges.keys, edges.starts, bnd, vb.off, vb.v2b, nb), nb;
                     name = "fillholes/loops")
    Mantle.dispatch!(g, loopsums!, (perimeter, centre, count, comp, edges.keys, edges.starts, bnd, V, nb), nb;
                     name = "fillholes/sums")
    Mantle.dispatch!(g, keeploops!, (keep, isloop, perimeter, comp, fixed(maxperimeter), nb), nb;
                     name = "fillholes/keep")
    runonce!(g)
    Mantle.free!(vb)
    pos = prefixsum(dev, keep)
    nnew = lastentry(pos)
    if nnew == 0
        foreach(Mantle.free!, (edges, bnd, comp, isloop, perimeter, centre, count, keep, pos))
        return V, F
    end
    fflag = Mantle.Buffer(dev, Int32, nb)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, edgekept!, (fflag, keep, comp, nb), nb; name = "fillholes/fan edges")
    runonce!(g)
    fpos = prefixsum(dev, fflag)
    nadd = lastentry(fpos)
    V2 = Mantle.Buffer(dev, Float32, (3, nv + nnew))
    F2 = Mantle.Buffer(dev, Int32, (3, nf + nadd))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, copycols!, (V2, V, nv), nv; name = "fillholes/copy vertices")
    Mantle.dispatch!(g, copycols!, (F2, F, nf), nf; name = "fillholes/copy faces")
    Mantle.dispatch!(g, newvertices!, (V2, keep, pos, centre, count, nv, nb), nb; name = "fillholes/centres")
    Mantle.dispatch!(g, newfaces!, (F2, fflag, fpos, comp, pos, edges.keys, edges.starts, bnd, nv, nf, nb), nb;
                     name = "fillholes/fans")
    runonce!(g)
    foreach(Mantle.free!, (edges, bnd, comp, isloop, perimeter, centre, count, keep, pos, fflag, fpos, V, F))
    return V2, F2
end

function fillholes(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                   kw...)
    V, F = fillholes(dev, upload(dev, vertices), upload(dev, faces); kw...)
    return download(V), download(F)
end

"""Other boundary edges of `v`'s component than `b` at `v`: none means `b`'s
component does not close at that end."""
@inline function othersatvertex(comp, off, v2b, v, b, self)
    c = Int32(0)
    @inbounds for i in (off[v] + Int32(1)):off[v + Int32(1)]
        o = v2b[i]
        (o != b && comp[o] == self) && (c += Int32(1))
    end
    return c
end

function loopcheck!(isloop, comp, keys, starts, bnd, off, v2b, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds begin
        key = keys[starts[bnd[b]]]
        self = comp[b]
        if othersatvertex(comp, off, v2b, edgelow(key), b, self) == Int32(0) ||
           othersatvertex(comp, off, v2b, edgehigh(key), b, self) == Int32(0)
            isloop[self] = Int32(0)
        end
    end
    return nothing
end

function loopsums!(perimeter, centre, count, comp, keys, starts, bnd, V, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds begin
        key = keys[starts[bnd[b]]]
        r = comp[b]
        p = vertex(V, edgelow(key))
        q = vertex(V, edgehigh(key))
        m = (p + q) * 0.5f0
        addfixed!(perimeter, r, fixed(norm(q - p)))
        base = Int32(3) * (r - Int32(1))
        addfixed!(centre, base + Int32(1), fixed(m[1]))
        addfixed!(centre, base + Int32(2), fixed(m[2]))
        addfixed!(centre, base + Int32(3), fixed(m[3]))
        Atomix.@atomic :monotonic count[r] += Int32(1)
    end
    return nothing
end

"""A component's root, when it is a loop below the perimeter, gets a centre vertex."""
function keeploops!(keep, isloop, perimeter, comp, maxfixed, nb)
    r = Int32(KI.get_global_id().x)
    r <= nb || return nothing
    @inbounds keep[r] = (comp[r] == r && isloop[r] == Int32(1) && fixedsum(perimeter, r) < maxfixed) ? Int32(1) : Int32(0)
    return nothing
end

function edgekept!(fflag, keep, comp, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds fflag[b] = keep[comp[b]]
    return nothing
end

"""
A kept loop's centre vertex, its fixed-point sum over its edge count. In
Float32: the sum rounds once to Float32 and once more in the division (the
scale is exact), at most an ulp from the once-rounded quotient.
"""
function newvertices!(V2, keep, pos, centre, count, nv, nb)
    r = Int32(KI.get_global_id().x)
    r <= nb || return nothing
    @inbounds if keep[r] == Int32(1)
        j = nv + pos[r]
        s = Float32(count[r]) * 4294967296f0
        base = Int32(3) * (r - Int32(1))
        V2[1, j] = Float32(fixedsum(centre, base + Int32(1))) / s
        V2[2, j] = Float32(fixedsum(centre, base + Int32(2))) / s
        V2[3, j] = Float32(fixedsum(centre, base + Int32(3))) / s
    end
    return nothing
end

function newfaces!(F2, fflag, fpos, comp, pos, keys, starts, bnd, nv, nf, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds if fflag[b] == Int32(1)
        j = nf + fpos[b]
        key = keys[starts[bnd[b]]]
        F2[1, j] = edgehigh(key)
        F2[2, j] = edgelow(key)
        F2[3, j] = nv + pos[comp[b]]
    end
    return nothing
end

# ── degenerate, duplicate and unreferenced ────────────────────────────────────

"""
    removedegenerate(dev, V, F; absolute = 1e-24, relative = 1e-12) -> (V, F)

cumesh's `remove_degenerate_faces`: a face goes when two of its vertices are
one, or when its area is below `absolute` and below `relative` times its
longest edge squared. Such a face has no plane, so a chart cannot map it.
Vertices no face is left on go with it ([`removeunreferenced`](@ref)), as after
every removal in cumesh.
"""
function removedegenerate(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer;
                          absolute::Real = 1e-24, relative::Real = 1e-12)
    nf = size(F, 2)
    flags = Mantle.Buffer(dev, Int32, nf)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markdegenerate!, (flags, V, F, Float32(absolute), Float32(relative), nf), nf;
                     name = "degenerate/mark")
    runonce!(g)
    F1, _ = compactcolumns(dev, F, flags, nf)
    Mantle.free!(flags)
    Mantle.free!(F)
    return removeunreferenced(dev, V, F1)
end

function removedegenerate(dev::Mantle.Device, vertices::AbstractMatrix{Float32},
                          faces::AbstractMatrix{<:Integer}; kw...)
    V, F = removedegenerate(dev, upload(dev, vertices), upload(dev, faces); kw...)
    return download(V), download(F)
end

"""A face to keep is flagged 1; cumesh's `mark_degenerate_faces_kernel`, in Float32 as there."""
function markdegenerate!(flags, V, F, absolute, relative, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        i, j, k = F[1, f], F[2, f], F[3, f]
        if i == j || j == k || k == i
            flags[f] = Int32(0)
            return nothing
        end
        a, b, c = vertex(V, i), vertex(V, j), vertex(V, k)
        e0, e1, e2 = b - a, c - b, a - c
        longest = max(norm(e0), norm(e1), norm(e2))
        area = norm(cross(e0, e1)) * 0.5f0
        flags[f] = area < min(relative * longest * longest, absolute) ? Int32(0) : Int32(1)
    end
    return nothing
end

@inline function sorted3(a, b, c)
    a, b = min(a, b), max(a, b)
    b, c = min(b, c), max(b, c)
    a, b = min(a, b), max(a, b)
    return (a, b, c)
end

"""
    removeduplicates(dev, F) -> F

cumesh's `remove_duplicate_faces`: a face over the same three vertices as an earlier one,
whichever way round, goes; the faces left keep their order. Two of them make every edge of
the pair look non-manifold. The faces are sorted by their sorted vertex triples; the first
face of each run of equal triples stays.
"""
function removeduplicates(dev::Mantle.Device, F::Mantle.Buffer)
    nf = size(F, 2)
    keys = Mantle.Buffer(dev, NTuple{3,Int32}, max(nf, 1))
    perm = Mantle.Buffer(dev, Int32, max(nf, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, facekeys!, (keys, F, nf), nf; name = "duplicates/keys")
    Mantle.dispatch!(g, iota!, (perm, nf, 1), nf; name = "duplicates/iota")
    runonce!(g)
    nf > 1 && AK.sort_by_key!(view(Mantle.storage(keys), 1:nf), view(Mantle.storage(perm), 1:nf))
    starts, nr = runs(dev, keys, nf)
    first = fill!(Mantle.Buffer(dev, Int32, max(nr, 1)), typemax(Int32))
    flags = fill!(Mantle.Buffer(dev, Int32, max(nf, 1)), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, firstofrun!, (first, perm, starts, nr), nr; name = "duplicates/first")
    Mantle.dispatch!(g, flagfirst!, (flags, first, nr), nr; name = "duplicates/flags")
    runonce!(g)
    F1, _ = compactcolumns(dev, F, flags, nf)
    foreach(Mantle.free!, (keys, perm, starts, first, flags, F))
    return F1
end

removeduplicates(dev::Mantle.Device, faces::AbstractMatrix{<:Integer}) = download(removeduplicates(dev, upload(dev, faces)))

function facekeys!(keys, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds keys[f] = sorted3(F[1, f], F[2, f], F[3, f])
    return nothing
end

"""The least face of run `r` (the sort need not be stable)."""
function firstofrun!(first, perm, starts, nr)
    r = Int32(KI.get_global_id().x)
    r <= nr || return nothing
    m = typemax(Int32)
    @inbounds for i in starts[r]:(starts[r + Int32(1)] - Int32(1))
        m = min(m, perm[i])
    end
    @inbounds first[r] = m
    return nothing
end

function flagfirst!(flags, first, nr)
    r = Int32(KI.get_global_id().x)
    r <= nr || return nothing
    @inbounds flags[first[r]] = Int32(1)
    return nothing
end

"""
    removeunreferenced(dev, V, F) -> (V, F)

cumesh's `remove_unreferenced_vertices`: the vertices some face uses,
renumbered in order, and `F` remapped in place.
"""
function removeunreferenced(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer)
    nv, nf = size(V, 2), size(F, 2)
    used = fill!(Mantle.Buffer(dev, Int32, nv), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markused!, (used, F, nf), nf; name = "unreferenced/mark")
    runonce!(g)
    pos = prefixsum(dev, used)
    nv2 = lastentry(pos)
    V2 = Mantle.Buffer(dev, eltype(V), (3, max(nv2, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactcols!, (V2, V, used, pos, nv), nv; name = "unreferenced/vertices")
    Mantle.dispatch!(g, remapfaces!, (F, pos, nf), nf; name = "unreferenced/faces")
    runonce!(g)
    foreach(Mantle.free!, (used, pos, V))
    return V2, F
end

function markused!(used, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        used[F[1, f]] = Int32(1)
        used[F[2, f]] = Int32(1)
        used[F[3, f]] = Int32(1)
    end
    return nothing
end

function remapfaces!(F, pos, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        F[1, f] = pos[F[1, f]]
        F[2, f] = pos[F[2, f]]
        F[3, f] = pos[F[3, f]]
    end
    return nothing
end

# ── orientation ───────────────────────────────────────────────────────────────

"""
    orientfaces(dev, F) -> F

cumesh's `unify_face_orientations`: across every edge two faces share, they
have to walk the edge in opposite directions. The pairs that do not are hooked
with a flip ([`orientedcomponents`](@ref)), and every face whose parity to its
component's root is odd is turned over. Edges of more than two faces are not
walked: a non-manifold edge does not say which way its faces should turn. Which
way round a component ends up is its lowest face's; nothing here looks at the
volume.

As in cumesh, the hooking races: two pairs hooking the same root in one round
carry each its own parity, and the atomic min keeps one of them, so where the
surface has odd cycles (sheets of the dual grid meeting) which faces end up
inconsistent varies between runs.
"""
function orientfaces(dev::Mantle.Device, F::Mantle.Buffer)
    nf = size(F, 2)
    edges = Edges(dev, F, nf)
    adj, m = manifoldfaceadjacency(dev, edges)
    Mantle.free!(edges)
    flipped = Mantle.Buffer(dev, Int32, max(m, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, flipflags!, (flipped, adj, F, m), max(m, 1); name = "orient/flags")
    runonce!(g)
    ids = orientedcomponents(dev, adj, flipped, m, nf)
    F2 = Mantle.Buffer(dev, Int32, (3, nf))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, flipfaces!, (F2, F, ids, nf), nf; name = "orient/flip")
    runonce!(g)
    foreach(Mantle.free!, (adj, flipped, ids, F))
    return F2
end

orientfaces(dev::Mantle.Device, faces::AbstractMatrix{<:Integer}) = download(orientfaces(dev, upload(dev, faces)))

"""Where face `f` has vertex `v`, 0 when it does not."""
@inline function localindex(F, f, v)
    @inbounds begin
        F[1, f] == v && return Int32(1)
        F[2, f] == v && return Int32(2)
        F[3, f] == v && return Int32(3)
    end
    return Int32(0)
end

"""cumesh's `get_flip_flags_kernel`: the two faces of pair `i` walk their
shared edge the same way round, so one must flip."""
function flipflags!(flipped, adj, F, m)
    i = Int32(KI.get_global_id().x)
    i <= m || return nothing
    @inbounds begin
        f1, f2 = adj[1, i], adj[2, i]
        a1 = a2 = b1 = b2 = Int32(0)
        found = Int32(0)
        for k in Int32(1):Int32(3)
            l = localindex(F, f2, F[k, f1])
            l == Int32(0) && continue
            if found == Int32(0)
                a1, a2 = k, l
            else
                b1, b2 = k, l
            end
            found += Int32(1)
            found == Int32(2) && break
        end
        d1 = (b1 - a1 + Int32(3)) % Int32(3)
        d2 = (b2 - a2 + Int32(3)) % Int32(3)
        flipped[i] = d1 == d2 ? Int32(1) : Int32(0)
    end
    return nothing
end

function flipfaces!(F2, F, ids, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        flip = ids[f] & Int32(1) == Int32(1)
        F2[1, f] = F[1, f]
        F2[2, f] = flip ? F[3, f] : F[2, f]
        F2[3, f] = flip ? F[2, f] : F[3, f]
    end
    return nothing
end

# ── non-manifold edges ────────────────────────────────────────────────────────

"""
    splitnonmanifold(dev, V, F) -> (V, F)

cumesh's `repair_non_manifold_edges`, by splitting vertices: around every vertex, the faces
that reach each other across edges of exactly two faces form one fan, and every fan after the
first gets a copy of the vertex. Where two sheets meet along an edge they come apart, each a
manifold of its own, with the copies at the same place, so nothing moves.

A face corner (`3(f - 1) + k`, vertex `F[k, f]`) is joined to the corner of the same vertex
across every edge of two faces at both its ends; the [`components`](@ref) of the corners are
the fans. Sorted by (vertex, fan root), a vertex's first fan (the one with its lowest face)
keeps it and the others get new vertices, numbered by vertex and then by fan.
"""
function splitnonmanifold(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer)
    nv, nf = size(V, 2), size(F, 2)
    nc = 3nf
    edges = Edges(dev, F, nf)
    eids, m = edgeswithfaces(dev, edges, 2)
    adj = Mantle.Buffer(dev, Int32, (2, max(2m, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, fancorners!, (adj, eids, edges.keys, edges.starts, edges.perm, F, m), m; name = "nonmanifold/pairs")
    runonce!(g)
    foreach(Mantle.free!, (edges, eids))
    ids = components(dev, adj, 2m, nc)
    Mantle.free!(adj)
    keys = Mantle.Buffer(dev, UInt64, max(nc, 1))
    corner = Mantle.Buffer(dev, Int32, max(nc, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, fankeys!, (keys, F, ids, nc), nc; name = "nonmanifold/keys")
    Mantle.dispatch!(g, iota!, (corner, nc, 1), nc; name = "nonmanifold/iota")
    runonce!(g)
    Mantle.free!(ids)
    nc > 1 && AK.sort_by_key!(view(Mantle.storage(keys), 1:nc), view(Mantle.storage(corner), 1:nc))
    starts, nfan = runs(dev, keys, nc)
    extra = Mantle.Buffer(dev, Int32, max(nfan, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, extrafans!, (extra, keys, starts, nfan), nfan; name = "nonmanifold/extra")
    runonce!(g)
    pos = prefixsum(dev, extra)
    nnew = lastentry(pos)
    if nnew == 0
        foreach(Mantle.free!, (keys, corner, starts, extra, pos))
        return V, F
    end
    V2 = Mantle.Buffer(dev, eltype(V), (3, nv + nnew))
    F2 = Mantle.Buffer(dev, Int32, (3, nf))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, copycols!, (V2, V, nv), nv; name = "nonmanifold/copy vertices")
    Mantle.dispatch!(g, copycols!, (F2, F, nf), nf; name = "nonmanifold/copy faces")
    Mantle.dispatch!(g, splitfans!, (V2, F2, V, extra, pos, keys, starts, corner, nv, nfan), nfan; name = "nonmanifold/split")
    runonce!(g)
    foreach(Mantle.free!, (keys, corner, starts, extra, pos, V, F))
    return V2, F2
end

function splitnonmanifold(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    V, F = splitnonmanifold(dev, upload(dev, vertices), upload(dev, faces))
    return download(V), download(F)
end

"""The corner of face `f` at vertex `v`."""
@inline cornerof(F, f, v) = Int32(3) * (f - Int32(1)) + localindex(F, f, v)

"""The two corners at each end of manifold edge `eids[i]` are one fan."""
function fancorners!(adj, eids, keys, starts, perm, F, m)
    i = Int32(KI.get_global_id().x)
    i <= m || return nothing
    @inbounds begin
        s = starts[eids[i]]
        key = keys[s]
        lo, hi = edgelow(key), edgehigh(key)
        f1, f2 = edgeface(perm, s), edgeface(perm, s + Int32(1))
        adj[1, 2i - Int32(1)] = cornerof(F, f1, lo)
        adj[2, 2i - Int32(1)] = cornerof(F, f2, lo)
        adj[1, 2i] = cornerof(F, f1, hi)
        adj[2, 2i] = cornerof(F, f2, hi)
    end
    return nothing
end

function fankeys!(keys, F, ids, nc)
    c = Int32(KI.get_global_id().x)
    c <= nc || return nothing
    @inbounds keys[c] = (UInt64(F[c] % UInt32) << 32) | UInt64(ids[c] % UInt32)
    return nothing
end

"""A fan that is not its vertex's first."""
function extrafans!(extra, keys, starts, nfan)
    r = Int32(KI.get_global_id().x)
    r <= nfan || return nothing
    @inbounds extra[r] = (r > Int32(1) && keys[starts[r]] >> 32 == keys[starts[r - Int32(1)]] >> 32) ? Int32(1) : Int32(0)
    return nothing
end

function splitfans!(V2, F2, V, extra, pos, keys, starts, corner, nv, nfan)
    r = Int32(KI.get_global_id().x)
    r <= nfan || return nothing
    @inbounds if extra[r] == Int32(1)
        v = (keys[starts[r]] >> 32) % Int32
        w = nv + pos[r]
        V2[1, w], V2[2, w], V2[3, w] = V[1, v], V[2, v], V[3, v]
        for i in starts[r]:(starts[r + Int32(1)] - Int32(1))
            F2[corner[i]] = w
        end
    end
    return nothing
end

# ── small components ──────────────────────────────────────────────────────────

"""
    removesmallcomponents(dev, V, F; minarea = 1e-5) -> (V, F)

cumesh's `remove_small_connected_components`: pieces of the surface (faces joined through
shared vertices) whose total area is below `minarea` go, and the vertices only they used.
Areas are summed per component in fixed point, so the sums do not depend on the order.
"""
function removesmallcomponents(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer; minarea::Real = 1e-5)
    nv, nf = size(V, 2), size(F, 2)
    adj = Mantle.Buffer(dev, Int32, (2, max(2nf, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, facevertexpairs!, (adj, F, nf), nf; name = "components/face pairs")
    runonce!(g)
    ids = components(dev, adj, 2nf, nv)
    Mantle.free!(adj)
    area = fill!(Mantle.Buffer(dev, UInt32, 2nv), 0)
    flags = Mantle.Buffer(dev, Int32, max(nf, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, componentarea!, (area, ids, V, F, nf), nf; name = "components/area")
    Mantle.dispatch!(g, flaglarge!, (flags, area, ids, F, fixed(minarea), nf), nf; name = "components/keep")
    runonce!(g)
    F1, _ = compactcolumns(dev, F, flags, nf)
    foreach(Mantle.free!, (ids, area, flags, F))
    return removeunreferenced(dev, V, F1)
end

function removesmallcomponents(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                               kw...)
    V, F = removesmallcomponents(dev, upload(dev, vertices), upload(dev, faces); kw...)
    return download(V), download(F)
end

function facevertexpairs!(adj, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        adj[1, 2f - Int32(1)], adj[2, 2f - Int32(1)] = F[1, f], F[2, f]
        adj[1, 2f], adj[2, 2f] = F[1, f], F[3, f]
    end
    return nothing
end

function componentarea!(area, ids, V, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        a, b, c = vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f])
        addfixed!(area, ids[F[1, f]], fixed(0.5f0 * norm(cross(b - a, c - a))))
    end
    return nothing
end

function flaglarge!(flags, area, ids, F, minfixed, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds flags[f] = fixedsum(area, ids[F[1, f]]) >= minfixed ? Int32(1) : Int32(0)
    return nothing
end
