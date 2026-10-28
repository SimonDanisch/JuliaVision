"""
Mesh simplification on the device: rounds of parallel edge collapses, cumesh's `simplify`.
Each round prices every edge by the error quadric of its ends at the point it would collapse
to, and collapses at once every edge that is cheap enough and the cheapest on every face
around both its ends: those edges' rings do not overlap.

Two ways to stop:

- `simplify(dev, V, F, target)`: cumesh's, until at most `target` faces are left, the cost
  threshold raised tenfold whenever a round removes less than 1% of the faces. Quadrics are
  taken afresh from the current faces each round, and edge length and sliver penalties steer
  the order, as upstream does.
- `simplify(dev, V, F; tolerance)`: within `tolerance` of the original surface, with the
  error checked at every collapse as Klein, Liebich and Straßer do (*Mesh reduction with
  error control*, 1996). Each original vertex belongs to a face of the current mesh within
  `tolerance` of it; a collapse goes ahead only if every original vertex of the faces it
  changes stays within `tolerance` of the faces it makes, and those faces' corners, centroids
  and edge midpoints stay within `tolerance` of the original ([`SurfaceGrid`](@ref)). A
  point whose face changed then belongs to the nearest face around the collapse's vertex.
  Each vertex carries the quadric of every original face it absorbed (Garland and Heckbert's)
  and how many those are; it orders the collapses (the mean squared distance of an edge's
  collapse point to those planes) and places them: the cheapest of the edge's midpoint, its
  two ends and the point the quadric is least at (solved around the midpoint with
  Lindstrom's truncated pseudo-inverse, [`qefsolve`](@ref)). An end lets a vertex in the
  middle of a flat face or a straight edge go into a corner for free, the quadric's point lets
  a curved surface's error fall on both sides of it. A collapse keeps a closed mesh closed and
  two-manifold (the link condition: the ends share exactly the two neighbours of the edge's
  faces) and turns no face over. At the end the distance is measured both ways over the whole
  mesh.

Each vertex's quadric is kept relative to the vertex itself and moved along with it, so its
terms stay as small as the distances it measures: in Float32, a quadric about the origin
would cancel away a 14 µm tolerance on a part 50 mm out.

`(3, V)` Float32 vertices and `(3, F)` one-based Int32 faces, as `Mantle.Buffer`s (consumed);
returns new ones with the vertices that are still used.
"""

function simplify(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, target::Int;
                  thresh::Float64 = 1e-8, lambda_edge_length::Float32 = 1f-2, lambda_skinny::Float32 = 1f-3)
    nf = size(F, 2)
    nf <= target && return V, F
    while true
        V, F, _, _ = simplifystep(dev, V, F, nothing, Float32(thresh), lambda_edge_length, lambda_skinny, false)
        n = size(F, 2)
        n <= target && break
        (nf - n) / nf < 1e-2 && (thresh *= 10)
        nf = n
    end
    return V, F
end

function simplify(dev::Mantle.Device, V0::Mantle.Buffer, F0::Mantle.Buffer; tolerance::Real, rounds::Int = 1000,
                  part::Union{Nothing,Mantle.Buffer{Int32,1}} = nothing)
    tol = Float32(tolerance)
    cell = checkcell(dev, V0, F0, tol)
    parts = part === nothing ? fill!(Mantle.Buffer(dev, Int32, size(V0, 2)), 0) : part
    w = Witness(dev, V0, F0, parts, tol, cell)
    V = Mantle.Buffer(dev, copy(Mantle.storage(V0)))
    F = Mantle.Buffer(dev, copy(Mantle.storage(F0)))
    Q = facequadrics(dev, V, F)
    for _ in 1:rounds
        # The quadric only gates which collapses get checked: a mean squared plane distance
        # four times the tolerance's square is past it.
        V, F, Q, n = simplifystep(dev, V, F, Q, 4tol^2, 0f0, 0f0, true; witness = w)
        n == 0 && break
    end
    Mantle.free!(Q)
    d = surfacedistance(dev, w.grid, V0, parts, V, F, w.part; cell = checkcell(dev, V, F, tol))
    P = w.part
    w.part = Mantle.Buffer(dev, Int32, 1)
    Mantle.free!(w)
    d <= tol || error("simplify: $d from the original, past the tolerance $tolerance")
    foreach(Mantle.free!, (V0, F0, parts))
    part === nothing || return V, F, P
    Mantle.free!(P)
    return V, F
end

"""
The original surface a tolerance-bounded simplification answers to: a [`SurfaceGrid`](@ref)
of it, and for each original vertex (`grid.vertices`) the face of the current mesh it
belongs to (`owner`), or the vertex around which it picks one in the next round (`target`,
0 for none), with the faces' points listed by face (`points[pstart[f] + 1:pend[f]]`).
For each vertex of the current mesh, `part` is the mesh it belongs to (several simplify
together, each against its own original), and `stale` says that a collapse at it failed the
check since a collapse last changed a face around it: an edge between two stale vertices is
not tried again, it would fail again (or, its ends stale from other edges, merely wait).
"""
mutable struct Witness
    grid::SurfaceGrid
    part::Mantle.Buffer{Int32,1}
    stale::Mantle.Buffer{Int32,1}
    owner::Mantle.Buffer{Int32,1}
    target::Mantle.Buffer{Int32,1}
    points::Mantle.Buffer{Int32,1}
    pend::Mantle.Buffer{Int32,1}
    tol::Float32
    width::Float32
    maxpoints::Int32
end

function Witness(dev::Mantle.Device, V0::Mantle.Buffer, F0::Mantle.Buffer, part::Mantle.Buffer{Int32,1}, tol::Float32, cell::Real;
                 width::Real = 4, maxpoints::Integer = 8192)
    nv, nf = size(V0, 2), size(F0, 2)
    grid = SurfaceGrid(dev, V0, F0; cell, part)
    v2f, off = vertexfaces(dev, F0, nv, nf)
    owner = Mantle.Buffer(dev, Int32, nv)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, firstface!, (owner, v2f, off, nv), nv; name = "simplify/first faces")
    runonce!(g)
    foreach(Mantle.free!, (v2f, off))
    return Witness(grid, Mantle.Buffer(dev, copy(Mantle.storage(part))), fill!(Mantle.Buffer(dev, Int32, nv), 0), owner, fill!(Mantle.Buffer(dev, Int32, nv), 0), Mantle.Buffer(dev, Int32, max(nv, 1)),
                   Mantle.Buffer(dev, Int32, 1), tol, Float32(width), Int32(maxpoints))
end

function Mantle.free!(w::Witness)
    Mantle.free!(w.grid)
    foreach(Mantle.free!, (w.part, w.stale, w.owner, w.target, w.points, w.pend))
    return nothing
end

"""
Points whose face changed pick the nearest face around their `target` vertex (the current
mesh's `v2f`, `off`), and every face's points are listed: `points`, `pend`.
"""
function ownpoints!(dev::Mantle.Device, w::Witness, V::Mantle.Buffer, F::Mantle.Buffer, v2f, off)
    np, nf = size(w.grid.vertices, 2), size(F, 2)
    cnt = fill!(Mantle.Buffer(dev, Int32, nf), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, reassign!, (w.owner, w.target, w.grid.vertices, V, F, v2f, off, np), np; name = "simplify/reassign")
    Mantle.dispatch!(g, ownercount!, (cnt, w.owner, np), np; name = "simplify/owner count")
    runonce!(g)
    Mantle.free!(w.pend)
    w.pend = prefixsum(dev, cnt)
    fill!(cnt, 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, ownerfill!, (w.points, w.pend, cnt, w.owner, np), np; name = "simplify/owner fill")
    runonce!(g)
    Mantle.free!(cnt)
    return nothing
end

"""
The cell of the grids that check the distance for the mesh `V`, `F`: twice the side of its
mean face, so a cell holds a few faces and a face reaches into a few cells, and never below
four times the tolerance, so a point within it finds its closest point in the 27 cells
around it.
"""
function checkcell(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, tol::Float32)
    area, _ = areavolume(dev, V, F)
    return max(4tol, 2f0 * sqrt(Float32(area) / max(size(F, 2), 1)))
end

"""
The larger of the two one-sided distances between the original surface (in `original`, a
[`SurfaceGrid`](@ref) of it, with its vertices `V0` of parts `P0`) and the mesh `V`, `F` of
parts `P` (`cell` its grid's): from every original vertex to the mesh, and from the mesh's
vertices, face centroids and edge midpoints to the original, each within its own part.
`Inf32` where a point has nothing within one cell.
"""
function surfacedistance(dev::Mantle.Device, original::SurfaceGrid, V0::Mantle.Buffer, P0::Mantle.Buffer, V::Mantle.Buffer,
                         F::Mantle.Buffer, P::Mantle.Buffer; cell::Real)
    result = SurfaceGrid(dev, V, F; cell, part = P)
    d1 = distances(dev, result, V0, size(V0, 2); part = P0)
    nf = size(F, 2)
    S = Mantle.Buffer(dev, Float32, (3, 4nf))
    SP = Mantle.Buffer(dev, Int32, 4nf)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, facesamples!, (S, SP, V, F, P, nf), nf; name = "simplify/samples")
    runonce!(g)
    d2 = distances(dev, original, S, 4nf; part = SP)
    d3 = distances(dev, original, V, size(V, 2); part = P)
    m = max(maximum(Mantle.storage(d1)), maximum(Mantle.storage(d2)), maximum(Mantle.storage(d3)))
    foreach(Mantle.free!, (result, d1, S, SP, d2, d3))
    return m
end

"""Each face's centroid and the midpoints of its three edges, four columns per face, with the face's part."""
function facesamples!(S, SP, V, F, P, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        a, b, c = vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f])
        j = Int32(4) * (f - Int32(1))
        q = P[F[1, f]]
        for (k, p) in ((Int32(1), (a + b + c) / 3f0), (Int32(2), (a + b) / 2f0), (Int32(3), (b + c) / 2f0), (Int32(4), (c + a) / 2f0))
            S[1, j + k], S[2, j + k], S[3, j + k] = p
            SP[j + k] = q
        end
    end
    return nothing
end

"""The faces at each vertex: `v2f[off[v] + 1:off[v + 1]]`."""
function vertexfaces(dev::Mantle.Device, F::Mantle.Buffer, nv::Int, nf::Int)
    cnt = fill!(Mantle.Buffer(dev, Int32, nv + 1), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, facecount!, (cnt, F, nf), nf; name = "simplify/facecount")
    runonce!(g)
    off = prefixsum(dev, cnt)
    fill!(cnt, 0)
    v2f = Mantle.Buffer(dev, Int32, max(3nf, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, fillfaces!, (v2f, off, cnt, F, nf), nf; name = "simplify/vertexfaces")
    runonce!(g)
    Mantle.free!(cnt)
    return v2f, off
end

"""Each vertex's quadric about itself and plane count, `(11, nv)`: the planes of its faces, unweighted."""
function facequadrics(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer)
    nv, nf = size(V, 2), size(F, 2)
    v2f, off = vertexfaces(dev, F, nv, nf)
    Q = Mantle.Buffer(dev, Float32, (11, nv))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, quadrics!, (Q, V, F, v2f, off, nv), nv; name = "simplify/quadrics")
    runonce!(g)
    foreach(Mantle.free!, (v2f, off))
    return Q
end

"""
    simplifystep(dev, V, F, Q, thresh, lambda_edge_length, lambda_skinny, link; witness) -> (V, F, Q, collapsed)

One round of parallel edge collapses. With `Q === nothing` the quadrics are taken from the
current faces (cumesh's `simplify_step`); otherwise `Q` holds each vertex's carried quadric,
merged into the kept end on a collapse and returned compacted with the vertices. `link`
refuses collapses that break the link condition and selects `passes` times, each pass among
the edges no collapse of the earlier ones came near. A [`Witness`](@ref) checks each pass's
winners, the edges that would collapse, before they do: one that would move the surface
farther than its tolerance from the original drops out, and the next pass may take a
neighbour instead. Only the winners, a few in a hundred, pay for the check.
"""
function simplifystep(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, Qin, thresh::Float32,
                      lambda_edge_length::Float32, lambda_skinny::Float32, link::Bool; witness::Union{Nothing,Witness} = nothing,
                      passes::Int = 4)
    nv, nf = size(V, 2), size(F, 2)
    v2f, off = vertexfaces(dev, F, nv, nf)
    witness === nothing || ownpoints!(dev, witness, V, F, v2f, off)
    # The edges, each once from its lower end: `ecnt[u + 1]` of them from `u`.
    ecnt = fill!(Mantle.Buffer(dev, Int32, nv + 1), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, edgecount!, (ecnt, F, v2f, off, nv), nv; name = "simplify/edgecount")
    runonce!(g)
    eoff = prefixsum(dev, ecnt)
    ne = lastentry(eoff)
    edges = Mantle.Buffer(dev, Int32, (2, max(ne, 1)))
    efaces = Mantle.Buffer(dev, Int32, max(ne, 1))
    boundary = fill!(Mantle.Buffer(dev, UInt8, nv), 0)
    Q = Qin === nothing ? Mantle.Buffer(dev, Float32, (11, nv)) : Qin
    cost = Mantle.Buffer(dev, Float32, max(ne, 1))
    choice = fill!(Mantle.Buffer(dev, Int32, max(ne, 1)), 0)
    best = fill!(Mantle.Buffer(dev, UInt64, nf), typemax(UInt64))
    vkept = fill!(Mantle.Buffer(dev, Int32, nv), 1)
    fkept = fill!(Mantle.Buffer(dev, Int32, nf), 1)
    merged = fill!(Mantle.Buffer(dev, Int32, nv), 0)
    win = Mantle.Buffer(dev, Int32, max(ne, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, edges!, (edges, efaces, eoff, F, v2f, off, nv), nv; name = "simplify/edges")
    Mantle.dispatch!(g, boundary!, (boundary, edges, efaces, ne), ne; name = "simplify/boundary")
    Qin === nothing && Mantle.dispatch!(g, quadrics!, (Q, V, F, v2f, off, nv), nv; name = "simplify/quadrics")
    Mantle.dispatch!(g, collapsecost!, (cost, choice, V, F, v2f, off, edges, boundary, Q, lambda_edge_length,
                                        lambda_skinny, link ? Int32(1) : Int32(0), ne), ne; name = "simplify/cost")
    witness === nothing || Mantle.dispatch!(g, dropstale!, (cost, edges, witness.stale, ne), ne; name = "simplify/stale")
    scramble = link ? Int32(1) : Int32(0)
    unit = witness === nothing ? thresh : witness.tol^2
    width = witness === nothing ? 0f0 : witness.width
    # cumesh's one pass; the tolerance mode passes again over the edges whose faces the earlier
    # passes left alone.
    for pass in 1:(link ? passes : 1)
        pass == 1 || Mantle.dispatch!(g, resetbest!, (best, nf), nf; name = "simplify/reset")
        Mantle.dispatch!(g, propagate!, (best, edges, v2f, off, cost, scramble, unit, width, ne), ne; name = "simplify/propagate")
        Mantle.dispatch!(g, winners!, (win, edges, v2f, off, cost, best, thresh, scramble, unit, width, ne), ne;
                         name = "simplify/winners")
        if witness !== nothing
            Mantle.dispatch!(g, withinbound!, (cost, win, choice, V, F, v2f, off, edges, boundary, Q, witness.tol^2, witness.part,
                                               witness.stale, witness.points, witness.pend, gridargs(witness.grid)...,
                                               witness.maxpoints, ne), ne; name = "simplify/bound")
        end
        Mantle.dispatch!(g, collapse!, (V, F, Q, edges, v2f, off, win, choice, boundary, vkept, fkept, merged, ne), ne;
                         name = "simplify/collapse")
        link && Mantle.dispatch!(g, untouched!, (cost, edges, v2f, off, F, merged, ne), ne; name = "simplify/untouched")
    end
    runonce!(g)
    vid = prefixsum(dev, vkept)
    fid = prefixsum(dev, fkept)
    nv2, nf2 = lastentry(vid), lastentry(fid)
    V2 = Mantle.Buffer(dev, Float32, (3, nv2))
    F2 = Mantle.Buffer(dev, Int32, (3, nf2))
    Q2 = Qin === nothing ? nothing : Mantle.Buffer(dev, Float32, (11, nv2))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactvertices!, (V2, V, vkept, vid, nv), nv; name = "simplify/compact vertices")
    Qin === nothing || Mantle.dispatch!(g, compactquadrics!, (Q2, Q, vkept, vid, nv), nv; name = "simplify/compact quadrics")
    Mantle.dispatch!(g, compactfaces!, (F2, F, fkept, fid, vid, nf), nf; name = "simplify/compact faces")
    if witness !== nothing
        np = size(witness.grid.vertices, 2)
        Mantle.dispatch!(g, ownerupdate!, (witness.owner, witness.target, F, fid, merged, vid, np), np; name = "simplify/owners")
        Mantle.dispatch!(g, freshen!, (witness.stale, F, merged, nf), nf; name = "simplify/freshen")
        part2, stale2 = Mantle.Buffer(dev, Int32, nv2), Mantle.Buffer(dev, Int32, nv2)
        Mantle.dispatch!(g, compactparts!, (part2, stale2, witness.part, witness.stale, vkept, vid, nv), nv;
                         name = "simplify/compact parts")
    end
    runonce!(g)
    if witness !== nothing
        Mantle.free!(witness.part); Mantle.free!(witness.stale)
        witness.part, witness.stale = part2, stale2
    end
    for b in (off, v2f, ecnt, eoff, edges, efaces, boundary, Q, cost, choice, best, vkept, fkept, merged, win, vid, fid, V, F)
        Mantle.free!(b)
    end
    return V2, F2, Q2, nf - nf2
end

# ── kernels ──────────────────────────────────────────────────────────────────

"""
cumesh's `QEM`: the error quadric of a set of planes, the upper triangle of
`Σ p pᵀ` for `p = (n, d)`, in Float32.
"""
struct Quadric
    e00::Float32; e01::Float32; e02::Float32; e03::Float32
    e11::Float32; e12::Float32; e13::Float32
    e22::Float32; e23::Float32
    e33::Float32
end

"""The quadric of the plane `n ⋅ p + d = 0`."""
function Quadric(n::Vec3f, d::Float32)
    a, b, c = n
    return Quadric(a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d)
end

Base.:+(p::Quadric, q::Quadric) =
    Quadric(p.e00 + q.e00, p.e01 + q.e01, p.e02 + q.e02, p.e03 + q.e03, p.e11 + q.e11, p.e12 + q.e12,
            p.e13 + q.e13, p.e22 + q.e22, p.e23 + q.e23, p.e33 + q.e33)

"""`[p; 1]ᵀ Q [p; 1]`, summed in cumesh's order."""
function evaluate(q::Quadric, p::Vec3f)
    x, y, z = p
    r = q.e00 * x * x
    r += 2f0 * q.e01 * x * y
    r += 2f0 * q.e02 * x * z
    r += 2f0 * q.e03 * x
    r += q.e11 * y * y
    r += 2f0 * q.e12 * y * z
    r += 2f0 * q.e13 * y
    r += q.e22 * z * z
    r += 2f0 * q.e23 * z
    r += q.e33
    return r
end

"""
The quadric `q` about a point `o` as a quadric about `o + δ`: `p ↦ q(p + δ)`, with `A` the
same, `b + A δ` and `c + 2 b ⋅ δ + δᵀ A δ`.
"""
@inline function moved(q::Quadric, δ::Vec3f)
    x, y, z = δ
    ax = q.e00 * x + q.e01 * y + q.e02 * z
    ay = q.e01 * x + q.e11 * y + q.e12 * z
    az = q.e02 * x + q.e12 * y + q.e22 * z
    c = q.e33 + 2f0 * (q.e03 * x + q.e13 * y + q.e23 * z) + (ax * x + ay * y + az * z)
    return Quadric(q.e00, q.e01, q.e02, q.e03 + ax, q.e11, q.e12, q.e13 + ay, q.e22, q.e23 + az, c)
end

@inline quadric(Q, v) = @inbounds Quadric(Q[1, v], Q[2, v], Q[3, v], Q[4, v], Q[5, v], Q[6, v], Q[7, v],
                                          Q[8, v], Q[9, v], Q[10, v])

@inline function setquadric!(Q, v, q::Quadric)
    @inbounds Q[1, v], Q[2, v], Q[3, v], Q[4, v], Q[5, v] = q.e00, q.e01, q.e02, q.e03, q.e11
    @inbounds Q[6, v], Q[7, v], Q[8, v], Q[9, v], Q[10, v] = q.e12, q.e13, q.e22, q.e23, q.e33
    return nothing
end

@inline hasvertex(F, f, v) = @inbounds(F[1, f] == v || F[2, f] == v || F[3, f] == v)

"""Whether a face among `v2f[lo:i - 1]` has `v`: a neighbour met before."""
@inline function seenbefore(F, v2f, lo, i, v)
    for k in lo:(i - 1)
        hasvertex(F, @inbounds(v2f[k]), v) && return true
    end
    return false
end

"""Whether some face around `w` (`v2f[off[w] + 1:off[w + 1]]`) has vertex `u`."""
@inline function aroundhas(F, v2f, off, w, u)
    @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
        hasvertex(F, v2f[i], u) && return true
    end
    return false
end

"""
The link condition of edge `(a, b)` on a closed two-manifold: the vertices adjacent to both
ends are exactly the two opposite the edge's two faces. Counts each common neighbour of `a`
once.
"""
@inline function linkholds(F, v2f, off, a, b)
    common = Int32(0)
    @inbounds lo, hi = off[a] + Int32(1), off[a + Int32(1)]
    @inbounds for i in lo:hi
        f = v2f[i]
        for k in Int32(1):Int32(3)
            u = F[k, f]
            (u == a || u == b) && continue
            seenbefore(F, v2f, lo, i, u) && continue
            aroundhas(F, v2f, off, b, u) && (common += Int32(1))
        end
    end
    return common == Int32(2)
end

function facecount!(cnt, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    for k in Int32(1):Int32(3)
        @inbounds Atomix.@atomic :monotonic cnt[F[k, f] + Int32(1)] += Int32(1)
    end
    return nothing
end

function fillfaces!(v2f, off, cnt, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    for k in Int32(1):Int32(3)
        @inbounds v = F[k, f]
        slot = @inbounds Atomix.@atomic :monotonic cnt[v] += Int32(1)
        @inbounds v2f[off[v] + slot] = f
    end
    return nothing
end

"""How many distinct neighbours above it vertex `u` has: its edges, from the lower end."""
function edgecount!(ecnt, F, v2f, off, nv)
    u = Int32(KI.get_global_id().x)
    u <= nv || return nothing
    @inbounds lo, hi = off[u] + Int32(1), off[u + Int32(1)]
    n = Int32(0)
    for i in lo:hi
        @inbounds f = v2f[i]
        for k in Int32(1):Int32(3)
            @inbounds v = F[k, f]
            (v > u && !seenbefore(F, v2f, lo, i, v)) && (n += Int32(1))
        end
    end
    @inbounds ecnt[u + Int32(1)] = n
    return nothing
end

"""
Vertex `u`'s edges `(u, v)`, `v > u`, in increasing `v`, and how many faces
each is on: every edge once, in the order cumesh's radix sort of `(min, max)`
keys gives, which decides which of two equal costs wins.
"""
function edges!(edges, efaces, eoff, F, v2f, off, nv)
    u = Int32(KI.get_global_id().x)
    u <= nv || return nothing
    @inbounds lo, hi = off[u] + Int32(1), off[u + Int32(1)]
    @inbounds base = eoff[u]
    n = Int32(0)
    for i in lo:hi
        @inbounds f = v2f[i]
        for k in Int32(1):Int32(3)
            @inbounds v = F[k, f]
            (v > u && !seenbefore(F, v2f, lo, i, v)) || continue
            c = Int32(0)
            for j in lo:hi
                hasvertex(F, @inbounds(v2f[j]), v) && (c += Int32(1))
            end
            # Insertion into the sorted run so far.
            j = base + n
            @inbounds while j > base && edges[2, j] > v
                edges[1, j + Int32(1)] = u
                edges[2, j + Int32(1)] = edges[2, j]
                efaces[j + Int32(1)] = efaces[j]
                j -= Int32(1)
            end
            @inbounds edges[1, j + Int32(1)] = u
            @inbounds edges[2, j + Int32(1)] = v
            @inbounds efaces[j + Int32(1)] = c
            n += Int32(1)
        end
    end
    return nothing
end

"""Both ends of an edge on one face only are boundary vertices."""
function boundary!(boundary, edges, efaces, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds if efaces[e] == Int32(1)
        boundary[edges[1, e]] = 0x01
        boundary[edges[2, e]] = 0x01
    end
    return nothing
end

"""Each vertex's quadric about itself: the planes of its faces, unweighted."""
function quadrics!(Q, V, F, v2f, off, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    q = Quadric(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0)
    planes = 0f0
    o = vertex(V, v)
    @inbounds for i in (off[v] + Int32(1)):off[v + Int32(1)]
        f = v2f[i]
        a = vertex(V, F[1, f])
        n = cross(vertex(V, F[2, f]) - a, vertex(V, F[3, f]) - a)
        l = sqrt(dot(n, n))
        # A degenerate face has no plane; its neighbours carry the surface there.
        l > 0f0 || continue
        n = n * (1f0 / l)
        q += Quadric(n, -dot(n, a - o))
        planes += 1f0
    end
    setquadric!(Q, v, q)
    @inbounds Q[11, v] = planes
    return nothing
end

"""
Where edge `(a, b)` collapses to: with `choice` 0 its midpoint, or the end on a boundary when
only one is (cumesh's rule); with 1 or 2 the end `a` or `b`; with 3 the least point of the
two ends' quadrics, solved around the midpoint.
"""
@inline function collapsepoint(V, boundary, Q, a, b, choice)
    choice == Int32(1) && return vertex(V, a)
    choice == Int32(2) && return vertex(V, b)
    if choice == Int32(3)
        pa, pb = vertex(V, a), vertex(V, b)
        mid = (pa + pb) * 0.5f0
        q = moved(quadric(Q, a), mid - pa) + moved(quadric(Q, b), mid - pb)
        return mid + qefsolve((q.e00, q.e01, q.e02, q.e11, q.e12, q.e22), Vec3f(-q.e03, -q.e13, -q.e23), Vec3f(0f0))
    end
    @inbounds ba, bb = boundary[a] != 0, boundary[b] != 0
    w = ba && !bb ? 1f0 : (!ba && bb ? 0f0 : 0.5f0)
    return vertex(V, a) * w + vertex(V, b) * (1f0 - w)
end

"""
The cost of collapsing edge `(a, b)` to the point `choice` names (see [`collapsepoint`](@ref)):
the quadric there plus the length and sliver terms, `Inf32` when a face would turn over.
"""
@inline function pointcost(V, F, v2f, off, boundary, Q, a, b, choice, lambda_edge_length, lambda_skinny, strict)
    p = collapsepoint(V, boundary, Q, a, b, choice)
    c = evaluate(quadric(Q, a), p - vertex(V, a)) + evaluate(quadric(Q, b), p - vertex(V, b))
    # The tolerance mode prices the mean over the planes the two ends carry.
    strict && (c /= @inbounds max(Q[11, a] + Q[11, b], 1f0))
    isnan(c) && return Inf32
    d = vertex(V, b) - vertex(V, a)
    len2 = dot(d, d)
    c += lambda_edge_length * len2
    skinny = 0f0
    n = Int32(0)
    for (keep, other) in ((a, b), (b, a))
        @inbounds for i in (off[keep] + Int32(1)):off[keep + Int32(1)]
            flips, s, k = afterward(V, F, v2f[i], keep, other, p, strict)
            flips && return Inf32
            skinny += s
            n += k
        end
    end
    n > Int32(0) && (skinny /= n)
    return c + lambda_skinny * skinny * len2
end

"""
The thinnest face the tolerance mode makes, as `4 √3 A / Σ l²` (1 for an equilateral face),
unless it was already thinner: about a 50 mm face 2 µm high. Thinner caps and needles are a
few rounding errors from no area (moved and rotated in Float32, some flatten to none), and
contact and point–triangle distances take their normals from that area.
"""
const MINSHAPE = 1f-4

"""`4 √3 A / Σ l²` of the face `a b c` with normal `n = (b - a) × (c - a)`: 1 when equilateral, 0 when flat."""
@inline function faceshape(a, b, c, n)
    e0, e1, e2 = c - b, b - a, c - a
    return 2f0 * sqrt(3f0) * norm(n) / max(dot(e0, e0) + dot(e1, e1) + dot(e2, e2), 1f-30)
end

"""
cumesh's `process_incident_tri`: face `f` at `keep` after `keep` moves to `p`, as
`(flips, skinniness, faces)`: whether it turns over, and its skinniness
`1 - 4 √3 A / Σ l²` (clamped to [0, 1]) counted as one face. A face that also
has `other` goes with the edge and counts as none. `strict` also refuses a face the
collapse would make thinner than [`MINSHAPE`](@ref) and than it was.
"""
@inline function afterward(V, F, f, keep, other, p, strict)
    hasvertex(F, f, other) && return (false, 0f0, Int32(0))
    @inbounds i, j, k = F[1, f], F[2, f], F[3, f]
    a, b, c = vertex(V, i), vertex(V, j), vertex(V, k)
    na = i == keep ? p : a
    nb = j == keep ? p : b
    nc = k == keep ? p : c
    old = cross(b - a, c - a)
    new = cross(nb - na, nc - na)
    dot(old, new) < 0f0 && return (true, 0f0, Int32(0))
    shape = faceshape(na, nb, nc, new)
    strict && shape < min(faceshape(a, b, c, old), MINSHAPE) && return (true, 0f0, Int32(0))
    return (false, 1f0 - min(max(shape, 0f0), 1f0), Int32(1))
end

"""
Each edge's cost and collapse point. cumesh's mode (`link = 0`) prices the midpoint (or the
boundary end); the tolerance mode (`link = 1`) refuses edges that break the link condition
and takes the cheapest of the midpoint, both ends and the quadric's least point.
"""
function collapsecost!(cost, choice, V, F, v2f, off, edges, boundary, Q, lambda_edge_length, lambda_skinny, link, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds a, b = edges[1, e], edges[2, e]
    strict = link == Int32(1)
    if strict && !linkholds(F, v2f, off, a, b)
        @inbounds cost[e] = Inf32
        return nothing
    end
    if strict
        best, which = cheapestpoint(V, F, v2f, off, boundary, Q, a, b)
    else
        best, which = pointcost(V, F, v2f, off, boundary, Q, a, b, Int32(0), lambda_edge_length, lambda_skinny, strict), Int32(0)
    end
    @inbounds cost[e] = best
    @inbounds choice[e] = which
    return nothing
end

"""The tolerance mode's price of collapsing `(a, b)` to point `choice`: the mean squared distance to the ends' planes."""
@inline function quadricprice(V, boundary, Q, a, b, choice)
    p = collapsepoint(V, boundary, Q, a, b, choice)
    c = (evaluate(quadric(Q, a), p - vertex(V, a)) + evaluate(quadric(Q, b), p - vertex(V, b))) /
        @inbounds(max(Q[11, a] + Q[11, b], 1f0))
    return isnan(c) ? Inf32 : c
end

"""Whether collapsing `(a, b)` to `p` keeps every face around both ends the right way up and of some area."""
@inline function keepsfaces(V, F, v2f, off, a, b, p)
    for (keep, other) in ((a, b), (b, a))
        @inbounds for i in (off[keep] + Int32(1)):off[keep + Int32(1)]
            first(afterward(V, F, v2f[i], keep, other, p, true)) && return false
        end
    end
    return true
end

"""
The tolerance mode's choice of collapse point for `(a, b)`, `(price, choice)`: the four points
priced by the quadric first, then turned faces tested from the cheapest up until one keeps
them all (the face tests cost more than the prices); `(Inf32, 0)` if none does.
"""
@inline function cheapestpoint(V, F, v2f, off, boundary, Q, a, b)
    c0 = quadricprice(V, boundary, Q, a, b, Int32(0))
    c1 = quadricprice(V, boundary, Q, a, b, Int32(1))
    c2 = quadricprice(V, boundary, Q, a, b, Int32(2))
    c3 = quadricprice(V, boundary, Q, a, b, Int32(3))
    for _ in Int32(1):Int32(4)
        c, k = c0, Int32(0)
        c1 < c && ((c, k) = (c1, Int32(1)))
        c2 < c && ((c, k) = (c2, Int32(2)))
        c3 < c && ((c, k) = (c3, Int32(3)))
        isfinite(c) || break
        keepsfaces(V, F, v2f, off, a, b, collapsepoint(V, boundary, Q, a, b, k)) && return c, k
        k == Int32(0) && (c0 = Inf32)
        k == Int32(1) && (c1 = Inf32)
        k == Int32(2) && (c2 = Inf32)
        k == Int32(3) && (c3 = Inf32)
    end
    return Inf32, Int32(0)
end

"""`(cost, edge)` as one key whose unsigned order is the cost's, then the edge's."""
@inline packcost(c::Float32, e) = (UInt64(reinterpret(UInt32, c)) << 32) | UInt64(e % UInt32)

"""Murmur3's 32-bit finaliser: a bijection that scatters neighbouring indices."""
@inline function fmix32(h::UInt32)
    h ⊻= h >> 16
    h *= 0x85ebca6b
    h ⊻= h >> 13
    h *= 0xc2b2ae35
    h ⊻= h >> 16
    return h
end

"""
The bucket of cost `c` against `unit`: `width` octaves each, counted up from `2^-20 unit`
(0 for anything cheaper), all finite costs in one with `width` 0.
"""
@inline function costoctave(c::Float32, unit::Float32, width::Float32)
    isfinite(c) || return typemax(UInt32)
    (width == 0f0 || c <= unit * 9.536743f-7) && return UInt32(0)
    return unsafe_trunc(UInt32, floor((log2(c / unit) + 20f0) / width)) + UInt32(1)
end

"""
An edge's key. cumesh's (`scramble` 0): its cost, then its index. Otherwise the octave of its
cost against `unit` ([`costoctave`](@ref)), then its index scattered by [`fmix32`](@ref).
A round collapses the edges whose key is least around them, and costs that vary smoothly
over the surface have few such minima: the collapses creep across it as a front, a ring per
round. Within an octave the scattered index decides, so the winners are a random
independent set (Luby's), a fixed fraction of the edges in every round, and the cheaper
octaves still go first.
"""
@inline edgekey(c::Float32, e, scramble, unit, width) =
    scramble == Int32(1) ? (UInt64(costoctave(c, unit, width)) << 32) | UInt64(fmix32(e % UInt32)) : packcost(c, e)

"""Every face at either end of an edge keeps the least key of those edges."""
function propagate!(best, edges, v2f, off, cost, scramble, unit, width, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds key = edgekey(cost[e], e, scramble, unit, width)
    for w in Int32(1):Int32(2)
        @inbounds v = edges[w, e]
        @inbounds for i in (off[v] + Int32(1)):off[v + Int32(1)]
            Atomix.@atomic :monotonic min(best[v2f[i]], key)
        end
    end
    return nothing
end

"""
Marks (`win[e] = 1`) every edge at most `thresh` whose key is the least on every face at both
its ends: those edges' rings do not overlap, so they collapse together.
"""
function winners!(win, edges, v2f, off, cost, best, thresh, scramble, unit, width, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds win[e] = Int32(0)
    @inbounds c = cost[e]
    c > thresh && return nothing
    @inbounds a, b = edges[1, e], edges[2, e]
    key = edgekey(c, e, scramble, unit, width)
    for w in (a, b)
        @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
            best[v2f[i]] == key || return nothing
        end
    end
    @inbounds win[e] = Int32(1)
    return nothing
end

"""
Collapses every winning edge: `b` onto `a` at the collapse point, the faces with both gone,
`b`'s other faces moved to `a`, both quadrics moved to the point and added.
"""
function collapse!(V, F, Q, edges, v2f, off, win, choice, boundary, vkept, fkept, merged, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds win[e] == Int32(1) || return nothing
    @inbounds a, b = edges[1, e], edges[2, e]
    p = collapsepoint(V, boundary, Q, a, b, @inbounds(choice[e]))
    setquadric!(Q, a, moved(quadric(Q, a), p - vertex(V, a)) + moved(quadric(Q, b), p - vertex(V, b)))
    @inbounds V[1, a], V[2, a], V[3, a] = p
    @inbounds Q[11, a] += Q[11, b]
    @inbounds vkept[b] = Int32(0)
    @inbounds merged[a] = Int32(1)
    @inbounds for i in (off[a] + Int32(1)):off[a + Int32(1)]
        f = v2f[i]
        hasvertex(F, f, b) && (fkept[f] = Int32(0))
    end
    @inbounds for i in (off[b] + Int32(1)):off[b + Int32(1)]
        f = v2f[i]
        if F[1, f] == b
            F[1, f] = a
        elseif F[2, f] == b
            F[2, f] = a
        elseif F[3, f] == b
            F[3, f] = a
        end
    end
    return nothing
end

function compactvertices!(V2, V, vkept, vid, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    @inbounds if vkept[v] != Int32(0)
        j = vid[v]
        V2[1, j], V2[2, j], V2[3, j] = V[1, v], V[2, v], V[3, v]
    end
    return nothing
end

function compactquadrics!(Q2, Q, vkept, vid, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    @inbounds if vkept[v] != Int32(0)
        setquadric!(Q2, vid[v], quadric(Q, v))
        Q2[11, vid[v]] = Q[11, v]
    end
    return nothing
end

function compactfaces!(F2, F, fkept, fid, vid, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds if fkept[f] != Int32(0)
        j = fid[f]
        F2[1, j], F2[2, j], F2[3, j] = vid[F[1, f]], vid[F[2, f]], vid[F[3, f]]
    end
    return nothing
end

# ── error control ────────────────────────────────────────────────────────────

function firstface!(owner, v2f, off, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    @inbounds owner[v] = v2f[off[v] + Int32(1)]
    return nothing
end

"""
The centroid of face `f` once edge `(a, b)` collapses to `p`, and the midpoint of its edge
from `p` to the next corner: around a fan of new faces each such spoke comes once, and the
fan's rim edges are old ones, sampled when they were made.
"""
@inline function collapsedsamples(V, F, f, a, b, p)
    x, y, z = collapsedface(V, F, f, a, b, p)
    @inbounds i, j = F[1, f], F[2, f]
    spoke = (i == a || i == b) ? (x + y) / 2f0 : (j == a || j == b) ? (y + z) / 2f0 : (z + x) / 2f0
    return (x + y + z) / 3f0, spoke
end

"""The corners of face `f` once edge `(a, b)` collapses to `p`."""
@inline function collapsedface(V, F, f, a, b, p)
    @inbounds i, j, k = F[1, f], F[2, f], F[3, f]
    x = i == a || i == b ? p : vertex(V, i)
    y = j == a || j == b ? p : vertex(V, j)
    z = k == a || k == b ? p : vertex(V, k)
    return x, y, z
end

"""Whether `q` lies within `sqrt(tol2)` of a face around `a` or `b` once `(a, b)` collapses to `p`."""
@inline function nearcollapsed(q, V, F, v2f, off, a, b, p, tol2)
    for w in (a, b)
        @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
            f = v2f[i]
            hasvertex(F, f, a) && hasvertex(F, f, b) && continue
            x, y, z = collapsedface(V, F, f, a, b, p)
            r = closestpoint(q, x, y, z) - q
            dot(r, r) <= tol2 && return true
        end
    end
    return false
end

"""
Takes a winning edge out of the round (no longer a winner, cost `Inf32`) whose collapse would
take the surface past the tolerance: a point of the original on a face around either end farther from the faces the
collapse leaves, or the collapse point, or a centroid or new edge's midpoint of a face it
leaves ([`collapsedsamples`](@ref)), farther from the original surface. It also refuses an edge whose faces hold more than
`maxpoints` original points: checking them all would take the round seconds, and faces that
large gain the simulation nothing by merging further.
"""
function withinbound!(cost, win, choice, V, F, v2f, off, edges, boundary, Q, tol2, part, stale, points, pend, X, F0, gkeys,
                      gstart, gtris, gslots, gcap, lox, loy, loz, cell, maxpoints, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds win[e] == Int32(1) || return nothing
    @inbounds a, b = edges[1, e], edges[2, e]
    held = Int32(0)
    for w in (a, b)
        @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
            f = v2f[i]
            w == b && hasvertex(F, f, a) && continue
            held += pend[f] - (f == Int32(1) ? Int32(0) : pend[f - Int32(1)])
        end
    end
    if held > maxpoints
        @inbounds cost[e] = Inf32
        @inbounds win[e] = Int32(0)
        @inbounds stale[a], stale[b] = Int32(1), Int32(1)
        return nothing
    end
    p = collapsepoint(V, boundary, Q, a, b, @inbounds(choice[e]))
    q = @inbounds part[a]
    ok = nearwithin(p, tol2, q, X, F0, gkeys, gstart, gtris, gslots, gcap, lox, loy, loz, cell)
    for w in (a, b)
        @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
            ok || break
            f = v2f[i]
            hasvertex(F, f, a) && hasvertex(F, f, b) && continue
            centroid, spoke = collapsedsamples(V, F, f, a, b, p)
            ok = nearwithin(centroid, tol2, q, X, F0, gkeys, gstart, gtris, gslots, gcap, lox, loy, loz, cell) &&
                 nearwithin(spoke, tol2, q, X, F0, gkeys, gstart, gtris, gslots, gcap, lox, loy, loz, cell)
        end
    end
    for w in (a, b)
        @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
            ok || break
            f = v2f[i]
            # A face of both ends, the collapse removes it; its points are seen from `a`.
            w == b && hasvertex(F, f, a) && continue
            for k in (f == Int32(1) ? Int32(1) : pend[f - Int32(1)] + Int32(1)):pend[f]
                ok || break
                ok = nearcollapsed(vertex(X, points[k]), V, F, v2f, off, a, b, p, tol2)
            end
        end
    end
    if !ok
        @inbounds cost[e] = Inf32
        @inbounds win[e] = Int32(0)
        @inbounds stale[a], stale[b] = Int32(1), Int32(1)
    end
    return nothing
end

"""A point whose face changed picks the nearest face around its target vertex."""
function reassign!(owner, target, X, V, F, v2f, off, np)
    i = Int32(KI.get_global_id().x)
    i <= np || return nothing
    t = @inbounds target[i]
    t == Int32(0) && return nothing
    q = vertex(X, i)
    best, face = Inf32, Int32(0)
    @inbounds for k in (off[t] + Int32(1)):off[t + Int32(1)]
        f = v2f[k]
        r = closestpoint(q, vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f])) - q
        d = dot(r, r)
        d < best && ((best, face) = (d, f))
    end
    @inbounds owner[i] = face
    @inbounds target[i] = Int32(0)
    return nothing
end

function ownercount!(cnt, owner, np)
    i = Int32(KI.get_global_id().x)
    i <= np || return nothing
    @inbounds Atomix.@atomic :monotonic cnt[owner[i]] += Int32(1)
    return nothing
end

function ownerfill!(points, pend, cursor, owner, np)
    i = Int32(KI.get_global_id().x)
    i <= np || return nothing
    f = @inbounds owner[i]
    slot = @inbounds Atomix.@atomic :monotonic cursor[f] += Int32(1)
    @inbounds points[(f == Int32(1) ? Int32(0) : pend[f - Int32(1)]) + slot] = i
    return nothing
end

"""
After a round's collapses: a point whose face holds a vertex that took in another picks again
around it next round; the others follow their face to its compacted index.
"""
function ownerupdate!(owner, target, F, fid, merged, vid, np)
    i = Int32(KI.get_global_id().x)
    i <= np || return nothing
    f = @inbounds owner[i]
    t = Int32(0)
    @inbounds for k in Int32(1):Int32(3)
        v = F[k, f]
        merged[v] != Int32(0) && (t = vid[v])
    end
    if t != Int32(0)
        @inbounds target[i] = t
    else
        @inbounds owner[i] = fid[f]
    end
    return nothing
end

function compactparts!(part2, stale2, part, stale, vkept, vid, nv)
    v = Int32(KI.get_global_id().x)
    v <= nv || return nothing
    @inbounds if vkept[v] != Int32(0)
        part2[vid[v]] = part[v]
        stale2[vid[v]] = stale[v]
    end
    return nothing
end

"""An edge between two stale vertices is not tried this round."""
function dropstale!(cost, edges, stale, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds (stale[edges[1, e]] != Int32(0) && stale[edges[2, e]] != Int32(0)) && (cost[e] = Inf32)
    return nothing
end

"""Every vertex of a face that holds a vertex that took in another is fresh again: its faces changed."""
function freshen!(stale, F, merged, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds if merged[F[1, f]] != Int32(0) || merged[F[2, f]] != Int32(0) || merged[F[3, f]] != Int32(0)
        stale[F[1, f]], stale[F[2, f]], stale[F[3, f]] = Int32(0), Int32(0), Int32(0)
    end
    return nothing
end

function resetbest!(best, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds best[f] = typemax(UInt64)
    return nothing
end

"""
Drops (cost `Inf32`) every edge with a face around either end that holds a vertex a collapse
of this round took another into: its check no longer holds.
"""
function untouched!(cost, edges, v2f, off, F, merged, ne)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    @inbounds isfinite(cost[e]) || return nothing
    for w in (@inbounds(edges[1, e]), @inbounds(edges[2, e]))
        @inbounds for i in (off[w] + Int32(1)):off[w + Int32(1)]
            f = v2f[i]
            if merged[F[1, f]] != Int32(0) || merged[F[2, f]] != Int32(0) || merged[F[3, f]] != Int32(0)
                cost[e] = Inf32
                return nothing
            end
        end
    end
    return nothing
end
