"""
    simplify(dev, vertices, faces, target; thresh = 1e-8, lambda_edge_length = 1f-2,
             lambda_skinny = 1f-3) -> (vertices, faces)

cumesh's `simplify`, on the device: [`simplifystep`](@ref)s until at most `target`
faces are left, the cost threshold ten times higher whenever a step removes less
than 1% of the faces. A step may overshoot `target`, as upstream's does.

`(3, V)` Float32 vertices and `(3, F)` one-based faces in as `Mantle.Buffer`s
(consumed) or host matrices, the same kind out, with the vertices no collapse
removed (an unreferenced one stays, as upstream).
"""
function simplify(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, target::Int;
                  thresh::Float64 = 1e-8, lambda_edge_length::Float32 = 1f-2, lambda_skinny::Float32 = 1f-3)
    nf = size(F, 2)
    nf <= target && return V, F
    while true
        V, F = simplifystep(dev, V, F, Float32(thresh), lambda_edge_length, lambda_skinny)
        n = size(F, 2)
        n <= target && break
        (nf - n) / nf < 1e-2 && (thresh *= 10)
        nf = n
    end
    return V, F
end

function simplify(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer},
                  target::Int; kw...)
    V, F = simplify(dev, upload(dev, vertices), upload(dev, faces), target; kw...)
    return download(V), download(F)
end

"""
    simplifystep(dev, V, F, thresh, lambda_edge_length, lambda_skinny) -> (V, F)

cumesh's `simplify_step`: one round of parallel edge collapses.

Every edge gets a cost, the error quadric of its two ends at the point it would
collapse to (the midpoint, or the boundary end when one end is on a boundary),
plus `lambda_edge_length` times its squared length and `lambda_skinny` times
that times the mean skinniness of the faces around it after the collapse;
infinite when one of those faces would turn over. Every face keeps the least
`(cost, edge)` of the edges at its corners, and an edge whose cost is at most
`thresh` and which is that least on every face around both its ends collapses.
Those edges' rings do not overlap, so they all collapse at once.
"""
function simplifystep(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, thresh::Float32,
                      lambda_edge_length::Float32, lambda_skinny::Float32)
    nv, nf = size(V, 2), size(F, 2)
    # The faces at each vertex, `v2f[off[v] + 1:off[v + 1]]`: counted into
    # `cnt[v + 1]` so the running total is `off` with `off[1] = 0`.
    cnt = fill!(Mantle.Buffer(dev, Int32, nv + 1), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, facecount!, (cnt, F, nf), nf; name = "simplify/facecount")
    runonce!(g)
    off = prefixsum(dev, cnt)
    fill!(cnt, 0)
    v2f = Mantle.Buffer(dev, Int32, 3nf)
    # The edges, each once from its lower end: `ecnt[u + 1]` of them from `u`.
    ecnt = fill!(Mantle.Buffer(dev, Int32, nv + 1), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, fillfaces!, (v2f, off, cnt, F, nf), nf; name = "simplify/vertexfaces")
    Mantle.dispatch!(g, edgecount!, (ecnt, F, v2f, off, nv), nv; name = "simplify/edgecount")
    runonce!(g)
    eoff = prefixsum(dev, ecnt)
    ne = lastentry(eoff)
    edges = Mantle.Buffer(dev, Int32, (2, ne))
    efaces = Mantle.Buffer(dev, Int32, ne)
    boundary = fill!(Mantle.Buffer(dev, UInt8, nv), 0)
    Q = Mantle.Buffer(dev, Float32, (10, nv))
    cost = Mantle.Buffer(dev, Float32, ne)
    best = fill!(Mantle.Buffer(dev, UInt64, nf), typemax(UInt64))
    vkept = fill!(Mantle.Buffer(dev, Int32, nv), 1)
    fkept = fill!(Mantle.Buffer(dev, Int32, nf), 1)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, edges!, (edges, efaces, eoff, F, v2f, off, nv), nv; name = "simplify/edges")
    Mantle.dispatch!(g, boundary!, (boundary, edges, efaces, ne), ne; name = "simplify/boundary")
    Mantle.dispatch!(g, quadrics!, (Q, V, F, v2f, off, nv), nv; name = "simplify/quadrics")
    Mantle.dispatch!(g, collapsecost!, (cost, V, F, v2f, off, edges, boundary, Q, lambda_edge_length,
                                        lambda_skinny, ne), ne; name = "simplify/cost")
    Mantle.dispatch!(g, propagate!, (best, edges, v2f, off, cost, ne), ne; name = "simplify/propagate")
    Mantle.dispatch!(g, collapse!, (V, F, edges, v2f, off, cost, best, boundary, thresh, vkept, fkept, ne), ne;
                     name = "simplify/collapse")
    runonce!(g)
    vid = prefixsum(dev, vkept)
    fid = prefixsum(dev, fkept)
    V2 = Mantle.Buffer(dev, Float32, (3, lastentry(vid)))
    F2 = Mantle.Buffer(dev, Int32, (3, lastentry(fid)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactvertices!, (V2, V, vkept, vid, nv), nv; name = "simplify/compact vertices")
    Mantle.dispatch!(g, compactfaces!, (F2, F, fkept, fid, vid, nf), nf; name = "simplify/compact faces")
    runonce!(g)
    for b in (cnt, off, v2f, ecnt, eoff, edges, efaces, boundary, Q, cost, best, vkept, fkept, vid, fid, V, F)
        Mantle.free!(b)
    end
    return V2, F2
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

@inline quadric(Q, v) = @inbounds Quadric(Q[1, v], Q[2, v], Q[3, v], Q[4, v], Q[5, v], Q[6, v], Q[7, v],
                                          Q[8, v], Q[9, v], Q[10, v])

@inline hasvertex(F, f, v) = @inbounds(F[1, f] == v || F[2, f] == v || F[3, f] == v)

"""Whether a face among `v2f[lo:i - 1]` has `v`: a neighbour met before."""
@inline function seenbefore(F, v2f, lo, i, v)
    for k in lo:(i - 1)
        hasvertex(F, @inbounds(v2f[k]), v) && return true
    end
    return false
end

function facecount!(cnt, F, nf)
    f = KI.get_global_id().x
    f <= nf || return nothing
    for k in 1:3
        @inbounds Atomix.@atomic cnt[F[k, f] + 1] += Int32(1)
    end
    return nothing
end

function fillfaces!(v2f, off, cnt, F, nf)
    f = KI.get_global_id().x
    f <= nf || return nothing
    for k in 1:3
        @inbounds v = F[k, f]
        slot = @inbounds Atomix.@atomic cnt[v] += Int32(1)
        @inbounds v2f[off[v] + slot] = f % Int32
    end
    return nothing
end

"""How many distinct neighbours above it vertex `u` has: its edges, from the lower end."""
function edgecount!(ecnt, F, v2f, off, nv)
    u = KI.get_global_id().x
    u <= nv || return nothing
    @inbounds lo, hi = off[u] + 1, off[u + 1]
    n = Int32(0)
    for i in lo:hi
        @inbounds f = v2f[i]
        for k in 1:3
            @inbounds v = F[k, f]
            (v > u && !seenbefore(F, v2f, lo, i, v)) && (n += Int32(1))
        end
    end
    @inbounds ecnt[u + 1] = n
    return nothing
end

"""
Vertex `u`'s edges `(u, v)`, `v > u`, in increasing `v`, and how many faces
each is on: every edge once, in the order cumesh's radix sort of `(min, max)`
keys gives, which decides which of two equal costs wins.
"""
function edges!(edges, efaces, eoff, F, v2f, off, nv)
    u = KI.get_global_id().x
    u <= nv || return nothing
    @inbounds lo, hi = off[u] + 1, off[u + 1]
    @inbounds base = eoff[u]
    n = Int32(0)
    for i in lo:hi
        @inbounds f = v2f[i]
        for k in 1:3
            @inbounds v = F[k, f]
            (v > u && !seenbefore(F, v2f, lo, i, v)) || continue
            c = Int32(0)
            for j in lo:hi
                hasvertex(F, @inbounds(v2f[j]), v) && (c += Int32(1))
            end
            # Insertion into the sorted run so far.
            j = base + n
            @inbounds while j > base && edges[2, j] > v
                edges[1, j + 1] = u % Int32
                edges[2, j + 1] = edges[2, j]
                efaces[j + 1] = efaces[j]
                j -= Int32(1)
            end
            @inbounds edges[1, j + 1] = u % Int32
            @inbounds edges[2, j + 1] = v
            @inbounds efaces[j + 1] = c
            n += Int32(1)
        end
    end
    return nothing
end

"""Both ends of an edge on one face only are boundary vertices."""
function boundary!(boundary, edges, efaces, ne)
    e = KI.get_global_id().x
    e <= ne || return nothing
    @inbounds if efaces[e] == 1
        boundary[edges[1, e]] = 0x01
        boundary[edges[2, e]] = 0x01
    end
    return nothing
end

"""Each vertex's quadric: the planes of its faces, unweighted."""
function quadrics!(Q, V, F, v2f, off, nv)
    v = KI.get_global_id().x
    v <= nv || return nothing
    q = Quadric(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0)
    @inbounds for i in (off[v] + 1):off[v + 1]
        f = v2f[i]
        a = vertex(V, F[1, f])
        n = cross(vertex(V, F[2, f]) - a, vertex(V, F[3, f]) - a)
        n = n * (1f0 / sqrt(dot(n, n)))
        q += Quadric(n, -dot(n, a))
    end
    @inbounds Q[1, v], Q[2, v], Q[3, v], Q[4, v], Q[5, v] = q.e00, q.e01, q.e02, q.e03, q.e11
    @inbounds Q[6, v], Q[7, v], Q[8, v], Q[9, v], Q[10, v] = q.e12, q.e13, q.e22, q.e23, q.e33
    return nothing
end

"""Where edge `(a, b)` collapses to: its midpoint, or the end on a boundary when only one is."""
@inline function collapsepoint(V, boundary, a, b)
    @inbounds ba, bb = boundary[a] != 0, boundary[b] != 0
    w = ba && !bb ? 1f0 : (!ba && bb ? 0f0 : 0.5f0)
    return vertex(V, a) * w + vertex(V, b) * (1f0 - w)
end

"""
cumesh's `process_incident_tri`: face `f` at `keep` after `keep` moves to `p`, as
`(flips, skinniness, faces)`: whether it turns over, and its skinniness
`1 - 4 √3 A / Σ l²` (clamped to [0, 1]) counted as one face. A face that also
has `other` goes with the edge and counts as none.
"""
@inline function afterward(V, F, f, keep, other, p)
    hasvertex(F, f, other) && return (false, 0f0, Int32(0))
    @inbounds i, j, k = F[1, f], F[2, f], F[3, f]
    a, b, c = vertex(V, i), vertex(V, j), vertex(V, k)
    na = i == keep ? p : a
    nb = j == keep ? p : b
    nc = k == keep ? p : c
    old = cross(b - a, c - a)
    e1, e2 = nb - na, nc - na
    new = cross(e1, e2)
    dot(old, new) < 0f0 && return (true, 0f0, Int32(0))
    area = 0.5f0 * norm(new)
    e0 = nc - nb
    denom = max(dot(e0, e0) + dot(e1, e1) + dot(e2, e2), 1f-12)
    shape = 4f0 * sqrt(3f0) * area / denom
    return (false, 1f0 - min(max(shape, 0f0), 1f0), Int32(1))
end

function collapsecost!(cost, V, F, v2f, off, edges, boundary, Q, lambda_edge_length, lambda_skinny, ne)
    e = KI.get_global_id().x
    e <= ne || return nothing
    @inbounds a, b = edges[1, e], edges[2, e]
    p = collapsepoint(V, boundary, a, b)
    c = evaluate(quadric(Q, a) + quadric(Q, b), p)
    d = vertex(V, b) - vertex(V, a)
    len2 = dot(d, d)
    c += lambda_edge_length * len2
    skinny = 0f0
    n = Int32(0)
    for (keep, other) in ((a, b), (b, a))
        @inbounds for i in (off[keep] + 1):off[keep + 1]
            flips, s, k = afterward(V, F, v2f[i], keep, other, p)
            if flips
                cost[e] = Inf32
                return nothing
            end
            skinny += s
            n += k
        end
    end
    n > 0 && (skinny /= n)
    @inbounds cost[e] = c + lambda_skinny * skinny * len2
    return nothing
end

"""`(cost, edge)` as one key whose unsigned order is the cost's, then the edge's."""
@inline packcost(c::Float32, e) = (UInt64(reinterpret(UInt32, c)) << 32) | UInt64(e % UInt32)

"""Every face at either end of an edge keeps the least key of those edges."""
function propagate!(best, edges, v2f, off, cost, ne)
    e = KI.get_global_id().x
    e <= ne || return nothing
    @inbounds key = packcost(cost[e], e)
    for w in 1:2
        @inbounds v = edges[w, e]
        @inbounds for i in (off[v] + 1):off[v + 1]
            Atomix.@atomic min(best[v2f[i]], key)
        end
    end
    return nothing
end

"""
Collapse every edge at most `thresh` whose key is the least on every face at
both its ends: `b` onto `a` at the collapse point, the faces with both gone,
`b`'s other faces moved to `a`.
"""
function collapse!(V, F, edges, v2f, off, cost, best, boundary, thresh, vkept, fkept, ne)
    e = KI.get_global_id().x
    e <= ne || return nothing
    @inbounds c = cost[e]
    c > thresh && return nothing
    @inbounds a, b = edges[1, e], edges[2, e]
    key = packcost(c, e)
    for w in (a, b)
        @inbounds for i in (off[w] + 1):off[w + 1]
            best[v2f[i]] == key || return nothing
        end
    end
    p = collapsepoint(V, boundary, a, b)
    @inbounds V[1, a], V[2, a], V[3, a] = p
    @inbounds vkept[b] = Int32(0)
    @inbounds for i in (off[a] + 1):off[a + 1]
        f = v2f[i]
        hasvertex(F, f, b) && (fkept[f] = Int32(0))
    end
    @inbounds for i in (off[b] + 1):off[b + 1]
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
    v = KI.get_global_id().x
    v <= nv || return nothing
    @inbounds if vkept[v] != 0
        j = vid[v]
        V2[1, j], V2[2, j], V2[3, j] = V[1, v], V[2, v], V[3, v]
    end
    return nothing
end

function compactfaces!(F2, F, fkept, fid, vid, nf)
    f = KI.get_global_id().x
    f <= nf || return nothing
    @inbounds if fkept[f] != 0
        j = fid[f]
        F2[1, j], F2[2, j], F2[3, j] = vid[F[1, f]], vid[F[2, f]], vid[F[3, f]]
    end
    return nothing
end
