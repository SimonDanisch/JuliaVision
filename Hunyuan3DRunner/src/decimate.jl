"""
Quadric edge-collapse decimation.

`FaceReducer` upstream is MeshLab's `meshing_decimation_quadric_edge_collapse`
with `targetfacenum`, `qualitythr = 1.0`, `preserveboundary`, `boundaryweight = 3`,
`preservenormal` and `preservetopology`. This is the same algorithm —
Garland-Heckbert quadric error metrics — with the same three constraints, and it
is not bit-identical to vcglib's: the cost of a collapse depends on how the
quadrics are weighted and on the order a heap pops equal costs in, and neither is
part of the algorithm's definition. What is checked instead is what a decimator
is actually for: the target face count, that the surface does not move, and that
a closed mesh stays closed. `test/runtests.jl` measures all three against
`pymeshlab` on the same mesh.

**Area-weighted quadrics.** Each face contributes its plane's outer product
scaled by twice its area, so a large flat region resists being pulled out of
plane more than a sliver does. This is the usual choice and is what makes the
metric independent of how finely a flat area happens to be triangulated —
important here, because marching cubes triangulates flat regions as finely as
curved ones.

**`qualitythr` is not modelled.** In MeshLab it multiplies a collapse's cost by a
penalty for the aspect ratio of the triangles it would leave behind, which
changes *which* collapse is cheapest but never whether one is legal. Leaving it
out costs triangle shape, not correctness or position, and putting it in without
vcglib's exact penalty curve would be a different heuristic wearing its name.
"""

"""
    Quadric

A symmetric 4x4 error quadric as its 10 distinct entries, in `Float64`.

Double precision on purpose: a quadric is a sum of outer products over a whole
one-ring and is then evaluated as a difference of large nearly-equal terms, so it
loses far more precision than the coordinates it is built from. In `Float32` the
cost of a good collapse goes negative, and a heap ordered by a negative cost
collapses the wrong edges first.
"""
const Quadric = NTuple{10,Float64}

const ZEROQ = ntuple(_ -> 0.0, 10)

"""
    planequadric(n, d, w) -> Quadric

`w * outer([n; d], [n; d])` for the plane `n·x + d = 0`.
"""
@inline function planequadric(a::Float64, b::Float64, c::Float64, d::Float64, w::Float64)
    return (w * a * a, w * a * b, w * a * c, w * a * d,
            w * b * b, w * b * c, w * b * d,
            w * c * c, w * c * d,
            w * d * d)
end

@inline qadd(p::Quadric, q::Quadric) = ntuple(i -> p[i] + q[i], 10)

"""
    qerror(q, x, y, z) -> Float64

`v' Q v` for `v = [x, y, z, 1]`. Clamped at zero: the quadric is positive
semidefinite in exact arithmetic, and a small negative value is rounding, but a
negative cost in the heap reorders the whole decimation.
"""
@inline function qerror(q::Quadric, x::Float64, y::Float64, z::Float64)
    e = q[1] * x * x + 2q[2] * x * y + 2q[3] * x * z + 2q[4] * x +
        q[5] * y * y + 2q[6] * y * z + 2q[7] * y +
        q[8] * z * z + 2q[9] * z + q[10]
    return max(e, 0.0)
end

"""
    optimalplace(q, ax, ay, az, bx, by, bz) -> (x, y, z, cost)

Where to put the merged vertex. Solves `A v = -b` from the quadric's upper 3x3,
which is the position of least error; when that block is near-singular — a
straight edge or a flat region, where the minimum is a line or a plane rather
than a point — falls back to the cheapest of the two endpoints and their
midpoint, which is what `optimalplacement` does in MeshLab too.
"""
function optimalplace(q::Quadric, ax, ay, az, bx, by, bz)
    a11, a12, a13 = q[1], q[2], q[3]
    a22, a23, a33 = q[5], q[6], q[8]
    det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13) +
          a13 * (a12 * a23 - a22 * a13)
    if abs(det) > 1e-12
        c1, c2, c3 = -q[4], -q[7], -q[9]
        x = (c1 * (a22 * a33 - a23 * a23) - a12 * (c2 * a33 - a23 * c3) +
             a13 * (c2 * a23 - a22 * c3)) / det
        y = (a11 * (c2 * a33 - a23 * c3) - c1 * (a12 * a33 - a23 * a13) +
             a13 * (a12 * c3 - c2 * a13)) / det
        z = (a11 * (a22 * c3 - c2 * a23) - a12 * (a12 * c3 - c2 * a13) +
             c1 * (a12 * a23 - a22 * a13)) / det
        return (x, y, z, qerror(q, x, y, z))
    end
    mx, my, mz = (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2
    ea, eb, em = qerror(q, ax, ay, az), qerror(q, bx, by, bz), qerror(q, mx, my, mz)
    em <= ea && em <= eb && return (mx, my, mz, em)
    ea <= eb && return (ax, ay, az, ea)
    return (bx, by, bz, eb)
end

"""
    decimate(verts, faces; maxfaces = MAX_FACES, boundaryweight = 3.0) -> (verts, faces)

Collapse edges cheapest-first until at most `maxfaces` faces remain. Returns the
mesh unchanged when it is already small enough, as `reduce_face` does.

Three constraints, matching the flags upstream passes:

  * **preservetopology** — a collapse is refused when `u` and `v` share a
    neighbour that is not opposite the edge. Without this check the collapse
    welds two sheets that only touched at that vertex and the mesh stops being
    manifold; it is also the check that keeps the result's genus.
  * **preservenormal** — refused when any surviving incident face would turn over.
    This is what stops a decimator from folding thin features inside out, and it
    is the difference between a mesh that looks decimated and one that looks
    broken.
  * **preserveboundary** — every edge with one incident face contributes a plane
    perpendicular to that face, weighted by `boundaryweight`, so an open edge
    holds its shape. A closed mesh has none, which is the normal case here.
"""
function decimate(verts::AbstractArray{<:Real,2}, faces::AbstractArray{<:Integer,2};
                  maxfaces::Integer = MAX_FACES, boundaryweight::Real = 3.0)
    nf0 = size(faces, 2)
    nf0 > maxfaces || return verts, faces

    nv = size(verts, 2)
    P = Array{Float64}(undef, 3, nv)
    @inbounds for i in 1:nv, d in 1:3
        P[d, i] = Float64(verts[d, i])
    end
    F = Array{Int32}(undef, 3, nf0)
    @inbounds for j in 1:nf0, r in 1:3
        F[r, j] = Int32(faces[r, j])
    end

    quads = fill(ZEROQ, nv)
    alive = trues(nf0)
    incident = [Int32[] for _ in 1:nv]           # faces touching each vertex
    @inbounds for j in 1:nf0
        a, b, c = F[1, j], F[2, j], F[3, j]
        n = facenormal(P, a, b, c)
        n === nothing && continue
        nx, ny, nz, area2 = n
        q = planequadric(nx, ny, nz, -(nx * P[1, a] + ny * P[2, a] + nz * P[3, a]), area2)
        quads[a] = qadd(quads[a], q)
        quads[b] = qadd(quads[b], q)
        quads[c] = qadd(quads[c], q)
        push!(incident[a], Int32(j)); push!(incident[b], Int32(j)); push!(incident[c], Int32(j))
    end

    # Boundary edges: one incident face. Each contributes a plane through the
    # edge, perpendicular to the face.
    edgefaces = Dict{UInt64,Int}()
    sizehint!(edgefaces, 3nf0)
    @inbounds for j in 1:nf0, r in 1:3
        edgefaces[edgekey(F[r, j], F[mod1(r + 1, 3), j])] =
            get(edgefaces, edgekey(F[r, j], F[mod1(r + 1, 3), j]), 0) + 1
    end
    @inbounds for j in 1:nf0, r in 1:3
        u, v = F[r, j], F[mod1(r + 1, 3), j]
        edgefaces[edgekey(u, v)] == 1 || continue
        n = facenormal(P, F[1, j], F[2, j], F[3, j])
        n === nothing && continue
        ex, ey, ez = P[1, v] - P[1, u], P[2, v] - P[2, u], P[3, v] - P[3, u]
        # perpendicular to both the face normal and the edge
        px = n[2] * ez - n[3] * ey
        py = n[3] * ex - n[1] * ez
        pz = n[1] * ey - n[2] * ex
        L = sqrt(px^2 + py^2 + pz^2)
        L > 0 || continue
        px, py, pz = px / L, py / L, pz / L
        w = Float64(boundaryweight) * (ex^2 + ey^2 + ez^2)
        q = planequadric(px, py, pz, -(px * P[1, u] + py * P[2, u] + pz * P[3, u]), w)
        quads[u] = qadd(quads[u], q)
        quads[v] = qadd(quads[v], q)
    end

    return collapseloop!(P, F, quads, alive, incident, nf0, maxfaces, verts, faces)
end

"""
    facenormal(P, a, b, c) -> (nx, ny, nz, 2*area) or nothing

Unit normal and twice the area. `nothing` for a degenerate face, which
contributes no plane rather than a division by zero.
"""
@inline function facenormal(P, a, b, c)
    ux, uy, uz = P[1, b] - P[1, a], P[2, b] - P[2, a], P[3, b] - P[3, a]
    vx, vy, vz = P[1, c] - P[1, a], P[2, c] - P[2, a], P[3, c] - P[3, a]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    L = sqrt(nx^2 + ny^2 + nz^2)
    L > 0 || return nothing
    return (nx / L, ny / L, nz / L, L)
end

@inline edgekey(a::Integer, b::Integer) =
    (UInt64(min(a, b)) << 32) | UInt64(max(a, b))

"""
    collapseloop!(...) -> (verts, faces)

The heap loop. Costs are computed once and re-checked on pop against the
vertices' current versions, so an edge whose endpoints have moved is recomputed
and pushed back rather than kept up to date eagerly — the usual lazy-deletion
arrangement, and the reason this is `O(n log n)` rather than quadratic.
"""
function collapseloop!(P, F, quads, alive, incident, nf, maxfaces, verts, faces)
    nv = size(P, 2)
    version = zeros(Int32, nv)
    merged = collect(Int32(1):Int32(nv))         # union-find into the survivor
    root(x) = begin
        r = x
        while merged[r] != r
            r = merged[r]
        end
        while merged[x] != r
            merged[x], x = r, merged[x]
        end
        r
    end

    # (cost, u, v, versionu, versionv)
    heap = Vector{Tuple{Float64,Int32,Int32,Int32,Int32}}()
    sizehint!(heap, 3size(F, 2))
    seen = Set{UInt64}()
    @inbounds for j in 1:size(F, 2), r in 1:3
        u, v = F[r, j], F[mod1(r + 1, 3), j]
        k = edgekey(u, v)
        k in seen && continue
        push!(seen, k)
        push!(heap, edgecost(P, quads, u, v, version))
    end
    empty!(seen)
    heapify!(heap)

    live = nf
    while live > maxfaces && !isempty(heap)
        cost, u, v, vu, vv = heappop!(heap)
        (version[u] != vu || version[v] != vv) && continue      # stale
        (root(u) != u || root(v) != v) && continue              # already merged away
        u == v && continue

        q = qadd(quads[u], quads[v])
        x, y, z, _ = optimalplace(q, P[1, u], P[2, u], P[3, u], P[1, v], P[2, v], P[3, v])
        legal(P, F, alive, incident, u, v, x, y, z) || continue

        # commit: v folds into u
        P[1, u], P[2, u], P[3, u] = x, y, z
        quads[u] = q
        merged[v] = u
        live -= killfaces!(F, alive, incident, u, v)
        append!(incident[u], incident[v])
        empty!(incident[v])
        @inbounds for j in incident[u]
            alive[j] || continue
            for r in 1:3
                F[r, j] == v && (F[r, j] = u)
            end
        end
        version[u] += Int32(1)
        version[v] += Int32(1)

        # re-cost the ring
        touched = Set{Int32}()
        @inbounds for j in incident[u]
            alive[j] || continue
            for r in 1:3
                w = F[r, j]
                w == u || push!(touched, w)
            end
        end
        for w in touched
            heappush!(heap, edgecost(P, quads, u, w, version))
        end
    end

    return harvest(P, F, alive, verts, faces)
end

@inline function edgecost(P, quads, u::Integer, v::Integer, version)
    q = qadd(quads[u], quads[v])
    _, _, _, c = optimalplace(q, P[1, u], P[2, u], P[3, u], P[1, v], P[2, v], P[3, v])
    return (c, Int32(u), Int32(v), version[u], version[v])
end

"""
    legal(...) -> Bool

The two refusals: the link condition (topology) and the normal test.
"""
function legal(P, F, alive, incident, u, v, x, y, z)
    # Link condition, stated as it actually is: the vertices adjacent to BOTH
    # endpoints must be exactly the ones opposite the edge. For a manifold
    # interior edge that is two of them, one per incident face.
    #
    # The tempting shortcut — refuse when any face of `v` that does not contain
    # `u` touches `u`'s ring — is wrong and refuses every legal collapse: the
    # face `(v, w1, w2)` on the far side of an opposite vertex `w1` does not
    # contain `u`, yet `w1` is in `u`'s ring by construction. That mistake makes
    # the decimator a no-op, which is not obviously a failure since it returns a
    # perfectly good mesh, just the one it was given.
    nfaces = 0
    Nu = Set{Int32}()
    @inbounds for j in incident[u]
        alive[j] || continue
        a, b, c = F[1, j], F[2, j], F[3, j]
        (a == v || b == v || c == v) && (nfaces += 1)
        for w in (a, b, c)
            (w == u || w == v) || push!(Nu, w)
        end
    end
    nfaces == 2 || return false          # not a manifold interior edge
    shared = 0
    @inbounds for j in incident[v]
        alive[j] || continue
        for w in (F[1, j], F[2, j], F[3, j])
            (w == u || w == v) && continue
            w in Nu && (delete!(Nu, w); shared += 1)
        end
    end
    shared == 2 || return false
    # Normal test: no surviving incident face may turn over.
    @inbounds for src in (incident[u], incident[v])
        for j in src
            alive[j] || continue
            a, b, c = F[1, j], F[2, j], F[3, j]
            (a == v || b == v || c == v) && (a == u || b == u || c == u) && continue
            before = facenormal(P, a, b, c)
            before === nothing && continue
            ax, ay, az = movedcoords(P, a, u, v, x, y, z)
            bx, by, bz = movedcoords(P, b, u, v, x, y, z)
            cx, cy, cz = movedcoords(P, c, u, v, x, y, z)
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            L = sqrt(nx^2 + ny^2 + nz^2)
            L > 0 || return false
            (before[1] * nx + before[2] * ny + before[3] * nz) / L > 0 || return false
        end
    end
    return true
end

@inline function movedcoords(P, w, u, v, x, y, z)
    (w == u || w == v) && return (x, y, z)
    return (P[1, w], P[2, w], P[3, w])
end

"""
    killfaces!(F, alive, incident, u, v) -> Int

Retire the faces containing both endpoints — the two triangles the edge belonged
to — and report how many died.
"""
function killfaces!(F, alive, incident, u, v)
    n = 0
    @inbounds for j in incident[u]
        alive[j] || continue
        a, b, c = F[1, j], F[2, j], F[3, j]
        if (a == v || b == v || c == v)
            alive[j] = false
            n += 1
        end
    end
    return n
end

"""
    harvest(P, F, alive, verts, faces) -> (verts, faces)

Surviving faces, renumbered onto a compact vertex array, back in the element type
the caller handed in.
"""
function harvest(P, F, alive, verts::AbstractArray{T}, faces::AbstractArray{I}) where {T,I}
    cols = [j for j in 1:size(F, 2) if alive[j]]
    out = Array{T}(undef, 3, size(P, 2))
    @inbounds for i in 1:size(P, 2), d in 1:3
        out[d, i] = T(P[d, i])
    end
    return compact(out, Array{I}(F[:, cols]))
end

# --------------------------------------------------------------------- the heap
#
# A plain binary heap on tuples rather than DataStructures.PriorityQueue: this
# package's dependencies are the runtime and nothing else, and lazy deletion
# needs push and pop and nothing more.

function heapify!(h)
    for i in (length(h) ÷ 2):-1:1
        siftdown!(h, i)
    end
    return h
end

function heappush!(h, x)
    push!(h, x)
    i = length(h)
    @inbounds while i > 1
        p = i ÷ 2
        h[p][1] <= h[i][1] && break
        h[p], h[i] = h[i], h[p]
        i = p
    end
    return h
end

function heappop!(h)
    @inbounds begin
        top = h[1]
        last = pop!(h)
        if !isempty(h)
            h[1] = last
            siftdown!(h, 1)
        end
        return top
    end
end

function siftdown!(h, i)
    n = length(h)
    @inbounds while true
        l, r = 2i, 2i + 1
        m = i
        l <= n && h[l][1] < h[m][1] && (m = l)
        r <= n && h[r][1] < h[m][1] && (m = r)
        m == i && return
        h[i], h[m] = h[m], h[i]
        i = m
    end
end
