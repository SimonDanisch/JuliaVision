"""
A face that could join the growing piece: across the piece's border edge
`(a, b)`, with its third vertex at `position`. `orient` is the side of `(a, b)`
the piece's own face lies on, `cost` how far the placement stretches the face.
"""
struct Candidate
    face::Int32
    a::Int32
    b::Int32
    position::Vec2f
    cost::Float32
    orient::Float32
end

"""
    PiecewiseParam(vertices, faces, nbr)

xatlas's `PiecewiseParam`: the parameterisation of a chart that no single map
flattens without folding or laying part of it over another. It grows a piece
from the chart's first free face by unfolding one neighbouring face at a time
into the plane (turned about the edge they share, so every face keeps its
shape), cheapest stretch first, and refuses a face that would fold, stretch by
more than half or cross the piece's border. Faces that share the new vertex go
in together at the mean of their placements. When nothing more fits, the next
piece starts.

The state spans the whole mesh and is reset for just a chart's faces and
vertices, so one instance serves every chart.
"""
struct PiecewiseParam{V<:AbstractMatrix{Float32},F<:AbstractMatrix{<:Integer}}
    vertices::V
    faces::F
    nbr::Vector{NTuple{3,Int32}}
    uv::Vector{Vec2f}
    inchart::BitVector
    inany::BitVector       # in a piece of this chart
    inpatch::BitVector     # in the piece being grown
    invalid::BitVector     # refused by the piece being grown
    vinpatch::BitVector
    candidate::Vector{Int32}   # face -> the new vertex of its candidate, 0 if none
    groups::Dict{Int32,Vector{Candidate}}   # new vertex -> the candidates placing it
    # Border edges `(face, k)` of the piece bucketed by cell; stale entries (no
    # longer on the border) are skipped when read.
    grid::Dict{Tuple{Int32,Int32},Vector{Tuple{Int32,Int8}}}
    cell::Base.RefValue{Float32}
    patch::Vector{Int32}
end

function PiecewiseParam(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer},
                        nbr::Vector{NTuple{3,Int32}})
    nv, nf = size(vertices, 2), size(faces, 2)
    return PiecewiseParam(vertices, faces, nbr, Vector{Vec2f}(undef, nv), falses(nf), falses(nf),
                          falses(nf), falses(nf), falses(nv), zeros(Int32, nf),
                          Dict{Int32,Vector{Candidate}}(), Dict{Tuple{Int32,Int32},Vector{Tuple{Int32,Int8}}}(),
                          Ref(1f0), Int32[])
end

position3(pp::PiecewiseParam, v) = Vec3f(view(pp.vertices, :, v))

"""Positive on one side of the line `a -> b`, negative on the other (xatlas's `orientToEdge`)."""
orient(a::Vec2f, b::Vec2f, p::Vec2f) = (a[1] - p[1]) * (b[2] - p[2]) - (a[2] - p[2]) * (b[1] - p[1])

"""
Whether segments `a1 a2` and `b1 b2` cross inside both, `epsilon` in the
segments' own parameter (xatlas's `linesIntersect`). The parallel test is
relative to the lengths: xatlas's absolute one declares every pair of this
mesh's edges, a few thousandths long, parallel.
"""
function crosses(a1::Vec2f, a2::Vec2f, b1::Vec2f, b2::Vec2f; epsilon::Float32 = 1f-4)
    v0, v1 = a2 - a1, b2 - b1
    denom = -v1[1] * v0[2] + v0[1] * v1[2]
    abs(denom) <= epsilon * norm(v0) * norm(v1) && return false
    s = (-v0[2] * (a1[1] - b1[1]) + v0[1] * (a1[2] - b1[2])) / denom
    epsilon < s < 1 - epsilon || return false
    t = (v1[1] * (a1[2] - b1[2]) - v1[2] * (a1[1] - b1[1])) / denom
    return epsilon < t < 1 - epsilon
end

edgevertices(pp::PiecewiseParam, f, k) = (Int32(pp.faces[k, f]), Int32(pp.faces[k % 3 + 1, f]))

"""The vertex of face `f` that is neither `a` nor `b`."""
function thirdvertex(pp::PiecewiseParam, f, a, b)
    for k in 1:3
        v = Int32(pp.faces[k, f])
        (v == a || v == b) || return v
    end
    return Int32(0)
end

cellof(pp::PiecewiseParam, p::Vec2f) =
    (unsafe_trunc(Int32, floor(p[1] / pp.cell[])), unsafe_trunc(Int32, floor(p[2] / pp.cell[])))

"""The grid cells the box of segment `p q` covers."""
function cells(pp::PiecewiseParam, p::Vec2f, q::Vec2f)
    ca, cb = cellof(pp, p), cellof(pp, q)
    return Iterators.product(min(ca[1], cb[1]):max(ca[1], cb[1]), min(ca[2], cb[2]):max(ca[2], cb[2]))
end

function gridinsert!(pp::PiecewiseParam, f, k)
    a, b = edgevertices(pp, f, k)
    for c in cells(pp, pp.uv[a], pp.uv[b])
        push!(get!(Vector{Tuple{Int32,Int8}}, pp.grid, c), (Int32(f), Int8(k)))
    end
end

"""Edge `k` of piece face `f` is on the piece's border."""
function onborder(pp::PiecewiseParam, f, k)
    g = pp.nbr[f][k]
    return g == 0 || !pp.inpatch[g]
end

"""
xatlas's `addFaceToPatch`: the face joins the piece, and every chart face
across one of its edges that is in no piece and no candidate yet becomes one.
A face whose third vertex is in the piece already is enclosed by it and left,
as xatlas leaves it.
"""
function addface!(pp::PiecewiseParam, f)
    push!(pp.patch, f)
    pp.inpatch[f] = true
    pp.inany[f] = true
    for k in 1:3
        g = pp.nbr[f][k]
        (g == 0 || !pp.inchart[g] || pp.inany[g] || pp.candidate[g] != 0) && continue
        a, b = edgevertices(pp, f, k)
        free = thirdvertex(pp, g, a, b)
        pp.vinpatch[free] && continue
        pp.invalid[g] && continue
        o = orient(pp.uv[a], pp.uv[b], pp.uv[thirdvertex(pp, f, a, b)])
        addcandidate!(pp, a, b, o, g, free)
    end
end

"""
xatlas's `addCandidateFace`: face `g` unfolded against the piece's edge `(a, b)`,
its third vertex on the side away from the piece's face (`o` the side that face
is on) and the triangle scaled with the edge's length in the plane. Refused for
good when that stretches it by more than half or leaves it without area.
"""
function addcandidate!(pp::PiecewiseParam, a, b, o, g, free)
    P0, P1, Pf = position3(pp, a), position3(pp, b), position3(pp, free)
    Q0, Q1 = pp.uv[a], pp.uv[b]
    L3, L2 = norm(P1 - P0), norm(Q1 - Q0)
    (L3 > 0 && L2 > 0) || return
    e = (P1 - P0) / L3
    t = dot(Pf - P0, e)
    h = norm(Pf - P0 - t * e)
    scale = L2 / L3
    d = (Q1 - Q0) / L2
    n = Vec2f(-d[2], d[1])
    side = orient(Q0, Q1, Q0 + n) * o > 0 ? -1f0 : 1f0
    p = Q0 + scale * (t * d + side * h * n)
    if !all(isfinite, p) || orient(Q0, Q1, p) * o > 0
        pp.invalid[g] = true
        return
    end
    parametric = scale * scale * L3 * h / 2
    geometric = L3 * h / 2
    if !(parametric > eps(Float32) * L2 * L2)
        pp.invalid[g] = true
        return
    end
    stretch = parametric <= geometric ? parametric / geometric : geometric / parametric
    cost = abs(stretch - 1f0)
    if cost > 0.5f0
        pp.invalid[g] = true
        return
    end
    push!(get!(Vector{Candidate}, pp.groups, free), Candidate(g, a, b, p, cost, o))
    pp.candidate[g] = free
    return
end

"""
The candidates placing vertex `free` taken together at `p`: none may fold
or lose its area, and none of their edges that stay on the border may cross the
piece's border (edges they share with the piece stop being border and are left
out, as are edges sharing a vertex).
"""
function fits(pp::PiecewiseParam, group::Vector{Candidate}, free, p::Vec2f)
    for c in group
        fo = orient(pp.uv[c.a], pp.uv[c.b], p)
        (fo == 0 || fo * c.orient > 0) && return false
    end
    pos(v) = v == free ? p : pp.uv[v]
    joining(g) = any(c -> c.face == g, group)
    for c in group, k in 1:3
        g = pp.nbr[c.face][k]
        (g != 0 && pp.inpatch[g]) && continue
        a, b = edgevertices(pp, c.face, k)
        pa, pb = pos(a), pos(b)
        for cell in cells(pp, pa, pb), (f2, k2) in get(pp.grid, cell, ())
            onborder(pp, f2, k2) || continue
            joining(pp.nbr[f2][k2]) && continue
            c2, d2 = edgevertices(pp, f2, k2)
            (c2 == a || c2 == b || d2 == a || d2 == b) && continue
            crosses(pa, pb, pp.uv[c2], pp.uv[d2]) && return false
        end
    end
    return true
end

"""
xatlas's `computeChart`: one piece from the chart's first face in no piece,
grown until no candidate fits. `false` when every face has its piece.
"""
function growpiece!(pp::PiecewiseParam, members::AbstractVector{<:Integer})
    empty!(pp.patch)
    empty!(pp.groups)
    empty!(pp.grid)
    for f in members
        pp.inpatch[f] = false
        pp.invalid[f] = false
        pp.candidate[f] = 0
        for k in 1:3
            pp.vinpatch[pp.faces[k, f]] = false
        end
    end
    i = findfirst(f -> !pp.inany[f], members)
    i === nothing && return false
    seed = members[i]
    a, b = edgevertices(pp, seed, 1)
    c = thirdvertex(pp, seed, a, b)
    P0, P1, P2 = position3(pp, a), position3(pp, b), position3(pp, c)
    L = norm(P1 - P0)
    e = L > 0 ? (P1 - P0) / L : Vec3f(1, 0, 0)
    t = dot(P2 - P0, e)
    pp.uv[a] = Vec2f(0, 0)
    pp.uv[b] = Vec2f(L, 0)
    pp.uv[c] = Vec2f(t, norm(P2 - P0 - t * e))
    for v in (a, b, c)
        pp.vinpatch[v] = true
    end
    addface!(pp, seed)
    for k in 1:3
        gridinsert!(pp, seed, k)
    end
    while !isempty(pp.groups)
        free, cost = Int32(0), Inf32
        for (v, group) in pp.groups
            m = maximum(c -> c.cost, group)
            m < cost && ((free, cost) = (v, m))
        end
        group = pop!(pp.groups, free)
        p = sum(c -> c.position, group) / length(group)
        for c in group
            pp.candidate[c.face] = 0
        end
        if fits(pp, group, free, p)
            pp.uv[free] = p
            pp.vinpatch[free] = true
            for c in group
                addface!(pp, c.face)
            end
            for c in group, k in 1:3
                onborder(pp, c.face, k) && gridinsert!(pp, c.face, k)
            end
        else
            for c in group
                pp.invalid[c.face] = true
            end
        end
    end
    return true
end

"""
    piecewise!(pieces, pp, members) -> pieces

Every piece [`PiecewiseParam`](@ref) cuts the chart `members` into, pushed as
`(faces, vids, uv)`.
"""
function piecewise!(pieces, pp::PiecewiseParam, members::Vector{Int32})
    pp.cell[] = 2 * sum(f -> sum(k -> norm(position3(pp, pp.faces[k % 3 + 1, f]) - position3(pp, pp.faces[k, f])), 1:3),
                        members) / (3 * length(members))
    pp.inchart[members] .= true
    while growpiece!(pp, members)
        vids = unique!([Int32(pp.faces[k, f]) for f in pp.patch for k in 1:3])
        uv = Matrix{Float32}(undef, 2, length(vids))
        for (i, v) in enumerate(vids)
            uv[1, i], uv[2, i] = pp.uv[v]
        end
        push!(pieces, (copy(pp.patch), vids, uv))
    end
    pp.inchart[members] .= false
    pp.inany[members] .= false
    return pieces
end
