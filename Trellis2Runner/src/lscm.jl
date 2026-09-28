"""
    lscm(vertices, faces, members, normals, areas) -> (vids, uv) or nothing

Least squares conformal map (Lévy et al. 2002) of one chart, the faces
`members` of the mesh: the planar coordinates that keep every triangle as close
to a similar copy of itself as possible, with the two vertices farthest apart
pinned. What xatlas parameterises a chart with, and what lets a chart curve
where a projection onto one plane would squash it.

Returns the chart's vertices (mesh indices) and their coordinates `(2, n)`,
scaled so the chart's area in the plane is its area on the surface; `nothing`
when the map folds a triangle over (the chart is not a disk, or bends too far),
which the caller answers by splitting the chart.
"""
function lscm(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer},
              members::AbstractVector{<:Integer}, normals::Vector{Vec3f}, areas::Vector{Float32})
    # Faces wound against the chart are mapped as if turned over: the map
    # cares which side is up only so it can tell a fold from a face.
    up = chartnormal(normals, areas, members)
    corners(f) = dot(normals[f], up) >= 0 ? (faces[1, f], faces[2, f], faces[3, f]) :
                                             (faces[1, f], faces[3, f], faces[2, f])
    local_index = Dict{Int32,Int32}()
    vids = Int32[]
    for f in members, k in 1:3
        v = Int32(faces[k, f])
        haskey(local_index, v) || (push!(vids, v); local_index[v] = length(vids))
    end
    n = length(vids)
    p(v) = Vec3(Float64.(view(vertices, :, v))...)
    # Pin the pair farthest apart, found from the vertex farthest from an
    # arbitrary one and the vertex farthest from that.
    far(from) = argmax(i -> norm(p(vids[i]) - p(vids[from])), 1:n)
    pa = far(1)
    pb = far(pa)
    pa == pb && return nothing
    free = setdiff(1:n, (pa, pb))
    col = zeros(Int, n)
    for (i, v) in enumerate(free)
        col[v] = i
    end
    nfree = length(free)
    I, J, V = Int[], Int[], Float64[]
    rhs = Float64[]
    area3d = 0.0
    row = 0
    # A vertex that only faces without a plane hold has no equation.
    held = falses(n)
    for f in members
        v1, v2, v3 = (local_index[Int32(x)] for x in corners(f))
        a, b, c = p(vids[v1]), p(vids[v2]), p(vids[v3])
        e1 = b - a
        nrm = cross(e1, c - a)
        dbl = norm(nrm)
        dbl > 0 || continue
        held[v1] = held[v2] = held[v3] = true
        area3d += dbl / 2
        x = normalize(e1)
        y = normalize(cross(nrm, e1))
        # The triangle in its own plane, and its gradient weights W_j.
        xs = (0.0, norm(e1), dot(c - a, x))
        ys = (0.0, 0.0, dot(c - a, y))
        s = 1 / sqrt(dbl)
        W = ((xs[3] - xs[2], ys[3] - ys[2]), (xs[1] - xs[3], ys[1] - ys[3]), (xs[2] - xs[1], ys[2] - ys[1]))
        re, im = row + 1, row + 2
        row += 2
        br, bi = 0.0, 0.0
        for (j, v) in enumerate((v1, v2, v3))
            wr, wi = W[j][1] * s, W[j][2] * s
            # Re: wr u - wi v; Im: wi u + wr v, with U = u + i v.
            if v == pa || v == pb
                u0, v0 = v == pa ? (0.0, 0.0) : (1.0, 0.0)
                br -= wr * u0 - wi * v0
                bi -= wi * u0 + wr * v0
            else
                cu, cv = col[v], nfree + col[v]
                append!(I, (re, re, im, im)); append!(J, (cu, cv, cu, cv)); append!(V, (wr, -wi, wi, wr))
            end
        end
        append!(rhs, (br, bi))
    end
    (row == 0 || !all(held)) && return nothing
    A = SparseArrays.sparse(I, J, V, row, 2nfree)
    x = nfree == 0 ? Float64[] : (A' * A) \ (A' * rhs)
    uv = Matrix{Float64}(undef, 2, n)
    uv[:, pa] .= (0.0, 0.0)
    uv[:, pb] .= (1.0, 0.0)
    for (i, v) in enumerate(free)
        uv[1, v], uv[2, v] = x[i], x[nfree + i]
    end
    # Folds: every triangle has to keep one orientation in the plane.
    area2d = 0.0
    pos, neg = 0, 0
    for f in members
        v1, v2, v3 = (local_index[Int32(x)] for x in corners(f))
        s2 = (uv[1, v2] - uv[1, v1]) * (uv[2, v3] - uv[2, v1]) -
             (uv[2, v2] - uv[2, v1]) * (uv[1, v3] - uv[1, v1])
        s2 > 0 ? (pos += 1) : s2 < 0 ? (neg += 1) : nothing
        area2d += abs(s2) / 2
    end
    (pos > 0 && neg > 0) && return nothing
    all(isfinite, uv) || return nothing
    area2d > 0 || return nothing
    # Mirrored as a whole is not a fold; turn it back and scale to surface area.
    neg > 0 && (uv[2, :] .*= -1)
    uv .*= sqrt(area3d / area2d)
    return vids, Float32.(uv)
end

"""
    selfoverlaps(vids, uv, faces, members) -> Bool

Whether a chart's map lays part of it over another part. With no triangle
turned over (what [`lscm`](@ref) checks), that happens exactly where the
chart's border crosses itself, so the border edges (those of one member face)
are bucketed into a grid in the plane and every two in a cell that do not share
a vertex are tested for a crossing. xatlas makes the same check before it
accepts a chart.
"""
function selfoverlaps(vids::AbstractVector{<:Integer}, uv::AbstractMatrix{Float32},
                      faces::AbstractMatrix{<:Integer}, members::AbstractVector{<:Integer})
    length(members) <= 1 && return false
    slot = Dict{Int32,Int32}(Int32(v) => Int32(i) for (i, v) in enumerate(vids))
    uses = Dict{Tuple{Int32,Int32},Int32}()
    for f in members, k in 1:3
        a, b = slot[faces[k, f]], slot[faces[k % 3 + 1, f]]
        e = a < b ? (a, b) : (b, a)
        uses[e] = get(uses, e, Int32(0)) + Int32(1)
    end
    border = [e for (e, n) in uses if n == 1]
    length(border) < 4 && return false
    p(i) = Vec2f(uv[1, i], uv[2, i])
    lo = Vec2f(minimum(view(uv, 1, :)), minimum(view(uv, 2, :)))
    cellsize = max(sum(e -> norm(p(e[2]) - p(e[1])), border) / length(border), 1f-12)
    cells = Dict{Tuple{Int,Int},Vector{Int32}}()
    cell(q) = (unsafe_trunc(Int, (q[1] - lo[1]) / cellsize), unsafe_trunc(Int, (q[2] - lo[2]) / cellsize))
    for (i, (a, b)) in enumerate(border)
        ca, cb = cell(p(a)), cell(p(b))
        for y in min(ca[2], cb[2]):max(ca[2], cb[2]), x in min(ca[1], cb[1]):max(ca[1], cb[1])
            push!(get!(Vector{Int32}, cells, (x, y)), i)
        end
    end
    orient(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
    for es in values(cells), i in eachindex(es), j in (i + 1):length(es)
        (a, b), (c, d) = border[es[i]], border[es[j]]
        (a == c || a == d || b == c || b == d) && continue
        pa, pb, pc, pd = p(a), p(b), p(c), p(d)
        if orient(pa, pb, pc) * orient(pa, pb, pd) < 0 && orient(pc, pd, pa) * orient(pc, pd, pb) < 0
            return true
        end
    end
    return false
end
