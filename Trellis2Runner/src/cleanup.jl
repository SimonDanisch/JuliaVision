edgekey(a, b) = a < b ? (Int32(a), Int32(b)) : (Int32(b), Int32(a))

"""Every undirected edge and how many faces use it."""
function edgecounts(faces::AbstractMatrix{<:Integer})
    counts = Dict{Tuple{Int32,Int32},Int32}()
    sizehint!(counts, 2 * size(faces, 2))
    for f in axes(faces, 2), k in 1:3
        e = edgekey(faces[k, f], faces[k % 3 + 1, f])
        counts[e] = get(counts, e, Int32(0)) + Int32(1)
    end
    return counts
end

"""
    removeduplicates(faces) -> faces

cumesh's `remove_duplicate_faces`: a face over the same three vertices as an
earlier one, whichever way round, goes. Two of them make every edge of the pair
look non-manifold.
"""
function removeduplicates(faces::AbstractMatrix{<:Integer})
    seen = Set{NTuple{3,Int32}}()
    keep = falses(size(faces, 2))
    for f in axes(faces, 2)
        key = Tuple(sort(Int32[faces[1, f], faces[2, f], faces[3, f]]))
        key in seen && continue
        push!(seen, key)
        keep[f] = true
    end
    return Matrix{Int32}(faces[:, keep])
end

"""
    removedegenerate(vertices, faces; absolute = 1e-24, relative = 1e-12) -> faces

cumesh's `remove_degenerate_faces`: a face goes when its area is below
`absolute` and below `relative` times its longest edge squared. Such a face
has no plane, so a chart cannot map it, and the vertex only it holds has no
equation in [`lscm`](@ref).
"""
function removedegenerate(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                          absolute::Real = 1e-24, relative::Real = 1e-12)
    keep = map(axes(faces, 2)) do f
        a, b, c = (Vec3(Float64.(view(vertices, :, faces[k, f]))...) for k in 1:3)
        area = norm(cross(b - a, c - a)) / 2
        longest = max(norm(b - a), norm(c - b), norm(a - c))
        return !(area < absolute && area < relative * longest^2)
    end
    return Matrix{Int32}(faces[:, keep])
end

"""
    fillholes(vertices, faces; maxperimeter = 3e-2) -> (vertices, faces)

cumesh's `fill_holes`: every hole whose perimeter is below `maxperimeter` (in
the unit cube's units) is closed by a fan, one face per boundary edge, around the
mean of its edges' midpoints. The dual grid leaves a hole wherever a voxel of an
edge's four is missing.

A hole is cumesh's boundary loop: the boundary edges (on one face) joined at
every vertex that has exactly two of them and no edge of three or more faces,
kept when every edge meets another of its own at both ends. So two holes that
touch at a vertex are two holes, each filled; tracing a loop from edge to edge
stops at such a vertex, and that left 1,316 of upstream's 14,312 holes open on
the 1024 example. cumesh also joins at a vertex with ONE boundary edge, reading
the next vertex's entry as the second; that is left out.

A fan face is wound like the face its edge belongs to, which cumesh leaves to
the edge's sorted vertex order.
"""
function fillholes(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                   maxperimeter::Real = 3e-2)
    counts = edgecounts(faces)
    nv = size(vertices, 2)
    # Boundary edges as their faces walk them, a -> b.
    ea, eb = Int32[], Int32[]
    for f in axes(faces, 2), k in 1:3
        a, b = Int32(faces[k, f]), Int32(faces[k % 3 + 1, f])
        counts[edgekey(a, b)] == 1 || continue
        push!(ea, a); push!(eb, b)
    end
    isempty(ea) && return Matrix{Float32}(vertices), Matrix{Int32}(faces)
    # A vertex joins its boundary edges only when it has exactly two and sits on
    # no edge of three or more faces.
    nbound = zeros(Int32, nv)
    pinched = falses(nv)
    for ((u, v), c) in counts
        c == 1 && (nbound[u] += Int32(1); nbound[v] += Int32(1))
        c > 2 && (pinched[u] = pinched[v] = true)
    end
    firstedge = zeros(Int32, nv)
    parent = collect(Int32(1):Int32(length(ea)))
    find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for e in eachindex(ea), v in (ea[e], eb[e])
        (nbound[v] == 2 && !pinched[v]) || continue
        if firstedge[v] == 0
            firstedge[v] = e
        else
            r, q = find(Int32(e)), find(firstedge[v])
            r == q || (parent[r] = q)
        end
    end
    # A component is a hole when each of its edges meets another of its own at
    # both ends; at a joining vertex that is the other edge, elsewhere any
    # boundary edge at the vertex that happens to be in the same component.
    root = [find(Int32(e)) for e in eachindex(ea)]
    atvertex = Dict{Int32,Vector{Int32}}()
    for e in eachindex(ea), v in (ea[e], eb[e])
        push!(get!(() -> Int32[], atvertex, v), e)
    end
    isloop = Dict{Int32,Bool}(r => true for r in root)
    for e in eachindex(ea), v in (ea[e], eb[e])
        any(o -> o != e && root[o] == root[e], atvertex[v]) || (isloop[root[e]] = false)
    end
    V = [Vec3f(view(vertices, :, v)) for v in axes(vertices, 2)]
    perimeter = Dict{Int32,Float32}()
    centre = Dict{Int32,Vec3f}()
    nedges = Dict{Int32,Int}()
    for e in eachindex(ea)
        r = root[e]
        isloop[r] || continue
        perimeter[r] = get(perimeter, r, 0f0) + norm(V[eb[e]] - V[ea[e]])
        centre[r] = get(centre, r, Vec3f(0)) + (V[ea[e]] + V[eb[e]]) * 0.5f0
        nedges[r] = get(nedges, r, 0) + 1
    end
    index = Dict{Int32,Int32}()
    for r in sort!(collect(keys(perimeter)))
        perimeter[r] < maxperimeter || continue
        push!(V, centre[r] / nedges[r])
        index[r] = Int32(length(V))
    end
    newfaces = Int32[]
    for e in eachindex(ea)
        c = get(index, root[e], Int32(0))
        # The fan face runs the edge b -> a, against its face's a -> b.
        c == 0 || append!(newfaces, (eb[e], ea[e], c))
    end
    isempty(newfaces) && return Matrix{Float32}(vertices), Matrix{Int32}(faces)
    return reduce(hcat, V), hcat(Matrix{Int32}(faces), reshape(newfaces, 3, :))
end

"""
    splitnonmanifold(vertices, faces) -> (vertices, faces)

cumesh's `repair_non_manifold_edges`, by splitting vertices: around every
vertex, the faces that reach each other across edges of exactly two faces form
one fan, and every fan after the first gets a copy of the vertex. Where two
sheets of the dual-grid surface meet along an edge they come apart, each a
manifold of its own, with the copies at the same place, so nothing moves.
"""
function splitnonmanifold(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    # The fans are read from `faces`; `F` gets the copies.
    F = Matrix{Int32}(faces)
    counts = edgecounts(faces)
    nv = size(vertices, 2)
    vfaces = [Int32[] for _ in 1:nv]
    for f in axes(faces, 2), k in 1:3
        push!(vfaces[faces[k, f]], f)
    end
    extra = Int32[]         # the vertex each new copy copies
    parent = Int32[]
    find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for v in 1:nv
        fs = vfaces[v]
        length(fs) <= 1 && continue
        resize!(parent, length(fs))
        parent .= 1:length(fs)
        # Two faces of the fan meet across an edge v-w they both hold.
        for i in eachindex(fs), j in (i + 1):length(fs)
            fi, fj = fs[i], fs[j]
            for k in 1:3
                w = faces[k, fi]
                w == v && continue
                (w == faces[1, fj] || w == faces[2, fj] || w == faces[3, fj]) || continue
                counts[edgekey(v, w)] == 2 || continue
                ri, rj = find(i), find(j)
                ri == rj || (parent[ri] = rj)
            end
        end
        roots = unique(find(i) for i in eachindex(fs))
        length(roots) <= 1 && continue
        for r in roots[2:end]
            push!(extra, v)
            twin = Int32(nv + length(extra))
            for i in eachindex(fs)
                find(i) == r || continue
                f = fs[i]
                for k in 1:3
                    faces[k, f] == v && (F[k, f] = twin)
                end
            end
        end
    end
    isempty(extra) && return Matrix{Float32}(vertices), F
    return hcat(Matrix{Float32}(vertices), vertices[:, extra]), F
end

"""
    removesmallcomponents(vertices, faces; minarea = 1e-5) -> (vertices, faces)

cumesh's `remove_small_connected_components`: pieces of the surface (faces
joined through shared vertices) whose total area is below `minarea` go, and the
vertices only they used.
"""
function removesmallcomponents(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                               minarea::Real = 1e-5)
    nv = size(vertices, 2)
    parent = collect(Int32(1):Int32(nv))
    find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for f in axes(faces, 2)
        a = find(faces[1, f])
        for k in 2:3
            b = find(faces[k, f])
            a == b || (parent[b] = a)
        end
    end
    _, areas = facegeometry(vertices, faces)
    area = Dict{Int32,Float64}()
    for f in axes(faces, 2)
        r = find(faces[1, f])
        area[r] = get(area, r, 0.0) + areas[f]
    end
    keep = [area[find(faces[1, f])] >= minarea for f in axes(faces, 2)]
    return compact(vertices, faces[:, keep])
end

"""The vertices `faces` use, renumbered in order."""
function compact(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    used = falses(size(vertices, 2))
    for x in faces
        used[x] = true
    end
    remap = Int32.(cumsum(used))
    return vertices[:, used], remap[faces]
end

"""
    repair(vertices, faces) -> (vertices, faces)

cumesh's clean-up between two simplifications: duplicate and degenerate faces
out, non-manifold edges split, specks below `1e-5` of area out.
"""
function repair(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    F = removedegenerate(vertices, removeduplicates(faces))
    return removesmallcomponents(splitnonmanifold(vertices, F)...)
end

"""
    cleanmesh(dev, vertices, faces; target) -> (vertices, faces)

`o_voxel.postprocess.to_glb`'s mesh preparation, from the mesh its first
[`fillholes`](@ref) left (the surface [`bake`](@ref) projects onto): simplify
to three times `target`, [`repair`](@ref), fill, simplify to `target`, repair,
fill, and, as `uv_unwrap` starts with, degenerate faces out; then wound once
more, since the splits cut the pieces the winding was made consistent over.
"""
function cleanmesh(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                   target::Int = 500_000)
    V, F = fillholes(repair(simplify(dev, vertices, faces, 3target)...)...)
    V, F = fillholes(repair(simplify(dev, V, F, target)...)...)
    F = removedegenerate(V, F)
    return V, orientfaces(V, F)
end
