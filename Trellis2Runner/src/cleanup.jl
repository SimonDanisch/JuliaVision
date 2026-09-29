"""
cumesh's mesh clean-up on the device: `fill_holes`, `remove_degenerate_faces`,
`remove_unreferenced_vertices`, `unify_face_orientations`, on the connectivity
of `topology.jl`. Every function takes `Mantle.Buffer`s, consumes the ones it
replaces, and returns the mesh it made; the host-matrix methods upload, run and
download for callers that hold arrays.
"""

upload(dev::Mantle.Device, x::AbstractMatrix{Float32}) = Mantle.Buffer(dev, Matrix{Float32}(x))
upload(dev::Mantle.Device, x::AbstractMatrix{<:Integer}) = Mantle.Buffer(dev, Matrix{Int32}(x))
download(b::Mantle.Buffer) = Array(Mantle.storage(b))

"""Fixed point with 32 fractional bits, what the fill's sums accumulate in:
integer atomics are exact and order-free where float ones are neither."""
@inline fixed(x) = unsafe_trunc(Int64, Float64(x) * 4294967296.0 + (x < 0 ? -0.5 : 0.5))
const FIXED_ONE = 4294967296.0

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
    perimeter = fill!(Mantle.Buffer(dev, Int64, nb), 0)
    centre = fill!(Mantle.Buffer(dev, Int64, 3nb), 0)
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
        Atomix.@atomic perimeter[r] += fixed(norm(q - p))
        base = Int32(3) * (r - Int32(1))
        Atomix.@atomic centre[base + Int32(1)] += fixed(m[1])
        Atomix.@atomic centre[base + Int32(2)] += fixed(m[2])
        Atomix.@atomic centre[base + Int32(3)] += fixed(m[3])
        Atomix.@atomic count[r] += Int32(1)
    end
    return nothing
end

"""A component's root, when it is a loop below the perimeter, gets a centre vertex."""
function keeploops!(keep, isloop, perimeter, comp, maxfixed, nb)
    r = Int32(KI.get_global_id().x)
    r <= nb || return nothing
    @inbounds keep[r] = (comp[r] == r && isloop[r] == Int32(1) && perimeter[r] < maxfixed) ? Int32(1) : Int32(0)
    return nothing
end

function edgekept!(fflag, keep, comp, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    @inbounds fflag[b] = keep[comp[b]]
    return nothing
end

function newvertices!(V2, keep, pos, centre, count, nv, nb)
    r = Int32(KI.get_global_id().x)
    r <= nb || return nothing
    @inbounds if keep[r] == Int32(1)
        j = nv + pos[r]
        s = Float64(count[r]) * FIXED_ONE
        base = Int32(3) * (r - Int32(1))
        V2[1, j] = Float32(Float64(centre[base + Int32(1)]) / s)
        V2[2, j] = Float32(Float64(centre[base + Int32(2)]) / s)
        V2[3, j] = Float32(Float64(centre[base + Int32(3)]) / s)
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

# ── degenerate and unreferenced ───────────────────────────────────────────────

"""
    removedegenerate(dev, V, F; absolute = 1e-24, relative = 1e-12) -> (V, F)

cumesh's `remove_degenerate_faces`: a face goes when two of its vertices are
one, or when its area is below `absolute` and below `relative` times its
longest edge squared. Such a face has no plane, so a chart cannot map it, and
the vertex only it holds has no equation in [`lscm`](@ref). Vertices no face
is left on go with it ([`removeunreferenced`](@ref)), as after every removal
in cumesh.
"""
function removedegenerate(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer;
                          absolute::Real = 1e-24, relative::Real = 1e-12)
    nf = size(F, 2)
    flags = Mantle.Buffer(dev, Int32, nf)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markdegenerate!, (flags, V, F, Float32(absolute), Float32(relative), nf), nf;
                     name = "degenerate/mark")
    runonce!(g)
    F1, nf1 = compactcolumns(dev, F, flags, nf)
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
    V2 = Mantle.Buffer(dev, Float32, (3, max(nv2, 1)))
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
inconsistent varies between runs. On the 512 example 19.7k of 4.2M two-face
edges stay walked the same way, against 13.5k for a host flood fill.
`generate` does not take this path: the remesh winds its shell itself.
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

# ── host clean-up of the non-remesh path ──────────────────────────────────────
#
# `cleanmesh` is the `to_glb(remesh = False)` branch, which `generate` does not
# take; its duplicate-face and non-manifold repair still run on the host.

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
    repair(dev, vertices, faces) -> (vertices, faces)

cumesh's clean-up between two simplifications: duplicate and degenerate faces
out, non-manifold edges split, specks below `1e-5` of area out.
"""
function repair(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    V, F = removedegenerate(dev, vertices, removeduplicates(faces))
    return removesmallcomponents(splitnonmanifold(V, F)...)
end

"""
    cleanmesh(dev, vertices, faces; target) -> (vertices, faces)

`o_voxel.postprocess.to_glb`'s mesh preparation without remeshing, from the
mesh its first [`fillholes`](@ref) left: simplify to three times `target`,
[`repair`](@ref), fill, simplify to `target`, repair, fill, and, as `uv_unwrap`
starts with, degenerate faces out; then wound once more, since the splits cut
the pieces the winding was made consistent over.
"""
function cleanmesh(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                   target::Int = 500_000)
    V, F = fillholes(dev, repair(dev, simplify(dev, vertices, faces, 3target)...)...)
    V, F = fillholes(dev, repair(dev, simplify(dev, V, F, target)...)...)
    V, F = removedegenerate(dev, V, F)
    return V, orientfaces(dev, F)
end
