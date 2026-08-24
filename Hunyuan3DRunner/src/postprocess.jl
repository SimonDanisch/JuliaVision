"""
Mesh clean-up, after marching cubes.

Upstream's `postprocessors.py` runs three of these through `pymeshlab`.

**On `assets/demo.png` the floater filter removes nothing** — that mesh is a
single connected component of all 692 956 faces, so there are no specks to drop.
It is a guard against a failure mode this field can have, not a fix for one it
does have, and it is worth knowing which before reaching for it. What the same
mesh *does* have is four degenerate faces, from one point where three separate
grid edges all crossed the level at the same grid vertex.

**The filters' semantics were determined by running MeshLab, not by reading it.**
`compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.005)`
has three things about it that the name does not settle and that the docstring
gets wrong in at least two ways if guessed:

  * the ratio is against the **largest component**, not the whole mesh. A speck
    that is 0.13% of a mesh's faces survives `nbfaceratio = 0.002` when the mesh
    has three equal large components, because it is 0.39% of the biggest one.
  * the threshold is **truncated to an integer** before the comparison, and the
    comparison is strict: a component is dropped when
    `faces < trunc(ratio * largest)`. With a largest of 1024 the flip point for a
    4-face speck is exactly 5/1024 = 0.00488281, not 4/1024 — verified either
    side of it.
  * components are joined across **shared edges**, not shared vertices. Two
    triangles meeting at one vertex are two components.

Each of those is pinned in `test/runtests.jl` against the value that
discriminates it.
"""

"""
    FLOATER_RATIO, MAX_FACES

Upstream's defaults: `remove_floater`'s `nbfaceratio` and `FaceReducer`'s
`max_facenum`. The reducer's *call* default is 40000 while `reduce_face`'s own is
200000; 40000 is the one the pipeline uses.
"""
const FLOATER_RATIO = 0.005
const MAX_FACES = 40000

"""
    facecomponents(faces, nverts) -> (label, ncomponents)

Connected components of the face graph, joined across shared **edges**, as
`vcglib`'s `ConnectedComponents` does. `label[f]` is the component of face `f`.

An edge with more than two incident faces — which marching cubes does not
produce but a repaired mesh might — puts all of them in one component, which is
what a face-face adjacency ring does too.
"""
function facecomponents(faces::AbstractArray{<:Integer,2}, nverts::Integer)
    nf = size(faces, 2)
    parent = collect(Int32(1):Int32(nf))
    find(x) = begin
        r = x
        while parent[r] != r
            r = parent[r]
        end
        while parent[x] != r          # path compression, iterative
            parent[x], x = r, parent[x]
        end
        r
    end
    union!(a, b) = begin
        ra, rb = find(a), find(b)
        ra == rb || (parent[ra] = rb)
    end

    # edge -> the first face that claimed it; every later claimant unions in.
    first = Dict{UInt64,Int32}()
    sizehint!(first, 3nf)
    @inbounds for f in 1:nf
        for r in 1:3
            a, b = faces[r, f], faces[mod1(r + 1, 3), f]
            lo, hi = minmax(UInt64(a), UInt64(b))
            key = (lo << 32) | hi
            g = get(first, key, Int32(0))
            g == 0 ? (first[key] = Int32(f)) : union!(Int32(f), g)
        end
    end

    label = Vector{Int32}(undef, nf)
    seen = Dict{Int32,Int32}()
    n = Int32(0)
    @inbounds for f in 1:nf
        r = find(Int32(f))
        label[f] = get!(seen, r) do
            n += Int32(1)
        end
    end
    return label, Int(n)
end

"""
    compact(verts, faces) -> (verts, faces)

Drop vertices no face references and renumber. Every filter here removes faces,
and a vertex array with orphans in it is what makes a decimated mesh look like it
did not shrink.
"""
function compact(verts::AbstractArray{T,2}, faces::AbstractArray{I,2}) where {T,I}
    used = falses(size(verts, 2))
    @inbounds for j in 1:size(faces, 2), r in 1:3
        used[faces[r, j]] = true
    end
    remap = zeros(I, size(verts, 2))
    n = 0
    @inbounds for v in 1:size(verts, 2)
        used[v] || continue
        n += 1
        remap[v] = I(n)
    end
    out = Array{T}(undef, size(verts, 1), n)
    @inbounds for v in 1:size(verts, 2)
        used[v] || continue
        for d in 1:size(verts, 1)
            out[d, remap[v]] = verts[d, v]
        end
    end
    nf = Array{I}(undef, 3, size(faces, 2))
    @inbounds for j in 1:size(faces, 2), r in 1:3
        nf[r, j] = remap[faces[r, j]]
    end
    return out, nf
end

"""
    removefloaters(verts, faces; nbfaceratio = FLOATER_RATIO) -> (verts, faces)

`FloaterRemover`: drop every connected component with fewer than
`trunc(nbfaceratio * largest)` faces, then compact.

The three details that are not guessable from the filter's name — ratio against
the largest component, integer truncation with a strict `<`, and edge rather than
vertex connectivity — are in this file's header, with the experiments that
settled them.
"""
function removefloaters(verts::AbstractArray{<:Real,2}, faces::AbstractArray{<:Integer,2};
                        nbfaceratio::Real = FLOATER_RATIO)
    label, n = facecomponents(faces, size(verts, 2))
    n <= 1 && return verts, faces
    counts = zeros(Int, n)
    @inbounds for f in 1:size(faces, 2)
        counts[label[f]] += 1
    end
    threshold = trunc(Int, nbfaceratio * maximum(counts))
    keep = [c >= threshold for c in counts]
    all(keep) && return verts, faces
    cols = [f for f in 1:size(faces, 2) if keep[label[f]]]
    return compact(verts, faces[:, cols])
end

"""
    weld(verts, faces) -> (verts, faces)

Merge vertices that share a position exactly, renumbering the faces onto the
survivor.

`marchingcubes` identifies a vertex by the grid *edge* it lies on, which is what
makes it shared between neighbouring cells. When the field is exactly at the
level on a grid *vertex*, several distinct edges cross at that one point and
produce several distinct indices for it. On the demo mesh there is one such
point, carrying three indices.

Exact equality, not a tolerance: these are the same float, produced by the same
interpolation hitting `t = 0` or `t = 1`, not two values that happen to be close.
A tolerance here would weld real geometry at the grid scale.
"""
function weld(verts::AbstractArray{T,2}, faces::AbstractArray{I,2}) where {T,I}
    key = Dict{NTuple{3,T},I}()
    sizehint!(key, size(verts, 2))
    remap = Vector{I}(undef, size(verts, 2))
    n = I(0)
    @inbounds for i in 1:size(verts, 2)
        remap[i] = get!(key, (verts[1, i], verts[2, i], verts[3, i])) do
            n += I(1)
        end
    end
    n == size(verts, 2) && return verts, faces
    out = Array{T}(undef, size(verts, 1), Int(n))
    @inbounds for i in 1:size(verts, 2), d in 1:size(verts, 1)
        out[d, remap[i]] = verts[d, i]
    end
    nf = Array{I}(undef, 3, size(faces, 2))
    @inbounds for j in 1:size(faces, 2), r in 1:3
        nf[r, j] = remap[faces[r, j]]
    end
    return out, nf
end

"""
    removedegenerate(verts, faces) -> (verts, faces)

`DegenerateFaceRemover`: [`weld`](@ref) coincident vertices, then drop faces left
with a repeated index.

**Welding first is what keeps the mesh closed, and skipping it is a hole.** The
degenerate faces here are not stray triangles; they are the faces around a point
that marching cubes gave three indices to. Dropping them on index-inequality
alone leaves those three indices distinct and tears a gap: measured on the demo
mesh, 6 of 1 039 434 edges stopped having exactly two incident faces. Welded
first, the neighbours close over the point and all 1 039 428 remaining edges have
exactly two.

Upstream's own version is a save-and-reload through PLY, which leans on MeshLab's
importer to weld and drop on the way in — the same two steps, named here instead
of implied by a file format.
"""
function removedegenerate(verts::AbstractArray{<:Real,2}, faces::AbstractArray{<:Integer,2})
    v, f = weld(verts, faces)
    keep = [j for j in 1:size(f, 2)
            if f[1, j] != f[2, j] && f[2, j] != f[3, j] && f[1, j] != f[3, j]]
    length(keep) == size(f, 2) && return v, f
    return compact(v, f[:, keep])
end
