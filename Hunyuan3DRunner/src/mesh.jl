"""
The implicit field to a triangle mesh.

`MCSurfaceExtractor` upstream is `skimage.measure.marching_cubes(..., method =
"lewiner")` followed by a rescale from grid indices into the sampling box. This is
the same two steps, without the dependency.

**Built from face contours rather than from a 256-entry case table.** The usual
marching cubes is a lookup table, and a table is 256 rows of hand-written indices
where one wrong entry is a hole in a mesh rather than an error. The construction
here derives the same surface: on each of the cube's six faces the sign changes
come in pairs and define line segments, every intersected edge belongs to exactly
two faces so every segment endpoint has degree two, and the segments therefore
close into loops that trianguate directly. Nothing is transcribed, and
`test/runtests.jl` checks all 256 sign patterns produce a closed surface plus a
sphere against its analytic area and volume.

Faces with four sign changes are the ambiguous case, and they are resolved with
the **asymptotic decider**: the bilinear interpolant over the face has a saddle,
and which pair of corners it joins is decided by the saddle's value rather than
by a fixed choice. That matters for more than aesthetics — the decision depends
only on the four values of the shared face, so two cubes either side of it always
make the *same* decision, which is what keeps the surface watertight. A fixed
choice is where the classic table's holes come from.
"""

# Corner `c` of a cell sits at offset `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`.
const CORNER = ntuple(c -> ((c - 1) & 1, ((c - 1) >> 1) & 1, ((c - 1) >> 2) & 1), 8)

# The twelve edges as (low corner, high corner), grouped by the axis they run
# along. The low corner is the one whose bit for that axis is 0, which is what
# makes the edge's identity `(cell of the low corner, axis)` — the key two
# neighbouring cells agree on, and so the key vertices are shared by.
const EDGE = ((1, 2, 1), (3, 4, 1), (5, 6, 1), (7, 8, 1),
              (1, 3, 2), (2, 4, 2), (5, 7, 2), (6, 8, 2),
              (1, 5, 3), (2, 6, 3), (3, 7, 3), (4, 8, 3))

"""
    FACE

The six faces, each as its four corners in cyclic order. Direction is irrelevant
— only the cyclic adjacency is used, and the winding is fixed afterwards from the
field's own gradient — but the order within each must be a genuine cycle around
the face or the segments pair up wrongly.
"""
const FACE = ((1, 3, 7, 5), (2, 4, 8, 6),      # x = 0, x = 1
              (1, 2, 6, 5), (3, 4, 8, 7),      # y = 0, y = 1
              (1, 2, 4, 3), (5, 6, 8, 7))      # z = 0, z = 1

"""
    edgeof(a, b) -> Int

Index into [`EDGE`](@ref) of the edge joining corners `a` and `b`.
"""
function edgeof(a::Int, b::Int)
    for (i, e) in enumerate(EDGE)
        ((e[1] == a && e[2] == b) || (e[1] == b && e[2] == a)) && return i
    end
    error("corners $a and $b are not an edge")
end

"""
    FACEEDGE

For each face, its four edges in the same cyclic order as [`FACE`](@ref)'s
corners: edge `k` joins corner `k` to corner `k + 1`.
"""
const FACEEDGE = ntuple(f -> ntuple(k -> edgeof(FACE[f][k], FACE[f][mod1(k + 1, 4)]), 4), 6)

"""
    facepairs(f0, f1, f2, f3, crossed) -> ((a, b), (c, d)) or ((a, b), (0, 0))

Which of a face's crossed edges join to which, given the face's four corner
values in cyclic order. Edges are named 1..4 by position in the cycle.

With two crossings there is one segment and no choice. With four the face is
ambiguous, and the **asymptotic decider** makes it: the bilinear interpolant
    B(s, t) = f0(1-s)(1-t) + f1 s(1-t) + f2 s t + f3 (1-s)t
has a saddle whose value is `(f0 f2 - f1 f3) / (f0 + f2 - f1 - f3)`. If the
saddle has the same sign as the diagonal pair `f0, f2`, that pair is joined
through the middle of the face and the segments cut off `f1` and `f3` separately;
otherwise the other way round.

Because this reads only the four shared values, the cell on the other side of the
face computes the same answer, and the two triangulations meet.
"""
function facepairs(f0::Float32, f1::Float32, f2::Float32, f3::Float32, crossed::NTuple{4,Bool})
    n = count(crossed)
    if n == 2
        a = crossed[1] ? 1 : crossed[2] ? 2 : 3
        b = crossed[4] ? 4 : crossed[3] ? 3 : 2
        return ((a, b), (0, 0))
    elseif n == 4
        den = f0 + f2 - f1 - f3
        # A zero denominator is a degenerate bilinear patch with no saddle; either
        # pairing is as good, and the fixed one keeps the two sides of the face
        # agreeing, which is the only property that matters.
        s = den == 0 ? f0 : (f0 * f2 - f1 * f3) / den
        joined = signbit(s) == signbit(f0)
        # edges in cycle order are (c0c1, c1c2, c2c3, c3c0) = 1, 2, 3, 4
        return joined ? ((4, 1), (2, 3)) : ((1, 2), (3, 4))
    end
    return ((0, 0), (0, 0))
end

"""
    marchingcubes(field; level = 0.0) -> (vertices, faces)

The `level` iso-surface of `field`, a 3-D array of scalars.

`vertices` is `(3, V)` in **grid index space**, zero-based, the way
`skimage.measure.marching_cubes` returns them — so `[0, size(field, d) - 1]` on
each axis. [`meshbox`](@ref) is what maps them into the sampling box.
`faces` is `(3, F)` of 1-based indices into `vertices`.

Vertices are shared: an intersection is identified by the grid edge it lies on,
so the two to six cells around that edge produce one vertex, not one each.

Winding is fixed afterwards from the field's own finite-difference gradient, so
face normals point **toward decreasing values** — `skimage`'s default
`gradient_direction = "descent"`, and for an occupancy field where inside is
positive that is outward.
"""
function marchingcubes(field::AbstractArray{T,3}; level::Real = 0.0) where {T<:Real}
    nx, ny, nz = size(field)
    (nx > 1 && ny > 1 && nz > 1) || throw(ArgumentError("field must be at least 2 on every axis"))
    lv = Float32(level)

    # Triangles as triples of edge ids; the ids become vertex indices once the
    # whole grid is done and the unique set is known.
    tris = Vector{NTuple{3,UInt64}}()
    sizehint!(tris, 1 << 16)
    vals = Vector{Float32}(undef, 8)
    adj1 = Vector{Int}(undef, 12)
    adj2 = Vector{Int}(undef, 12)
    loop = Vector{Int}(undef, 12)

    @inbounds for k in 1:(nz - 1), j in 1:(ny - 1), i in 1:(nx - 1)
        below = false
        above = false
        for c in 1:8
            dx, dy, dz = CORNER[c]
            v = Float32(field[i + dx, j + dy, k + dz]) - lv
            vals[c] = v
            v < 0 ? (below = true) : (above = true)
        end
        (below && above) || continue

        fill!(adj1, 0)
        fill!(adj2, 0)
        for f in 1:6
            cs = FACE[f]
            es = FACEEDGE[f]
            f0, f1, f2, f3 = vals[cs[1]], vals[cs[2]], vals[cs[3]], vals[cs[4]]
            crossed = (signbit(f0) != signbit(f1), signbit(f1) != signbit(f2),
                       signbit(f2) != signbit(f3), signbit(f3) != signbit(f0))
            (p, q) = facepairs(f0, f1, f2, f3, crossed)
            for pair in (p, q)
                pair[1] == 0 && continue
                a, b = es[pair[1]], es[pair[2]]
                adj1[a] == 0 ? (adj1[a] = b) : (adj2[a] = b)
                adj1[b] == 0 ? (adj1[b] = a) : (adj2[b] = a)
            end
        end

        # Walk the loops. Every crossed edge has degree exactly two, so following
        # `adj` from an unvisited one and never stepping back where it came from
        # returns to the start.
        seen = 0x0000
        for start in 1:12
            (adj1[start] == 0 || (seen >> (start - 1)) & 1 == 1) && continue
            n = 0
            e, prev = start, 0
            while true
                n += 1
                loop[n] = e
                seen |= 0x0001 << (e - 1)
                nxt = adj1[e] == prev ? adj2[e] : adj1[e]
                prev, e = e, nxt
                (e == start || e == 0 || n >= 12) && break
            end
            n >= 3 || continue
            a = edgeid(loop[1], i, j, k)
            for t in 2:(n - 1)
                push!(tris, (a, edgeid(loop[t], i, j, k), edgeid(loop[t + 1], i, j, k)))
            end
        end
    end

    return assemble(field, tris, lv)
end

"""
    edgeid(e, i, j, k) -> UInt64

A grid edge's identity, shared by every cell that touches it: the low endpoint's
grid position and the axis it runs along. This is what makes vertices shared
rather than duplicated per cell.
"""
@inline function edgeid(e::Int, i::Int, j::Int, k::Int)
    lo, _, axis = EDGE[e]
    dx, dy, dz = CORNER[lo]
    return ((UInt64(i + dx) << 42) | (UInt64(j + dy) << 21) | UInt64(k + dz)) * 3 + UInt64(axis)
end

"""
    assemble(field, tris, level) -> (vertices, faces)

Turn edge-id triples into a shared vertex array and an index array, placing each
vertex by linear interpolation along its edge and orienting each triangle by the
field's gradient.
"""
function assemble(field::AbstractArray{T,3}, tris::Vector{NTuple{3,UInt64}}, level::Float32) where {T}
    ids = Vector{UInt64}(undef, 3 * length(tris))
    @inbounds for (t, tri) in enumerate(tris)
        ids[3t - 2], ids[3t - 1], ids[3t] = tri
    end
    uniq = sort!(unique(ids))
    index = Dict{UInt64,Int32}(id => Int32(n) for (n, id) in enumerate(uniq))

    nx, ny, nz = size(field)
    verts = Array{Float32}(undef, 3, length(uniq))
    @inbounds for (n, id) in enumerate(uniq)
        axis = Int((id - 1) % 3) + 1
        p = (id - UInt64(axis)) ÷ 3
        i = Int(p >> 42)
        j = Int((p >> 21) & 0x1fffff)
        k = Int(p & 0x1fffff)
        di, dj, dk = axis == 1 ? (1, 0, 0) : axis == 2 ? (0, 1, 0) : (0, 0, 1)
        a = Float32(field[i, j, k]) - level
        b = Float32(field[i + di, j + dj, k + dk]) - level
        # skimage's placement: the linear root along the edge. `a == b` cannot
        # happen for a crossed edge, since the two have opposite signs.
        t = a / (a - b)
        verts[1, n] = (i - 1) + t * di
        verts[2, n] = (j - 1) + t * dj
        verts[3, n] = (k - 1) + t * dk
    end

    faces = Array{Int32}(undef, 3, length(tris))
    @inbounds for (t, tri) in enumerate(tris)
        i1, i2, i3 = index[tri[1]], index[tri[2]], index[tri[3]]
        # Orient toward decreasing field. The winding out of the loop walk is
        # consistent within a cell but its sign depends on which face the walk
        # started from, so it is settled here against the gradient rather than
        # tracked through the walk.
        cx = (verts[1, i1] + verts[1, i2] + verts[1, i3]) / 3
        cy = (verts[2, i1] + verts[2, i2] + verts[2, i3]) / 3
        cz = (verts[3, i1] + verts[3, i2] + verts[3, i3]) / 3
        g = gradient(field, cx, cy, cz, nx, ny, nz)
        ux, uy, uz = verts[1, i2] - verts[1, i1], verts[2, i2] - verts[2, i1], verts[3, i2] - verts[3, i1]
        vx, vy, vz = verts[1, i3] - verts[1, i1], verts[2, i3] - verts[2, i1], verts[3, i3] - verts[3, i1]
        nrmx, nrmy, nrmz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        flip = nrmx * g[1] + nrmy * g[2] + nrmz * g[3] > 0
        faces[1, t], faces[2, t], faces[3, t] = flip ? (i1, i3, i2) : (i1, i2, i3)
    end
    return verts, faces
end

"""
    gradient(field, x, y, z, nx, ny, nz) -> (gx, gy, gz)

Central differences at the nearest grid point to a zero-based position, clamped
at the boundary. Only its direction is used, so the spacing is not divided out.
"""
@inline function gradient(field, x, y, z, nx, ny, nz)
    i = clamp(round(Int, x) + 1, 1, nx)
    j = clamp(round(Int, y) + 1, 1, ny)
    k = clamp(round(Int, z) + 1, 1, nz)
    @inbounds gx = Float32(field[min(i + 1, nx), j, k]) - Float32(field[max(i - 1, 1), j, k])
    @inbounds gy = Float32(field[i, min(j + 1, ny), k]) - Float32(field[i, max(j - 1, 1), k])
    @inbounds gz = Float32(field[i, j, min(k + 1, nz)]) - Float32(field[i, j, max(k - 1, 1)])
    return (gx, gy, gz)
end

"""
    meshbox(vertices, grid, octree; box_v = 1.01) -> vertices

`MCSurfaceExtractor.run`'s rescale from grid indices into the sampling box:

    vertices = vertices / grid_size * bbox_size + bbox_min

**Divided by `grid_size`, which is `octree + 1`, not by `octree`.** The grid's
last sample sits exactly on the box's far face, so mapping index `octree` back to
`box_v` would divide by `octree`. Upstream divides by one more than that, which
shrinks the mesh by a factor of `octree / (octree + 1)` — 0.26% at the default
384 — and shifts it. It is reproduced because it is what the reference mesh is,
not because it is right.
"""
function meshbox(verts::AbstractArray{<:Real,2}, octree::Integer; box_v::Real = 1.01)
    G = Float32(octree + 1)
    lo, size_ = Float32(-box_v), Float32(2box_v)
    return @. Float32(verts) / G * size_ + lo
end

"""
    latents2mesh(m, latents; box_v, octree, level, progress) -> (vertices, faces)

`ShapeVAE.latents2mesh`: sweep the geometry decoder over the grid, then extract
the surface. `vertices` is `(3, V)` in the sampling box, `faces` is `(3, F)`.

The field is brought to the host for the extraction — [`marchingcubes`](@ref) is
host code. At the default `octree = 384` that is 228 MB across the bus once,
against the 7134 decoder calls that produced it.

**The axes are reversed on the way out.** [`occupancy`](@ref) returns the field in
the runtime's layout, where `field[k, j, i]` is upstream's `grid_logits[i, j, k]`,
so marching cubes over it produces vertices ordered `(z, y, x)`. Reversing the
three components is what puts them back in the `(x, y, z)` the box mapping
expects; skipping it yields a mesh that is a transpose of the right one, which is
a plausible-looking object rather than an error.

Reversing three components is a *reflection*, not a rotation, so every triangle
turns inside out with it. Swapping two indices of each face undoes that. Leaving
it out has a recognisable symptom — geometry that is exactly right and renders
black, or lit from within.
"""
function latents2mesh(m::Hunyuan3D, latents; box_v::Real = 1.01, octree::Integer = 384,
                      level::Real = 0.0, progress = nothing)
    field = Array(occupancy(m, latents; box_v, octree, progress))
    verts, faces = marchingcubes(field; level)
    return meshbox(reverse(verts; dims = 1), octree; box_v), faces[[1, 3, 2], :]
end
