"""
Where two triangle meshes cross: every edge of one tested against the triangles of the other
that a [`SurfaceGrid`](@ref) lists in the cells along the edge. Two closed meshes intersect
exactly when an edge of one crosses a triangle of the other, so the crossing points show
where they do even when no vertex lies inside the other mesh.
"""

"""
    edgecrossings(dev, grid, V, F) -> (points, count)

The points where the edges of the mesh `V` `(3, nv)` Float32, `F` `(3, nf)` Int32 cross the
triangles of `grid`'s mesh, `(3, count)`: one per face edge that crosses (an edge shared by
two faces counts twice), at its crossing nearest its first end.
"""
function edgecrossings(dev::Mantle.Device, s::SurfaceGrid, V::Mantle.Buffer, F::Mantle.Buffer)
    ne = 3 * size(F, 2)
    hits = Mantle.Buffer(dev, Float32, (3, max(ne, 1)))
    flags = fill!(Mantle.Buffer(dev, Int32, max(ne, 1)), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, edgecrossing!, (hits, flags, V, F, ne, gridargs(s)...), ne; name = "intersect/edges")
    runonce!(g)
    points, count = compactcolumns(dev, hits, flags, ne)
    foreach(Mantle.free!, (hits, flags))
    return points, count
end

"""
Where the segment `a b` crosses triangle `p q r` (Möller and Trumbore), as `(hit, t)` with
the crossing at `a + t (b - a)`; a segment in the triangle's plane does not cross it.
"""
@inline function segmenttriangle(a::Vec3f, b::Vec3f, p::Vec3f, q::Vec3f, r::Vec3f)
    d = b - a
    e1, e2 = q - p, r - p
    h = cross(d, e2)
    det = dot(e1, h)
    abs(det) <= 1f-12 * dot(e1, e1) * sqrt(dot(d, d) * dot(e2, e2)) && return (false, 0f0)
    s = a - p
    u = dot(s, h) / det
    (u < 0f0 || u > 1f0) && return (false, 0f0)
    c = cross(s, e1)
    v = dot(d, c) / det
    (v < 0f0 || u + v > 1f0) && return (false, 0f0)
    t = dot(e2, c) / det
    return (0f0 <= t <= 1f0, t)
end

function edgecrossing!(hits, flags, V, F, ne, VT, FT, keys, start, tris, slots, cap, lox, loy, loz, cell)
    e = Int32(KI.get_global_id().x)
    e <= ne || return nothing
    f = (e - Int32(1)) ÷ Int32(3) + Int32(1)
    k = (e - Int32(1)) % Int32(3) + Int32(1)
    @inbounds a = vertex(V, F[k, f])
    @inbounds b = vertex(V, F[k % Int32(3) + Int32(1), f])
    x0, x1 = cellof(min(a[1], b[1]), lox, cell), cellof(max(a[1], b[1]), lox, cell)
    y0, y1 = cellof(min(a[2], b[2]), loy, cell), cellof(max(a[2], b[2]), loy, cell)
    z0, z1 = cellof(min(a[3], b[3]), loz, cell), cellof(max(a[3], b[3]), loz, cell)
    best = Inf32
    @inbounds for z in z0:z1, y in y0:y1, x in x0:x1
        incells(x, y, z) || continue
        for i in cellrange(keys, start, slots, cap, cellkey64(Int32(0), x, y, z))
            t3 = tris[i]
            hit, t = segmenttriangle(a, b, vertex(VT, FT[1, t3]), vertex(VT, FT[2, t3]), vertex(VT, FT[3, t3]))
            hit && t < best && (best = t)
        end
    end
    if best < Inf32
        p = a + best * (b - a)
        @inbounds hits[1, e], hits[2, e], hits[3, e] = p
        @inbounds flags[e] = Int32(1)
    end
    return nothing
end
