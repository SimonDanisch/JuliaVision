"""
    SurfaceGrid(vertices, faces; res)

A triangle mesh bucketed for closest-point queries, the stand-in for upstream's
`cuBVH.unsigned_distance`: a `res`^3 grid of cells over the unit cube
[-0.5, 0.5]^3, every triangle listed in every cell its bounding box touches.
Cell `c`'s triangles are `tris[start[c]:start[c + 1] - 1]`, `c` numbered like
the voxel grid (`x + res (y + res z) + 1`).

A query searches the 27 cells around its point, so it finds the closest point
whenever that lies within one cell; farther out it finds nothing and the point
stays where it is.
"""
struct SurfaceGrid
    vertices::Matrix{Float32}
    faces::Matrix{Int32}
    start::Vector{Int32}
    tris::Vector{Int32}
    res::Int
end

function SurfaceGrid(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer}; res::Int)
    cell(x) = clamp(unsafe_trunc(Int, floor((x + 0.5f0) * res)), 0, res - 1)
    function box(f)
        lo = ntuple(i -> cell(minimum(vertices[i, faces[k, f]] for k in 1:3)), 3)
        hi = ntuple(i -> cell(maximum(vertices[i, faces[k, f]] for k in 1:3)), 3)
        return lo, hi
    end
    start = zeros(Int32, res^3 + 1)
    for f in axes(faces, 2)
        lo, hi = box(f)
        for z in lo[3]:hi[3], y in lo[2]:hi[2], x in lo[1]:hi[1]
            start[x + res * (y + res * z) + 2] += 1
        end
    end
    start[1] = 1
    cumsum!(start, start)
    fill = start[1:end-1]
    tris = Vector{Int32}(undef, start[end] - 1)
    for f in axes(faces, 2)
        lo, hi = box(f)
        for z in lo[3]:hi[3], y in lo[2]:hi[2], x in lo[1]:hi[1]
            c = x + res * (y + res * z) + 1
            tris[fill[c]] = f
            fill[c] += 1
        end
    end
    return SurfaceGrid(Matrix{Float32}(vertices), Matrix{Int32}(faces), start, tris, res)
end

"""
The point of triangle `a b c` closest to `p` (Ericson, *Real-Time Collision
Detection*, 5.1.5): by the Voronoi region of the triangle `p` falls in.
"""
function closestpoint(p::Vec3f, a::Vec3f, b::Vec3f, c::Vec3f)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = dot(ab, ap), dot(ac, ap)
    (d1 <= 0f0 && d2 <= 0f0) && return a
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    (d3 >= 0f0 && d4 <= d3) && return b
    vc = d1 * d4 - d3 * d2
    (vc <= 0f0 && d1 >= 0f0 && d3 <= 0f0) && return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    (d6 >= 0f0 && d5 <= d6) && return c
    vb = d5 * d2 - d1 * d6
    (vb <= 0f0 && d2 >= 0f0 && d6 <= 0f0) && return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    (va <= 0f0 && d4 - d3 >= 0f0 && d5 - d6 >= 0f0) &&
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    s = 1f0 / (va + vb + vc)
    return a + ab * (vb * s) + ac * (vc * s)
end

"""
The point of the surface in `vertices`/`faces`, bucketed by `start`/`tris` into
a `res`^3 grid, closest to `p`; `p` itself when no triangle is within reach. A
degenerate triangle's `NaN` answer loses every comparison and drops out.
"""
project(p::Vec3f, vertices, faces, start, tris, res) = nearest(p, vertices, faces, start, tris, res)[1]

"""
`(q, d2)`: the point `q` of the surface closest to `p` and the squared distance
to it, `(p, Inf32)` when no triangle is within reach (see [`project`](@ref)).
"""
function nearest(p::Vec3f, vertices, faces, start, tris, res)
    @inbounds begin
        cx = unsafe_trunc(Int32, floor((p[1] + 0.5f0) * res))
        cy = unsafe_trunc(Int32, floor((p[2] + 0.5f0) * res))
        cz = unsafe_trunc(Int32, floor((p[3] + 0.5f0) * res))
        best = Inf32
        q = p
        for z in (cz - Int32(1)):(cz + Int32(1)), y in (cy - Int32(1)):(cy + Int32(1)),
            x in (cx - Int32(1)):(cx + Int32(1))
            (0 <= x < res && 0 <= y < res && 0 <= z < res) || continue
            c = x + res * (y + res * z) + 1
            for i in start[c]:(start[c + 1] - Int32(1))
                f = tris[i]
                a = Vec3f(vertices[1, faces[1, f]], vertices[2, faces[1, f]], vertices[3, faces[1, f]])
                b = Vec3f(vertices[1, faces[2, f]], vertices[2, faces[2, f]], vertices[3, faces[2, f]])
                d = Vec3f(vertices[1, faces[3, f]], vertices[2, faces[3, f]], vertices[3, faces[3, f]])
                r = closestpoint(p, a, b, d)
                e = r - p
                dist = dot(e, e)
                if dist < best
                    best = dist
                    q = r
                end
            end
        end
    end
    return q, best
end
