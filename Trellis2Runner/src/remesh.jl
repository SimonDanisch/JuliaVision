"""
    remesh(dev, vertices, faces; resolution, scale, center = Vec3f(0), band = 1) -> (vertices, faces)

cumesh's `remesh_narrow_band_dc`, what `to_glb(remesh = True)` rebuilds the
mesh with: the surface `band` voxels away from the mesh, the zero set of its
unsigned distance minus `band` voxels, extracted by dual contouring on a
`resolution`^3 grid over the cube of side `scale` around `center`. Only voxels
near that surface are visited, those whose centre is within 0.87 voxels of it.

Holes and slits narrower than about two voxels close, and the result is a
closed, consistently wound shell whatever the input was: the dual-grid mesh's
open sheets and edges of three or more faces are gone. A thin part becomes a
shell two voxels thick.

Each voxel gets one vertex, the mean of the points where the distance crosses
`band` voxels on its edges; each crossed edge a quad of the four voxels around
it, split along the diagonal whose two triangles agree best in their normals.
(cumesh means to choose the split so, but its comparison takes the cross
product of vertices 2-4 of the six instead of the second triangle's, which
always picks the first split.)
"""
function remesh(dev::Mantle.Device, vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer};
                resolution::Int, scale::Real, center::Vec3f = Vec3f(0), band::Real = 1)
    res = resolution
    cell = Float32(scale / res)
    eps = Float32(band) * cell
    lo = center .- Float32(scale) / 2
    # Distances are exact out to a cell of this grid, four voxels; the farthest
    # asked for is a band voxel's corner, under three.
    grid = SurfaceGrid(vertices, faces; res = cld(res, 4))
    voxel(i) = ((i - 1) % res, (i - 1) ÷ res % res, (i - 1) ÷ (res * res))
    # Candidates: voxels whose centre is in a triangle's box grown by the band.
    nf = size(faces, 2)
    mask = fill!(Mantle.Buffer(dev, UInt8, res^3), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, markband!, (mask, Mantle.Buffer(dev, grid.vertices), Mantle.Buffer(dev, grid.faces),
                                    res, lo[1], lo[2], lo[3], cell, eps + 0.87f0 * cell, nf),
                     nf; name = "remesh/band")
    runonce!(g)
    candidates = findall(!iszero, Array(Mantle.storage(mask)))
    centres = Matrix{Float32}(undef, 3, length(candidates))
    for (j, i) in enumerate(candidates)
        x, y, z = voxel(i)
        centres[:, j] .= lo .+ (Vec3f(x, y, z) .+ 0.5f0) .* cell
    end
    d = distances(dev, centres, grid)
    active = candidates[abs.(d .- eps) .< 0.87f0 * cell]
    n = length(active)
    voxbits = falses(res^3)
    voxbits[active] .= true
    voxid = GridSet(voxbits)
    # Distance minus the band at every corner of an active voxel.
    R = res + 1
    cornerbits = falses(R^3)
    for i in active
        x, y, z = voxel(i)
        for k in 0:1, j in 0:1, l in 0:1
            cornerbits[(x + l) + R * ((y + j) + R * (z + k)) + 1] = true
        end
    end
    cornerid = GridSet(cornerbits)
    corners = findall(cornerbits)
    cpos = Matrix{Float32}(undef, 3, length(corners))
    for (j, i) in enumerate(corners)
        cpos[:, j] .= lo .+ Vec3f((i - 1) % R, (i - 1) ÷ R % R, (i - 1) ÷ (R * R)) .* cell
    end
    value = distances(dev, cpos, grid) .- eps
    val(x, y, z) = value[cornerid[x + R * (y + R * z) + 1]]
    # One vertex per voxel, and the sign change along its three far edges.
    dual = Vector{Vec3f}(undef, n)
    crossed = zeros(Int8, 3, n)
    unit = (Vec{3,Int}(1, 0, 0), Vec{3,Int}(0, 1, 0), Vec{3,Int}(0, 0, 1))
    Threads.@threads for i in 1:n
        c = Vec{3,Int}(voxel(active[i])...)
        s, m = Vec3f(0), 0
        for a in 1:3, u in 0:1, v in 0:1
            o = a == 1 ? Vec{3,Int}(0, u, v) : a == 2 ? Vec{3,Int}(u, 0, v) : Vec{3,Int}(u, v, 0)
            p = c + o
            v1, v2 = val(p...), val((p + unit[a])...)
            if (v1 < 0) != (v2 < 0)
                s += Vec3f(p) + (-v1 / (v2 - v1)) * Vec3f(unit[a])
                m += 1
            end
            (u == 1 && v == 1) && (crossed[a, i] = v1 < 0 ? (v2 >= 0 ? 1 : 0) : (v2 < 0 ? -1 : 0))
        end
        dual[i] = m > 0 ? s / m : Vec3f(c) .+ 0.5f0
    end
    # A quad of the four voxels around every crossed edge.
    around = ((Vec{3,Int}(0, 0, 0), Vec{3,Int}(0, 0, 1), Vec{3,Int}(0, 1, 1), Vec{3,Int}(0, 1, 0)),
              (Vec{3,Int}(0, 0, 0), Vec{3,Int}(1, 0, 0), Vec{3,Int}(1, 0, 1), Vec{3,Int}(0, 0, 1)),
              (Vec{3,Int}(0, 0, 0), Vec{3,Int}(0, 1, 0), Vec{3,Int}(1, 1, 0), Vec{3,Int}(1, 0, 0)))
    tris = Int32[]
    pos(q) = dual[q]
    normal(a, b, c) = normalize(cross(pos(b) - pos(a), pos(c) - pos(a)))
    for i in 1:n, a in 1:3
        dir = crossed[a, i]
        dir == 0 && continue
        c = Vec{3,Int}(voxel(active[i])...)
        q = ntuple(4) do k
            p = c + around[a][k]
            all(0 .<= p .< res) ? voxid[p[1] + res * (p[2] + res * p[3]) + 1] : Int32(0)
        end
        all(!=(0), q) || continue
        # cumesh's two splits, each wound so the quad faces out.
        s1 = dir == 1 ? (q[1], q[3], q[2], q[1], q[4], q[3]) : (q[1], q[2], q[3], q[1], q[3], q[4])
        s2 = dir == 1 ? (q[1], q[4], q[2], q[4], q[3], q[2]) : (q[1], q[2], q[4], q[4], q[2], q[3])
        align(s) = dot(normal(s[1], s[2], s[3]), normal(s[4], s[5], s[6]))
        append!(tris, align(s1) >= align(s2) ? s1 : s2)
    end
    # Voxels no quad reaches drop out.
    used = zeros(Int32, n)
    for t in tris
        used[t] = 1
    end
    keep = findall(!iszero, used)
    used[keep] .= 1:length(keep)
    V = Matrix{Float32}(undef, 3, length(keep))
    for (j, i) in enumerate(keep)
        V[:, j] .= lo .+ dual[i] .* cell
    end
    return V, reshape(used[tris], 3, :)
end

"""
    GridSet(bits)

Cells of a dense grid, a set bit each, numbered in order: `s[i]` is how many set
cells there are up to and including cell `i`, zero when `i` is not set. One bit
and 1/64 of a word per cell, where an index array takes a word: at 1024^3 a
tenth of a gigabyte against four.
"""
struct GridSet
    bits::BitVector
    rank::Vector{Int32}   # set cells before each 64-cell chunk
end

function GridSet(bits::BitVector)
    chunks = bits.chunks
    rank = Vector{Int32}(undef, length(chunks))
    s = Int32(0)
    for (k, w) in enumerate(chunks)
        rank[k] = s
        s += Int32(count_ones(w))
    end
    return GridSet(bits, rank)
end

function Base.getindex(s::GridSet, i::Integer)
    k, b = (i - 1) >> 6 + 1, (i - 1) & 63
    w = s.bits.chunks[k]
    (w >> b) & 1 == 0 && return Int32(0)
    # Bits 0..b; at b = 63 the shift gives zero and the mask all ones.
    return s.rank[k] + Int32(count_ones(w & ((UInt64(2) << b) - UInt64(1))))
end

"""Mark, in the `res`^3 `mask`, every voxel whose centre lies in a triangle's
bounding box grown by `reach`; the grid's corner is at `lo`, voxels `cell` wide."""
function markband!(mask, vertices, faces, res, lox, loy, loz, cell, reach, nf)
    f = KI.get_global_id().x
    f <= nf || return nothing
    @inbounds begin
        ia, ib, ic = faces[1, f], faces[2, f], faces[3, f]
        # Voxel units, voxel centres at the integers.
        gx(v) = (vertices[1, v] - lox) / cell - 0.5f0
        gy(v) = (vertices[2, v] - loy) / cell - 0.5f0
        gz(v) = (vertices[3, v] - loz) / cell - 0.5f0
        r = reach / cell
        x0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gx(ia), gx(ib), gx(ic)) - r)))
        x1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gx(ia), gx(ib), gx(ic)) + r)))
        y0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gy(ia), gy(ib), gy(ic)) - r)))
        y1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gy(ia), gy(ib), gy(ic)) + r)))
        z0 = max(Int32(0), unsafe_trunc(Int32, ceil(min(gz(ia), gz(ib), gz(ic)) - r)))
        z1 = min(Int32(res - 1), unsafe_trunc(Int32, floor(max(gz(ia), gz(ib), gz(ic)) + r)))
        for z in z0:z1, y in y0:y1, x in x0:x1
            mask[x + res * (y + res * z) + 1] = 0x01
        end
    end
    return nothing
end

"""The unsigned distance from every point `(3, n)` to the surface of `grid`,
`Inf` beyond its reach (see [`nearest`](@ref))."""
function distances(dev::Mantle.Device, points::Matrix{Float32}, grid::SurfaceGrid)
    n = size(points, 2)
    out = fill!(Mantle.Buffer(dev, Float32, n), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, udf!, (out, Mantle.Buffer(dev, points), Mantle.Buffer(dev, grid.vertices),
                               Mantle.Buffer(dev, grid.faces), Mantle.Buffer(dev, grid.start),
                               Mantle.Buffer(dev, grid.tris), grid.res, n),
                     n; name = "remesh/distance")
    runonce!(g)
    return Array(Mantle.storage(out))
end

function udf!(out, points, sv, sf, start, tris, sres, n)
    t = KI.get_global_id().x
    t <= n || return nothing
    @inbounds begin
        _, d2 = nearest(Vec3f(points[1, t], points[2, t], points[3, t]), sv, sf, start, tris, sres)
        out[t] = sqrt(d2)
    end
    return nothing
end
