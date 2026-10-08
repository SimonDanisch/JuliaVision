"""
Marching cubes over a dense field, on the device or the host.

**The case table is derived, not transcribed.** A transcribed table is 256 rows
of hand-written indices where one wrong entry is a hole in a mesh rather than an
error. [`celltriangles`](@ref) builds each case from the cube's faces instead: on
each of the six faces the sign changes come in pairs and define line segments,
every crossed edge belongs to exactly two faces so every segment endpoint has
degree two, and the segments therefore close into loops, which are fanned into
triangles. [`MCTABLE`](@ref) is that construction evaluated for every case.

**Ambiguous faces use the asymptotic decider.** A face with four sign changes
has a saddle in its bilinear interpolant, and which diagonal pair of corners it
joins is decided by the saddle's value rather than by a fixed choice. The
decision reads only the four values of the shared face, so the two cells either
side of it make the same one, which is what keeps the surface watertight. The
table is therefore keyed by the eight corner signs and one decision bit per face:
`256 * 64` cases.

**Winding is part of the table.** Each loop is walked so the side of the field
below `level` lies to its left, seen from outside the cube, which makes every
triangle face toward decreasing values (`skimage`'s `gradient_direction =
"descent"`). A face is walked in opposite directions by the two cells sharing it,
so the orientation is consistent across the whole mesh.

**Vertices are shared.** One vertex per crossed grid edge, at skimage's linear
root along the edge. Vertices and triangles are numbered brick by brick, see
[`BRICK`](@ref), the same on the host and the device.
"""

# Corner `c` of a cell sits at offset `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`, `c` zero-based.
const CORNER = ntuple(c -> ((c - 1) & 1, ((c - 1) >> 1) & 1, ((c - 1) >> 2) & 1), 8)

# The twelve edges as (low corner, high corner, axis), grouped by axis. The low
# corner is the one whose bit for that axis is 0, so an edge's identity is
# `(grid point of the low corner, axis)`: the key every cell around it agrees on.
# [`edgeowner`](@ref) is the same table as arithmetic, for the device.
const EDGE = ((1, 2, 1), (3, 4, 1), (5, 6, 1), (7, 8, 1),
              (1, 3, 2), (2, 4, 2), (5, 7, 2), (6, 8, 2),
              (1, 5, 3), (2, 6, 3), (3, 7, 3), (4, 8, 3))

"""
    FACE

The six faces, each as its four corners in cyclic order, starting from the same
grid point on both cells that share the face. The decider reads the corners in
this order, so both cells see the same four values in the same places.
"""
const FACE = ((1, 3, 7, 5), (2, 4, 8, 6),      # x = 0, x = 1
              (1, 2, 6, 5), (3, 4, 8, 7),      # y = 0, y = 1
              (1, 2, 4, 3), (5, 6, 8, 7))      # z = 0, z = 1

"""The outward normal of each face in [`FACE`](@ref)."""
const FACENORMAL = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))

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
    facepairs(crossed, joined) -> ((a, b), (c, d))

Which of a face's crossed edges join to which, edges named 1..4 by position in
the face's cycle, `(0, 0)` for no segment. With two crossings there is one
segment and no choice. With four, `joined` says the corners `c0` and `c2` are
connected through the face's middle, so the segments cut off `c1` (edges 1 and 2)
and `c3` (edges 3 and 4); otherwise they cut off `c0` and `c2`.
"""
function facepairs(crossed::NTuple{4,Bool}, joined::Bool)
    n = count(crossed)
    if n == 2
        a = crossed[1] ? 1 : crossed[2] ? 2 : 3
        b = crossed[4] ? 4 : crossed[3] ? 3 : 2
        return ((a, b), (0, 0))
    elseif n == 4
        return joined ? ((1, 2), (3, 4)) : ((4, 1), (2, 3))
    end
    return ((0, 0), (0, 0))
end

"""The midpoint of edge `e` in cell units."""
function edgemidpoint(e::Int)
    lo, _, axis = EDGE[e]
    return ntuple(d -> CORNER[lo][d] + (d == axis ? 0.5 : 0.0), 3)
end

"""
    walkspositive(a, b, cube) -> Bool

Whether the segment from edge `a` to edge `b` has the corners below the level on
its left, seen from outside the face the two edges share. That is the walk
direction whose fan faces toward decreasing values.
"""
function walkspositive(a::Int, b::Int, cube::Int)
    f = findfirst(f -> a in FACEEDGE[f] && b in FACEEDGE[f], 1:6)
    lo, hi, _ = EDGE[a]
    below = (cube >> (lo - 1)) & 1 == 1 ? lo : hi
    pa, pb, c = edgemidpoint(a), edgemidpoint(b), CORNER[below]
    u = pb .- pa
    w = c .- pa
    n = (u[2] * w[3] - u[3] * w[2], u[3] * w[1] - u[1] * w[3], u[1] * w[2] - u[2] * w[1])
    return sum(FACENORMAL[f] .* n) > 0
end

"""
    celltriangles(cube, joined) -> Vector{NTuple{3,Int}}

The triangles of one cell as triples of edge indices into [`EDGE`](@ref). Bit
`c - 1` of `cube` says corner `c` is below the level; bit `f - 1` of `joined` is
the decider's answer on face `f`, read only where that face has four crossings.
"""
function celltriangles(cube::Int, joined::Int)
    below(c) = (cube >> (c - 1)) & 1 == 1
    adj = zeros(Int, 2, 12)
    for f in 1:6
        cs, es = FACE[f], FACEEDGE[f]
        crossed = ntuple(k -> below(cs[k]) != below(cs[mod1(k + 1, 4)]), 4)
        for (p, q) in facepairs(crossed, (joined >> (f - 1)) & 1 == 1)
            p == 0 && continue
            a, b = es[p], es[q]
            adj[adj[1, a] == 0 ? 1 : 2, a] = b
            adj[adj[1, b] == 0 ? 1 : 2, b] = a
        end
    end
    tris = NTuple{3,Int}[]
    seen = falses(12)
    for start in 1:12
        (adj[1, start] == 0 || seen[start]) && continue
        loop = Int[]
        e, prev = start, 0
        while true
            push!(loop, e)
            seen[e] = true
            next = adj[1, e] == prev ? adj[2, e] : adj[1, e]
            prev, e = e, next
            e == start && break
        end
        walkspositive(loop[1], loop[2], cube) || (loop = [loop[1]; reverse(loop[2:end])])
        for t in 2:(length(loop) - 1)
            push!(tris, (loop[1], loop[t], loop[t + 1]))
        end
    end
    return tris
end

"""
    mctable() -> Matrix{Int32}

[`celltriangles`](@ref) for every case, column `cube + 256 joined + 1`: row 1 is
the triangle count, row `t + 1` triangle `t` with its three edges packed four
bits each, first edge lowest.
"""
function mctable()
    cases = [celltriangles(key & 255, key >> 8) for key in 0:(256 * 64 - 1)]
    table = zeros(Int32, 1 + maximum(length, cases), length(cases))
    for (col, tris) in enumerate(cases)
        table[1, col] = length(tris)
        for (t, (a, b, c)) in enumerate(tris)
            table[t + 1, col] = a | b << 4 | c << 8
        end
    end
    return table
end

"""
    MCTABLE

[`mctable`](@ref), evaluated once when the package is compiled.
"""
const MCTABLE = mctable()

# ── shared by the host and the device ─────────────────────────────────────────
#
# Every per-point and per-cell step is one method taking its index, called by a
# host loop and by a kernel. The host path is therefore the device path, run on
# the CPU, which is what lets the tests compare the two exactly. Every index is
# Int32 and the field is indexed linearly.

@inline fieldat(field, p, lv) = Float32(@inbounds field[p]) - lv

"""Whether the grid edges from point `p` at `(i, j, k)` along x, y and z cross the level."""
@inline function crossings(field, p, i, j, k, nx, ny, nz, lv)
    s = signbit(fieldat(field, p, lv))
    cx = i < nx && signbit(fieldat(field, p + Int32(1), lv)) != s
    cy = j < ny && signbit(fieldat(field, p + nx, lv)) != s
    cz = k < nz && signbit(fieldat(field, p + nx * ny, lv)) != s
    return cx, cy, cz
end

@inline function gridindex(p, nx, ny)
    p0 = p - Int32(1)
    return p0 % nx + Int32(1), (p0 ÷ nx) % ny + Int32(1), p0 ÷ (nx * ny) + Int32(1)
end

"""Edge `e`'s low corner as an offset from the cell, and its axis; [`EDGE`](@ref) as arithmetic."""
@inline function edgeowner(e)
    a = (e - Int32(1)) ÷ Int32(4) + Int32(1)
    k0 = (e - Int32(1)) % Int32(4)
    u = k0 & Int32(1)
    w = k0 >> Int32(1)
    a == Int32(1) && return Int32(0), u, w, a
    a == Int32(2) && return u, Int32(0), w, a
    return u, w, Int32(0), a
end

@inline cornerbit(v, w) = signbit(v) ? Int32(w) : Int32(0)

"""The decider's answer on a face with corners `f0..f3` in cycle order, as bit
`w`; zero unless the face has four crossings. See [`facepairs`](@ref)."""
@inline function facebit(f0, f1, f2, f3, w)
    s0, s1, s2, s3 = signbit(f0), signbit(f1), signbit(f2), signbit(f3)
    (s0 != s1 && s1 != s2 && s2 != s3) || return Int32(0)
    den = f0 + f2 - f1 - f3
    # A zero denominator is a degenerate patch with no saddle. Either pairing
    # does, as long as both cells pick the same one, and they read the same values.
    saddle = den == 0f0 ? f0 : (f0 * f2 - f1 * f3) / den
    return signbit(saddle) == s0 ? Int32(w) : Int32(0)
end

"""The [`MCTABLE`](@ref) column of the cell whose low corner is grid point `p`, zero-based."""
@inline function cellkey(field, p, nx, ny, lv)
    dy, dz = nx, nx * ny
    return cellkey(fieldat(field, p, lv), fieldat(field, p + Int32(1), lv),
                   fieldat(field, p + dy, lv), fieldat(field, p + Int32(1) + dy, lv),
                   fieldat(field, p + dz, lv), fieldat(field, p + Int32(1) + dz, lv),
                   fieldat(field, p + dy + dz, lv), fieldat(field, p + Int32(1) + dy + dz, lv))
end

"""The [`MCTABLE`](@ref) column of a cell with corner values `v1..v8`, less the level, in [`CORNER`](@ref) order."""
@inline function cellkey(v1::Float32, v2::Float32, v3::Float32, v4::Float32,
                         v5::Float32, v6::Float32, v7::Float32, v8::Float32)
    cube = cornerbit(v1, 1) + cornerbit(v2, 2) + cornerbit(v3, 4) + cornerbit(v4, 8) +
           cornerbit(v5, 16) + cornerbit(v6, 32) + cornerbit(v7, 64) + cornerbit(v8, 128)
    # Most cells are all on one side, and those have no faces to decide.
    (cube == Int32(0) || cube == Int32(255)) && return cube
    joined = facebit(v1, v3, v7, v5, 1) + facebit(v2, v4, v8, v6, 2) + facebit(v1, v2, v6, v5, 4) +
             facebit(v3, v4, v8, v7, 8) + facebit(v1, v2, v4, v3, 16) + facebit(v5, v6, v8, v7, 32)
    return cube + Int32(256) * joined
end

"""How many grid edges from point `p` at `(i, j, k)` cross the level: the vertices `p` owns."""
@inline function vertexcount(field, p, i, j, k, nx, ny, nz, lv)
    cx, cy, cz = crossings(field, p, i, j, k, nx, ny, nz, lv)
    return Int32(cx) + Int32(cy) + Int32(cz)
end

"""How many triangles the cell whose low corner is `p` has. A point on the grid's
far side is no cell's low corner and has none."""
@inline function trianglecount(table, field, p, i, j, k, nx, ny, nz, lv)
    (i < nx && j < ny && k < nz) || return Int32(0)
    return @inbounds table[1, cellkey(field, p, nx, ny, lv) + Int32(1)]
end

"""The vertex on the edge from grid point `q` at `(i, j, k)` along axis `a`, one-based,
where `vbase[q]` vertices come before `q`'s."""
@inline function vertexindex(vbase, field, q, i, j, k, a, nx, ny, nz, lv)
    cx, cy, _ = crossings(field, q, i, j, k, nx, ny, nz, lv)
    rank = a == Int32(1) ? Int32(0) : a == Int32(2) ? Int32(cx) : Int32(cx) + Int32(cy)
    return @inbounds vbase[q] + rank + Int32(1)
end

"""The field's difference across grid point `p`, at index `i` of `n` along an axis
whose neighbours are `stride` apart: central, and one-sided on the grid's faces."""
@inline function difference(field, p, i, n, stride)
    hi = i < n ? p + stride : p
    lo = i > Int32(1) ? p - stride : p
    span = Float32(Int32(i < n) + Int32(i > Int32(1)))
    return (Float32(@inbounds field[hi]) - Float32(@inbounds field[lo])) / span
end

"""The field's gradient at grid point `p` at `(i, j, k)`, per grid step."""
@inline gradient(field, p, i, j, k, nx, ny, nz) =
    Vec3f(difference(field, p, i, nx, Int32(1)), difference(field, p, j, ny, nx),
          difference(field, p, k, nz, nx * ny))

"""
Write the vertex on the edge from `p` along `a` as element `j` of `V` and `N`:
skimage's linear root along the edge at `origin + spacing * x` for `x` its
zero-based grid position, and the normal, the field's gradient interpolated
along the edge, pointing toward decreasing values as the faces do.
"""
@inline function edgepoint!(V, N, j, field, p, i, jj, k, a, nx, ny, nz, lv, origin, spacing)
    q = a == Int32(1) ? p + Int32(1) : a == Int32(2) ? p + nx : p + nx * ny
    va, vb = fieldat(field, p, lv), fieldat(field, q, lv)
    t = va / (va - vb)
    x = Vec3f(Float32(i - Int32(1)) + (a == Int32(1) ? t : 0f0),
              Float32(jj - Int32(1)) + (a == Int32(2) ? t : 0f0),
              Float32(k - Int32(1)) + (a == Int32(3) ? t : 0f0))
    g0 = gradient(field, p, i, jj, k, nx, ny, nz)
    g1 = gradient(field, q, i + Int32(a == Int32(1)), jj + Int32(a == Int32(2)), k + Int32(a == Int32(3)), nx, ny, nz)
    # Per grid step to per unit of the output's space.
    g = ((1f0 - t) .* g0 .+ t .* g1) ./ spacing
    len = sqrt(dot(g, g))
    @inbounds V[j] = Point3f(origin .+ spacing .* x)
    @inbounds N[j] = len > 0f0 ? Vec3f(-g ./ len) : Vec3f(0f0)
    return nothing
end

"""Write the vertices point `p` owns as elements `base + 1`, `base + 2`, … of `V`
and `N`, those that fit in their `cap`."""
@inline function writevertices!(V, N, cap, base, field, p, i, j, k, nx, ny, nz, lv, origin, spacing)
    cx, cy, cz = crossings(field, p, i, j, k, nx, ny, nz, lv)
    v = base
    cx && (v += Int32(1); v <= cap && edgepoint!(V, N, v, field, p, i, j, k, Int32(1), nx, ny, nz, lv, origin, spacing))
    cy && (v += Int32(1); v <= cap && edgepoint!(V, N, v, field, p, i, j, k, Int32(2), nx, ny, nz, lv, origin, spacing))
    cz && (v += Int32(1); v <= cap && edgepoint!(V, N, v, field, p, i, j, k, Int32(3), nx, ny, nz, lv, origin, spacing))
    return nothing
end

"""Write the triangles of the cell whose low corner is `p` as elements `base + 1`,
… of `F`, those that fit in its `cap`."""
@inline function writetriangles!(F, cap, base, vbase, table, field, p, i, j, k, nx, ny, nz, lv)
    col = cellkey(field, p, nx, ny, lv) + Int32(1)
    @inbounds for t in Int32(1):table[1, col]
        base + t > cap && break
        packed = table[t + Int32(1), col]
        corner(r) = begin
            ox, oy, oz, a = edgeowner((packed >> (Int32(4) * (r - Int32(1)))) & Int32(15))
            vertexindex(vbase, field, p + ox + nx * (oy + ny * oz), i + ox, j + oy, k + oz, a, nx, ny, nz, lv)
        end
        F[base + t] = GLTriangleFace(GLIndex(corner(Int32(1))), GLIndex(corner(Int32(2))), GLIndex(corner(Int32(3))))
    end
    return nothing
end

function griddims(field)
    nx, ny, nz = size(field)
    (nx > 1 && ny > 1 && nz > 1) || throw(ArgumentError("field must be at least 2 on every axis, got $(size(field))"))
    # Every vertex index is an Int32, and a grid point owns up to three.
    3 * nx * ny * nz < typemax(Int32) ||
        throw(ArgumentError("field of $(size(field)) has more edges than Int32 vertex indices reach"))
    return Int32(nx), Int32(ny), Int32(nz)
end

"""`counts` in place as their exclusive prefix sum, and the total."""
function exclusivescan!(counts::Vector{Int32})
    total = Int32(0)
    for p in eachindex(counts)
        n = counts[p]
        counts[p] = total
        total += n
    end
    return total
end

# ── the order of the output ───────────────────────────────────────────────────
#
# The device runs one workgroup per brick of `BRICK` grid points, so a workgroup
# whose brick the surface misses can skip the passes that write, and most do:
# a surface touches a few percent of the bricks. Vertices and triangles come out
# brick by brick, and within a brick in grid order. The host path walks the
# points in the same order, which is what makes the two meshes the same.

"""The workgroup size of the device passes; [`groupscan`](@ref) is written for it."""
const GROUP = 256

"""The grid points one workgroup covers, `GROUP` of them."""
const BRICK = (Int32(32), Int32(4), Int32(2))

"""How many bricks cover the grid along x and y, and in all."""
@inline function brickcounts(nx, ny, nz)
    nbx = (nx + BRICK[1] - Int32(1)) ÷ BRICK[1]
    nby = (ny + BRICK[2] - Int32(1)) ÷ BRICK[2]
    nbz = (nz + BRICK[3] - Int32(1)) ÷ BRICK[3]
    return nbx, nby, nbx * nby * nbz
end

"""Grid point `(i, j, k)` of the `l`th point, zero-based, of brick `b`; past the
grid's end in a brick that overhangs it."""
@inline function brickpoint(b, l, nbx, nby)
    b0 = b - Int32(1)
    bx, by, bz = b0 % nbx, (b0 ÷ nbx) % nby, b0 ÷ (nbx * nby)
    lx, ly, lz = l % BRICK[1], (l ÷ BRICK[1]) % BRICK[2], l ÷ (BRICK[1] * BRICK[2])
    return bx * BRICK[1] + lx + Int32(1), by * BRICK[2] + ly + Int32(1), bz * BRICK[3] + lz + Int32(1)
end

@inline pointindex(i, j, k, nx, ny) = i + nx * ((j - Int32(1)) + ny * (k - Int32(1)))

"""
    marchingcubes(field; level = 0, origin = Vec3f(0), spacing = Vec3f(1)) -> GeometryBasics.Mesh
    marchingcubes(dev, field::Mantle.Buffer{Float32,3}; level, origin, spacing) -> GeometryBasics.Mesh

The `level` iso-surface of a 3-D array of scalars, on the host, or of a device
buffer, on the device. Either way it is a `GeometryBasics.Mesh` of `Point3f`
positions, `GLTriangleFace` faces and `Vec3f` normals, so `mesh!(ax, m)` draws
it; from the device its arrays are device arrays. The two run the same per-point
code in the same order and give the same mesh. To extract from a field that
changes, an animation, set up a [`MarchingCubes`](@ref) once instead.

Grid point `(i, j, k)` is at `origin + spacing * (i - 1, j - 1, k - 1)`; the
defaults give **grid index space**, zero-based, the way
`skimage.measure.marching_cubes` returns vertices. Faces are wound to face toward
decreasing values, and the normals, the field's gradient interpolated along each
edge, point the same way.

A point exactly at the level counts as above it unless it is `-0.0`; several
edges then meet in one point and produce one vertex each, at the same position.
"""
function marchingcubes(field::AbstractArray{<:Real,3}; level::Real = 0, origin = Vec3f(0), spacing = Vec3f(1))
    nx, ny, nz = griddims(field)
    lv = Float32(level)
    o, h = Vec3f(origin), Vec3f(spacing)
    np = nx * ny * nz
    vbase = Vector{Int32}(undef, np)
    tbase = Vector{Int32}(undef, np)
    Threads.@threads for p in Int32(1):np
        i, j, k = gridindex(p, nx, ny)
        vbase[p] = vertexcount(field, p, i, j, k, nx, ny, nz, lv)
        tbase[p] = trianglecount(MCTABLE, field, p, i, j, k, nx, ny, nz, lv)
    end
    # Counts to where each point's output starts, in the device's order: the
    # bricks' totals, scanned, then each brick's points from its start.
    nbx, nby, nb = brickcounts(nx, ny, nz)
    bv = Vector{Int32}(undef, nb)
    bf = Vector{Int32}(undef, nb)
    Threads.@threads for b in Int32(1):nb
        sv = sf = Int32(0)
        for l in Int32(0):Int32(GROUP - 1)
            i, j, k = brickpoint(b, l, nbx, nby)
            (i <= nx && j <= ny && k <= nz) || continue
            p = pointindex(i, j, k, nx, ny)
            sv += vbase[p]
            sf += tbase[p]
        end
        bv[b], bf[b] = sv, sf
    end
    nv, nf = exclusivescan!(bv), exclusivescan!(bf)
    Threads.@threads for b in Int32(1):nb
        v, f = bv[b], bf[b]
        for l in Int32(0):Int32(GROUP - 1)
            i, j, k = brickpoint(b, l, nbx, nby)
            (i <= nx && j <= ny && k <= nz) || continue
            p = pointindex(i, j, k, nx, ny)
            v, vbase[p] = v + vbase[p], v
            f, tbase[p] = f + tbase[p], f
        end
    end
    V = Vector{Point3f}(undef, nv)
    N = Vector{Vec3f}(undef, nv)
    F = Vector{GLTriangleFace}(undef, nf)
    Threads.@threads for p in Int32(1):np
        i, j, k = gridindex(p, nx, ny)
        writevertices!(V, N, nv, vbase[p], field, p, i, j, k, nx, ny, nz, lv, o, h)
    end
    Threads.@threads for p in Int32(1):np
        i, j, k = gridindex(p, nx, ny)
        (i < nx && j < ny && k < nz) &&
            writetriangles!(F, nf, tbase[p], vbase, MCTABLE, field, p, i, j, k, nx, ny, nz, lv)
    end
    return GeometryBasics.Mesh(V, F; normal = N)
end

function marchingcubes(dev::Mantle.Device, field::Mantle.Buffer{Float32,3}; level::Real = 0,
                       origin = Vec3f(0), spacing = Vec3f(1))
    mc = MarchingCubes(dev, field; headroom = 1, origin, spacing)
    shared = marchingcubes!(mc; level)
    # Copies that own their memory, so the mesh outlives the workspace.
    m = GeometryBasics.Mesh(copy(coordinates(shared)), copy(faces(shared)); normal = copy(normals(shared)))
    Mantle.free!(mc)
    return m
end

# ── the device passes ─────────────────────────────────────────────────────────
#
# A grid point owns the up to three grid edges leaving it in +x, +y and +z, so
# its vertices, and is the low corner of at most one cell, so its triangles. A
# frame is eight passes, recorded once:
#
# 1. `countvertices!`: every brick's vertex count, four points per thread.
# 2. `scanbricks!`: where each brick's vertices start, within its workgroup of
#    bricks, and which bricks are candidates: those that or whose upper
#    neighbours have vertices. Only a candidate can have triangles, since a
#    crossed cell has a crossed edge, owned by one of its corners.
# 3. `scansums!`: the same across workgroups of bricks, the vertex total, and
#    the candidate count, which sizes every pass after it on the device.
# 4. `listbricks!`: the candidates, in brick order.
# 5. `candidatevertices!`: each candidate's vertices, and its triangle count.
# 6. `scantriangles!` and 7. `scantrianglesums!`: where each candidate's
#    triangles start, and the triangle total.
# 8. `candidatetriangles!`: each candidate's triangles.
#
# Passes 5 to 8 run on the few percent of bricks the surface is near, so the
# frame costs about one read of the field.

"""The inclusive sum of `x` over the workgroup up to this thread. Every thread of
the workgroup must call it."""
@inline function groupscan(x::Int32, lid::Int32)
    tmp = KI.localmemory(Int32, Val((GROUP,)), Val(1))
    @inbounds tmp[lid] = x
    KI.barrier()
    offset = Int32(1)
    while offset < Int32(GROUP)
        @inbounds v = lid > offset ? tmp[lid - offset] : Int32(0)
        KI.barrier()
        @inbounds tmp[lid] += v
        KI.barrier()
        offset *= Int32(2)
    end
    return @inbounds tmp[lid]
end

"""The sum of `x` over a workgroup of `N` threads, on its first thread. Every
thread of the workgroup must call it."""
@inline function groupsum(x::Int32, lid::Int32, ::Val{N}) where {N}
    partial = KI.localmemory(Int32, Val((N,)), Val(2))
    s = KI.sub_group_reduce_add(x)
    KI.get_sub_group_local_id() == UInt32(1) && (@inbounds partial[Int32(KI.get_sub_group_id())] = s)
    KI.barrier()
    total = Int32(0)
    if lid == Int32(1)
        for q in Int32(1):(Int32(N) ÷ Int32(KI.get_sub_group_size()))
            total += @inbounds partial[q]
        end
    end
    return total
end

"""Points per thread in [`countvertices!`](@ref), consecutive along x."""
const WINDOW = Int32(4)

"""Threads per brick in [`countvertices!`](@ref)."""
const COUNTGROUP = GROUP ÷ Int(WINDOW)

"""The level-relative field at the `WINDOW + 1` points along x from point `p` at
x index `i`, where `inside` says the row is in the grid; points past the grid's
end read nothing."""
@inline window(field, p, i, nx, lv, inside) = ntuple(Val(Int(WINDOW) + 1)) do q
    (inside && i + Int32(q - 1) <= nx) ? fieldat(field, p + Int32(q - 1), lv) : 1f0
end

"""Each brick's vertex count, as `vcount[brick]`. A thread counts `WINDOW`
consecutive points of one row of the brick, reading each value once."""
function countvertices!(vcount, field, nx, ny, nz, nbx, nby, level)
    b = Int32(KI.get_group_id().x)
    lid = Int32(KI.get_local_id().x)
    lv = @inbounds level[1]
    perrow = BRICK[1] ÷ WINDOW
    t0 = lid - Int32(1)
    i, j, k = brickpoint(b, (t0 % perrow) * WINDOW + (t0 ÷ perrow) * BRICK[1], nbx, nby)
    n = Int32(0)
    if i <= nx && j <= ny && k <= nz
        p = pointindex(i, j, k, nx, ny)
        a = window(field, p, i, nx, lv, true)
        y = window(field, p + nx, i, nx, lv, j < ny)
        z = window(field, p + nx * ny, i, nx, lv, k < nz)
        counts = ntuple(Val(Int(WINDOW))) do q
            ii = i + Int32(q - 1)
            ii <= nx || return Int32(0)
            return Int32(ii < nx && signbit(a[q]) != signbit(a[q + 1])) +
                   Int32(j < ny && signbit(a[q]) != signbit(y[q])) +
                   Int32(k < nz && signbit(a[q]) != signbit(z[q]))
        end
        n = reduce(+, counts)                       # `sum` would widen to Int64
    end
    total = groupsum(n, lid, Val(COUNTGROUP))
    lid == Int32(1) && (@inbounds vcount[b] = total)
    return nothing
end

"""Whether brick `b` can have triangles: it or one of its seven upper neighbours
has vertices."""
@inline function candidate(vcount, b, nbx, nby, nb)
    b0 = b - Int32(1)
    bx, by = b0 % nbx, (b0 ÷ nbx) % nby
    @inbounds for dz in Int32(0):Int32(1), dy in Int32(0):Int32(1), dx in Int32(0):Int32(1)
        (bx + dx < nbx && by + dy < nby) || continue
        q = b + dx + nbx * (dy + nby * dz)
        q <= nb && vcount[q] > Int32(0) && return true
    end
    return false
end

"""Where each brick's vertices start and its place among the candidates (-1 for
none), both within its workgroup of bricks, and each workgroup's totals as
`sums[:, group]`."""
function scanbricks!(voffset, cpos, sums, vcount, nbx, nby, nb)
    b = Int32(KI.get_global_id().x)
    lid = Int32(KI.get_local_id().x)
    v = c = Int32(0)
    if b <= nb
        v = @inbounds vcount[b]
        c = candidate(vcount, b, nbx, nby, nb) ? Int32(1) : Int32(0)
    end
    sv = groupscan(v, lid)
    sc = groupscan(c, lid)
    if b <= nb
        @inbounds voffset[b] = sv - v
        @inbounds cpos[b] = c == Int32(1) ? sc - c : Int32(-1)
    end
    if lid == Int32(GROUP)
        g = Int32(KI.get_group_id().x)
        @inbounds sums[1, g] = sv
        @inbounds sums[2, g] = sc
    end
    return nothing
end

"""One workgroup: where each workgroup of bricks' vertices and candidates start,
the vertex total and the candidate count, and the candidate count again as the
thread counts the candidate passes are sized by."""
function scansums!(sumoffsets, totals, candidates, candidatethreads, sums, ns)
    lid = Int32(KI.get_local_id().x)
    run = (ns + Int32(GROUP) - Int32(1)) ÷ Int32(GROUP)
    lo = (lid - Int32(1)) * run + Int32(1)
    hi = min(lid * run, ns)
    sv = sc = Int32(0)
    @inbounds for s in lo:hi
        sv += sums[1, s]
        sc += sums[2, s]
    end
    v = groupscan(sv, lid)
    c = groupscan(sc, lid)
    if lid == Int32(GROUP)
        @inbounds totals[1] = v
        @inbounds candidates[1] = c
        @inbounds candidatethreads[1] = c * Int32(GROUP)
    end
    v -= sv
    c -= sc
    @inbounds for s in lo:hi
        sumoffsets[1, s] = v
        sumoffsets[2, s] = c
        v += sums[1, s]
        c += sums[2, s]
    end
    return nothing
end

"""Every brick's vertex start made global, and the candidates listed in brick order."""
function listbricks!(list, voffset, cpos, sumoffsets, nb)
    b = Int32(KI.get_global_id().x)
    b <= nb || return nothing
    g = (b - Int32(1)) ÷ Int32(GROUP) + Int32(1)
    @inbounds begin
        voffset[b] += sumoffsets[1, g]
        cpos[b] >= Int32(0) && (list[cpos[b] + sumoffsets[2, g] + Int32(1)] = b)
    end
    return nothing
end

"""Each candidate's vertices, `vbase` for the points that have any, and each
candidate's triangle count as `tcount[candidate]`."""
function candidatevertices!(V, N, cap, vbase, tcount, list, candidates, voffset, table, field,
                            nx, ny, nz, nbx, nby, level, origin, spacing)
    c = Int32(KI.get_group_id().x)
    # The same for the whole workgroup, so all of it returns before any barrier.
    @inbounds c > candidates[1] && return nothing
    lid = Int32(KI.get_local_id().x)
    lv = @inbounds level[1]
    b = @inbounds list[c]
    i, j, k = brickpoint(b, lid - Int32(1), nbx, nby)
    p = nv = nt = Int32(0)
    if i <= nx && j <= ny && k <= nz
        p = pointindex(i, j, k, nx, ny)
        nv = vertexcount(field, p, i, j, k, nx, ny, nz, lv)
        nt = trianglecount(table, field, p, i, j, k, nx, ny, nz, lv)
    end
    s = groupscan(nv, lid)
    if nv > Int32(0)
        base = @inbounds(voffset[b]) + s - nv
        @inbounds vbase[p] = base
        writevertices!(V, N, cap, base, field, p, i, j, k, nx, ny, nz, lv, origin, spacing)
    end
    total = groupsum(nt, lid, Val(GROUP))
    lid == Int32(1) && (@inbounds tcount[c] = total)
    return nothing
end

"""Where each candidate's triangles start within its workgroup of candidates, and
each workgroup's total as `tsums[group]`."""
function scantriangles!(toffset, tsums, tcount, candidates)
    c = Int32(KI.get_global_id().x)
    lid = Int32(KI.get_local_id().x)
    n = @inbounds candidates[1]
    t = c <= n ? @inbounds(tcount[c]) : Int32(0)
    s = groupscan(t, lid)
    c <= n && (@inbounds toffset[c] = s - t)
    lid == Int32(GROUP) && (@inbounds tsums[Int32(KI.get_group_id().x)] = s)
    return nothing
end

"""One workgroup: where each workgroup of candidates' triangles start, and the
triangle total."""
function scantrianglesums!(tsumoffsets, totals, tsums, candidates)
    lid = Int32(KI.get_local_id().x)
    ns = (@inbounds(candidates[1]) + Int32(GROUP) - Int32(1)) ÷ Int32(GROUP)
    run = (ns + Int32(GROUP) - Int32(1)) ÷ Int32(GROUP)
    lo = (lid - Int32(1)) * run + Int32(1)
    hi = min(lid * run, ns)
    st = Int32(0)
    @inbounds for s in lo:hi
        st += tsums[s]
    end
    t = groupscan(st, lid)
    lid == Int32(GROUP) && (@inbounds totals[2] = t)
    t -= st
    @inbounds for s in lo:hi
        tsumoffsets[s] = t
        t += tsums[s]
    end
    return nothing
end

"""Each candidate's triangles."""
function candidatetriangles!(F, cap, vbase, list, candidates, tcount, toffset, tsumoffsets, table, field,
                             nx, ny, nz, nbx, nby, level)
    c = Int32(KI.get_group_id().x)
    @inbounds (c > candidates[1] || tcount[c] == Int32(0)) && return nothing
    lid = Int32(KI.get_local_id().x)
    lv = @inbounds level[1]
    b = @inbounds list[c]
    i, j, k = brickpoint(b, lid - Int32(1), nbx, nby)
    p = n = Int32(0)
    if i <= nx && j <= ny && k <= nz
        p = pointindex(i, j, k, nx, ny)
        n = trianglecount(table, field, p, i, j, k, nx, ny, nz, lv)
    end
    s = groupscan(n, lid)
    if n > Int32(0)
        base = @inbounds(toffset[c] + tsumoffsets[(c - Int32(1)) ÷ Int32(GROUP) + Int32(1)]) + s - n
        writetriangles!(F, cap, base, vbase, table, field, p, i, j, k, nx, ny, nz, lv)
    end
    return nothing
end

"""
    MarchingCubes(dev, field; origin = Vec3f(0), spacing = Vec3f(1), vertices = 0, triangles = 0, headroom = 1.5)

Marching cubes over the device buffer `field`, set up once to run every time the
field changes. [`marchingcubes!`](@ref) extracts the current surface as a
`GeometryBasics.Mesh` of device arrays; `origin` and `spacing` place it as
[`marchingcubes`](@ref) describes.

    mc = MarchingCubes(dev, field)
    for frame in frames
        simulate!(field, frame)                    # writes the field on the device
        m = marchingcubes!(mc; level = 0)
        draw(m)
    end

The scratch buffers and the recorded plan are made here, and the output buffers
start at room for `vertices` and `triangles`. A frame is one submission and one
read of the two counts. A frame that needs more room grows the output to
`headroom` times what it needed, records the plan again and runs it a second
time; every other frame allocates and records nothing. `field` is bound by
address, so it has to stay the same buffer; its contents are what changes.

A grid of `n` points keeps `4n` bytes of scratch besides the field. Free it all
with `Mantle.free!(mc)`.
"""
mutable struct MarchingCubes
    dev::Mantle.Device
    field::Mantle.Buffer{Float32,3}
    dims::NTuple{3,Int32}
    level::Mantle.GPURef{Float32}
    table::Mantle.Buffer{Int32,2}
    # per brick
    vcount::Mantle.Buffer{Int32,1}
    voffset::Mantle.Buffer{Int32,1}
    cpos::Mantle.Buffer{Int32,1}
    sums::Mantle.Buffer{Int32,2}
    sumoffsets::Mantle.Buffer{Int32,2}
    # per candidate brick
    list::Mantle.Buffer{Int32,1}
    tcount::Mantle.Buffer{Int32,1}
    toffset::Mantle.Buffer{Int32,1}
    tsums::Mantle.Buffer{Int32,1}
    tsumoffsets::Mantle.Buffer{Int32,1}
    candidates::Mantle.Buffer{Int32,1}
    candidatethreads::Mantle.Buffer{Int32,1}
    totals::Mantle.Buffer{Int32,1}
    # per grid point
    vbase::Mantle.Buffer{Int32,1}
    # the mesh, with room to spare
    V::Mantle.Buffer{Point3f,1}
    N::Mantle.Buffer{Vec3f,1}
    F::Mantle.Buffer{GLTriangleFace,1}
    origin::Vec3f
    spacing::Vec3f
    headroom::Float64
    plan::Mantle.Plan
    function MarchingCubes(dev::Mantle.Device, field::Mantle.Buffer{Float32,3}; origin = Vec3f(0),
                           spacing = Vec3f(1), vertices::Integer = 0, triangles::Integer = 0, headroom::Real = 1.5)
        dims = griddims(field)
        _, _, nb32 = brickcounts(dims...)
        nb = Int(nb32)
        ns = cld(nb, GROUP)
        buf(T, dims...) = Mantle.Buffer(dev, T, dims)
        mc = new(dev, field, dims, Mantle.GPURef(dev, 0f0), Mantle.Buffer(dev, MCTABLE),
                 buf(Int32, nb), buf(Int32, nb), buf(Int32, nb), buf(Int32, 2, ns), buf(Int32, 2, ns),
                 buf(Int32, nb), buf(Int32, nb), buf(Int32, nb), buf(Int32, ns), buf(Int32, ns),
                 buf(Int32, 1), buf(Int32, 1), buf(Int32, 2), buf(Int32, prod(Int.(dims))),
                 buf(Point3f, max(Int(vertices), 1)), buf(Vec3f, max(Int(vertices), 1)),
                 buf(GLTriangleFace, max(Int(triangles), 1)), Vec3f(origin), Vec3f(spacing), Float64(headroom))
        mc.plan = frameplan(mc)
        return mc
    end
end

"""The passes of one frame, as `(kernel, arguments, threads, group, name)`."""
function passes(mc::MarchingCubes)
    nx, ny, nz = mc.dims
    nbx, nby, nb = brickcounts(nx, ny, nz)
    capv, capf = Int32(Mantle.capacity(mc.V)), Int32(Mantle.capacity(mc.F))
    perbrick = Mantle.DeviceRange(mc.candidatethreads; max = Int(nb) * GROUP)
    percandidate = Mantle.DeviceRange(mc.candidates; max = Int(nb))
    return ((countvertices!, (mc.vcount, mc.field, nx, ny, nz, nbx, nby, mc.level), Int(nb) * COUNTGROUP,
             COUNTGROUP, "marchingcubes/count vertices"),
            (scanbricks!, (mc.voffset, mc.cpos, mc.sums, mc.vcount, nbx, nby, nb), Int(nb), GROUP,
             "marchingcubes/scan bricks"),
            (scansums!, (mc.sumoffsets, mc.totals, mc.candidates, mc.candidatethreads, mc.sums,
                         Int32(size(mc.sums, 2))), GROUP, GROUP, "marchingcubes/scan brick sums"),
            (listbricks!, (mc.list, mc.voffset, mc.cpos, mc.sumoffsets, nb), Int(nb), GROUP,
             "marchingcubes/list candidates"),
            (candidatevertices!, (mc.V, mc.N, capv, mc.vbase, mc.tcount, mc.list, mc.candidates, mc.voffset, mc.table,
                                  mc.field, nx, ny, nz, nbx, nby, mc.level, mc.origin, mc.spacing), perbrick, GROUP,
             "marchingcubes/vertices"),
            (scantriangles!, (mc.toffset, mc.tsums, mc.tcount, mc.candidates), percandidate, GROUP,
             "marchingcubes/scan triangles"),
            (scantrianglesums!, (mc.tsumoffsets, mc.totals, mc.tsums, mc.candidates), GROUP, GROUP,
             "marchingcubes/scan triangle sums"),
            (candidatetriangles!, (mc.F, capf, mc.vbase, mc.list, mc.candidates, mc.tcount, mc.toffset,
                                   mc.tsumoffsets, mc.table, mc.field, nx, ny, nz, nbx, nby, mc.level), perbrick,
             GROUP, "marchingcubes/triangles"))
end

function frameplan(mc::MarchingCubes)
    g = Mantle.Graph(mc.dev)
    for (kernel, args, threads, group, name) in passes(mc)
        Mantle.dispatch!(g, kernel, args, threads; group, name)
    end
    return Mantle.record!(Mantle.Plan(g))
end

"""Room for `nv` vertices and `nf` triangles: grow the output and record the plan
again when there is not. Returns whether it grew."""
function reserve!(mc::MarchingCubes, nv::Int, nf::Int)
    (nv <= Mantle.capacity(mc.V) && nf <= Mantle.capacity(mc.F)) && return false
    grown(n) = max(n, ceil(Int, mc.headroom * n))
    Mantle.free!(mc.plan)
    if nv > Mantle.capacity(mc.V)
        foreach(Mantle.free!, (mc.V, mc.N))
        mc.V = Mantle.Buffer(mc.dev, Point3f, grown(nv))
        mc.N = Mantle.Buffer(mc.dev, Vec3f, grown(nv))
    end
    if nf > Mantle.capacity(mc.F)
        Mantle.free!(mc.F)
        mc.F = Mantle.Buffer(mc.dev, GLTriangleFace, grown(nf))
    end
    mc.plan = frameplan(mc)
    return true
end

"""
    marchingcubes!(mc::MarchingCubes; level = 0) -> GeometryBasics.Mesh

The `level` iso-surface of `mc`'s field as it is now, as [`marchingcubes`](@ref)
describes it. The mesh's arrays are device arrays over `mc`'s own buffers, no
copy: they hold this frame's surface until the next call, and are only valid
while `mc` is alive.
"""
function marchingcubes!(mc::MarchingCubes; level::Real = 0)
    mc.level[] = Float32(level)
    Mantle.run!(mc.plan)
    Mantle.waitfor!(mc.plan)
    nv, nf = Int.(Array(mc.totals))
    if reserve!(mc, nv, nf)
        Mantle.run!(mc.plan)
        Mantle.waitfor!(mc.plan)
    end
    return GeometryBasics.Mesh(view(Mantle.storage(mc.V), 1:nv), view(Mantle.storage(mc.F), 1:nf);
                               normal = view(Mantle.storage(mc.N), 1:nv))
end

function Mantle.free!(mc::MarchingCubes)
    foreach(Mantle.free!, (mc.plan, mc.level, mc.table, mc.vcount, mc.voffset, mc.cpos, mc.sums, mc.sumoffsets,
                           mc.list, mc.tcount, mc.toffset, mc.tsums, mc.tsumoffsets, mc.candidates,
                           mc.candidatethreads, mc.totals, mc.vbase, mc.V, mc.N, mc.F))
    return nothing
end
