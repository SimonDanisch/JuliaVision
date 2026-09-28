"""
    dualgridmesh(level, out; margin = 0.5) -> (vertices, faces)

`FlexiDualGridVaeDecoder`'s mesh: `flexible_dual_grid_to_mesh` (o-voxel) on the
shape decoder's output `out` `(7, N)` at the voxels of `level`.

Every voxel carries one dual vertex, `(1 + 2 margin) sigmoid(out[1:3]) - margin`
inside it; an axis whose intersection logit (`out[4:6]`) is positive has a quad
through the four voxels around that edge, when all four exist; the quad is split
along the diagonal whose `softplus(out[7])` weights multiply to more. Quads in
voxel order, axis by axis, as upstream's boolean mask walks `(N, 3)`.
"""
function dualgridmesh(level::Level, out::AbstractMatrix{Float32}; margin = 0.5f0)
    n = length(level)
    res = level.res
    vs = 1f0 / res
    vertices = Matrix{Float32}(undef, 3, n)
    for j in 1:n, a in 1:3
        d = (1 + 2margin) / (1 + exp(-out[a, j])) - margin
        vertices[a, j] = (level.coords[a, j] + d) * vs - 0.5f0
    end
    lerp = [log1p(exp(out[7, j])) for j in 1:n]
    key(x, y, z) = Int64(x) + res * (Int64(y) + res * Int64(z))
    index = Dict{Int64,Int32}()
    sizehint!(index, n)
    for j in 1:n
        index[key(level.coords[1, j], level.coords[2, j], level.coords[3, j])] = j
    end
    offsets = (((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),   # x
               ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),   # y
               ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)))   # z
    faces = Int32[]
    q = zeros(Int32, 4)
    for j in 1:n, a in 1:3
        out[3 + a, j] > 0 || continue
        found = true
        for (k, (dx, dy, dz)) in enumerate(offsets[a])
            i = get(index, key(level.coords[1, j] + dx, level.coords[2, j] + dy,
                               level.coords[3, j] + dz), Int32(0))
            i == 0 && (found = false; break)
            q[k] = i
        end
        found || continue
        if lerp[q[1]] * lerp[q[3]] > lerp[q[2]] * lerp[q[4]]
            append!(faces, (q[1], q[2], q[3], q[1], q[3], q[4]))
        else
            append!(faces, (q[1], q[2], q[4], q[4], q[2], q[3]))
        end
    end
    return vertices, reshape(faces, 3, :)
end

"""
    orientfaces(vertices, faces) -> faces

Consistent winding, outward: cumesh's `unify_face_orientations`. The dual grid
emits every quad in a fixed per-axis order, so neighbouring triangles disagree
about which side is out. Across every edge two faces share, they have to walk
the edge in opposite directions; a flood fill over each connected piece flips the
ones that do not, and a piece whose signed volume comes out negative is turned
over whole. Edges of more than two faces are not walked: a non-manifold edge
does not say which way its faces should turn.
"""
function orientfaces(vertices::AbstractMatrix{Float32}, faces::AbstractMatrix{<:Integer})
    F = Matrix{Int32}(faces)
    nf = size(F, 2)
    # Every undirected edge's faces.
    owners = Dict{Tuple{Int32,Int32},Vector{Int32}}()
    edgeof(a, b) = a < b ? (a, b) : (b, a)
    for f in 1:nf, k in 1:3
        push!(get!(Vector{Int32}, owners, edgeof(F[k, f], F[k % 3 + 1, f])), f)
    end
    # Whether face `f` walks the edge `a -> b`.
    walks(f, a, b) = any(k -> F[k, f] == a && F[k % 3 + 1, f] == b, 1:3)
    function flip!(f)
        F[2, f], F[3, f] = F[3, f], F[2, f]
        return nothing
    end
    seen = falses(nf)
    queue = Int32[]
    piece = Int32[]
    for start in 1:nf
        seen[start] && continue
        seen[start] = true
        empty!(piece)
        push!(queue, start)
        while !isempty(queue)
            f = popfirst!(queue)
            push!(piece, f)
            for k in 1:3
                a, b = F[k, f], F[k % 3 + 1, f]
                fs = owners[edgeof(a, b)]
                length(fs) == 2 || continue
                g = fs[1] == f ? fs[2] : fs[1]
                seen[g] && continue
                walks(g, a, b) && flip!(g)
                seen[g] = true
                push!(queue, g)
            end
        end
        volume = 0.0
        for f in piece
            a, b, c = (Vec3(Float64.(view(vertices, :, F[k, f]))...) for k in 1:3)
            volume += dot(a, cross(b, c))
        end
        volume < 0 && foreach(flip!, piece)
    end
    return F
end
