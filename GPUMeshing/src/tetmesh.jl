"""
Tetrahedral meshes on the device: their faces and edges each once, which of those bound the
mesh, and which way a boundary face points. `(3, nv)` vertices and `(4, nt)` one-based Int32
tetrahedra, as `Mantle.Buffer`s.
"""

"""A vertex of a `(3, n)` buffer of any float type, as a `Vec` of that type."""
@inline point(V, v) = @inbounds Vec{3,eltype(V)}(V[1, v], V[2, v], V[3, v])

"""
    TetFaces(dev, T, nt)

Every face of every tetrahedron as its sorted vertex triple, sorted, and where each came
from: `keys[i]` is face `(perm[i] - 1) % 4 + 1` of tetrahedron [`facetet`](@ref)`(perm, i)`.
The runs of equal keys are the mesh's `nf` distinct faces, face `r` being
`keys[starts[r]:starts[r + 1] - 1]`: a run of one is a face on the boundary, a run of two a
face between two tetrahedra.
"""
struct TetFaces
    keys::Mantle.Buffer{NTuple{3,Int32},1}
    perm::Mantle.Buffer{Int32,1}
    starts::Mantle.Buffer{Int32,1}
    nf::Int
end

function TetFaces(dev::Mantle.Device, T::Mantle.Buffer, nt::Int)
    n = 4nt
    keys = Mantle.Buffer(dev, NTuple{3,Int32}, max(n, 1))
    perm = Mantle.Buffer(dev, Int32, max(n, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, expandtetfaces!, (keys, T, nt), nt; name = "tetfaces/expand")
    Mantle.dispatch!(g, iota!, (perm, n, 1), n; name = "tetfaces/iota")
    runonce!(g)
    n > 1 && AK.sort_by_key!(view(Mantle.storage(keys), 1:n), view(Mantle.storage(perm), 1:n))
    starts, nf = runs(dev, keys, n)
    return TetFaces(keys, perm, starts, nf)
end

function Mantle.free!(f::TetFaces)
    Mantle.free!(f.keys)
    Mantle.free!(f.perm)
    Mantle.free!(f.starts)
    return nothing
end

"""The tetrahedron sorted position `i` of `faces.keys` belongs to."""
@inline facetet(perm, i) = @inbounds (perm[i] - Int32(1)) ÷ Int32(4) + Int32(1)

function expandtetfaces!(keys, T, nt)
    t = Int32(KI.get_global_id().x)
    t <= nt || return nothing
    @inbounds begin
        a, b, c, d = T[1, t], T[2, t], T[3, t], T[4, t]
        base = Int32(4) * (t - Int32(1))
        keys[base + Int32(1)] = sorted3(a, b, c)
        keys[base + Int32(2)] = sorted3(a, b, d)
        keys[base + Int32(3)] = sorted3(a, c, d)
        keys[base + Int32(4)] = sorted3(b, c, d)
    end
    return nothing
end

"""
    TetBoundary

What [`tetboundary`](@ref) finds: the faces `F` `(3, nf)` of the tetrahedra, each once as its
sorted vertex triple and in sorted order, how many tetrahedra each is on (`facecount`, 1 on
the boundary) and one of them (`parent`, the least); `tetflag`, set on the tetrahedra with a
boundary face; the edges `E` `(2, ne)` of the faces, each once and sorted, with `edgeflag` set
on those of a boundary face; and `vertexflag`, set on the vertices of a boundary face.
"""
struct TetBoundary
    F::Mantle.Buffer{Int32,2}
    facecount::Mantle.Buffer{Int32,1}
    parent::Mantle.Buffer{Int32,1}
    tetflag::Mantle.Buffer{Int32,1}
    E::Mantle.Buffer{Int32,2}
    edgeflag::Mantle.Buffer{Int32,1}
    vertexflag::Mantle.Buffer{Int32,1}
    nf::Int
    ne::Int
end

function Mantle.free!(b::TetBoundary)
    foreach(Mantle.free!, (b.F, b.facecount, b.parent, b.tetflag, b.E, b.edgeflag, b.vertexflag))
    return nothing
end

"""
    tetboundary(dev, T, nt, nv) -> TetBoundary

The faces and edges of the tetrahedra `T` `(4, nt)` over `nv` vertices and which of them,
their tetrahedra and vertices lie on the boundary: a face bounds the mesh when one
tetrahedron has it, an edge or vertex when a boundary face has it.
"""
function tetboundary(dev::Mantle.Device, T::Mantle.Buffer, nt::Int, nv::Int)
    faces = TetFaces(dev, T, nt)
    nf = faces.nf
    F = Mantle.Buffer(dev, Int32, (3, max(nf, 1)))
    facecount = Mantle.Buffer(dev, Int32, max(nf, 1))
    parent = Mantle.Buffer(dev, Int32, max(nf, 1))
    tetflag = fill!(Mantle.Buffer(dev, Int32, max(nt, 1)), 0)
    vertexflag = fill!(Mantle.Buffer(dev, Int32, max(nv, 1)), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, tetfacerows!, (F, facecount, parent, tetflag, vertexflag, faces.keys, faces.starts, faces.perm, nf),
                     nf; name = "tetboundary/faces")
    runonce!(g)
    Mantle.free!(faces)
    edges = Edges(dev, F, nf)
    E, ukeys = edgelist(dev, edges)
    ne = edges.ne
    Mantle.free!(edges)
    edgeflag = fill!(Mantle.Buffer(dev, Int32, max(ne, 1)), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, boundaryedges!, (edgeflag, ukeys, ne, F, facecount, nf), nf; name = "tetboundary/edges")
    runonce!(g)
    Mantle.free!(ukeys)
    return TetBoundary(F, facecount, parent, tetflag, E, edgeflag, vertexflag, nf, ne)
end

function tetfacerows!(F, facecount, parent, tetflag, vertexflag, keys, starts, perm, nf)
    r = Int32(KI.get_global_id().x)
    r <= nf || return nothing
    @inbounds begin
        s, e = starts[r], starts[r + Int32(1)]
        a, b, c = keys[s]
        F[1, r], F[2, r], F[3, r] = a, b, c
        facecount[r] = e - s
        t = typemax(Int32)
        for i in s:(e - Int32(1))
            t = min(t, facetet(perm, i))
        end
        parent[r] = t
        if e - s == Int32(1)
            tetflag[t] = Int32(1)
            vertexflag[a] = Int32(1)
            vertexflag[b] = Int32(1)
            vertexflag[c] = Int32(1)
        end
    end
    return nothing
end

function boundaryedges!(edgeflag, ukeys, ne, F, facecount, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds if facecount[f] == Int32(1)
        a, b, c = F[1, f], F[2, f], F[3, f]
        edgeflag[edgeindex(ukeys, ne, a, b)] = Int32(1)
        edgeflag[edgeindex(ukeys, ne, b, c)] = Int32(1)
        edgeflag[edgeindex(ukeys, ne, a, c)] = Int32(1)
    end
    return nothing
end

"""
    faceorientation(dev, V, T, F, parent, nf) -> orient

Which way each face `F[:, f]` (as its vertex order winds it) points relative to its
tetrahedron `parent[f]`: 1 away from the tetrahedron's fourth vertex (outward), -1 towards
it, 0 when that vertex lies in the face's plane. In the vertices' own precision.
"""
function faceorientation(dev::Mantle.Device, V::Mantle.Buffer, T::Mantle.Buffer, F::Mantle.Buffer, parent::Mantle.Buffer,
                         nf::Int)
    orient = Mantle.Buffer(dev, Int32, max(nf, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, faceorientation!, (orient, V, T, F, parent, nf), nf; name = "tetboundary/orientation")
    runonce!(g)
    return orient
end

function faceorientation!(orient, V, T, F, parent, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        t = parent[f]
        a, b, c = F[1, f], F[2, f], F[3, f]
        o = T[1, t]
        for k in Int32(2):Int32(4)
            (o == a || o == b || o == c) || break
            o = T[k, t]
        end
        A = point(V, a)
        n = cross(point(V, b) - A, point(V, c) - A)
        d = dot(n, point(V, o) - A)
        orient[f] = d < 0 ? Int32(1) : d > 0 ? Int32(-1) : Int32(0)
    end
    return nothing
end
