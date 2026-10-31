"""
The volume moments of a solid, `∫ 1`, `∫ x` and `∫ x xᵀ` over it, from its closed surface (by
the divergence theorem, face by face) or its tetrahedra: mass, centre of mass and inertia for
a unit density. In Float64 whatever the vertices' precision, per element on the device and
summed there.
"""

"""
    VolumeMoments

`volume` (`∫ 1`), `first` (`∫ x`) and `second` (`∫ x xᵀ`) of a solid, in Float64.
"""
struct VolumeMoments
    volume::Float64
    first::Vec{3,Float64}
    second::Mat{3,3,Float64,9}
end

"""The moments from the ten sums `∫1, ∫x, ∫y, ∫z, ∫xx, ∫yy, ∫zz, ∫xy, ∫xz, ∫yz`."""
function VolumeMoments(s::AbstractVector{Float64})
    xy, xz, yz = s[8], s[9], s[10]
    return VolumeMoments(s[1], Vec{3,Float64}(s[2], s[3], s[4]),
                         Mat{3,3,Float64,9}(s[5], xy, xz, xy, s[6], yz, xz, yz, s[7]))
end

@inline point64(V, v) = @inbounds Vec{3,Float64}(V[1, v], V[2, v], V[3, v])

"""
    volumemoments(dev, V, F; orient = nothing) -> VolumeMoments

The moments of the solid the closed triangle mesh `V` `(3, nv)`, `F` `(3, nf)` bounds, its
faces pointing outward; a face whose `orient` entry (Int32, `nf`) is negative counts turned
over. The per-face terms are libuipc's `compute_trimesh_dyadic_mass` (its
`affine_body_quantity` notebook): with `N = (p1 - p0) × (p2 - p0)`, `∫ 1 = Σ p0·N / 6`,
`∫ x_a = Σ N_a / 2 · (Σ_{i ≤ j} p_ia p_ja) / 12`, and the second moments from the cubic
integrals of each face.
"""
function volumemoments(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer;
                       orient::Union{Nothing,Mantle.Buffer{Int32,1}} = nothing)
    nf = size(F, 2)
    terms = fill!(Mantle.Buffer(dev, Float64, (10, max(nf, 1))), 0.0)
    g = Mantle.Graph(dev)
    o = orient === nothing ? fill!(Mantle.Buffer(dev, Int32, 1), 1) : orient
    Mantle.dispatch!(g, surfacemoments!, (terms, V, F, o, orient !== nothing, nf), nf; name = "moments/faces")
    runonce!(g)
    s = vec(Array(sum(Mantle.storage(terms); dims = 2)))
    Mantle.free!(terms)
    orient === nothing && Mantle.free!(o)
    return VolumeMoments(s)
end

function surfacemoments!(terms, V, F, orient, oriented, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        p0, p1, p2 = point64(V, F[1, f]), point64(V, F[2, f]), point64(V, F[3, f])
        N = cross(p1 - p0, p2 - p0)
        (oriented && orient[f] < 0) && (N = -N)
        terms[Int32(1), f] = dot(p0, N) / 6.0
        for a in Int32(1):Int32(3)
            x0, x1, x2 = p0[a], p1[a], p2[a]
            terms[Int32(1) + a, f] = N[a] / 2 * (x0 * x0 + x0 * x1 + x0 * x2 + x1 * x1 + x1 * x2 + x2 * x2) / 12
            cubic = x0 * x0 * x0 + x0 * x0 * x1 + x0 * x0 * x2 + x0 * x1 * x1 + x0 * x1 * x2 + x0 * x2 * x2 +
                    x1 * x1 * x1 + x1 * x1 * x2 + x1 * x2 * x2 + x2 * x2 * x2
            terms[Int32(4) + a, f] = N[a] / 3 * cubic / 20
        end
        terms[Int32(8), f] = mixedmoment(N, p0, p1, p2, Int32(1), Int32(2))
        terms[Int32(9), f] = mixedmoment(N, p0, p1, p2, Int32(1), Int32(3))
        terms[Int32(10), f] = mixedmoment(N, p0, p1, p2, Int32(2), Int32(3))
    end
    return nothing
end

"""`∫ x_a x_b` of one face's prism (a ≠ b), libuipc's off-diagonal term."""
@inline function mixedmoment(N, p0, p1, p2, a, b)
    p0a, p1a, p2a = p0[a], p1[a], p2[a]
    p0b, p1b, p2b = p0[b], p1[b], p2[b]
    p0a2, p1a2, p2a2 = p0a * p0a, p1a * p1a, p2a * p2a
    v = p0a2 * p0b / 20 + p0a2 * p1b / 60 + p0a2 * p2b / 60 + p0a * p0b * p1a / 30 + p0a * p0b * p2a / 30 +
        p0a * p1a * p1b / 30 + p0a * p1a * p2b / 60 + p0a * p1b * p2a / 60 + p0a * p2a * p2b / 30 +
        p0b * p1a2 / 60 + p0b * p1a * p2a / 60 + p0b * p2a2 / 60 + p1a2 * p1b / 20 + p1a2 * p2b / 60 +
        p1a * p1b * p2a / 30 + p1a * p2a * p2b / 30 + p1b * p2a2 / 60 + p2a2 * p2b / 20
    return N[a] / 2 * v
end

"""
    tetvolumemoments(dev, V, T) -> VolumeMoments

The moments of the solid the tetrahedra `T` `(4, nt)` fill, libuipc's
`compute_tetmesh_dyadic_mass`: with `D = (p1 - p0)·((p2 - p0) × (p3 - p0))`, `∫ 1 = Σ D / 6`,
`∫ x_a = Σ D (Σ_i p_ia) / 24`, `∫ x_a x_b = Σ D (Σ_i p_ia p_ib / 60 + Σ_{i ≠ j} p_ia p_jb / 120)`.
"""
function tetvolumemoments(dev::Mantle.Device, V::Mantle.Buffer, T::Mantle.Buffer)
    nt = size(T, 2)
    terms = fill!(Mantle.Buffer(dev, Float64, (10, max(nt, 1))), 0.0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, tetmoments!, (terms, V, T, nt), nt; name = "moments/tetrahedra")
    runonce!(g)
    s = vec(Array(sum(Mantle.storage(terms); dims = 2)))
    Mantle.free!(terms)
    return VolumeMoments(s)
end

function tetmoments!(terms, V, T, nt)
    t = Int32(KI.get_global_id().x)
    t <= nt || return nothing
    @inbounds begin
        p0, p1, p2, p3 = point64(V, T[1, t]), point64(V, T[2, t]), point64(V, T[3, t]), point64(V, T[4, t])
        D = dot(p1 - p0, cross(p2 - p0, p3 - p0))
        terms[Int32(1), t] = D / 6
        for a in Int32(1):Int32(3)
            terms[Int32(1) + a, t] = D * (p0[a] + p1[a] + p2[a] + p3[a]) / 24
            terms[Int32(4) + a, t] = D * tetsecond(p0, p1, p2, p3, a, a)
        end
        terms[Int32(8), t] = D * tetsecond(p0, p1, p2, p3, Int32(1), Int32(2))
        terms[Int32(9), t] = D * tetsecond(p0, p1, p2, p3, Int32(1), Int32(3))
        terms[Int32(10), t] = D * tetsecond(p0, p1, p2, p3, Int32(2), Int32(3))
    end
    return nothing
end

"""`Σ_i p_ia p_ib / 60 + Σ_{i ≠ j} p_ia p_jb / 120` over a tetrahedron's corners."""
@inline function tetsecond(p0, p1, p2, p3, a, b)
    v = (p0[a] * p0[b] + p1[a] * p1[b] + p2[a] * p2[b] + p3[a] * p3[b]) / 60
    v += (p0[a] * p1[b] + p0[a] * p2[b] + p0[a] * p3[b] + p0[b] * p1[a] + p0[b] * p2[a] + p0[b] * p3[a] +
          p1[a] * p2[b] + p1[a] * p3[b] + p1[b] * p2[a] + p1[b] * p3[a] + p2[a] * p3[b] + p2[b] * p3[a]) / 120
    return v
end
