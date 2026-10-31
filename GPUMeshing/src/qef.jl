"""
Vertex placement from Hermite data, for dual contouring that keeps sharp edges and corners
(Ju, Losasso, Schaefer and Warren, *Dual Contouring of Hermite Data*, 2002): a voxel's vertex
minimises the summed squared distance to the planes through its edges' crossing points along
their normals. The quadratic error function is solved around the crossings' mass point with
a truncated pseudo-inverse (Lindstrom's), so directions the planes leave free keep the mass
point: on a flat face the vertex stays at the crossings' mean, on an edge it moves onto the
edge's line, at a corner onto the corner.

A caller accumulates, per voxel and crossing `p` with unit normal `n`: `AᵀA += n nᵀ`,
`Aᵀb += n (n ⋅ p)`, the mass point's sum `+= p` and the count, and solves with
[`qefsolve`](@ref).
"""

"""Symmetric 3×3 matrix as its upper triangle `(a11, a12, a13, a22, a23, a33)`."""
const Sym3 = NTuple{6,Float32}

@inline symmul(A::Sym3, x::Vec3f) = Vec3f(A[1] * x[1] + A[2] * x[2] + A[3] * x[3],
                                         A[2] * x[1] + A[4] * x[2] + A[5] * x[3],
                                         A[3] * x[1] + A[5] * x[2] + A[6] * x[3])

"""
Eigen decomposition of a symmetric 3×3 matrix by cyclic Jacobi rotations: `(λ, v1, v2, v3)`,
the eigenvalues and the orthonormal eigenvectors as columns, `sweeps` sweeps of the three
off-diagonal entries.
"""
@inline function symeigen(A::Sym3; sweeps::Int = 6)
    a11, a12, a13, a22, a23, a33 = A
    # V = identity, by columns.
    v11, v21, v31 = 1f0, 0f0, 0f0
    v12, v22, v32 = 0f0, 1f0, 0f0
    v13, v23, v33 = 0f0, 0f0, 1f0
    for _ in 1:sweeps
        # Rotation in the (1, 2) plane.
        if abs(a12) > 1f-30
            θ = (a22 - a11) / (2f0 * a12)
            t = sign(θ) / (abs(θ) + sqrt(θ * θ + 1f0))
            θ == 0f0 && (t = 1f0)
            c = 1f0 / sqrt(t * t + 1f0); s = t * c
            a11, a22, a12 = a11 - t * a12, a22 + t * a12, 0f0
            a13, a23 = c * a13 - s * a23, s * a13 + c * a23
            v11, v12 = c * v11 - s * v12, s * v11 + c * v12
            v21, v22 = c * v21 - s * v22, s * v21 + c * v22
            v31, v32 = c * v31 - s * v32, s * v31 + c * v32
        end
        # Rotation in the (1, 3) plane.
        if abs(a13) > 1f-30
            θ = (a33 - a11) / (2f0 * a13)
            t = sign(θ) / (abs(θ) + sqrt(θ * θ + 1f0))
            θ == 0f0 && (t = 1f0)
            c = 1f0 / sqrt(t * t + 1f0); s = t * c
            a11, a33, a13 = a11 - t * a13, a33 + t * a13, 0f0
            a12, a23 = c * a12 - s * a23, s * a12 + c * a23
            v11, v13 = c * v11 - s * v13, s * v11 + c * v13
            v21, v23 = c * v21 - s * v23, s * v21 + c * v23
            v31, v33 = c * v31 - s * v33, s * v31 + c * v33
        end
        # Rotation in the (2, 3) plane.
        if abs(a23) > 1f-30
            θ = (a33 - a22) / (2f0 * a23)
            t = sign(θ) / (abs(θ) + sqrt(θ * θ + 1f0))
            θ == 0f0 && (t = 1f0)
            c = 1f0 / sqrt(t * t + 1f0); s = t * c
            a22, a33, a23 = a22 - t * a23, a33 + t * a23, 0f0
            a12, a13 = c * a12 - s * a13, s * a12 + c * a13
            v12, v13 = c * v12 - s * v13, s * v12 + c * v13
            v22, v23 = c * v22 - s * v23, s * v22 + c * v23
            v32, v33 = c * v32 - s * v33, s * v32 + c * v33
        end
    end
    return Vec3f(a11, a22, a33), Vec3f(v11, v21, v31), Vec3f(v12, v22, v32), Vec3f(v13, v23, v33)
end

"""
    qefsolve(ata, atb, mass; truncate = 0.1f0) -> Vec3f

The point minimising `xᵀ AᵀA x - 2 xᵀ Aᵀb` nearest the mass point `mass`: the mass point
moved by the pseudo-inverse of `AᵀA` (a [`Sym3`](@ref)) applied to the residual there, with
eigenvalues below `truncate` times the largest left out (Lindstrom). With unit normals an
eigenvalue counts the planes along its direction, so the default keeps any direction at
least a tenth of the planes constrain.
"""
@inline function qefsolve(ata::Sym3, atb::Vec3f, mass::Vec3f; truncate::Float32 = 0.1f0)
    λ, v1, v2, v3 = symeigen(ata)
    top = max(abs(λ[1]), abs(λ[2]), abs(λ[3]))
    top > 0f0 || return mass
    r = atb - symmul(ata, mass)
    x = mass
    for (l, v) in ((λ[1], v1), (λ[2], v2), (λ[3], v3))
        abs(l) > truncate * top || continue
        x += v * (dot(v, r) / l)
    end
    return x
end
