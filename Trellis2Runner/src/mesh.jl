"""
    dualgridmesh(dev, level, out; margin = 0.5) -> (V, F)

`FlexiDualGridVaeDecoder`'s mesh: `flexible_dual_grid_to_mesh` (o-voxel) on the
shape decoder's output `out` `(7, N)` at the voxels of `level`, as
`Mantle.Buffer`s `(3, N)` and `(3, F)`.

Every voxel carries one dual vertex, `(1 + 2 margin) sigmoid(out[1:3]) - margin`
inside it; an axis whose intersection logit (`out[4:6]`) is positive has a quad
through the four voxels around that edge, when all four exist
([`GPUMeshing.edgequads`](@ref), as upstream); the quad is split along the
diagonal whose `softplus(out[7])` weights multiply to more. Quads in voxel
order, axis by axis, as upstream's boolean mask walks `(N, 3)`. Every voxel
keeps its vertex whether a quad reaches it or not, as upstream.
"""
function dualgridmesh(dev::Mantle.Device, level::Level, out::AbstractMatrix{Float32}; margin = 0.5f0)
    n = length(level)
    res = level.res
    outd = Mantle.Buffer(dev, Matrix{Float32}(out))
    V = Mantle.Buffer(dev, Float32, (3, n))
    lerp = Mantle.Buffer(dev, Float32, n)
    crossing = Mantle.Buffer(dev, Int32, (3, n))
    grid = SparseGrid(dev, Mantle.Buffer(dev, level.coords), n, res)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualvertices!, (V, lerp, crossing, grid.coords, outd, Float32(res), Float32(margin), n), n;
                     name = "dualgrid/vertices")
    runonce!(g)
    quads, qflag = edgequads(dev, grid, crossing)
    foreach(Mantle.free!, (grid, crossing, outd))
    ns = 3n
    qpos = prefixsum(dev, qflag)
    nq = lastentry(qpos)
    F = Mantle.Buffer(dev, Int32, (3, 2 * max(nq, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualtriangles!, (F, quads, qflag, qpos, lerp, ns), ns; name = "dualgrid/triangles")
    runonce!(g)
    foreach(Mantle.free!, (quads, qflag, qpos, lerp))
    return V, F
end

"""The dual vertex and split weight of every voxel, and which of its three far
edges the intersection logits cross."""
function dualvertices!(V, lerp, crossing, coords, out, res, margin, n)
    j = Int32(KI.get_global_id().x)
    j <= n || return nothing
    @inbounds begin
        vs = 1f0 / res
        for a in Int32(1):Int32(3)
            d = (1f0 + 2f0 * margin) / (1f0 + exp(-out[a, j])) - margin
            V[a, j] = (Float32(coords[a, j]) + d) * vs - 0.5f0
            crossing[a, j] = out[Int32(3) + a, j] > 0f0 ? Int32(1) : Int32(0)
        end
        # softplus, with torch's threshold
        x = out[7, j]
        lerp[j] = x > 20f0 ? x : log1p(exp(x))
    end
    return nothing
end

function dualtriangles!(F, quads, qflag, qpos, lerp, ns)
    s = Int32(KI.get_global_id().x)
    s <= ns || return nothing
    @inbounds begin
        qflag[s] == Int32(1) || return nothing
        q = qpos[s]
        i1, i2, i3, i4 = quads[1, s], quads[2, s], quads[3, s], quads[4, s]
        t1, t2 = Int32(2) * q - Int32(1), Int32(2) * q
        if lerp[i1] * lerp[i3] > lerp[i2] * lerp[i4]
            F[1, t1] = i1; F[2, t1] = i2; F[3, t1] = i3
            F[1, t2] = i1; F[2, t2] = i3; F[3, t2] = i4
        else
            F[1, t1] = i1; F[2, t1] = i2; F[3, t1] = i4
            F[1, t2] = i4; F[2, t2] = i2; F[3, t2] = i3
        end
    end
    return nothing
end
