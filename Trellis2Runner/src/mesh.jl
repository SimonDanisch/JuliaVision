"""
    dualgridmesh(dev, level, out; margin = 0.5) -> (V, F)

`FlexiDualGridVaeDecoder`'s mesh: `flexible_dual_grid_to_mesh` (o-voxel) on the
shape decoder's output `out` `(7, N)` at the voxels of `level`, as
`Mantle.Buffer`s `(3, N)` and `(3, F)`.

Every voxel carries one dual vertex, `(1 + 2 margin) sigmoid(out[1:3]) - margin`
inside it; an axis whose intersection logit (`out[4:6]`) is positive has a quad
through the four voxels around that edge, when all four exist (looked up in a
[`HashMap`](@ref) of the voxels, as upstream); the quad is split along the
diagonal whose `softplus(out[7])` weights multiply to more. Quads in voxel
order, axis by axis, as upstream's boolean mask walks `(N, 3)`. Every voxel
keeps its vertex whether a quad reaches it or not, as upstream.
"""
function dualgridmesh(dev::Mantle.Device, level::Level, out::AbstractMatrix{Float32}; margin = 0.5f0)
    n = length(level)
    res = level.res
    coords = Mantle.Buffer(dev, level.coords)
    outd = Mantle.Buffer(dev, Matrix{Float32}(out))
    V = Mantle.Buffer(dev, Float32, (3, n))
    lerp = Mantle.Buffer(dev, Float32, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualvertices!, (V, lerp, coords, outd, Float32(res), Float32(margin), n), n;
                     name = "dualgrid/vertices")
    runonce!(g)
    vox = insert!(dev, HashMap(dev, n), coords, n, res)
    ns = 3n
    quads = Mantle.Buffer(dev, Int32, (4, ns))
    qflag = Mantle.Buffer(dev, Int32, ns)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualquads!, (quads, qflag, coords, outd, vox.keys, vox.vals, vox.capacity, Int32(res), n), ns;
                     name = "dualgrid/quads")
    runonce!(g)
    foreach(Mantle.free!, (vox, coords, outd))
    qpos = prefixsum(dev, qflag)
    nq = lastentry(qpos)
    F = Mantle.Buffer(dev, Int32, (3, 2 * max(nq, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, dualtriangles!, (F, quads, qflag, qpos, lerp, ns), ns; name = "dualgrid/triangles")
    runonce!(g)
    foreach(Mantle.free!, (quads, qflag, qpos, lerp))
    return V, F
end

function dualvertices!(V, lerp, coords, out, res, margin, n)
    j = Int32(KI.get_global_id().x)
    j <= n || return nothing
    @inbounds begin
        vs = 1f0 / res
        for a in Int32(1):Int32(3)
            d = (1f0 + 2f0 * margin) / (1f0 + exp(-out[a, j])) - margin
            V[a, j] = (Float32(coords[a, j]) + d) * vs - 0.5f0
        end
        # softplus, with torch's threshold
        x = out[7, j]
        lerp[j] = x > 20f0 ? x : log1p(exp(x))
    end
    return nothing
end

"""The `k`th of the four voxels around the edge along axis `a` from a voxel,
as an offset; o-voxel's `edge_neighbor_voxel_offset`."""
@inline function aroundoffset(a, k)
    if a == Int32(1)
        return k == Int32(1) ? (Int32(0), Int32(0), Int32(0)) : k == Int32(2) ? (Int32(0), Int32(0), Int32(1)) :
               k == Int32(3) ? (Int32(0), Int32(1), Int32(1)) : (Int32(0), Int32(1), Int32(0))
    elseif a == Int32(2)
        return k == Int32(1) ? (Int32(0), Int32(0), Int32(0)) : k == Int32(2) ? (Int32(1), Int32(0), Int32(0)) :
               k == Int32(3) ? (Int32(1), Int32(0), Int32(1)) : (Int32(0), Int32(0), Int32(1))
    else
        return k == Int32(1) ? (Int32(0), Int32(0), Int32(0)) : k == Int32(2) ? (Int32(0), Int32(1), Int32(0)) :
               k == Int32(3) ? (Int32(1), Int32(1), Int32(0)) : (Int32(1), Int32(0), Int32(0))
    end
end

"""Slot `s` is voxel `(s - 1) ÷ 3 + 1`, axis `(s - 1) % 3 + 1`: its quad's four
voxel ids when the axis intersects and all four exist."""
function dualquads!(quads, qflag, coords, out, keys, vals, cap, res, n)
    s = Int32(KI.get_global_id().x)
    s <= Int32(3) * n || return nothing
    @inbounds begin
        j = (s - Int32(1)) ÷ Int32(3) + Int32(1)
        a = (s - Int32(1)) % Int32(3) + Int32(1)
        qflag[s] = Int32(0)
        out[Int32(3) + a, j] > 0f0 || return nothing
        x, y, z = coords[1, j], coords[2, j], coords[3, j]
        for k in Int32(1):Int32(4)
            o = aroundoffset(a, k)
            id = lookupcoord(keys, vals, x + o[1], y + o[2], z + o[3], res, cap)
            id == Int32(0) && return nothing
            quads[k, s] = id
        end
        qflag[s] = Int32(1)
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
