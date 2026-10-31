"""
Several meshes as one and back: one simplification or query over many parts, each part's
vertices tagged with its index.
"""

"""
    joinparts(dev, meshes) -> (V, F, P)

The meshes `(V_k, F_k)` (`(3, n)` Float32 and Int32 buffers, consumed) as one, their faces
renumbered, and each vertex's part `P` (0 for the first).
"""
function joinparts(dev::Mantle.Device, meshes)
    nv = sum(size(V, 2) for (V, _) in meshes)
    nf = sum(size(F, 2) for (_, F) in meshes)
    V = Mantle.Buffer(dev, Float32, (3, nv))
    F = Mantle.Buffer(dev, Int32, (3, nf))
    P = Mantle.Buffer(dev, Int32, nv)
    v, f = 0, 0
    for (k, (Vk, Fk)) in enumerate(meshes)
        a, b = size(Vk, 2), size(Fk, 2)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, placepart!, (V, F, P, Vk, Fk, Int32(v), Int32(f), Int32(k - 1), Int32(a), Int32(b)), max(a, b);
                         name = "parts/join")
        runonce!(g)
        v += a
        f += b
        foreach(Mantle.free!, (Vk, Fk))
    end
    return V, F, P
end

function placepart!(V, F, P, Vk, Fk, v, f, part, nv, nf)
    i = Int32(KI.get_global_id().x)
    @inbounds if i <= nv
        V[1, v + i], V[2, v + i], V[3, v + i] = Vk[1, i], Vk[2, i], Vk[3, i]
        P[v + i] = part
    end
    @inbounds if i <= nf
        F[1, f + i], F[2, f + i], F[3, f + i] = Fk[1, i] + v, Fk[2, i] + v, Fk[3, i] + v
    end
    return nothing
end

"""
    splitparts(dev, V, F, P, n) -> [(V_k, F_k)]

The `n` parts of a joined mesh as meshes of their own. A pass that keeps the order of
vertices and faces (compaction does) leaves each part's as one run: found by the first vertex
and face of each part, then copied out with the faces renumbered.
"""
function splitparts(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, P::Mantle.Buffer, n::Int)
    nv, nf = size(V, 2), size(F, 2)
    vstart = fill!(Mantle.Buffer(dev, Int32, n + 1), Int32(nv))
    fstart = fill!(Mantle.Buffer(dev, Int32, n + 1), Int32(nf))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, partstarts!, (vstart, fstart, P, F, Int32(nv), Int32(nf)), max(nv, nf); name = "parts/starts")
    runonce!(g)
    vs, fs = Int.(Array(Mantle.storage(vstart))), Int.(Array(Mantle.storage(fstart)))
    foreach(Mantle.free!, (vstart, fstart))
    out = Tuple{Mantle.Buffer{Float32,2},Mantle.Buffer{Int32,2}}[]
    for k in 1:n
        a, b = vs[k], vs[k + 1]
        c, d = fs[k], fs[k + 1]
        Vk = Mantle.Buffer(dev, Float32, (3, b - a))
        Fk = Mantle.Buffer(dev, Int32, (3, d - c))
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, takepart!, (Vk, Fk, V, F, Int32(a), Int32(c), Int32(b - a), Int32(d - c)), max(b - a, d - c);
                         name = "parts/split")
        runonce!(g)
        push!(out, (Vk, Fk))
    end
    return out
end

"""Where each part's run starts (zero-based): the vertex or face whose part differs from the one before."""
function partstarts!(vstart, fstart, P, F, nv, nf)
    i = Int32(KI.get_global_id().x)
    @inbounds if i <= nv && (i == Int32(1) || P[i] != P[i - Int32(1)])
        vstart[P[i] + Int32(1)] = i - Int32(1)
    end
    @inbounds if i <= nf && (i == Int32(1) || P[F[1, i]] != P[F[1, i - Int32(1)]])
        fstart[P[F[1, i]] + Int32(1)] = i - Int32(1)
    end
    return nothing
end

function takepart!(Vk, Fk, V, F, v, f, nv, nf)
    i = Int32(KI.get_global_id().x)
    @inbounds if i <= nv
        Vk[1, i], Vk[2, i], Vk[3, i] = V[1, v + i], V[2, v + i], V[3, v + i]
    end
    @inbounds if i <= nf
        Fk[1, i], Fk[2, i], Fk[3, i] = F[1, f + i] - v, F[2, f + i] - v, F[3, f + i] - v
    end
    return nothing
end
