"""
Device helpers every meshing pass is built from: run a graph once, scan, and
compact. Every index a kernel forms is Int32 (an Int64 one around adjacent loads
miscompiles on RADV, see DNNKernels' `q8gemm.jl`).
"""

"""
    runonce!(g::Mantle.Graph)

Record `g`, run it once and free the plan. Every resource the caller reads
afterwards is a `Buffer` it bound, so nothing of the plan is needed after it.
"""
function runonce!(g::Mantle.Graph)
    plan = Mantle.record!(Mantle.Plan(g))
    Mantle.run!(plan)
    Mantle.waitfor!(plan)
    Mantle.free!(plan)
    return nothing
end

"""The inclusive prefix sum of an Int32 buffer, on the device."""
function prefixsum(dev::Mantle.Device, b::Mantle.Buffer)
    out = Mantle.Buffer(dev, Int32, length(b))
    accumulate!(+, Mantle.storage(out), Mantle.storage(b))
    return out
end

"""A buffer's last entry, read back alone."""
lastentry(b::Mantle.Buffer) = Int(only(Array(view(Mantle.storage(b), length(b):length(b)))))

"""A mesh buffer's column `v` as a point."""
@inline vertex(V, v) = @inbounds Vec3f(V[1, v], V[2, v], V[3, v])

"""
    select(dev, flags, n) -> (indices, count)

The one-based indices of the set entries of `flags` (Int32, 0 or 1), in order:
cub's `DeviceSelect` as a scan and a scatter.
"""
function select(dev::Mantle.Device, flags::Mantle.Buffer, n::Int)
    pos = prefixsum(dev, flags)
    m = lastentry(pos)
    idx = Mantle.Buffer(dev, Int32, max(m, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactindex!, (idx, flags, pos, n), n; name = "select")
    runonce!(g)
    Mantle.free!(pos)
    return idx, m
end

"""
    compactcolumns(dev, src, flags, n) -> (dst, count)

The columns of `src` `(3, n)` whose flag is set, in order.
"""
function compactcolumns(dev::Mantle.Device, src::Mantle.Buffer{T}, flags::Mantle.Buffer, n::Int) where {T}
    pos = prefixsum(dev, flags)
    m = lastentry(pos)
    dst = Mantle.Buffer(dev, T, (3, max(m, 1)))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, compactcols!, (dst, src, flags, pos, n), n; name = "compact columns")
    runonce!(g)
    Mantle.free!(pos)
    return dst, m
end

function compactindex!(idx, flags, pos, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds flags[i] == Int32(1) && (idx[pos[i]] = i)
    return nothing
end

"""`dst[:, pos[i]] = src[:, i]` for every `i <= n` whose flag is set."""
function compactcols!(dst, src, flags, pos, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds if flags[i] == Int32(1)
        j = pos[i]
        dst[1, j] = src[1, i]
        dst[2, j] = src[2, i]
        dst[3, j] = src[3, i]
    end
    return nothing
end
