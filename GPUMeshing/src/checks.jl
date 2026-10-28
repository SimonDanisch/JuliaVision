"""
Mesh invariants checked on the device: area and enclosed volume, whether a mesh is a closed,
consistently wound two-manifold, and which points lie inside it.
"""

"""
    areavolume(dev, V, F) -> (area, volume)

Total area and enclosed signed volume of the mesh `(3, nv)` Float32 / `(3, nf)` Int32;
positive when the faces point outward: sums of per-face terms.
"""
function areavolume(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer)
    nf = size(F, 2)
    terms = fill!(Mantle.Buffer(dev, Float32, (2, max(nf, 1))), 0f0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, faceterms!, (terms, V, F, nf), nf; name = "checks/area volume")
    runonce!(g)
    s = Array(sum(Mantle.storage(terms); dims = 2))
    Mantle.free!(terms)
    return s[1], s[2]
end

function faceterms!(terms, V, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds begin
        a, b, c = vertex(V, F[1, f]), vertex(V, F[2, f]), vertex(V, F[3, f])
        n = cross(b - a, c - a)
        terms[1, f] = 0.5f0 * sqrt(dot(n, n))
        terms[2, f] = dot(a, n) / 6f0
    end
    return nothing
end

"""
    closedmanifold(dev, F) -> Bool

Whether every directed edge of the faces `(3, nf)` occurs exactly once and its reverse does
too: a closed, consistently wound two-manifold surface. The directed edges are sorted as
keys `(from << 32) | to`, and each looks for its neighbour and its reverse.
"""
function closedmanifold(dev::Mantle.Device, F::Mantle.Buffer)
    nf = size(F, 2)
    n = 3nf
    n == 0 && return true
    keys = Mantle.Buffer(dev, UInt64, n)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, directedkeys!, (keys, F, nf), nf; name = "checks/directed edges")
    runonce!(g)
    AK.sort!(Mantle.storage(keys))
    bad = fill!(Mantle.Buffer(dev, Int32, 1), 0)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, pairededges!, (bad, keys, n), n; name = "checks/paired edges")
    runonce!(g)
    ok = only(Array(Mantle.storage(bad))) == 0
    foreach(Mantle.free!, (keys, bad))
    return ok
end

function directedkeys!(keys, F, nf)
    f = Int32(KI.get_global_id().x)
    f <= nf || return nothing
    @inbounds for k in Int32(1):Int32(3)
        a, b = F[k, f], F[k % Int32(3) + Int32(1), f]
        keys[Int32(3) * (f - Int32(1)) + k] = (UInt64(a % UInt32) << 32) | UInt64(b % UInt32)
    end
    return nothing
end

"""Counts a directed edge that repeats or whose reverse is missing (binary search)."""
function pairededges!(bad, keys, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds begin
        k = keys[i]
        repeated = i > Int32(1) && keys[i - Int32(1)] == k
        r = (k << 32) | (k >> 32)
        lo, hi = Int32(1), n
        while lo < hi
            mid = (lo + hi) >> 1
            keys[mid] < r ? (lo = mid + Int32(1)) : (hi = mid)
        end
        (repeated || keys[lo] != r) && Atomix.@atomic :monotonic bad[1] += Int32(1)
    end
    return nothing
end

"""
    windingnumbers(dev, V, F, P, n) -> Mantle.Buffer{Float32}

The generalized winding number of the mesh `V` `(3, nv)` Float32, `F` `(3, nf)` Int32 at each
of the first `n` columns of `P` (Jacobson, Kavan and Sorkine-Hornung 2013): the solid angles
of all faces seen from the point, summed over `4π`. 1 inside a closed, outward-wound mesh,
0 outside, and in between near holes or where the mesh overlaps itself; above ½ counts as
inside. Every point sums every face, so it suits thousands of points against a part, not a
scene.
"""
function windingnumbers(dev::Mantle.Device, V::Mantle.Buffer, F::Mantle.Buffer, P::Mantle.Buffer, n::Int)
    out = Mantle.Buffer(dev, Float32, max(n, 1))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, windingnumber!, (out, V, F, size(F, 2), P, n), n; name = "checks/winding numbers")
    runonce!(g)
    return out
end

"""The solid angle of triangle `a b c` seen from the origin (Van Oosterom and Strackee)."""
@inline function solidangle(a::Vec3f, b::Vec3f, c::Vec3f)
    la, lb, lc = norm(a), norm(b), norm(c)
    num = dot(a, cross(b, c))
    den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb
    return 2f0 * atan(num, den)
end

function windingnumber!(out, V, F, nf, P, n)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    p = vertex(P, i)
    w = 0f0
    @inbounds for f in Int32(1):Int32(nf)
        w += solidangle(vertex(V, F[1, f]) - p, vertex(V, F[2, f]) - p, vertex(V, F[3, f]) - p)
    end
    @inbounds out[i] = w / (4f0 * Float32(π))
    return nothing
end
