"""
The marching cubes Hunyuan3DRunner shipped before this package: the face-contour
loop walk run cell by cell, with vertices shared through a dictionary keyed by
grid edge. It shares nothing with the package's path but `GPUMeshing.FACE`,
`FACEEDGE` and `CORNER`: no table, no Int32 index arithmetic, no prefix sums, so
agreeing with it checks those. Its asymptotic decider joined the wrong diagonal
pair; that is fixed here (`facepairs` is the package's). Its winding came from the
local gradient per triangle and is left out: the triangles come back unoriented.
"""
module Reference

using GPUMeshing: CORNER, EDGE, FACE, FACEEDGE, facepairs

function marchingcubes(field::AbstractArray{<:Real,3}; level::Real = 0)
    nx, ny, nz = size(field)
    lv = Float32(level)
    edgeid(e, i, j, k) = (i + CORNER[EDGE[e][1]][1], j + CORNER[EDGE[e][1]][2], k + CORNER[EDGE[e][1]][3], EDGE[e][3])
    tris = NTuple{3,NTuple{4,Int}}[]
    vals = zeros(Float32, 8)
    for k in 1:(nz - 1), j in 1:(ny - 1), i in 1:(nx - 1)
        for c in 1:8
            vals[c] = Float32(field[i + CORNER[c][1], j + CORNER[c][2], k + CORNER[c][3]]) - lv
        end
        adj = zeros(Int, 2, 12)
        for f in 1:6
            cs, es = FACE[f], FACEEDGE[f]
            fv = ntuple(n -> vals[cs[n]], 4)
            crossed = ntuple(n -> signbit(fv[n]) != signbit(fv[mod1(n + 1, 4)]), 4)
            den = fv[1] + fv[3] - fv[2] - fv[4]
            saddle = den == 0 ? fv[1] : (fv[1] * fv[3] - fv[2] * fv[4]) / den
            for (p, q) in facepairs(crossed, signbit(saddle) == signbit(fv[1]))
                p == 0 && continue
                a, b = es[p], es[q]
                adj[adj[1, a] == 0 ? 1 : 2, a] = b
                adj[adj[1, b] == 0 ? 1 : 2, b] = a
            end
        end
        seen = falses(12)
        for start in 1:12
            (adj[1, start] == 0 || seen[start]) && continue
            loop = Int[]
            e, prev = start, 0
            while true
                push!(loop, e)
                seen[e] = true
                prev, e = e, adj[1, e] == prev ? adj[2, e] : adj[1, e]
                e == start && break
            end
            for t in 2:(length(loop) - 1)
                push!(tris, (edgeid(loop[1], i, j, k), edgeid(loop[t], i, j, k), edgeid(loop[t + 1], i, j, k)))
            end
        end
    end
    point(id) = begin
        i, j, k, a = id
        d = ntuple(n -> n == a ? 1 : 0, 3)
        va = Float32(field[i, j, k]) - lv
        vb = Float32(field[i + d[1], j + d[2], k + d[3]]) - lv
        t = va / (va - vb)
        (Float32(i - 1) + (a == 1 ? t : 0f0), Float32(j - 1) + (a == 2 ? t : 0f0), Float32(k - 1) + (a == 3 ? t : 0f0))
    end
    ids = unique(id for t in tris for id in t)
    return [point(id) for id in ids], [sort([point(id) for id in t]) for t in tris]
end

end # module
