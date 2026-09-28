"""
    BitImage(width, height)

One bit per texel, rows of 64-bit words: xatlas's `BitImage`, what charts are
packed as.
"""
struct BitImage
    width::Int
    height::Int
    words::Matrix{UInt64}   # (cld(width, 64), height)
end

BitImage(width::Integer, height::Integer) = BitImage(width, height, zeros(UInt64, cld(width, 64), height))

"""A bit image of `mask` (`x, y`, one-based), or of its transpose with `transpose`."""
function BitImage(mask::BitMatrix; transpose::Bool = false)
    w, h = transpose ? reverse(size(mask)) : size(mask)
    img = BitImage(w, h)
    for y in 1:h, x in 1:w
        (transpose ? mask[y, x] : mask[x, y]) || continue
        img.words[(x - 1) >> 6 + 1, y] |= UInt64(1) << ((x - 1) & 63)
    end
    return img
end

"""
    canblit(atlas, img, x, y) -> Bool

Whether `img` placed with its corner at texel `(x, y)` (zero-based) of `atlas`
covers no texel that is set there; texels outside `atlas` count as free.
"""
function canblit(atlas::BitImage, img::BitImage, x::Int, y::Int)
    s = x & 63
    w0 = x >> 6
    na = size(atlas.words, 1)
    for row in 1:img.height
        ar = y + row
        ar > atlas.height && break
        for j in axes(img.words, 1)
            v = img.words[j, row]
            v == 0 && continue
            a = w0 + j
            a <= na && (atlas.words[a, ar] & (v << s)) != 0 && return false
            s != 0 && a + 1 <= na && (atlas.words[a + 1, ar] & (v >> (64 - s))) != 0 && return false
        end
    end
    return true
end

"""Set every texel of `img` placed at `(x, y)` in `atlas`, which has to hold it."""
function blit!(atlas::BitImage, img::BitImage, x::Int, y::Int)
    s = x & 63
    w0 = x >> 6
    for row in 1:img.height, j in axes(img.words, 1)
        v = img.words[j, row]
        v == 0 && continue
        atlas.words[w0 + j, y + row] |= v << s
        s != 0 && v >> (64 - s) != 0 && (atlas.words[w0 + j + 1, y + row] |= v >> (64 - s))
    end
    return atlas
end

"""
    footprint(texel, tris, width, height; reach = 1) -> BitMatrix

Every texel a chart's triangles (`texel` `(2, n)` positions, `tris` `(3, m)`
local indices) can be sampled at: those whose `2 reach` wide square around the
centre touches a triangle. `reach = 1` is xatlas's `bilinearExpand` of its
conservative raster: every texel bilinear filtering reads for a point of the
chart.
"""
function footprint(texel::AbstractMatrix{Float32}, tris::AbstractMatrix{<:Integer}, width::Int, height::Int;
                   reach::Float32 = 1f0)
    mask = falses(width, height)
    for t in axes(tris, 2)
        ia, ib, ic = tris[1, t], tris[2, t], tris[3, t]
        ax, ay, bx, by, cx, cy = texel[1, ia], texel[2, ia], texel[1, ib], texel[2, ib], texel[1, ic], texel[2, ic]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        g0x, g0y = (by - cy) / area, (cx - bx) / area
        g1x, g1y = (cy - ay) / area, (ax - cx) / area
        # A degenerate triangle still covers its segment: one without area is
        # taken as its box.
        flat = !(abs(area) > 1f-12)
        r0 = reach * (abs(g0x) + abs(g0y))
        r1 = reach * (abs(g1x) + abs(g1y))
        r2 = reach * (abs(g0x + g1x) + abs(g0y + g1y))
        x0 = max(0, floor(Int, min(ax, bx, cx) - reach - 0.5f0))
        x1 = min(width - 1, floor(Int, max(ax, bx, cx) + reach - 0.5f0) + 1)
        y0 = max(0, floor(Int, min(ay, by, cy) - reach - 0.5f0))
        y1 = min(height - 1, floor(Int, max(ay, by, cy) + reach - 0.5f0) + 1)
        for y in y0:y1, x in x0:x1
            px, py = x + 0.5f0, y + 0.5f0
            (min(ax, bx, cx) - reach <= px <= max(ax, bx, cx) + reach &&
             min(ay, by, cy) - reach <= py <= max(ay, by, cy) + reach) || continue
            if !flat
                w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
                w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
                (w0 >= -r0 && w1 >= -r1 && 1f0 - w0 - w1 >= -r2) || continue
            end
            mask[x + 1, y + 1] = true
        end
    end
    return mask
end

"""
xatlas's `findChartLocation`: of `attempts` random corners (every corner, while
the used area has fewer), the one where the chart (either way round) fits and
grows the used `w`×`h` least, by `extent^2 + area`; stops at the first that
leaves it unchanged. `(x, y, rotated)`, or `nothing`.
"""
function findlocation(rng, atlas::BitImage, img::BitImage, imgr::BitImage, w::Int, h::Int; attempts::Int = 4096)
    best, metric = (0, 0, false), typemax(Int)
    brute = attempts >= w * h
    corners = (w + 2) * (h + 2)
    for i in 1:(brute ? 2corners : attempts)
        if brute
            r = i > corners
            x, y = (i - 1) % corners % (w + 2), (i - 1) % corners ÷ (w + 2)
        else
            x, y, r = rand(rng, 0:w), rand(rng, 0:h), rand(rng, Bool)
        end
        cw, ch = r ? (imgr.width, imgr.height) : (img.width, img.height)
        W, H = max(w, x + cw), max(h, y + ch)
        m = max(W, H)^2 + W * H
        m > metric && continue
        m == metric && max(x, y) >= max(best[1], best[2]) && continue
        canblit(atlas, r ? imgr : img, x, y) || continue
        best, metric = (x, y, r), m
        W * H == w * h && break
    end
    return metric == typemax(Int) ? nothing : best
end

"""
    packcharts(uvs, tris, density; capacity, seed) -> (texels, width, height) or nothing

xatlas's `packCharts` without a resolution limit: every chart (`uvs[c]` its
coordinates in surface units, turned to its axes, `tris[c]` its triangles)
scaled by `density` texels per unit, rasterised with its bilinear
[`footprint`](@ref), and placed largest (by perimeter) first where it grows the
atlas least ([`findlocation`](@ref)), turned by 90° where that fits better. No
two footprints share a texel.

The chart positions in texels, and the atlas's size; `nothing` once it would
outgrow `capacity`.
"""
function packcharts(uvs::Vector{Matrix{Float32}}, tris::Vector{Matrix{Int32}}, density::Float32;
                    capacity::Int, seed::Integer = 0)
    n = length(uvs)
    inchart = Vector{Matrix{Float32}}(undef, n)   # texels from the chart's corner
    dims = Vector{Tuple{Int,Int}}(undef, n)
    for c in 1:n
        uv = uvs[c]
        lo = (minimum(view(uv, 1, :)), minimum(view(uv, 2, :)))
        q = similar(uv)
        for i in axes(uv, 2)
            # One texel of margin for the footprint on every side.
            q[1, i] = (uv[1, i] - lo[1]) * density + 1f0
            q[2, i] = (uv[2, i] - lo[2]) * density + 1f0
        end
        inchart[c] = q
        dims[c] = (ceil(Int, maximum(view(q, 1, :))) + 2, ceil(Int, maximum(view(q, 2, :))) + 2)
    end
    order = sortperm(dims; by = d -> -(d[1] + d[2]))
    rng = Random.Xoshiro(seed)
    atlas = BitImage(capacity, capacity)
    w = h = 0
    texels = Vector{Matrix{Float32}}(undef, n)
    for c in order
        mask = footprint(inchart[c], tris[c], dims[c]...)
        img, imgr = BitImage(mask), BitImage(mask; transpose = true)
        loc = findlocation(rng, atlas, img, imgr, w, h)
        loc === nothing && return nothing
        x, y, r = loc
        cw, ch = r ? (imgr.width, imgr.height) : (img.width, img.height)
        (x + cw > capacity || y + ch > capacity) && return nothing
        blit!(atlas, r ? imgr : img, x, y)
        w, h = max(w, x + cw), max(h, y + ch)
        q = inchart[c]
        t = similar(q)
        for i in axes(q, 2)
            a, b = r ? (q[2, i], q[1, i]) : (q[1, i], q[2, i])
            t[1, i], t[2, i] = x + a, y + b
        end
        texels[c] = t
    end
    return texels, w, h
end
