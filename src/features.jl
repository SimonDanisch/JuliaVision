# GPU Shi-Tomasi feature detection ("good features to track"): the missing
# piece for robust point tracking. Everything here runs on the KA backend
# (Lava/CPU) — the per-pixel structure tensor, min-eigenvalue cornerness, and
# per-cell argmax selection are all kernels; only the final threshold filter
# of a few hundred cell winners touches the host.

@kernel function tensorprod_kernel!(sxx, syy, sxy, @Const(ix), @Const(iy))
    I = @index(Global, Cartesian)
    a = ix[I]; b = iy[I]
    sxx[I] = a * a
    syy[I] = b * b
    sxy[I] = a * b
end

"Min eigenvalue of the (smoothed) structure tensor — Shi-Tomasi cornerness."
@kernel function cornerness_kernel!(corner, @Const(sxx), @Const(syy), @Const(sxy))
    I = @index(Global, Cartesian)
    a = sxx[I]; d = syy[I]; b = sxy[I]
    half = 0.5f0 * (a + d)
    disc = sqrt(max(0.25f0 * (a - d)^2 + b * b, 0.0f0))
    corner[I] = half - disc          # smaller eigenvalue
end

# One thread per CELL: find the strongest corner in its cell (distributes the
# selected points across the frame instead of clumping on the highest-contrast
# region). Writes the winner's value + pixel position per cell.
@kernel function cellmax_kernel!(vals, xs, ys, @Const(corner), cw::Int32, ch::Int32,
                                 border::Int32)
    C = @index(Global, Cartesian)
    cx, cy = Tuple(C)
    w = size(corner, 1); h = size(corner, 2)
    x0 = (cx - 1) * cw + 1; x1 = min(cx * cw, w)
    y0 = (cy - 1) * ch + 1; y1 = min(cy * ch, h)
    best = -Inf32; bx = Int32(0); by = Int32(0)
    j = y0
    while j <= y1
        i = x0
        while i <= x1
            # keep detections away from the frame edge (patches need room)
            if i > border && i <= w - border && j > border && j <= h - border
                v = corner[i, j]
                if v > best
                    best = v; bx = Int32(i); by = Int32(j)
                end
            end
            i += 1
        end
        j += 1
    end
    vals[C] = best
    xs[C] = bx
    ys[C] = by
end

"""
    goodfeatures(backend, gray; maxpoints=200, σ=3.0, border=64, quality=0.02) -> Vector{NTuple{2,Int}}

Shi-Tomasi "good features to track" on the GPU: the strongest, most trackable
corner in each cell of a grid sized to yield about `maxpoints` well-spread
points. `quality` keeps only corners at least that fraction of the frame's
peak cornerness (drops flat/low-texture cells); `border` keeps points far
enough from the edge to cut a template around them.
"""
function goodfeatures(backend, gray::AbstractMatrix{Float32}; maxpoints::Integer = 200,
                      σ::Real = 3.0, border::Integer = 64, quality::Real = 0.02)
    w, h = size(gray)
    ix = KA.allocate(backend, Float32, (w, h)); iy = similar(ix)
    sxx = similar(ix); syy = similar(ix); sxy = similar(ix)
    tmp = similar(ix); corner = similar(ix)
    gradients!(ix, iy, gray)
    tensorprod_kernel!(backend)(sxx, syy, sxy, ix, iy; ndrange = (w, h))
    smooth!(sxx, copy(sxx), σ; tmp = tmp)      # window-sum the tensor (gaussian)
    smooth!(syy, copy(syy), σ; tmp = tmp)
    smooth!(sxy, copy(sxy), σ; tmp = tmp)
    cornerness_kernel!(backend)(corner, sxx, syy, sxy; ndrange = (w, h))
    # cell grid ~ maxpoints cells, roughly square
    ncell = max(Int(maxpoints), 1)
    aspect = w / h
    ny = max(round(Int, sqrt(ncell / aspect)), 1)
    nx = max(cld(ncell, ny), 1)
    cw = Int32(cld(w, nx)); ch = Int32(cld(h, ny))
    vals = KA.allocate(backend, Float32, (nx, ny))
    xs = KA.allocate(backend, Int32, (nx, ny)); ys = similar(xs)
    cellmax_kernel!(backend)(vals, xs, ys, corner, cw, ch, Int32(border); ndrange = (nx, ny))
    KA.synchronize(backend)
    hv = Array(vals); hx = Array(xs); hy = Array(ys)
    thresh = Float32(quality) * maximum(hv)
    pts = NTuple{2, Int}[]
    for k in eachindex(hv)
        hv[k] >= thresh && hx[k] > 0 && push!(pts, (Int(hx[k]), Int(hy[k])))
    end
    return pts
end
