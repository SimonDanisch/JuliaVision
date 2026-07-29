"""
Dense pyramidal Lucas-Kanade optical flow (FOLKI-style iterative scheme,
after Le Besnerais & Champagnat / the FOLKI.jl reference implementation).

Everything operates on `Float32` grayscale matrices in the same (width,
height) layout as the color pipeline. The per-level update is deliberately
broadcast-heavy — the same code runs on CPU arrays and GPU arrays; only
sampling (warp/resize) and convolution are explicit kernels.
"""

@kernel function gray_kernel!(out, @Const(img))
    I = @index(Global, Cartesian)
    c = tofloat(img[I])
    out[I] = 0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b
end

"Rec.709 luma of an RGB image into a Float32 matrix of the same size."
function grayscale!(out::AbstractMatrix{Float32}, img::AbstractMatrix{<:AbstractRGB})
    gray_kernel!(KA.get_backend(img))(out, img; ndrange = size(img))
    return out
end

@kernel function scalarconv_kernel!(out, @Const(img), @Const(weights), radius::Int32, ::Val{DIM}) where {DIM}
    I = @index(Global, Cartesian)
    i, j = Tuple(I)
    len = Int32(size(img, DIM))
    acc = 0.0f0
    for k in -radius:radius
        ii = DIM == 1 ? clamp(i + k, Int32(1), len) : i
        jj = DIM == 2 ? clamp(j + k, Int32(1), len) : j
        acc += weights[k + radius + 1] * img[ii, jj]
    end
    out[I] = acc
end

"Separable gaussian smoothing for Float32 matrices (replicate border)."
function smooth!(out::AbstractMatrix{Float32}, img::AbstractMatrix{Float32}, σ::Real;
                 tmp::AbstractMatrix{Float32} = similar(img))
    σ <= 0 && return copyto!(out, img)
    backend = KA.get_backend(img)
    weights, radius = gaussianweights(σ)
    wdev = todevice(img, weights)
    kernel = scalarconv_kernel!(backend)
    kernel(tmp, img, wdev, Int32(radius), Val(1); ndrange = size(img))
    kernel(out, tmp, wdev, Int32(radius), Val(2); ndrange = size(img))
    return out
end

@kernel function gradient_kernel!(ix, iy, @Const(img))
    I = @index(Global, Cartesian)
    i, j = Tuple(I)
    w = size(img, 1)
    h = size(img, 2)
    ix[I] = 0.5f0 * (img[min(i + 1, w), j] - img[max(i - 1, 1), j])
    iy[I] = 0.5f0 * (img[i, min(j + 1, h)] - img[i, max(j - 1, 1)])
end

"Central-difference gradients of a Float32 image."
function gradients!(ix::AbstractMatrix{Float32}, iy::AbstractMatrix{Float32},
                    img::AbstractMatrix{Float32})
    gradient_kernel!(KA.get_backend(img))(ix, iy, img; ndrange = size(img))
    return ix, iy
end

@kernel function scalarresize_kernel!(out, @Const(img))
    I = @index(Global, Cartesian)
    w, h = size(img)
    ow, oh = size(out)
    x = clamp((Float32(I[1]) - 0.5f0) * w / ow + 0.5f0, 1.0f0, Float32(w))
    y = clamp((Float32(I[2]) - 0.5f0) * h / oh + 0.5f0, 1.0f0, Float32(h))
    x0 = unsafe_trunc(Int32, x)
    y0 = unsafe_trunc(Int32, y)
    x1 = min(x0 + Int32(1), Int32(w))
    y1 = min(y0 + Int32(1), Int32(h))
    fx = x - Float32(x0)
    fy = y - Float32(y0)
    out[I] = (img[x0, y0] * (1 - fx) + img[x1, y0] * fx) * (1 - fy) +
             (img[x0, y1] * (1 - fx) + img[x1, y1] * fx) * fy
end

"Bilinear resize of a Float32 image to `size(out)`."
function bilinearresize!(out::AbstractMatrix{Float32}, img::AbstractMatrix{Float32})
    scalarresize_kernel!(KA.get_backend(img))(out, img; ndrange = size(out))
    return out
end

@kernel function flowwarp_kernel!(out, @Const(img), @Const(u), @Const(v))
    I = @index(Global, Cartesian)
    w, h = size(img)
    x = clamp(Float32(I[1]) + u[I], 1.0f0, Float32(w))
    y = clamp(Float32(I[2]) + v[I], 1.0f0, Float32(h))
    x0 = unsafe_trunc(Int32, x)
    y0 = unsafe_trunc(Int32, y)
    x1 = min(x0 + Int32(1), Int32(w))
    y1 = min(y0 + Int32(1), Int32(h))
    fx = x - Float32(x0)
    fy = y - Float32(y0)
    out[I] = (img[x0, y0] * (1 - fx) + img[x1, y0] * fx) * (1 - fy) +
             (img[x0, y1] * (1 - fx) + img[x1, y1] * fx) * fy
end

"Sample `img` displaced by the flow field: `out[p] = img[p + (u[p], v[p])]`."
function flowwarp!(out::AbstractMatrix{Float32}, img::AbstractMatrix{Float32},
                   u::AbstractMatrix{Float32}, v::AbstractMatrix{Float32})
    flowwarp_kernel!(KA.get_backend(img))(out, img, u, v; ndrange = size(out))
    return out
end

"""
    fitaffine(u, v; step=4, trim=0.3) -> Mat3f

Fit a global affine transform to a dense flow field (least squares over a
subsampled grid, refit after trimming the worst `trim` fraction of
residuals — robust against moving subjects). Returns the sampling matrix
`T` with `T*(x, y, 1) ≈ (x + u[x,y], y + v[x,y], 1)`: warping frame 2 by
`T` (`warp!(out, frame2, T)`) aligns it back onto frame 1, correcting
translation, rotation, scale and shear in one step.
"""
function fitaffine(u::AbstractMatrix{Float32}, v::AbstractMatrix{Float32};
                   step::Integer = 4, trim::Real = 0.3)
    uh, vh = u isa Matrix ? u : Array(u), v isa Matrix ? v : Array(v)
    w, h = size(uh)
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    for j in 1:step:h, i in 1:step:w
        push!(xs, i); push!(ys, j); push!(us, uh[i, j]); push!(vs, vh[i, j])
    end

    function solve(idx)
        # same design matrix G = [x y 1] for both component equations
        gtg = zeros(3, 3); gu = zeros(3); gv = zeros(3)
        for k in idx
            g = (xs[k], ys[k], 1.0)
            for a in 1:3, b in 1:3
                gtg[a, b] += g[a] * g[b]
            end
            for a in 1:3
                gu[a] += g[a] * us[k]
                gv[a] += g[a] * vs[k]
            end
        end
        cu = gtg \ gu   # u ≈ cu[1]*x + cu[2]*y + cu[3]
        cv = gtg \ gv
        return cu, cv
    end

    idx = collect(eachindex(xs))
    if 0 < trim < 1
        # bootstrap with the median translation — robust against spatially
        # coherent outliers (a moving subject skews a plain least-squares
        # first fit so badly that its residuals no longer separate inliers) —
        # then iterate trimmed affine fits
        nkeep = ceil(Int, (1 - trim) * length(idx))
        mu, mv = median(us), median(vs)
        res = [abs(us[k] - mu) + abs(vs[k] - mv) for k in idx]
        keep = idx[partialsortperm(res, 1:nkeep)]
        cu, cv = solve(keep)
        for _ in 1:3
            res = [abs(cu[1]*xs[k] + cu[2]*ys[k] + cu[3] - us[k]) +
                   abs(cv[1]*xs[k] + cv[2]*ys[k] + cv[3] - vs[k]) for k in idx]
            keep = idx[partialsortperm(res, 1:nkeep)]
            cu, cv = solve(keep)
        end
    else
        cu, cv = solve(idx)
    end
    # T(p) = p + f(p): linear part I + [cu[1] cu[2]; cv[1] cv[2]], translation (cu[3], cv[3])
    return Mat3f(1 + cu[1], cv[1], 0,
                 cu[2], 1 + cv[2], 0,
                 cu[3], cv[3], 1)
end

"""
    fitsimilarity(xs, ys, us, vs; trim=0.3) -> Mat3f

Fit a similarity transform — translation, rotation and UNIFORM scale, 4
degrees of freedom — to point correspondences `(xs[k], ys[k]) →
(xs[k] + us[k], ys[k] + vs[k])`, with iterated trimmed least squares.
Same convention as [`fitaffine`](@ref): the returned sampling matrix
aligns the second frame back onto the first. The constrained model has no
shear and no keystone terms, so measurement noise cannot excite the
jelly-like warping a full affine or homography fit shows on scenes whose
motion the model cannot represent.
"""
function fitsimilarity(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
                       us::AbstractVector{<:Real}, vs::AbstractVector{<:Real};
                       trim::Real = 0.3)
    length(xs) >= 2 || throw(ArgumentError("need at least 2 correspondences"))
    function solve(idx)
        A = zeros(4, 4)
        b = zeros(4)
        for k in idx
            x, y = xs[k], ys[k]
            X, Y = x + us[k], y + vs[k]
            r1 = (x, -y, 1.0, 0.0)   # X = a·x − b·y + tx
            r2 = (y, x, 0.0, 1.0)    # Y = b·x + a·y + ty
            for a in 1:4, c in 1:4
                A[a, c] += r1[a] * r1[c] + r2[a] * r2[c]
            end
            for a in 1:4
                b[a] += r1[a] * X + r2[a] * Y
            end
        end
        return A \ b
    end
    idx = collect(eachindex(xs))
    nkeep = max(ceil(Int, (1 - trim) * length(idx)), 2)
    c = solve(idx)
    for _ in 1:2
        nkeep == length(idx) && break
        res = [abs(c[1] * xs[k] - c[2] * ys[k] + c[3] - xs[k] - us[k]) +
               abs(c[2] * xs[k] + c[1] * ys[k] + c[4] - ys[k] - vs[k]) for k in idx]
        keep = idx[partialsortperm(res, 1:nkeep)]
        c = solve(keep)
    end
    return Mat3f(c[1], c[2], 0, -c[2], c[1], 0, c[3], c[4], 1)
end

"""
    ransacsimilarity(xs, ys, us, vs; thresh=2.0, p=0.99, maxiters=200) -> Mat3f

Robust similarity fit via RANSAC — same correspondences, convention and
return type as [`fitsimilarity`], but instead of trimming a FIXED fraction
of residuals it repeatedly fits from a minimal 2-point sample, keeps the
model with the largest consensus set (reprojection residual < `thresh`
pixels), then refits least squares on those inliers. Unlike trimmed least
squares it assumes nothing about the outlier fraction, so an independently
moving subject (a bird crossing the patches) is rejected however many
points it covers — the inlier/outlier split a dedicated NLE stabiliser
uses. Deterministic (fixed RNG seed) so a re-analysis reproduces exactly.
"""
function ransacsimilarity(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
                          us::AbstractVector{<:Real}, vs::AbstractVector{<:Real};
                          thresh::Real = 2.0, p::Real = 0.99, maxiters::Integer = 200)
    n = length(xs)
    n >= 2 || return Mat3f(1, 0, 0, 0, 1, 0, 0, 0, 1)
    function solve(idx)
        A = zeros(4, 4); b = zeros(4)
        for k in idx
            x, y = xs[k], ys[k]
            X, Y = x + us[k], y + vs[k]
            r1 = (x, -y, 1.0, 0.0)
            r2 = (y, x, 0.0, 1.0)
            for a in 1:4, c in 1:4
                A[a, c] += r1[a] * r1[c] + r2[a] * r2[c]
            end
            for a in 1:4
                b[a] += r1[a] * X + r2[a] * Y
            end
        end
        return A \ b
    end
    resid(c, k) = hypot(c[1] * xs[k] - c[2] * ys[k] + c[3] - (xs[k] + us[k]),
                        c[2] * xs[k] + c[1] * ys[k] + c[4] - (ys[k] + vs[k]))
    best = Int[]
    bestc = solve(collect(eachindex(xs)))       # trim-free LS fallback
    lim = maxiters; it = 0; t = Float64(thresh)
    st = UInt32(2654435761)                      # deterministic LCG (no RNG dependency)
    while it < lim && it < maxiters
        it += 1
        st = st * UInt32(1664525) + UInt32(1013904223); i = Int(st >> 1) % n + 1
        st = st * UInt32(1664525) + UInt32(1013904223); j = Int(st >> 1) % n + 1
        i == j && continue
        local c
        try c = solve((i, j)) catch; continue end
        inl = [k for k in 1:n if resid(c, k) < t]
        if length(inl) > length(best)
            best = inl
            w = length(inl) / n
            w > 0 && (lim = min(maxiters, ceil(Int, log(1 - p) / log(1 - w^2 + 1.0e-9))))
        end
    end
    c = length(best) >= 2 ? solve(best) : bestc
    return Mat3f(c[1], c[2], 0, -c[2], c[1], 0, c[3], c[4], 1)
end

"""
    ransactranslation(us, vs; thresh=2.0) -> (dx, dy)

Dominant 2-D translation from displacement samples `(us[k], vs[k])`, RANSAC-robust:
the translation with the largest set of samples agreeing within `thresh` px, refined
to their mean. Rejects an independently-moving cluster (a bird crossing the patches)
that a plain mean or median would let bias the result. Used to correct residual
translation against a fixed reference without re-introducing rotation/scale.
"""
function ransactranslation(us::AbstractVector{<:Real}, vs::AbstractVector{<:Real};
                           thresh::Real = 2.0)
    m = length(us)
    m == 0 && return (0.0, 0.0)
    t2 = Float64(thresh)^2
    best = 0; bx = 0.0; by = 0.0
    for i in 1:m
        tx = us[i]; ty = vs[i]; c = 0
        for k in 1:m
            ((us[k] - tx)^2 + (vs[k] - ty)^2 < t2) && (c += 1)
        end
        if c > best
            sx = 0.0; sy = 0.0
            for k in 1:m
                if (us[k] - tx)^2 + (vs[k] - ty)^2 < t2
                    sx += us[k]; sy += vs[k]
                end
            end
            best = c; bx = sx / c; by = sy / c
        end
    end
    return (bx, by)
end

"""
    fithomography(u, v; step=4, trim=0.3) -> Mat3f

Fit a global homography to a dense flow field — [`fitaffine`](@ref) plus
the two perspective terms, for footage where the camera tilt makes the
scene keystone (a pure affine leaves the corners swimming). Same robust
scheme (median-translation bootstrap, iterated trimmed least squares) and
same convention: the returned sampling matrix `T` satisfies
`T*(x, y, 1) ∝ (x + u[x,y], y + v[x,y], 1)` projectively, so
`warp!(out, frame2, T)` aligns frame 2 back onto frame 1. On affine flow
the perspective terms vanish, so it degenerates gracefully.
"""
function fithomography(u::AbstractMatrix{Float32}, v::AbstractMatrix{Float32};
                       step::Integer = 4, trim::Real = 0.3)
    uh, vh = u isa Matrix ? u : Array(u), v isa Matrix ? v : Array(v)
    w, h = size(uh)
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    for j in 1:step:h, i in 1:step:w
        push!(xs, i); push!(ys, j); push!(us, uh[i, j]); push!(vs, vh[i, j])
    end
    # normalized coordinates: the DLT design matrix mixes x, x² and x·u
    # scales, which is hopeless conditioning in raw pixels
    cx, cy = (1 + w) / 2, (1 + h) / 2
    s = (w + h) / 2
    nx(k) = (xs[k] - cx) / s
    ny(k) = (ys[k] - cy) / s
    px(k) = (xs[k] + us[k] - cx) / s
    py(k) = (ys[k] + vs[k] - cy) / s

    # inhomogeneous DLT, h33 fixed to 1 (valid: stabilization homographies
    # are near-identity, h33 never approaches 0). Accumulated as 8×8 normal
    # equations like fitaffine — the fit runs per frame inside analysis
    # loops, so no per-solve matrix allocation or LAPACK calls on tall
    # matrices (fine numerically: the coordinates are normalized above)
    gtg = zeros(8, 8)
    gb = zeros(8)
    rowu = zeros(8)
    rowv = zeros(8)
    function solve(idx)
        fill!(gtg, 0.0); fill!(gb, 0.0)
        for k in idx
            x, y, xp, yp = nx(k), ny(k), px(k), py(k)
            rowu[1] = x; rowu[2] = y; rowu[3] = 1; rowu[4] = 0; rowu[5] = 0
            rowu[6] = 0; rowu[7] = -xp * x; rowu[8] = -xp * y
            rowv[1] = 0; rowv[2] = 0; rowv[3] = 0; rowv[4] = x; rowv[5] = y
            rowv[6] = 1; rowv[7] = -yp * x; rowv[8] = -yp * y
            for a in 1:8
                ru, rv = rowu[a], rowv[a]
                (ru == 0 && rv == 0) && continue
                for b in a:8
                    gtg[a, b] += ru * rowu[b] + rv * rowv[b]
                end
                gb[a] += ru * xp + rv * yp
            end
        end
        for a in 2:8, b in 1:(a - 1)  # symmetrize the upper-triangle sums
            gtg[a, b] = gtg[b, a]
        end
        return gtg \ gb
    end
    function residual(c, k)
        x, y = nx(k), ny(k)
        d = c[7] * x + c[8] * y + 1
        return abs((c[1] * x + c[2] * y + c[3]) / d - px(k)) +
               abs((c[4] * x + c[5] * y + c[6]) / d - py(k))
    end

    idx = collect(eachindex(xs))
    c = if 0 < trim < 1
        # same bootstrap as fitaffine: median translation separates a
        # coherent moving subject that a plain first fit absorbs
        nkeep = ceil(Int, (1 - trim) * length(idx))
        mu, mv = median(us), median(vs)
        res = [abs(us[k] - mu) + abs(vs[k] - mv) for k in idx]
        keep = idx[partialsortperm(res, 1:nkeep)]
        cc = solve(keep)
        for _ in 1:3
            res = [residual(cc, k) for k in idx]
            keep = idx[partialsortperm(res, 1:nkeep)]
            cc = solve(keep)
        end
        cc
    else
        solve(idx)
    end
    # un-normalize: H = N⁻¹ H̃ N with N the similarity used above
    Ht = [c[1] c[2] c[3]; c[4] c[5] c[6]; c[7] c[8] 1.0]
    N = [1/s 0 -cx/s; 0 1/s -cy/s; 0 0 1.0]
    Ninv = [s 0 cx; 0 s cy; 0 0 1.0]
    Hm = Ninv * Ht * N
    Hm ./= Hm[3, 3]
    return Mat3f(Hm[1, 1], Hm[2, 1], Hm[3, 1],
                 Hm[1, 2], Hm[2, 2], Hm[3, 2],
                 Hm[1, 3], Hm[2, 3], Hm[3, 3])
end

"""
    FlowWorkspace(backend, size; levels=3, σ=4.0)

Preallocated buffers for [`opticalflow!`](@ref). Reuse one workspace across
many frame pairs (e.g. a whole stabilization analysis): per call the flow
solver needs ~15 intermediate arrays per pyramid level plus the device-side
filter weights — allocating them per frame churns hundreds of megabytes
through the allocator.
"""
struct FlowWorkspace
    σ::Float64
    weights::Any
    radius::Int
    iterations::Int
    levels::Vector{NamedTuple}
end

function FlowWorkspace(backend, sz::Tuple{Int, Int}; levels::Integer = 3,
                       σ::Real = 4.0, iterations::Integer = 5)
    hostweights, radius = gaussianweights(σ)
    weights = KA.allocate(backend, Float32, length(hostweights))
    copyto!(weights, hostweights)
    lvls = NamedTuple[]
    w, h = sz
    for l in 1:levels
        l > 1 && (w < 24 || h < 24) && break
        alloc() = KA.allocate(backend, Float32, (w, h))
        push!(lvls, (g1 = alloc(), g2 = alloc(), ix = alloc(), iy = alloc(),
                     ixx = alloc(), iyy = alloc(), ixy = alloc(), det = alloc(),
                     w2 = alloc(), tmp = alloc(), prod = alloc(),
                     ixt = alloc(), iyt = alloc(), u = alloc(), v = alloc()))
        w ÷= 2
        h ÷= 2
    end
    return FlowWorkspace(Float64(σ), weights, radius, Int(iterations), lvls)
end

"Separable smoothing with preuploaded weights (no per-call allocation)."
function smoothws!(out, img, ws::FlowWorkspace, tmp)
    backend = KA.get_backend(img)
    kernel = scalarconv_kernel!(backend)
    kernel(tmp, img, ws.weights, Int32(ws.radius), Val(1); ndrange = size(img))
    kernel(out, tmp, ws.weights, Int32(ws.radius), Val(2); ndrange = size(img))
    return out
end

"""
    opticalflow!(ws::FlowWorkspace, u, v, i1, i2) -> (u, v)

Allocation-free dense flow using a preallocated workspace (see
[`FlowWorkspace`](@ref)); semantics identical to the convenience method.
"""
function opticalflow!(ws::FlowWorkspace, u::AbstractMatrix{Float32}, v::AbstractMatrix{Float32},
                      i1::AbstractMatrix{Float32}, i2::AbstractMatrix{Float32})
    size(i1) == size(i2) == size(u) == size(v) == size(ws.levels[1].g1) ||
        throw(DimensionMismatch("flow buffers must match the workspace size"))
    copyto!(ws.levels[1].g1, i1)
    copyto!(ws.levels[1].g2, i2)
    for l in 2:length(ws.levels)
        bilinearresize!(ws.levels[l].g1, ws.levels[l - 1].g1)
        bilinearresize!(ws.levels[l].g2, ws.levels[l - 1].g2)
    end

    top = length(ws.levels)
    fill!(ws.levels[top].u, 0.0f0)
    fill!(ws.levels[top].v, 0.0f0)
    for level in top:-1:1
        L = ws.levels[level]
        gradients!(L.ix, L.iy, L.g1)
        @. L.prod = L.ix * L.ix
        smoothws!(L.ixx, L.prod, ws, L.tmp)
        @. L.prod = L.iy * L.iy
        smoothws!(L.iyy, L.prod, ws, L.tmp)
        @. L.prod = L.ix * L.iy
        smoothws!(L.ixy, L.prod, ws, L.tmp)
        ε = 1.0f-3 * (sum(L.ixx) / length(L.ixx))^2 + 1.0f-12
        @. L.det = L.ixx * L.iyy - L.ixy * L.ixy + ε
        for _ in 1:ws.iterations
            flowwarp!(L.w2, L.g2, L.u, L.v)
            @. L.w2 = L.u * L.ix + L.v * L.iy + L.g1 - L.w2  # w2 becomes dI/dt
            @. L.prod = L.ix * L.w2
            smoothws!(L.ixt, L.prod, ws, L.tmp)
            @. L.prod = L.iy * L.w2
            smoothws!(L.iyt, L.prod, ws, L.tmp)
            @. L.u = (L.iyy * L.ixt - L.ixy * L.iyt) / L.det
            @. L.v = (L.ixx * L.iyt - L.ixy * L.ixt) / L.det
        end
        if level > 1
            up = ws.levels[level - 1]
            bilinearresize!(up.u, L.u)
            bilinearresize!(up.v, L.v)
            up.u .*= 2.0f0
            up.v .*= 2.0f0
        end
    end
    copyto!(u, ws.levels[1].u)
    copyto!(v, ws.levels[1].v)
    KA.synchronize(KA.get_backend(i1))
    return u, v
end

"""
    opticalflow!(u, v, i1, i2; levels=3, iterations=5, σ=4.0) -> (u, v)

Dense flow from `i1` to `i2` (Float32 grayscale, equal sizes): on
convergence `i2[p + (u[p], v[p])] ≈ i1[p]`. For a pure translation
`i2[x] = i1[x - t]` (content moved by `+t`) this yields `u = t` — the flow
IS the content motion from frame 1 to frame 2. Note the sampling duality:
`flowwarp!(out, img, U, V)` displaces content by `-(U, V)`.
Coarse-to-fine over `levels` pyramid
levels with `iterations` fixed-point updates per level; `σ` is the
integration window. `u`/`v` are overwritten (initial values ignored).
"""
function opticalflow!(u::AbstractMatrix{Float32}, v::AbstractMatrix{Float32},
                      i1::AbstractMatrix{Float32}, i2::AbstractMatrix{Float32};
                      levels::Integer = 3, iterations::Integer = 5, σ::Real = 4.0)
    ws = FlowWorkspace(KA.get_backend(i1), size(i1); levels, σ, iterations)
    return opticalflow!(ws, u, v, i1, i2)
end
