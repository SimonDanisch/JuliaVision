# Template matching: the tracking complement to dense optical flow. Flow
# fails silently on large displacements, low texture and motion blur — a
# normalized cross-correlation search finds the global best match within
# its radius and reports HOW good and HOW unique the match is, so callers
# can tell a lock from a guess.

"""
    samplewindow!(dst, img, M, xs, ys) -> dst

Fill `dst` with `img` sampled through the projective transform `M` on the
pixel grid `xs × ys`: `dst[a, b] = img(M(xs[a], ys[b]))`, bilinear with
replicate borders. The windowed CPU counterpart of [`warp!`](@ref) —
sampling a small window beats warping the whole frame when only a patch
is needed.
"""
function samplewindow!(dst::AbstractMatrix{Float32}, img::AbstractMatrix{Float32},
                       M::Mat3f, xs::AbstractRange, ys::AbstractRange)
    size(dst) == (length(xs), length(ys)) ||
        throw(DimensionMismatch("dst $(size(dst)) vs grid $((length(xs), length(ys)))"))
    W, H = size(img)
    @inbounds for (b, y) in enumerate(ys), (a, x) in enumerate(xs)
        den = M[3, 1] * x + M[3, 2] * y + M[3, 3]
        sx = (M[1, 1] * x + M[1, 2] * y + M[1, 3]) / den
        sy = (M[2, 1] * x + M[2, 2] * y + M[2, 3]) / den
        x0 = clamp(floor(Int, sx), 1, W - 1)
        y0 = clamp(floor(Int, sy), 1, H - 1)
        fx = clamp(sx - x0, 0.0f0, 1.0f0)
        fy = clamp(sy - y0, 0.0f0, 1.0f0)
        dst[a, b] = (1 - fx) * (1 - fy) * img[x0, y0] + fx * (1 - fy) * img[x0 + 1, y0] +
                    (1 - fx) * fy * img[x0, y0 + 1] + fx * fy * img[x0 + 1, y0 + 1]
    end
    return dst
end

"""
Best match in an NCC `score` map (any value < -1 marks unscored cells):
`(dx, dy, score, margin)` — the offset from the centered position with
parabolic sub-pixel refinement, the peak score, and the uniqueness margin
(peak minus the best score further than `excl` cells away — repetitive
texture matches itself well SOMEWHERE, only a unique peak is a lock).
"""
function peakmargin(score::AbstractMatrix{Float32}; excl::Integer = 5)
    S = size(score, 1)
    R = (S - 1) ÷ 2
    best = argmax(score)
    bx, by = best[1], best[2]
    side = -2.0f0
    @inbounds for oy in 1:S, ox in 1:S
        (abs(ox - bx) > excl || abs(oy - by) > excl) && (side = max(side, score[ox, oy]))
    end
    subpix(sm1, s0, sp1) = begin
        d = sm1 - 2s0 + sp1
        d >= 0 ? 0.0f0 : clamp(0.5f0 * (sm1 - sp1) / d, -0.5f0, 0.5f0)
    end
    fx = 1 < bx < S ? subpix(score[bx - 1, by], score[bx, by], score[bx + 1, by]) : 0.0f0
    fy = 1 < by < S ? subpix(score[bx, by - 1], score[bx, by], score[bx, by + 1]) : 0.0f0
    return (bx - R - 1 + fx, by - R - 1 + fy, score[bx, by], score[bx, by] - side)
end

"""
    nccpeak(region, template; excl=5) -> (dx, dy, score, margin)

Zero-normalized cross-correlation template match on the CPU: slide the
`w×w` `template` over `region` (`(w+2R)×(w+2R)`) and return
[`peakmargin`](@ref) of the score map. Lighting-invariant; unlike a
flow-based tracker it cannot converge to a local minimum — within `±R`
the best match is exact.
"""
function nccpeak(region::AbstractMatrix{Float32}, template::AbstractMatrix{Float32};
                 excl::Integer = 5)
    w = size(template, 1)
    size(template, 2) == w || throw(DimensionMismatch("template must be square"))
    R = (size(region, 1) - w) ÷ 2
    size(region) == (w + 2R, w + 2R) ||
        throw(DimensionMismatch("region $(size(region)) vs template $w + 2×$R"))
    tm = template .- Float32(sum(template) / length(template))
    tn = sqrt(sum(abs2, tm))
    S = 2R + 1
    score = fill(-2.0f0, S, S)
    for oy in -R:R, ox in -R:R
        num = 0.0f0
        den = 0.0f0
        psum = 0.0f0
        @inbounds for j in 1:w, i in 1:w
            psum += region[R + i + ox, R + j + oy]
        end
        pm = psum / (w * w)
        @inbounds for j in 1:w, i in 1:w
            p = region[R + i + ox, R + j + oy] - pm
            num += p * tm[i, j]
            den += p * p
        end
        score[ox + R + 1, oy + R + 1] = den > 0 ? num / (sqrt(den) * tn) : -2.0f0
    end
    return peakmargin(score; excl)
end

# ------------------------------------------------------- device patch tracking

@kernel function sampleregions_kernel!(regions, @Const(gray), M::Mat3f,
                                       @Const(cx), @Const(cy), half::Int32, R::Int32)
    I = @index(Global, Cartesian)   # (a, b, k) over (w + 2R) × (w + 2R) × K
    a, b, k = Int32(I[1]), Int32(I[2]), Int32(I[3])
    x = Float32(cx[k] - half - R - Int32(1) + a)
    y = Float32(cy[k] - half - R - Int32(1) + b)
    den = M[3, 1] * x + M[3, 2] * y + M[3, 3]
    sx = (M[1, 1] * x + M[1, 2] * y + M[1, 3]) / den
    sy = (M[2, 1] * x + M[2, 2] * y + M[2, 3]) / den
    W = Int32(size(gray, 1))
    H = Int32(size(gray, 2))
    x0 = clamp(floor(Int32, sx), Int32(1), W - Int32(1))
    y0 = clamp(floor(Int32, sy), Int32(1), H - Int32(1))
    fx = clamp(sx - Float32(x0), 0.0f0, 1.0f0)
    fy = clamp(sy - Float32(y0), 0.0f0, 1.0f0)
    @inbounds regions[a, b, k] =
        (1 - fx) * (1 - fy) * gray[x0, y0] + fx * (1 - fy) * gray[x0 + Int32(1), y0] +
        (1 - fx) * fy * gray[x0, y0 + Int32(1)] + fx * fy * gray[x0 + Int32(1), y0 + Int32(1)]
end

# Weighted zero-normalized cross-correlation. `tmpls` holds w·(t − μ_w(t))
# (so Σ tmpls == 0 and the numerator needs no patch-mean term), `weights`
# the per-template spatial weights, `wsum` their sums, `tnorm` the weighted
# template norms. Uniform weights reduce exactly to plain ZNCC; a centered
# window makes an object patch track the SUBJECT under the click instead of
# compromising with the background ring around it.
@kernel function nccscores_kernel!(scores, @Const(regions), @Const(tmpls), @Const(weights),
                                   @Const(wsum), @Const(tnorm), w::Int32, R::Int32)
    I = @index(Global, Cartesian)   # (ox, oy, k) over S × S × K, S = 2R + 1
    ox, oy, k = Int32(I[1]), Int32(I[2]), Int32(I[3])
    sp = 0.0f0    # Σ w·p
    spp = 0.0f0   # Σ w·p²
    num = 0.0f0   # Σ [w·(t−μ)]·p
    @inbounds for j in Int32(1):w, i in Int32(1):w
        p = regions[ox - Int32(1) + i, oy - Int32(1) + j, k]
        wij = weights[i, j, k]
        sp += wij * p
        spp += wij * p * p
        num += tmpls[i, j, k] * p
    end
    den = spp - sp * sp / wsum[k]
    @inbounds scores[ox, oy, k] = den > 0 ? num / (sqrt(den) * tnorm[k]) : -2.0f0
end

"""
    PatchTracker(backend, gray1, centers; window=96, maxradius=48, minstd=0.02,
                 centerweight=false)

Device-resident NCC tracker for a set of reference patches. Cuts a
`window`-sized template around every center of `gray1` (frame 1, host
Float32), drops textureless ones (std < `minstd`), and keeps weighted
mean-subtracted templates plus scratch on `backend` sized for searches up
to `±maxradius`. `centerweight = true` applies a Hann² window to the
match — an OBJECT patch then follows the subject under its center instead
of compromising with the background ring around it. Feed frames with
[`matchpatches!`](@ref).
"""
struct PatchTracker{A2, A3, V, VI}
    w::Int
    maxradius::Int
    centers::Vector{NTuple{2, Int}}   # host copy of the surviving centers
    cx::VI                            # device center coordinates
    cy::VI
    tmpls::A3                         # w × w × K, weight·(t − μ_w) (device)
    weights::A3                       # w × w × K spatial match weights (device)
    wsum::V                           # K weight sums (device)
    tnorm::V                          # K weighted template norms (device)
    regions::A3                       # (w+2maxradius)² × K scratch (device)
    scores::A3                        # (2maxradius+1)² × K scratch (device)
    hostscores::Array{Float32, 3}
    gray::A2                          # W × H frame buffer (device)
end

function PatchTracker(backend, gray1::AbstractMatrix{Float32},
                      centers::AbstractVector{<:Tuple{Integer, Integer}};
                      window::Integer = 96, maxradius::Integer = 48, minstd::Real = 0.02,
                      centerweight::Bool = false,
                      refcenters::AbstractVector{<:Tuple{Integer, Integer}} = centers)
    # `refcenters` decouples WHERE the template is cut (`centers`, in `gray1`)
    # from the reference position stored/searched/reported (`refcenters`). They
    # match by default; re-detection passes fresh templates cut from the current
    # frame with reference positions back-projected into frame-1 space, so the
    # lock stays drift-free while replacing stale templates.
    w = Int(window)
    iseven(w) || (w -= 1)
    half = w ÷ 2
    W, H = size(gray1)
    wgt = centerweight ?
          Float32[sinpi((i - 0.5f0) / w)^2 * sinpi((j - 0.5f0) / w)^2 for i in 1:w, j in 1:w] :
          ones(Float32, w, w)
    ws = sum(wgt)
    kept = NTuple{2, Int}[]
    tm = Float32[]
    tn = Float32[]
    for (idx, (cx, cy)) in enumerate(centers)
        (half + 1 <= cx <= W - half && half + 1 <= cy <= H - half) || continue
        t = gray1[(cx - half):(cx + half - 1), (cy - half):(cy + half - 1)]
        μ = sum(wgt .* t) / ws
        t .-= μ
        n = sqrt(sum(wgt .* t .^ 2))
        n / w < minstd && continue    # ≈ weighted std: textureless patch
        rc = refcenters[idx]
        push!(kept, (Int(rc[1]), Int(rc[2])))   # reference (frame-1-space) position
        append!(tm, wgt .* t)
        push!(tn, n)
    end
    K = length(kept)
    K > 0 || throw(ArgumentError("no usable patches — the frame is textureless"))
    R = Int(maxradius)
    S = 2R + 1
    tmpls = KA.allocate(backend, Float32, (w, w, K))
    copyto!(tmpls, reshape(tm, w, w, K))
    weights = KA.allocate(backend, Float32, (w, w, K))
    copyto!(weights, repeat(wgt, 1, 1, K))
    wsum = KA.allocate(backend, Float32, K)
    copyto!(wsum, fill(Float32(ws), K))
    tnorm = KA.allocate(backend, Float32, K)
    copyto!(tnorm, tn)
    cx = KA.allocate(backend, Int32, K)
    copyto!(cx, Int32[c[1] for c in kept])
    cy = KA.allocate(backend, Int32, K)
    copyto!(cy, Int32[c[2] for c in kept])
    regions = KA.allocate(backend, Float32, (w + 2R, w + 2R, K))
    scores = KA.allocate(backend, Float32, (S, S, K))
    gray = KA.allocate(backend, Float32, (W, H))
    return PatchTracker(w, R, kept, cx, cy, tmpls, weights, wsum, tnorm, regions, scores,
                        Array{Float32, 3}(undef, S, S, K), gray)
end

"""
    matchpatches!(tracker, gray, M; radius=16, excl=5) -> Vector{NamedTuple}

Upload the host `gray` frame, sample every patch's search region through
the sampling transform `M` on the device, NCC-score all offsets within
`±radius`, and return per patch
`(x, y, dx, dy, score, margin)` — the reference center, the measured
offset in the `M`-corrected domain, and the match quality
([`peakmargin`](@ref)). One fused device pass regardless of patch count.
"""
function matchpatches!(tracker::PatchTracker, gray::AbstractMatrix{Float32}, M::Mat3f;
                       radius::Integer = 16, excl::Integer = 5)
    R = Int(radius)
    R <= tracker.maxradius ||
        throw(ArgumentError("radius $R exceeds the tracker's maxradius $(tracker.maxradius)"))
    w = tracker.w
    K = length(tracker.centers)
    S = 2R + 1
    backend = KA.get_backend(tracker.tmpls)
    copyto!(tracker.gray, gray)
    sampleregions_kernel!(backend)(tracker.regions, tracker.gray, M, tracker.cx, tracker.cy,
                                   Int32(w ÷ 2), Int32(R); ndrange = (w + 2R, w + 2R, K))
    nccscores_kernel!(backend)(tracker.scores, tracker.regions, tracker.tmpls,
                               tracker.weights, tracker.wsum, tracker.tnorm,
                               Int32(w), Int32(R); ndrange = (S, S, K))
    KA.synchronize(backend)
    copyto!(tracker.hostscores, tracker.scores)  # scratch is maxradius-sized; slice below
    return [begin
                dx, dy, s, m = peakmargin(view(tracker.hostscores, 1:S, 1:S, k); excl)
                (x = tracker.centers[k][1], y = tracker.centers[k][2],
                 dx = dx, dy = dy, score = s, margin = m)
            end
            for k in 1:K]
end
