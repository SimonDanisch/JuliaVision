"""
SAM 2.1 with no first-call latency.

A thin package around `DNNKernels.SAM2` whose only job is the workload: it runs a
real segmentation during its own precompilation, so that both halves of the
cold-start cost are paid once, at `Pkg.precompile`, instead of by whoever clicks
first.

The two halves are separate problems and need both mechanisms:

  * **Julia inference** — `execute!`, `runop!`, the broadcast machinery, the KA
    launch path. Measured at ~62 s for SAM 2's encoder, and the larger half.
    `PrecompileTools` keeps it in the package image.
  * **SPIR-V** — 99 kernels. `Lava`'s frozen cache keys them on nothing but
    their signature and `KERNELS_VERSION`, so they load without Julia inferring
    anything to find them.

`Lava.@compile_workload` runs both at once, which is why the workload is written
once and not twice.

**Bump [`KERNELS_VERSION`](@ref) after changing any kernel.** The frozen cache
does not detect that for you — by design; see `Lava/src/runtime/frozen_cache.jl`.
"""
module SAM2Runner

using Lava, DNNKernels, KernelAbstractions, ColorTypes
using ColorTypes: RGB, red, green, blue
# Through ColorTypes rather than as a direct dependency: the workload needs the
# editor's exact frame element type and nothing else from FixedPointNumbers.
const N0f8 = ColorTypes.FixedPointNumbers.N0f8
using Lava: @setup_workload, @compile_workload
using DNNKernels: SAM2, encode, decode, segment, prompt, toback

# `segment` is `DNNKernels.segment` with methods added here, not a new function:
# same verb, different arguments, which is what dispatch is for.
export segment, defaultmodel, unloadmodel!
export SAM2Runner_VERSION, sam2model, runsam2, assetdir, sam2segmenter

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Where the exported graphs and weights live: the `sam2-large` artifact, unless
this checkout generates them itself or `JULIA_SAM2_ASSETS` says otherwise. See
[`DNNKernels.assetpath`](@ref) for the order and why it is that order.

942 MB of weights, so an artifact rather than anything in git, and lazy so that
installing this package does not download them.
"""
assetdir() = assetpath(; artifact = "sam2-large",
                       toml = joinpath(@__DIR__, "..", "Artifacts.toml"),
                       generated = joinpath("gen", "graphs", "sam2-large"),
                       env = "JULIA_SAM2_ASSETS", from = @__DIR__)

"""
    refsdir() -> String

The PyTorch reference activations the test suite compares against — 1.2 GB, and
*only* for tests, which is why they are a separate artifact from the weights. A
caller that just wants to segment a picture should never fetch these.
"""
refsdir() = assetpath(; artifact = "sam2-large-refs",
                      toml = joinpath(@__DIR__, "..", "Artifacts.toml"),
                      generated = joinpath("gen", "graphs", "sam2-large"),
                      env = "JULIA_SAM2_REFS", from = @__DIR__)

"""
    sam2model(; backend, dir, res) -> SAM2

Load the model. Separate from [`runsam2`](@ref) so the workload can build it in
`@setup_workload`, where the loading is *not* what is being cached.
"""
function sam2model(; backend = LavaBackend(), dir::AbstractString = assetdir(), res::Int = 1024)
    return SAM2(dir, joinpath(dir, "weights.safetensors"); backend, res)
end

"""
    runsam2(model, image, points, labels; pick = :best) -> (mask, score)

One frame, one set of clicks: embed, then decode. `image` is `(res, res, 3, 1)`
in 0..1 RGB on the model's backend, `points` are `(x, y)` normalized, `labels`
`true` for foreground.

Returns the 256x256 mask **logits** and the model's predicted IoU. Thresholding
and resampling belong to the caller, which knows the frame it came from.

This is the function the workload runs, and therefore the one whose whole call
tree is precompiled. A path it does not take is a path that still compiles on
first use — see the `missing_kernels` test.
"""
function runsam2(model::SAM2, image, points, labels; pick = :best)
    feats = encode(model, image)
    return segment(model, feats, points, labels; pick)
end

"""
    runsam2(model, image) -> (mask, score)

One centre click — the smallest complete call, and what the workload uses.
"""
runsam2(model::SAM2, image) = runsam2(model, image, [(0.5, 0.5)], [true])

"""
    resizeto!(img, frame, res) -> img

Nearest-neighbour resample of `frame` into the model's square `res × res × 3`
input buffer.

Its own function, and `img` concretely typed, because it is 3.1 million
iterations and the buffer used to reach it through a `Ref{Any}`. Every
`img[i, j, c, 1] = …` was then a dynamically dispatched, boxed `setindex!`:
**~740 ms per call**, three times the encode it feeds. The same loop with a
typed destination is 3.2 ms.

That cost was invisible to every GPU profile — `Lava.with_dispatch_timing` says
the encode is 98% GPU-bound and it is; this sits entirely outside it, in front.
"""
function resizeto!(img::Array{Float32,4}, frame::AbstractMatrix, res::Integer)
    w, h = size(frame)
    # SAM resizes to a *square*, so normalized coordinates map straight through
    # and a click needs no aspect correction.
    @inbounds for j in 1:res, i in 1:res
        c = frame[clamp(round(Int, (i - 0.5) * w / res + 0.5), 1, w),
                  clamp(round(Int, (j - 0.5) * h / res + 0.5), 1, h)]
        img[i, j, 1, 1] = Float32(red(c))
        img[i, j, 2, 1] = Float32(green(c))
        img[i, j, 3, 1] = Float32(blue(c))
    end
    img
end

"""
    sam2segmenter(model; pick = :confident) -> f

A segmenter matching `VideoEditor.registersegmenter!`'s contract:
`f(frame, points; key) -> Matrix{UInt8}`.

Lives in this package rather than in a script so it can be *precompiled*. That
is not tidiness: with the kernels frozen and this closure defined at the call
site, the editor's first click still cost 41 s, 97% of it Julia inferring this
function and everything it reaches. Code that is not in a package cannot be in a
package image.

`key` identifies the frame so the embedding is computed once per marked frame
rather than once per click — 0.9 s against 1.4 s here, and the difference grows
with every point the user adds.

`pick` defaults to `:confident` rather than SAM's own `:best`, because what this
produces is a matte *seed*: `DNNKernels.segment` documents the measurement, but in
short, argmax over a predicted score will hand back a mask that sits at the
threshold and speckles when it is resampled to frame resolution. `:best` remains
available and remains exactly what PyTorch does.
"""
function sam2segmenter(model::SAM2; pick = :confident)
    cachekey = Ref{Any}(nothing)
    cachefeats = Ref{Any}(nothing)
    # Concretely typed, unlike the two above: this one is *indexed* three million
    # times per call, and `Ref{Any}` made every one of those a dynamic dispatch.
    # The other two are read once each, where `Any` costs nothing.
    host = Ref{Array{Float32,4}}()
    res = model.res
    backend = model.model.backend
    return function (frame::AbstractMatrix, points; key = nothing)
        w, h = size(frame)
        feats = if key !== nothing && cachekey[] == key && cachefeats[] !== nothing
            cachefeats[]
        else
            # Resized inside the miss branch, because `img` feeds nothing but the
            # encode. It ran on every call for a while, and on the path that
            # matters — the second and later clicks on one marked frame — that is
            # 2.7 ms of a 16.7 ms click spent producing a buffer nobody reads.
            isassigned(host) || (host[] = zeros(Float32, res, res, 3, 1))
            img = resizeto!(host[], frame, res)
            f = encode(model, toback(backend, img))
            KA.synchronize(backend)
            cachekey[] = key; cachefeats[] = f
            f
        end
        logits, _ = segment(model, feats, [(p[1], p[2]) for p in points],
                            [p[3] for p in points]; pick)
        return maskatframe(Array(logits), w, h)
    end
end

# ─────────────────────────────────────────────────────────── the one-call form
#
# Everything above takes a model, or a backend, or a closure that owns an
# embedding cache. That is the right shape for the editor, which holds a model
# for the length of a session and knows which frame it is on. It is the wrong
# shape for "I have a picture and a click".

"""
The default model, built on first use and kept.

One `Ref`, not a registry: there is exactly one sensible default model and the
alternative — a dict keyed by resolution or backend — would be a cache nobody
asked for. Anything that wants a *different* model builds one with
[`sam2model`](@ref) and passes it, which every method below accepts.

Holding it is the point. Building the model reads a graph and ~900 MB of
weights; on this card that is ~3 s, and paying it per call would make the
convenience form useless for the thing it is for.
"""
const DEFAULT_MODEL = Ref{Any}(nothing)

"""
    defaultmodel(; backend = LavaBackend()) -> SAM2

The shared model, built on first call. Throws with the path it looked in when
the weights are not installed, rather than returning `nothing` for the caller to
trip over later.
"""
function defaultmodel(; backend = LavaBackend())
    m = DEFAULT_MODEL[]
    m === nothing || return m::SAM2
    dir = assetdir()
    isfile(joinpath(dir, "weights.safetensors")) || throw(ArgumentError(
        "SAM 2.1 weights not found at $dir. Set JULIA_SAM2_ASSETS, or generate " *
        "them with `uv run tools/export_sam2.py && uv run tools/convert_weights.py`."))
    m = sam2model(; backend, dir)
    DEFAULT_MODEL[] = m
    return m
end

"""
    segment(image, points; …) -> Matrix{UInt8}

Segment `image` from a few clicks. The whole API, for the case where you have a
picture and want the object under the cursor:

```julia
using SAM2Runner
mask = segment(img, [(0.5, 0.5)])              # one click, in the middle
mask = segment(img, [(0.3, 0.4), (0.8, 0.9)])  # two things to include
mask = segment(img, [(0.5, 0.5, true), (0.1, 0.1, false)])   # and one to exclude
```

`image` is any matrix of colours. `points` are **normalized** `(x, y)` in
`0..1`, optionally with a third element: `true` for a point that is on the
object (the default), `false` for one that is not. Returns a `UInt8` mask the
same size as `image`, `0xff` inside the object and `0x00` outside.

`key` identifies the picture so that repeated calls on the same one reuse its
embedding — 0.9 s against 1.4 s here, and the gap widens with every extra
click. Pass anything cheap that is equal for equal images; the frame number, in
an editor. Leave it out and every call re-embeds, which is correct but slower.

`model` takes a model other than the shared default, and `pick` chooses between
SAM's own argmax (`:best`) and the tie-break this defaults to (`:confident`) —
see [`sam2segmenter`](@ref) for why a matte seed wants the latter.

The first call pays for building the model, ~3 s of reading weights. Every call
after that is the network.
"""
function DNNKernels.segment(image::AbstractMatrix, points::AbstractVector;
                            key = nothing, model::Union{Nothing,SAM2} = nothing,
                            pick = :confident, backend = LavaBackend())
    m = model === nothing ? defaultmodel(; backend) : model
    seg = get!(SEGMENTERS, (objectid(m), pick)) do
        sam2segmenter(m; pick)
    end
    return seg(image, normalizepoints(points); key)
end

"""
    segment(image, x, y; …) -> Matrix{UInt8}

One click, spelled as two numbers. `segment(img, 0.5, 0.5)`.
"""
DNNKernels.segment(image::AbstractMatrix, x::Real, y::Real; kw...) =
    DNNKernels.segment(image, [(x, y)]; kw...)

"""
Segmenter closures by `(model, pick)`. They are not free — each owns the host
staging buffer and the embedding cache that makes `key` work — so a caller that
alternates between two pictures still gets the caching it came for.
"""
const SEGMENTERS = Dict{Tuple{UInt,Symbol},Any}()

"""
`(x, y)` and `(x, y, inside)` both accepted, because requiring the third element
makes the common case — every point is on the object — read worse than it is.
"""
normalizepoints(points) = [(Float64(p[1]), Float64(p[2]),
                            length(p) >= 3 ? Bool(p[3]) : true) for p in points]

"""
Forget the shared model and its segmenters, releasing the weights.

For a long-running process that is done with SAM 2 — the editor closing a
project, a script moving on — since the model holds ~900 MB of VRAM and the
`Ref` is what keeps it reachable.
"""
function unloadmodel!()
    DEFAULT_MODEL[] = nothing
    empty!(SEGMENTERS)
    return nothing
end

"""
`(256, 256)` mask logits as a 0/255 mask at frame resolution: bilinear, then
thresholded at zero — SAM's own convention, and why the decoder returns logits
rather than probabilities.

**This was the largest single cost of a click**, at 8.1 ms for 1920x1080 against
3.3 for the decode itself, because the original wrote the interpolation out
literally: two divides, two floors, four clamps and four scattered loads per
output pixel, 2.07 million times. None of it varies the way the loop assumed —
`x0`, `x1` and `tx` depend on `i` alone and are identical down every column, and
for one output row the two source rows are fixed. So the x mapping is tabulated
once, the two source rows are blended into a 256-long vector once per row, and
the inner loop is two loads and two multiply-adds against a cache-resident
vector. **8.10 -> 1.04 ms**, bit-identical output (checked over the full frame,
both loop orders interleaved), and the click it sits in went 16.7 -> 4.6 ms.
"""
function maskatframe(lg::AbstractMatrix, w::Integer, h::Integer)
    mw, mh = size(lg)
    out = Matrix{UInt8}(undef, w, h)
    x0s = Vector{Int}(undef, w); x1s = Vector{Int}(undef, w)
    txs = Vector{Float32}(undef, w)
    @inbounds for i in 1:w
        fx = ((i - 0.5) / w) * mw + 0.5
        x0 = clamp(floor(Int, fx), 1, mw)
        x0s[i] = x0; x1s[i] = min(x0 + 1, mw); txs[i] = Float32(fx - x0)
    end
    row = Vector{Float32}(undef, mw)
    @inbounds for j in 1:h
        fy = ((j - 0.5) / h) * mh + 0.5
        y0 = clamp(floor(Int, fy), 1, mh); y1 = min(y0 + 1, mh)
        ty = Float32(fy - y0); ty1 = 1 - ty
        for k in 1:mw
            row[k] = ty1 * lg[k, y0] + ty * lg[k, y1]
        end
        for i in 1:w
            tx = txs[i]
            v = (1 - tx) * row[x0s[i]] + tx * row[x1s[i]]
            out[i, j] = v > 0 ? 0xff : 0x00
        end
    end
    return out
end

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Lava.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
# Anything thrown here would otherwise break `using SAM2Runner` for everyone.
@setup_workload begin
    dir = assetdir()
    ready = isdir(dir) && isfile(joinpath(dir, "weights.safetensors")) &&
            isfile(joinpath(dir, "sam2_encoder.json"))
    if ready
        try
            backend = LavaBackend()
            model = sam2model(; backend, dir)
            res = model.res
            image = toback(backend, zeros(Float32, res, res, 3, 1))

            # The frame type the editor actually hands a segmenter, and the
            # point vector it actually builds. Small on purpose — the shape does
            # not matter, only that every method on the path gets inferred.
            frame = fill(RGB{N0f8}(0.4, 0.5, 0.6), 64, 48)
            points = [(0.5, 0.5, true), (0.25, 0.75, false)]

            @compile_workload KERNELS_VERSION begin
                mask, score = runsam2(model, image)
                # Force the results across the host boundary: the download path
                # has kernels and specialisations of its own, and it is on every
                # real call.
                Array(mask)
                # And the path the EDITOR takes, which is a different one: the
                # closure `sam2segmenter` returns specialises on the frame type,
                # and `runsam2` above never reaches it. Leaving it out was worth
                # 45 s on the first click even with the kernels frozen and every
                # method already in this package — being precompilable is not the
                # same as being precompiled.
                seg = sam2segmenter(model)
                seg(frame, points; key = 1)
                seg(frame, points; key = 1)          # the cached-embedding branch
                KA.synchronize(backend)
            end
        catch err
            @warn "SAM2Runner: workload skipped; first use will compile" exception = err
        end
    else
        @info "SAM2Runner: no assets at $dir — nothing precompiled"
    end
end

end # module
