"""
SAM 2.1 with no first-call latency.

A thin package around `LavaDNN.SAM2` whose only job is the workload: it runs a
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

using Lava, LavaDNN, KernelAbstractions, ColorTypes
using ColorTypes: RGB, red, green, blue
# Through ColorTypes rather than as a direct dependency: the workload needs the
# editor's exact frame element type and nothing else from FixedPointNumbers.
const N0f8 = ColorTypes.FixedPointNumbers.N0f8
using Lava: @setup_workload, @compile_workload
using LavaDNN: SAM2, encode, decode, segment, prompt, toback

export SAM2Runner_VERSION, sam2model, runsam2, assetdir, sam2segmenter

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`LavaDNN.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = LavaDNN.KERNELS_VERSION

"""
    assetdir() -> String

Where the exported graphs and weights live.

Overridable with `JULIA_SAM2_ASSETS` because precompilation has to find them
without a running program to ask, and a checkout somewhere else is the normal
case rather than the exception.
"""
assetdir() = get(ENV, "JULIA_SAM2_ASSETS",
                 normpath(joinpath(@__DIR__, "..", "..", "..", "gen", "graphs", "sam2-large")))

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
produces is a matte *seed*: `LavaDNN.segment` documents the measurement, but in
short, argmax over a predicted score will hand back a mask that sits at the
threshold and speckles when it is resampled to frame resolution. `:best` remains
available and remains exactly what PyTorch does.
"""
function sam2segmenter(model::SAM2; pick = :confident)
    cachekey = Ref{Any}(nothing)
    cachefeats = Ref{Any}(nothing)
    host = Ref{Any}(nothing)
    res = model.res
    backend = model.model.backend
    return function (frame::AbstractMatrix, points; key = nothing)
        w, h = size(frame)
        host[] === nothing && (host[] = zeros(Float32, res, res, 3, 1))
        img = host[]
        # SAM resizes to a *square*, so normalized coordinates map straight
        # through and a click needs no aspect correction.
        @inbounds for j in 1:res, i in 1:res
            c = frame[clamp(round(Int, (i - 0.5) * w / res + 0.5), 1, w),
                      clamp(round(Int, (j - 0.5) * h / res + 0.5), 1, h)]
            img[i, j, 1, 1] = Float32(red(c))
            img[i, j, 2, 1] = Float32(green(c))
            img[i, j, 3, 1] = Float32(blue(c))
        end
        feats = if key !== nothing && cachekey[] == key && cachefeats[] !== nothing
            cachefeats[]
        else
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

"""
`(256, 256)` mask logits as a 0/255 mask at frame resolution: bilinear, then
thresholded at zero — SAM's own convention, and why the decoder returns logits
rather than probabilities.
"""
function maskatframe(lg::AbstractMatrix, w::Integer, h::Integer)
    mw, mh = size(lg)
    out = Matrix{UInt8}(undef, w, h)
    @inbounds for j in 1:h, i in 1:w
        fx = ((i - 0.5) / w) * mw + 0.5
        fy = ((j - 0.5) / h) * mh + 0.5
        x0 = clamp(floor(Int, fx), 1, mw); y0 = clamp(floor(Int, fy), 1, mh)
        x1 = min(x0 + 1, mw); y1 = min(y0 + 1, mh)
        tx = Float32(fx - x0); ty = Float32(fy - y0)
        v = (1 - tx) * (1 - ty) * lg[x0, y0] + tx * (1 - ty) * lg[x1, y0] +
            (1 - tx) * ty * lg[x0, y1] + tx * ty * lg[x1, y1]
        out[i, j] = v > 0 ? 0xff : 0x00
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
