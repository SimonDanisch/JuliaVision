"""
MatAnyone2 with no first-call latency.

The same treatment as `SAM2Runner`, for the propagator half of the matte: run a
real propagation during precompilation so both the Julia specialisation and the
SPIR-V are paid once, at `Pkg.precompile`.

Deliberately sharing `LavaDNN.KERNELS_VERSION` with every other model on this
runtime. The two networks overlap heavily — every elementwise op is the same
`Lava` broadcast kernel, every reduction the same AcceleratedKernels one — so a
kernel frozen by whichever workload runs first is a hit for the other. Versioned
per package they would each freeze their own copy of the same bytes.

The workload drives `step!` rather than `matte`, and drives it in the order the
editor does: a mask-ingest step, then the re-runs that settle the memory bank,
then a plain propagation step. Those are three *different* graphs
(`encode_mask_deep`, `encode_mask_shallow`, the memory-read path), and a workload
that only ran one of them would leave the others compiling on first use — which
at ~33 s per uncached kernel is the whole problem back again.
"""
module MatAnyoneRunner

using Lava, LavaDNN, KernelAbstractions
# The propagator reads the editor's frames, which are `Matrix{RGB{N0f8}}`.
using ColorTypes: red, green, blue
using Lava: @setup_workload, @compile_workload
using LavaDNN: Model, initstate, step!, toback

export matanyonemodel, runmatanyone, matanyonepropagator

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`LavaDNN.KERNELS_VERSION`, shared with every other model. Bump it there.
"""
const KERNELS_VERSION = LavaDNN.KERNELS_VERSION

"""
    assetdir() -> String

Where MatAnyone2's exported graphs live.
"""
assetdir() = get(ENV, "JULIA_MATANYONE_ASSETS",
                 normpath(joinpath(@__DIR__, "..", "..", "..", "gen", "graphs", "aten-autocast")))

"""Path to the weights, which sit beside the graph directory rather than in it."""
weightpath() = get(ENV, "JULIA_MATANYONE_WEIGHTS",
                   normpath(joinpath(@__DIR__, "..", "..", "..", "gen", "weights.safetensors")))

"""
    matanyonemodel(; backend, dir, weights) -> Model

Load the propagator. Separate from the workload body so the loading is not what
gets cached.
"""
function matanyonemodel(; backend = LavaBackend(),
                        dir::AbstractString = assetdir(),
                        weights::AbstractString = weightpath())
    return Model(dir, weights; backend)
end

"""
    runmatanyone(model, image, mask; warmup = 2) -> alpha

Seed on `image` with `mask`, settle, then propagate one more frame — the exact
sequence the editor's matte tool performs, and therefore the one whose kernels
have to be frozen.

`image` is `(W, H, 3, 1)` in 0..1 on the model's backend and `W`, `H` must be
multiples of 16 (the encoder downsamples by 16 and `initstate` sizes the memory
bank from `W ÷ 16`). `mask` is `(W, H)` in **0..255** — a 0/1 mask is 255x too
faint and yields an all-zero matte with no error anywhere.
"""
function runmatanyone(model::Model, image, mask; warmup::Int = 2)
    W, H = size(image, 1), size(image, 2)
    state = initstate(model, W, H)
    # Ingest is its own step, WITHOUT `firstframe`: the two together reach a
    # path where `State`'s `lastpixfeat`/`lastmskvalue` are still `nothing` and
    # fail to compile.
    step!(model, state, image; mask, firstframe = true)
    alpha = nothing
    for _ in 1:warmup
        alpha = step!(model, state, image; firstframe = true)
    end
    alpha = step!(model, state, image)          # the ordinary propagation path
    return alpha
end

"""
    padto16(n) -> Int

The encoder downsamples by 16, so both extents must be multiples of it.
"""
padto16(n::Integer) = ((Int(n) + 15) ÷ 16) * 16

"""
    matanyonepropagator(; graphdir, weights, backend, warmup = 10) -> f

A propagator matching `VideoEditor.registermatte!`'s contract:
`f(frames, seeds; progress) -> Array{UInt8,3}`.

**Lives in this package rather than in `examples/matanyone.jl` so it can be
precompiled.** Measured through the editor's seam, the first propagation cost
94.3 s, 74.4 s of it Julia inferring this function and everything it reaches —
with every kernel already frozen. Code in a script cannot be in a package image.
`VideoEditorRunner`'s workload drives it, which is the level that matters: the
inference has to be cached on the far side of `VideoEditor` being loaded, or
loading the editor invalidates it again.
"""
function matanyonepropagator(;
        graphdir::AbstractString = assetdir(),
        weights::AbstractString = weightpath(),
        backend = LavaBackend(),
        # Re-runs of a seeded frame, settling the memory bank before its own matte
        # is read. `inference_matanyone2.py` uses 10 and `LavaDNN.matte` matches
        # it; it is also what makes a single-frame call (the live preview while
        # marking) return a matte rather than the seed.
        warmup::Int = 10)
    isdir(graphdir) || error("no exported graphs at $graphdir — run tools/ first")
    isfile(weights) || error("no weights at $weights")
    # Built on FIRST USE, not here. A Vulkan `BatchQueue` is single-writer and
    # belongs to whichever thread first touches the context, while the editor
    # calls a propagator from `runanalysis` — its pinned GPU worker for a GPU
    # backend, a plain task for a CPU one. Constructing the model at registration
    # time binds the queue to whoever happened to call `usematanyone!` (usually
    # the REPL's main thread), and every later call then dies on
    # "BatchQueue is single-writer; cross-thread sweep forbidden". Building it
    # inside the call puts the context on the executor that will use it.
    modelref = Ref{Any}(nothing)

    return function (frames, seeds; progress = nothing)
        modelref[] === nothing && (modelref[] = Model(graphdir, weights; backend))
        model = modelref[]
        n = length(frames)
        w, h = size(frames[1])
        W, H = padto16(w), padto16(h)
        img = KA.allocate(backend, Float32, W, H, 3, 1)
        host = zeros(Float32, W, H, 3, 1)
        maskhost = zeros(Float32, W, H)
        state = initstate(model, W, H)
        out = zeros(UInt8, w, h, n)
        # One device buffer per frame, downloaded once at the end. `collect`ing
        # each frame's alpha as it came cost a full queue drain per frame — the
        # host waits for the GPU, then the GPU waits for the host to ask for the
        # next frame, and neither overlaps. Device-to-device copies do not
        # synchronise, so the pipeline stays full.
        planes = [KA.allocate(backend, Float32, W, H) for _ in 1:n]
        got = falses(n)
        order = sort!(collect(keys(seeds)))
        isempty(order) && return out
        first = order[1]

        for k in 1:n
            # pad by edge replication rather than zeros: a black border reads as
            # background the model has to explain away, and it leaks into the
            # matte at the frame edge
            frame = frames[k]
            @inbounds for j in 1:H, i in 1:W
                c = frame[min(i, w), min(j, h)]
                host[i, j, 1, 1] = Float32(red(c))
                host[i, j, 2, 1] = Float32(green(c))
                host[i, j, 3, 1] = Float32(blue(c))
            end
            copyto!(img, host)

            alpha = if haskey(seeds, k)
                m = seeds[k]
                fill!(maskhost, 0.0f0)
                @inbounds for j in 1:min(h, H), i in 1:min(w, W)
                    # 0..255, NOT 0..1: the reference mask the model was
                    # validated against is `(0.0, 255.0)`, and a 0/1 mask is 255x
                    # too faint to register — it produces an all-zero matte with
                    # no error anywhere, which is the whole difficulty of this bug.
                    maskhost[i, j] = m[i, j] > 0x7f ? 255.0f0 : 0.0f0
                end
                dev = KA.allocate(backend, Float32, W, H)
                copyto!(dev, maskhost)
                # Mask ingest is its own step, WITHOUT `firstframe`. The two
                # together hit a path where `State`'s `lastpixfeat`/`lastmskvalue`
                # are still `nothing` and reach a broadcast, which fails to
                # compile ("call to jl_f_throw_methoderror" inside
                # `lava_broadcast_flat!`). The verified driver order is
                # ingest-then-run, so do that.
                # `firstframe` on the FIRST mark only: it resets the memory
                # bank, which is right when seeding and wrong when correcting a
                # drifting matte further into the clip.
                seeding = k == first
                step!(model, state, img; mask = dev, firstframe = seeding)
                # The ingest step's own alpha IS the mask: `step!` overwrites
                # `prob` with the supplied selection outright, so taking it would
                # hand the user's rough box back as the matte — on exactly the
                # frame they are looking at after marking. Re-run the frame to get
                # a segmented one, which is what `LavaDNN.matte` does for its own
                # first frame and the reason its result looks nothing like this
                # one did.
                a = nothing
                for _ in 1:(seeding ? warmup : 1)
                    a = step!(model, state, img; firstframe = seeding)
                end
                a
            elseif k < first
                # nothing marked yet — leave these frames transparent rather than
                # running the model on a bank that has never been seeded
                progress === nothing || progress(k, n)
                continue
            else
                step!(model, state, img)
            end

            copyto!(planes[k], alpha)
            got[k] = true
            progress === nothing || progress(k, n)
        end
        for k in 1:n
            got[k] || continue
            a = collect(planes[k])
            @inbounds for j in 1:h, i in 1:w
                out[i, j, k] = round(UInt8, clamp(a[i, j], 0.0f0, 1.0f0) * 255)
            end
        end
        return out
    end
end


function __init__()
    Lava.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

@setup_workload begin
    dir, w = assetdir(), weightpath()
    ready = isdir(dir) && isfile(w) && isfile(joinpath(dir, "encode_image.json"))
    if ready
        try
            backend = LavaBackend()
            model = matanyonemodel(; backend, dir, weights = w)
            # Small, but a real shape: 16-multiples, and big enough that the
            # memory bank's reductions take their normal paths.
            W, H = 128, 96
            image = toback(backend, fill(0.5f0, W, H, 3, 1))
            host = zeros(Float32, W, H)
            host[(W ÷ 4):(3W ÷ 4), (H ÷ 4):(3H ÷ 4)] .= 255.0f0
            mask = toback(backend, host)

            @compile_workload KERNELS_VERSION begin
                alpha = runmatanyone(model, image, mask)
                Array(alpha)
                KA.synchronize(backend)
            end
        catch err
            @warn "MatAnyoneRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "MatAnyoneRunner: no assets — nothing precompiled" dir weights = w
    end
end

end # module
