"""
MatAnyone2 with no first-call latency.

The same treatment as `SAM2Runner`, for the propagator half of the matte: run a
real propagation during precompilation so both the Julia specialisation and the
SPIR-V are paid once, at `Pkg.precompile`.

Deliberately sharing `DNNKernels.KERNELS_VERSION` with every other model on this
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

using Lava, DNNKernels, KernelAbstractions
# The propagator reads the editor's frames, which are `Matrix{RGB{N0f8}}`.
using ColorTypes: red, green, blue
using Lava: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: Model, initstate, step!, toback, loadgraph, readsafetensors

export matanyonemodel, runmatanyone, matanyonepropagator
export matanyonegraph, matanyoneweights, matanyonerefs, matanyonemanifest
export matanyoneprecisions, ready

const KA = KernelAbstractions
# Through DNNKernels rather than as a direct dependency, the same way the
# parity test reaches it: one JSON3 on the runtime, not two.
const JSON3 = DNNKernels.JSON3

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model. Bump it there.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    artifactdir() -> String

The artifact's root: downloaded on first use and cached across every environment
on this machine.

**Changing these assets means re-binding the artifact**, not editing a directory.
Re-export, then `julia --project=. tools/make_artifacts.jl matanyone` — that hashes
the new content and rewrites `../Artifacts.toml`, so this call resolves to it
immediately. Uploading is only needed to publish it to anyone else.
"""
artifactdir() = @artifact_str("matanyone")

"""
    assetdir() -> String

The **graph** directory, which is `graphs/` INSIDE the artifact.

`Model(dir, weights)` reads `joinpath(dir, "\$name.json")`, so it needs the
directory the JSONs are in, and this artifact keeps them one level down. SAM 2's
is flat and its `assetdir()` is the artifact root — the two layouts differ, and a
uniform `assetdir() = @artifact_str(...)` is right for one and wrong for the
other.

That is not hypothetical: it shipped that way for a moment and the symptom was
silent. `@setup_workload` guards on `isfile(joinpath(dir, "encode_image.json"))`,
which was false, so the guard took its else branch and logged *"no assets —
nothing precompiled"* while naming a path inside a fully downloaded artifact.
The package still loaded, still precompiled, and simply did none of the work it
exists to do — and the message read like a deliberate skip.
"""
assetdir() = joinpath(artifactdir(), "graphs")

"""
    refsdir() -> String

The PyTorch reference activations the layer-by-layer parity test compares
against — a **separate** artifact from the weights, the way `sam2-large-refs`
sits beside `sam2-large`. `tools/make_artifacts.jl` deliberately keeps reference
tensors out of a model's own tarball, because someone matting a clip should not
download the test fixtures.

Not bound yet, and that is the point of this function existing. The parity test
used to reach these through a walk up the filesystem for a `gen/` tree, which
meant it ran on the machine that generated them and silently skipped everywhere
else — so the one check that catches a kernel which is fast and subtly wrong was
invisible on every machine that could have disagreed. Throwing here names the
missing artifact instead.
"""
function refsdir()
    toml = joinpath(dirname(@__DIR__), "Artifacts.toml")
    h = artifact_hash("matanyone-refs", toml)
    h === nothing && error("""
        the `matanyone-refs` artifact is not bound in $toml.
        The layer-by-layer PyTorch parity test needs refs-<precision>.safetensors,
        refs_manifest-<precision>.json and graphs/aten-<precision>/, none of which
        ship in the `matanyone` model artifact. Re-export and bind them:
            uv run tools/dump_refs.py --precision autocast --max-size 128
            uv run tools/dump_refs.py --precision fp32     --max-size 128
            julia --project=. tools/make_artifacts.jl matanyone-refs""")
    artifact_exists(h) || error("""
        `matanyone-refs` is bound but not installed, and its download failed.
        Check the URL in $toml resolves, or re-bind from a local export.""")
    return artifact_path(h)
end

"""
Path to the weights, which sit beside the graph directory in a generated tree
but *inside* the artifact, since an artifact is one directory.

**The artifact and nothing else.** This had three paths — an environment
variable, then the artifact, then a walk up the filesystem for a `gen/` tree —
and the first and third are exactly the two fallbacks `DNNKernels/src/assets.jl`
says were removed. To point at a locally re-exported tree, re-bind the artifact
(`tools/make_artifacts.jl matanyone`); that is the override, and it is the only
one. A second path can only disagree with the first, and the way it disagrees is
silent: on a machine that happens to have a `gen/` tree the walk always answered,
which is what kept a broken download invisible the last time this existed.
"""
function weightpath()
    p = joinpath(artifactdir(), "weights.safetensors")
    isfile(p) || error("""
        no weights.safetensors in the `matanyone` artifact ($(artifactdir())).
        Re-export and re-bind: `julia --project=. tools/make_artifacts.jl matanyone`.""")
    return p
end

# ── What callers outside this package may ask for ────────────────────────────
#
# `artifactdir`, `assetdir`, `refsdir` and `weightpath` are INTERNAL: they name
# where the artifacts happen to put things. Nothing outside this file calls them
# or `joinpath`s onto them, because a re-export that moves a file would then
# break callers that never knew they depended on the layout. Callers ask for the
# graph, the weights or the references, and `dir` is a config keyword defaulting
# to the artifact — the shape `BasicVSRRunner` and the other scaffolded runners
# already use.

"""
    matanyoneprecisions(; dir = refsdir()) -> Vector{String}

Which precisions have reference activations installed — a subset of
`("autocast", "fp32")`, and **empty** when the `matanyone-refs` artifact is not
bound at all.

Empty rather than throwing, so the parity test loops over nothing and records a
skip. That test is the one check that catches a kernel which is fast and subtly
wrong, and it must be visibly absent rather than quietly passing when its
fixtures are.
"""
function matanyoneprecisions(; dir::Union{Nothing,AbstractString} = nothing)
    d = dir === nothing ? refsdir_or_nothing() : dir
    d === nothing && return String[]
    return filter(p -> isfile(joinpath(d, "refs-$p.safetensors")), ["autocast", "fp32"])
end

"""
    refsdir_or_nothing() -> String | Nothing

`refsdir()` when the artifact is bound and installed, `nothing` when it is not.

A PREDICATE, not a `try`. This asked `refsdir()` and swallowed whatever came back
with a bare `catch`, which is the pattern that is forbidden here for a reason
this session demonstrated twice: `pipeline_exec_stats` hid a genuine
`ConstructionBase` failure behind `catch ex; @debug; return nothing` and it read
as "the AMD driver reports no statistics" for hours.

The specific harm of the version this replaces: an unbound artifact and a
*corrupt* one were indistinguishable, and so was a typo in this file. Testing the
two conditions directly means the only way to get `nothing` is the one intended.
"""
function refsdir_or_nothing()
    toml = joinpath(dirname(@__DIR__), "Artifacts.toml")
    h = artifact_hash("matanyone-refs", toml)
    h === nothing && return nothing        # not bound
    artifact_exists(h) || return nothing   # bound, not installed
    return artifact_path(h)
end

"""
    matanyonerefs(precision; dir = refsdir()) -> Dict

The PyTorch reference activations for one precision.
"""
function matanyonerefs(precision::AbstractString; dir::AbstractString = refsdir())
    p = joinpath(dir, "refs-$precision.safetensors")
    isfile(p) || throw(ArgumentError("MatAnyone references not found at $p"))
    return readsafetensors(p)
end

"""
    matanyonemanifest(precision; dir = refsdir()) -> Any

The manifest beside the references — resolution and per-tensor metadata.
"""
function matanyonemanifest(precision::AbstractString; dir::AbstractString = refsdir())
    p = joinpath(dir, "refs_manifest-$precision.json")
    isfile(p) || throw(ArgumentError("MatAnyone reference manifest not found at $p"))
    return JSON3.read(read(p, String))
end

"""
    matanyonegraph(name, precision; dir = refsdir()) -> Graph

One exported ATen graph at one precision. These travel with the references
rather than with the model, because the autocast/fp32 split exists to *localise a
fault* and only the tests read it.
"""
function matanyonegraph(name::AbstractString, precision::AbstractString;
                        dir::AbstractString = refsdir())
    p = joinpath(dir, "graphs", "aten-$precision", "$name.json")
    isfile(p) || throw(ArgumentError(
        "MatAnyone graph `$name` at precision `$precision` not found at $p"))
    return loadgraph(p)
end

"""
    matanyoneweights(; dir = artifactdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
matanyoneweights(; dir::AbstractString = artifactdir()) =
    readsafetensors(joinpath(dir, "weights.safetensors"))

"""
    ready(; dir = assetdir()) -> Bool

Whether the model assets are installed. Says nothing about the references, which
are a separate artifact — ask [`matanyoneprecisions`](@ref) for those.
"""
ready(; dir::AbstractString = assetdir()) = isfile(joinpath(dir, "encode_image.json"))

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
`VideoEditor`'s own precompile workload drives it, which is the level that
matters: the inference has to be cached on the far side of `VideoEditor` being
loaded, or loading the editor invalidates it again — and since the editor
depends on this package directly, the editor is that far side.
"""
function matanyonepropagator(;
        graphdir::AbstractString = assetdir(),
        weights::AbstractString = weightpath(),
        backend = LavaBackend(),
        # Re-runs of a seeded frame, settling the memory bank before its own matte
        # is read. `inference_matanyone2.py` uses 10 and `DNNKernels.matte` matches
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
                # a segmented one, which is what `DNNKernels.matte` does for its own
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
