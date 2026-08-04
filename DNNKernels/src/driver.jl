"""
The per-frame driver.

Everything the Python `InferenceCore.step` did with dicts and object managers,
as ordinary Julia control flow over a fixed variant set. The guards are all
functions of the frame index (tools/enumerate.py verified none depend on tensor
contents), so they are plain conditionals decided before submission - no
conditional rendering, no indirect dispatch (lava-dnn.md, Driver).

Four variants, matching the trace:

  v0  mask ingest      encode_image, transform_key, encode_mask(deep)
  v1  first_frame_pred as v0 plus a segment that reads memory *without*
                       affinity (read_first_frame reuses last_msk_value)
  v2  normal frame     full memory read, encode_mask(shallow)
  v3  memory frame     full memory read, encode_mask(deep), bank update
"""

# Parameterised on the BACKEND TYPE, not `::Any`. With `backend::Any` every
# `ctx.backend` launch is a dynamic dispatch, so inference has to consider every
# `KA.Kernel{B}` method — including `Kernel{CPU}` and `KA.__run`, whose arguments
# are all `Any`. Those land in the package image even though nothing here ever
# runs on the CPU, and their call edges cover whole method tables, so loading any
# package that adds methods throws them away and everything inferred through them
# with it. Measured: `KA.__run` alone was 4 062 of the 20 200 extra CodeInstances
# rejected when SAM 2's image loads after VideoEditor.
struct Model{B}
    graphs::Dict{String,Graph}
    weights::Dict{String,Any}
    backend::B
    memevery::Int
    memframes::Int
    topk::Int
    # dims -> (slab, per-graph plans). Computed on first use at a resolution and
    # reused; the plan depends only on the graph and the resolved shapes.
    scratch::Dict{Any,Any}
    # Per-run instrumentation, off by default and free when off. On the model
    # rather than in a module `Ref` so two models in one process can be measuring
    # different things — `m.diag.optimes = Dict{String,Tuple{Int,Float64}}()` and
    # every graph this model runs starts accumulating. See `Diagnostics`.
    diag::Diagnostics
end

Model(graphs, weights, backend, memevery, memframes, topk) =
    Model(graphs, weights, backend, memevery, memframes, topk, Dict{Any,Any}(),
          Diagnostics())
Model(graphs, weights, backend, memevery, memframes, topk, scratch) =
    Model(graphs, weights, backend, memevery, memframes, topk, scratch, Diagnostics())

"""
    scratchfor(m, dims) -> (slab, plans, workspace, lazies, recyclers)

One slab for every graph at this resolution, sized to the largest peak. The
graphs run one after another inside a step, so they can share it; values that
escape a graph are excluded from the plan (see `escaping`).

The recyclers are *per graph*, unlike the slab and the workspace: their ordinals
count allocations within one graph call, so sharing one across graphs would make
a graph's ordinals depend on what ran before it in the step — and the step has
two shapes (`encode_mask_deep` on every fifth frame, `encode_mask_shallow`
otherwise), which would shift every ordinal after that point on alternate steps.

**Only graphs whose symbols `dims` resolves are planned.** A model's graphs need
not share an axis: Kokoro's text half is symbolic in the token count and its
vocoder in the frame count, and the frame count is not known until the text half
has run — the model predicts it. Planning every graph at every `dims` made that
model impossible to call at all, with a `FieldError` naming the missing field
rather than the graph that wanted it.
"""
function scratchfor(m::Model, dims)
    get!(m.scratch, dims) do
        plans = Dict(n => planslab(g, dims)
                     for (n, g) in m.graphs if all(s -> haskey(dims, Symbol(s)), g.symbols))
        nb = maximum(p -> p.bytes, values(plans); init = 0)
        slab = KernelAbstractions.allocate(m.backend, UInt8, max(nb, 1))
        @debug "DNNKernels: static scratch slab $(round(nb/2^20, digits=2)) MB at $dims"
        # One workspace for every graph at this resolution: it is reset per op,
        # so the graphs cannot collide over it any more than two ops can.
        # ... and one more for `step!` itself, which materialises the alpha and
        # the mask it carries to the next frame outside of any graph.
        (slab, plans, Workspace(m.backend), fusablesets(m.graphs),
         Dict(n => Recycler() for n in keys(m.graphs)), Recycler())
    end
end

"""
    toback(backend, a) -> array

Move a host array onto the execution backend. A no-op on the CPU backend, and a
no-op for an array that already lives on `backend` — the sampler hands its latent
straight back to the transformer, so without that check every step would download
and re-upload it.

Residency is judged by the *kind* of backend rather than by equality: KA's
`get_backend` rebuilds the descriptor, so `get_backend(a) == backend` is false
even for an array allocated on exactly that device.
"""
toback(::KernelAbstractions.CPU, a::AbstractArray) = a isa Array ? a : collect(a)
function toback(backend, a::AbstractArray)
    isempty(a) && return a
    KernelAbstractions.get_backend(a) isa typeof(backend) && return a
    d = KernelAbstractions.allocate(backend, eltype(a), size(a)...)
    copyto!(d, collect(a))
    d
end

"""
    Model(graphdir, weightpath; names, backend, ...)

Load a graph set and its weights, and run every host-side preparation pass over
them.

`names` selects which graphs in `graphdir` to load; it defaults to MatAnyone's
eight because that is what `step!` drives. Everything from here down is
model-agnostic — the passes, the slab planner, `call` — so another model is a
different `names` and its own driver, not another `Model`.
"""
function Model(graphdir::AbstractString, weightpath::AbstractString;
               backend=KernelAbstractions.CPU(), memevery=5, memframes=5, topk=30,
               names = ["encode_image", "transform_key", "encode_mask_deep",
                        "encode_mask_shallow", "pixel_fusion", "pred_uncertainty",
                        "segment", "readout_query"])
    graphs = Dict(n => loadgraph(joinpath(graphdir, "$n.json")) for n in names)
    # Host-side graph preparation, in order. Folding runs *before* the casts are
    # hoisted so it sees the fp32 master weights through `weightsource` and
    # rounds to the declared dtype exactly once; hoisting then turns every
    # remaining constant cast into a plain weight; the sweep removes whatever
    # both of them orphaned. All of it before upload, so only the final weights
    # ever reach the device.
    graphs, host, nfold = foldbatchnorm(graphs, readsafetensors(weightpath))
    graphs, nact = foldrelu(graphs)
    graphs, host, nhoist = hoistcasts(graphs, host)
    # After the casts: under autocast a weight's transposed view sits on top of
    # its fp16 cast, and hoisting the cast first turns that into a plain weight
    # this pass can then permute.
    graphs, host, nperm = hoistpermutes(graphs, host)
    graphs, host, nconst = hoistconstants(graphs, host)
    # The other side of `hoistcasts`: a cast that narrows something the graph
    # just computed, rather than a constant. After the weight passes, because
    # this one only ever looks at computed values and there is no point offering
    # it buffers the passes above are about to turn into weights.
    graphs, noutcast = foldoutcasts(graphs)
    graphs, ndead = dropdead(graphs)
    # Upload only the weights the surviving graphs still name. `dropdead` prunes
    # dead *ops*; without this the host dict keeps every orphan those passes
    # created — above all the fp32 masters whose `_to_copy` `hoistcasts` turned
    # into a plain fp16 weight. Uploading them anyway cost 849 MB of VRAM on
    # SAM 2 (1852 MB of weights resident against 1003 MB of parameters), for
    # tensors no op reads.
    live = livekeys(graphs)
    dropped = length(host) - count(k -> k in live, keys(host))
    host = Dict{String,Any}(k => v for (k, v) in host if k in live)
    weights = Dict{String,Any}(k => toback(backend, v) for (k, v) in host)
    # The one pass that has to *run* the ops it folds, so it comes after the
    # upload and works on the device weights: constant subgraphs, not just the
    # nullary constants `hoistconstants` took above. Then the same sweep again,
    # because folding a subgraph orphans whatever only it read.
    graphs, weights, nsub = hoistconstants(graphs, weights, backend)
    if nsub > 0
        graphs, nsubdead = dropdead(graphs)
        live2 = livekeys(graphs)
        weights = Dict{String,Any}(k => v for (k, v) in weights if k in live2)
        ndead += nsubdead
    end
    @debug "DNNKernels: folded $nfold batch-norms, $nact relus and $noutcast output casts, hoisted $nhoist casts, $nperm permutes, $nconst constants and $nsub constant-subgraph ops, dropped $ndead dead ops and $dropped orphaned weights"
    Model(graphs, weights, backend, memevery, memframes, topk)
end

"""
    livekeys(graphs) -> Set{String}

Host-weight keys still named in a graph's `order` after `dropdead`. Everything
else in the weight dict is an orphan of the rewrite passes — above all the fp32
masters whose `_to_copy` `hoistcasts` replaced — and must not be uploaded.
"""
function livekeys(graphs)
    live = Set{String}()
    for g in values(graphs), id in g.order
        b = get(g.buffers, id, nothing)
        b !== nothing && b.kind === :weight && !isempty(b.key) && push!(live, b.key)
    end
    live
end


"""Run one graph and return its outputs in declaration order."""
function call(m::Model, name::AbstractString, args...; dims, clampattn::Bool = false,
              noise::NoiseSource = RandomNoise())
    g = m.graphs[name]
    length(args) == length(g.inputs) ||
        error("$name expects $(length(g.inputs)) inputs, got $(length(args))")
    missing_ = filter(s -> !haskey(dims, Symbol(s)), g.symbols)
    isempty(missing_) || error(
        "$name is symbolic in $(join(g.symbols, ", ")) but dims = $dims " *
        "does not give $(join(missing_, ", "))")
    slab, plans, ws, lazies, recs, _ = scratchfor(m, dims)
    rec = startcall!(recs[name])
    vals = execute!(g, Dict{String,Any}(zip(g.inputs, args)), m.weights;
                    dims, backend=m.backend, slab=slab, plan=plans[name], ws=ws,
                    lazy=lazies[name], rec=rec, diag=m.diag, clampattn, noise)
    # The same recycler resolves the outputs: an output that is a view gets
    # materialised right here, and that copy needs a stable address as much as
    # anything inside the graph did. Ordinals carry on from where `execute!` left
    # them, which is deterministic because the op sequence is.
    ctx = Ctx(vals, g, dims, m.backend; slab, plan = plans[name], ws,
              lazy = lazies[name], rec, diag = m.diag, clampattn, noise)
    Tuple(value(ctx, o) for o in g.outputs)
end

mutable struct State
    bank::MemoryBank
    sensory::Any            # (w, h, S, NOBJ, B)
    lastmask::Any           # (W, H, NOBJ, B) full resolution
    lastpixfeat::Any
    lastmskvalue::Any       # (w, h, CV, NOBJ, B)
    ti::Int                 # curr_ti
    lastmemti::Int
    dims::NamedTuple
end

"""
    initstate(model, W, H; sensory_dim, ...) -> State

`W`, `H` are the padded full-resolution extents (multiples of 16).
"""
function initstate(m::Model, W::Int, H::Int; ck=64, cv=256, sensory=256,
                   nobj=1, bs=1, q=16, embed=256, T=Float32)
    w, h = W ÷ 16, H ÷ 16
    State(MemoryBank(m.backend, T, w * h, m.memframes, ck, cv, nobj, bs, q, embed),
          fill!(KernelAbstractions.allocate(m.backend, T, w, h, sensory, nobj, bs), 0),
          fill!(KernelAbstractions.allocate(m.backend, T, W, H, nobj, bs), 0),
          nothing, nothing, -1, 0, (h=h, w=w))
end

"""
    step!(model, state, image; mask=nothing, firstframe=false) -> alpha

`image` is `(W, H, 3, 1)` in [0,1]; normalisation happens inside encode_image.
`mask` is `(W, H)` in [0,255] and only on the very first call. Returns the alpha
matte as `(W, H)`.
"""
function step!(m::Model, s::State, image; mask=nothing, firstframe::Bool=false)
    dims = s.dims
    s.ti += 1
    # One flip per step, before any graph runs: this step's outputs land in the
    # other bank from the values it still has to read out of the last one.
    recs, steprec = scratchfor(m, dims)[5:6]
    foreach(flip!, values(recs))
    startcall!(flip!(steprec))

    # inference_core.py:288-301
    ismem = ((s.ti - s.lastmemti >= m.memevery) || mask !== nothing)
    needseg = mask === nothing
    if firstframe
        s.ti = 0
        s.lastmemti = 0
        ismem = true
        # NOT `needseg = true`. A supplied mask already *is* the segmentation —
        # the `mask !== nothing` branch below overwrites `prob` outright — so
        # segmenting first is wasted work, and on a fresh state it is a crash:
        # `s.ti == 0` sends `s.lastmskvalue` into `pixel_fusion`, and `initstate`
        # leaves that `nothing`, which reaches `lava_broadcast_flat!` as a
        # `LavaRefValue{Nothing}` and fails to compile
        # ("call to jl_f_throw_methoderror") rather than erroring in Julia.
        # Keeping the mask authoritative makes `step!(…; mask, firstframe = true)`
        # both legal and cheaper — it is the natural way to seed a clip.
        needseg = mask === nothing
    end

    f16, f8, f4, f2, f1, pixfeat = call(m, "encode_image", image; dims)
    key, shrinkage, selection = call(m, "transform_key", f16; dims)

    prob = nothing
    if needseg
        readout = if s.ti == 0
            # read_first_frame (memory_manager.py:115): no affinity, the
            # previous mask value is reused directly
            first(call(m, "pixel_fusion", pixfeat, s.lastmskvalue, s.sensory, s.lastmask; dims))
        else
            visual = readmemory(Ctx(m.backend; diag=m.diag), s.bank, key, selection,
                                dims.w, dims.h; topk=m.topk)
            # temporal-sparsity blend (memory_manager.py:249). Slices go through
            # `view` + broadcast rather than `getindex`; see `materialize`.
            diff = view(visual, :, :, :, 1, :) .- view(s.lastmskvalue, :, :, :, 1, :)
            p = first(call(m, "pred_uncertainty", s.lastpixfeat, pixfeat, s.lastmask, diff; dims))
            pu = reshape(p, size(p, 1), size(p, 2), 1, 1, size(p, 4))
            visual = visual .* pu .+ s.lastmskvalue .* (1 .- pu)
            first(call(m, "pixel_fusion", pixfeat, visual, s.sensory, s.lastmask; dims))
        end
        objmem = reshape(s.bank.objmem, size(s.bank.objmem, 1), size(s.bank.objmem, 2), 1,
                         size(s.bank.objmem, 3), size(s.bank.objmem, 4))
        memreadout = first(call(m, "readout_query", readout, objmem; dims))
        newsensory, prob = call(m, "segment", f16, f8, f4, f2, f1, memreadout, s.sensory; dims)
        s.sensory = newsensory
    end

    if mask !== nothing
        # matting path (inference_core.py:357): prob = [1-m, m], m in [0,1]
        a = reshape(mask, size(mask, 1), size(mask, 2), 1, 1) ./ 255
        prob = cat(1 .- a, a; dims=3)
    end

    s.lastmask = materialize(steprec, m.backend, view(prob, :, :, 2:2, :))
    s.lastpixfeat = pixfeat

    if ismem
        # first_frame_pred clears the temporary memory before re-adding
        firstframe && reset!(s.bank)
        mv, sens, summaries = call(m, "encode_mask_deep", image, pixfeat, s.sensory, s.lastmask; dims)
        add!(s.bank, key, shrinkage, mv, summaries)
        s.sensory = sens
        s.lastmskvalue = mv
        s.lastmemti = s.ti
    else
        mv, _, _ = call(m, "encode_mask_shallow", image, pixfeat, s.sensory, s.lastmask; dims)
        s.lastmskvalue = mv
    end

    materialize(steprec, m.backend, view(prob, :, :, 2, 1))
end

"""
    matte(model, frames, mask; warmup=10) -> Array{Float32,3}

The full clip. `frames` is `(W, H, 3, T)` in [0,1], `mask` is `(W, H)` in
[0,255]. Mirrors inference_matanyone2.py:90-104, including the warm-up that
re-runs the first frame to settle the memory.

Both ends of the loop reuse their device buffers, which is worth more than it
sounds: `toback` per frame is an allocation *and* an upload at 1.50 ms against
0.36 ms for a `copyto!` into a buffer that already exists, on a step of ~20 ms.
The editor's own propagator has always done it this way.

`chunk` frames of alpha are likewise held on the device and downloaded together,
so the queue is not drained every step. The chunk bounds what that costs in VRAM,
at `W*H*4` bytes a frame.
"""
function matte(m::Model, frames, mask; warmup::Int=10, chunk::Int=16)
    W, H, _, nframes = size(frames)
    T = eltype(frames)
    s = initstate(m, W, H; T)
    out = Array{T}(undef, W, H, nframes)
    gmask = toback(m.backend, collect(mask))
    img = KernelAbstractions.allocate(m.backend, T, W, H, 3, 1)
    hostimg = Array{T}(undef, W, H, 3, 1)
    planes = [KernelAbstractions.allocate(m.backend, T, W, H) for _ in 1:chunk]
    held = 0                                     # frames sitting in `planes`
    function flush!(upto)
        for k in 1:held
            out[:, :, upto - held + k] = collect(planes[k])
        end
        held = 0
    end
    for ti in 1:nframes
        copyto!(hostimg, view(frames, :, :, :, ti:ti))
        copyto!(img, hostimg)
        alpha = if ti == 1
            step!(m, s, img; mask=gmask)
            for _ in 1:warmup
                step!(m, s, img; firstframe=true)
            end
            step!(m, s, img; firstframe=true)
        else
            step!(m, s, img)
        end
        if alpha isa Array
            out[:, :, ti] = alpha
        else
            copyto!(planes[held += 1], alpha)
            held == chunk && flush!(ti)
        end
    end
    flush!(nframes)
    out
end
