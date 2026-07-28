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

struct Model
    graphs::Dict{String,Graph}
    weights::Dict{String,Any}
    backend::Any
    memevery::Int
    memframes::Int
    topk::Int
    # dims -> (slab, per-graph plans). Computed on first use at a resolution and
    # reused; the plan depends only on the graph and the resolved shapes.
    scratch::Dict{Any,Any}
end

Model(graphs, weights, backend, memevery, memframes, topk) =
    Model(graphs, weights, backend, memevery, memframes, topk, Dict{Any,Any}())

"""
    scratchfor(m, dims) -> (slab, plans)

One slab for every graph at this resolution, sized to the largest peak. The
graphs run one after another inside a step, so they can share it; values that
escape a graph are excluded from the plan (see `escaping`).
"""
function scratchfor(m::Model, dims)
    get!(m.scratch, dims) do
        plans = Dict(n => planslab(g, dims) for (n, g) in m.graphs)
        nb = maximum(p -> p.bytes, values(plans); init = 0)
        slab = KernelAbstractions.allocate(m.backend, UInt8, max(nb, 1))
        @debug "LavaDNN: static scratch slab $(round(nb/2^20, digits=2)) MB at $dims"
        # One workspace for every graph at this resolution: it is reset per op,
        # so the graphs cannot collide over it any more than two ops can.
        (slab, plans, Workspace(m.backend), fusablesets(m.graphs))
    end
end

"""
    toback(backend, a) -> array

Move a host array onto the execution backend. A no-op on the CPU backend.
"""
toback(::KernelAbstractions.CPU, a::AbstractArray) = a isa Array ? a : collect(a)
function toback(backend, a::AbstractArray)
    isempty(a) && return a
    d = KernelAbstractions.allocate(backend, eltype(a), size(a)...)
    copyto!(d, collect(a))
    d
end

function Model(graphdir::AbstractString, weightpath::AbstractString;
               backend=KernelAbstractions.CPU(), memevery=5, memframes=5, topk=30)
    names = ["encode_image", "transform_key", "encode_mask_deep", "encode_mask_shallow",
             "pixel_fusion", "pred_uncertainty", "segment", "readout_query"]
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
    graphs, ndead = dropdead(graphs)
    @debug "LavaDNN: folded $nfold batch-norms and $nact relus, hoisted $nhoist casts, $nperm permutes and $nconst constants, dropped $ndead dead ops"
    weights = Dict{String,Any}(k => toback(backend, v) for (k, v) in host)
    Model(graphs, weights, backend, memevery, memframes, topk)
end

"""Run one graph and return its outputs in declaration order."""
function call(m::Model, name::AbstractString, args...; dims)
    g = m.graphs[name]
    length(args) == length(g.inputs) ||
        error("$name expects $(length(g.inputs)) inputs, got $(length(args))")
    slab, plans, ws, lazies = scratchfor(m, dims)
    vals = execute!(g, Dict{String,Any}(zip(g.inputs, args)), m.weights;
                    dims, backend=m.backend, slab=slab, plan=plans[name], ws=ws,
                    lazy=lazies[name])
    ctx = Ctx(vals, g, dims, m.backend)
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

    # inference_core.py:288-301
    ismem = ((s.ti - s.lastmemti >= m.memevery) || mask !== nothing)
    needseg = mask === nothing
    if firstframe
        s.ti = 0
        s.lastmemti = 0
        ismem = needseg = true
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
            visual = readmemory(s.bank, key, selection, dims.w, dims.h;
                                topk=m.topk, backend=m.backend)
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

    s.lastmask = materialize(view(prob, :, :, 2:2, :))
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

    materialize(view(prob, :, :, 2, 1))
end

"""
    matte(model, frames, mask; warmup=10) -> Array{Float32,3}

The full clip. `frames` is `(W, H, 3, T)` in [0,1], `mask` is `(W, H)` in
[0,255]. Mirrors inference_matanyone2.py:90-104, including the warm-up that
re-runs the first frame to settle the memory.
"""
function matte(m::Model, frames, mask; warmup::Int=10)
    W, H, _, nframes = size(frames)
    s = initstate(m, W, H; T=eltype(frames))
    out = Array{eltype(frames)}(undef, W, H, nframes)
    gmask = toback(m.backend, collect(mask))
    for ti in 1:nframes
        img = toback(m.backend, collect(view(frames, :, :, :, ti:ti)))
        alpha = if ti == 1
            step!(m, s, img; mask=gmask)
            for _ in 1:warmup
                step!(m, s, img; firstframe=true)
            end
            step!(m, s, img; firstframe=true)
        else
            step!(m, s, img)
        end
        out[:, :, ti] = alpha isa Array ? alpha : Array(alpha)
    end
    out
end
