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
"""
function scratchfor(m::Model, dims)
    get!(m.scratch, dims) do
        plans = Dict(n => planslab(g, dims) for (n, g) in m.graphs)
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
function call(m::Model, name::AbstractString, args...; dims, clampattn::Bool = false)
    g = m.graphs[name]
    length(args) == length(g.inputs) ||
        error("$name expects $(length(g.inputs)) inputs, got $(length(args))")
    slab, plans, ws, lazies, recs, _ = scratchfor(m, dims)
    rec = startcall!(recs[name])
    vals = execute!(g, Dict{String,Any}(zip(g.inputs, args)), m.weights;
                    dims, backend=m.backend, slab=slab, plan=plans[name], ws=ws,
                    lazy=lazies[name], rec=rec, diag=m.diag, clampattn)
    # The same recycler resolves the outputs: an output that is a view gets
    # materialised right here, and that copy needs a stable address as much as
    # anything inside the graph did. Ordinals carry on from where `execute!` left
    # them, which is deterministic because the op sequence is.
    ctx = Ctx(vals, g, dims, m.backend; slab, plan = plans[name], ws,
              lazy = lazies[name], rec, diag = m.diag, clampattn)
    Tuple(value(ctx, o) for o in g.outputs)
end
