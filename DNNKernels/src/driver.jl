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
    # Whether this model's graphs are recorded into a Mantle plan and replayed —
    # see [`record`](@ref) for what a model has to satisfy to say yes. On the
    # model and not a module `Ref` for the same reason `diag` is: it is a
    # property of a model, two of them in one process disagree, and a global
    # would make one of them silently wrong.
    record_maxpasses::Dict{String,Int}
end

# `record` was a field and a keyword here. It is not a choice any more: `call`
# declares the graph, plans it and replays the plan, and there is no other path
# for it to take. The keyword is still ACCEPTED and ignored, because four call
# sites in the runners and the tools pass it and a model that asks for the only
# behaviour there is should not be an error.
Model(graphs, weights, backend, memevery, memframes, topk;
      record::Bool = true, record_maxpasses = Dict{String,Int}()) =
    Model(graphs, weights, backend, memevery, memframes, topk, Dict{Any,Any}(),
          Diagnostics(), Dict{String,Int}(record_maxpasses))
Model(graphs, weights, backend, memevery, memframes, topk, scratch;
      record::Bool = true, record_maxpasses = Dict{String,Int}()) =
    Model(graphs, weights, backend, memevery, memframes, topk, scratch,
          Diagnostics(), Dict{String,Int}(record_maxpasses))

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
    # `copyto!(d, a)`, NOT `copyto!(d, collect(a))`. `collect` on an `Array`
    # returns a COPY, so that form allocates a full anonymous duplicate of every
    # tensor on its way to the device. Invisible at SAM 2's 943 MB;
    # at K2 Horizon 32B's 64.78 GiB it is a second 64.78 GiB of host memory on
    # a unified-memory APU where host and device share one 122 GiB pool, and
    # the process is OOM-killed with no frame naming the cause.
    #
    # It also silently defeated mmap-backed weights: the mapping costs nothing
    # to read, and `collect` faulted every page into anonymous memory anyway.
    # A strided or lazy source still needs materialising, so only those collect.
    copyto!(d, a isa DenseArray ? a : collect(a))
    d
end

"""
    RowCat(parts)

Several weights stacked along their FIRST Julia axis, lazily.

`fuseqkv` replaces a group of matmuls that share an activation with one matmul
over the stacked weight. Stacking on the host would materialise it — 272 MB for a
gate/up pair, 17 GB over 64 layers — so this carries the parts and `toback`
assembles them straight into the device buffer, one at a time.
"""
struct RowCat{T,P} <: AbstractMatrix{T}
    parts::P
    m::Int
    k::Int
end
function RowCat(parts::Vector)
    k = size(first(parts), 2)
    all(p -> size(p, 2) == k, parts) ||
        throw(DimensionMismatch("RowCat: parts disagree on their second axis"))
    RowCat{eltype(first(parts)),typeof(parts)}(parts, sum(p -> size(p, 1), parts), k)
end
Base.size(A::RowCat) = (A.m, A.k)

# Assembled on the DEVICE, part by part, so the stacked weight never exists on
# the host. Each part is uploaded (or materialised, if it is one of
# `hoistpermutes`' lazy transposes) into its own row range and then dropped.
function toback(backend, A::RowCat)
    d = KernelAbstractions.allocate(backend, eltype(A), size(A)...)
    off = 0
    for p in A.parts
        rows = size(p, 1)
        copyto!(view(d, (off + 1):(off + rows), :), toback(backend, p))
        off += rows
    end
    d
end

# `hoistpermutes` leaves its transposed weights lazy so they are not all
# materialised at once. This is where one of them becomes real, and WHERE it
# happens is the whole cost of a cold load.
#
# `permutedims` on the host walks the mmap'd checkpoint by strides and measured
# 0.33 GiB/s, against 21.8 for the contiguous upload of the same tensor. Every
# large weight in a decoder arrives transposed, so the whole 64.78 GiB of K2
# Horizon 32B went through it: 224.7 s of a 227 s model build, disk and cache
# misses, with the device idle throughout.
#
# Uploading the parent as it lies and transposing on the DEVICE, through the
# same tiled kernel attention uses for its operands, moves that to 0.6 s of
# upload plus a dispatch. `permutedims!` from GPUArrays would also work and is
# one line, but it is an elementwise gather, and a 1.28 G-element one hard-hung
# the device here.
toback(backend, a::PermutedDimsArray{T,N,perm}) where {T,N,perm} =
    toback(backend, permutedims(parent(a), perm))

# The one that carries a decoder's weights. `Int32` indexing and an
# `(E, L, H, B)` operand are the kernel's contract, so a parent it cannot
# address falls back to the host transpose above: slow, and always right.
function toback(backend, a::PermutedDimsArray{T,2,(2,1)}) where {T}
    p = parent(a)
    T in (Float16, Float32) && length(p) <= typemax(Int32) ||
        return toback(backend, permutedims(p, (2, 1)))
    src = toback(backend, p)
    E, L = size(p)
    d = KernelAbstractions.allocate(backend, T, L, E)
    st = map(Int32, strides(src))
    k = T === Float16 ? toLE_tiled_Float16! : toLE_tiled_Float32!
    k(backend, (32, 4, 1))(reshape(d, L, E, 1, 1), reshape(src, length(src)),
        Int32(1), st[1], st[2], Int32(0), Int32(0), Int32(E), Int32(L), Int32(1);
        ndrange = (32 * cld(E, 32), 4 * cld(L, 32), 1))
    d
end

"""
    Model(graphs, weights; backend, ...)

Run every host-side preparation pass over an already-loaded graph set and its
weights, then upload what survives.

**Takes loaded objects, not paths.** Reading the JSONs and the safetensors is the
owning package's job — `SAM2Runner.sam2graphs()` / `sam2weights()` and the
equivalents beside them — because that package is the only thing that knows
which artifact holds them and which graphs it wants. Taking `(graphdir,
weightpath)` plus a list of JSON names to open would put model-specific
knowledge in the generic runtime, purely because the constructor did the
loading.

Two things follow. A model whose weights arrive as
several files — Hunyuan3D's denoiser is 6.1 GB, past what one GitHub release
asset can hold — merges them in its own `*weights()` and nothing here changes.
And there is exactly one way to hand a model its weights, rather than a path
form and a dict form that drift.
"""
function Model(graphs::Dict{String,Graph}, weights::AbstractDict;
               backend=KernelAbstractions.CPU(), memevery=5, memframes=5, topk=30,
               # Off is how the fusion passes get checked: build the model twice
               # and compare the numbers. Every other pass here is verified
               # against an invariant or a PyTorch reference, but a fusion is
               # verified against *the same model unfused*, and there has to be a
               # way to ask for that. It is also the switch to reach for first if
               # a model ever comes out wrong — see `2026-08-08-elementwise-fusion.md`
               # for why a bad fusion looks like a precision bug.
               fuse::Bool = true,
               # Store the matmul weights as int8 with a per-output-channel
               # scale. Decode reads every weight once per token and is
               # bandwidth-bound outright, so this is a straight halving of the
               # floor; see `quant.jl` for the scheme and what it costs.
               quantize::Bool = false,
               # Record each graph into a Mantle plan and replay it. See
               # [`record`](@ref) — it is off by default because it is not free
               # of preconditions, and a model that does not meet them comes out
               # WRONG rather than slow.
               record::Bool = false, record_maxpasses = Dict{String,Int}())
    # Host-side graph preparation, in order. Folding runs *before* the casts are
    # hoisted so it sees the fp32 master weights through `weightsource` and
    # rounds to the declared dtype exactly once; hoisting then turns every
    # remaining constant cast into a plain weight; the sweep removes whatever
    # both of them orphaned. All of it before upload, so only the final weights
    # ever reach the device.
    # Three phases, timed, because a cold load of a 32B checkpoint is minutes
    # long and reported nothing about where they went. Host passes read the
    # checkpoint through its mmap and can COPY it — `fuseqkv` stacking 1152
    # groups materialises tens of GiB — so the first number is disk as much as
    # it is CPU.
    t0 = time_ns()
    graphs, host, nfold = foldbatchnorm(graphs, Dict{String,Any}(weights))
    graphs, nact = foldrelu(graphs)
    graphs, host, nhoist = hoistcasts(graphs, host)
    # After the casts: under autocast a weight's transposed view sits on top of
    # its fp16 cast, and hoisting the cast first turns that into a plain weight
    # this pass can then permute.
    graphs, host, nperm = hoistpermutes(graphs, host)
    graphs, host, nconst = hoistconstants(graphs, host)
    # After the permutes are materialised, because the stack is over the weights
    # in the layout the GEMM reads, and before the live-key sweep so the parts
    # that are now only reachable through the stack get dropped.
    graphs, host, nqkv = fuseqkv(graphs, host)
    nqkv > 0 && @info "fuseqkv: $nqkv matmul group(s) stacked"
    # The other side of `hoistcasts`: a cast that narrows something the graph
    # just computed, rather than a constant. After the weight passes, because
    # this one only ever looks at computed values and there is no point offering
    # it buffers the passes above are about to turn into weights.
    graphs, noutcast = foldoutcasts(graphs)
    # And the other direction: a cast that *widens* a computed value in front of
    # a reduction whose accumulator is already the wide type. After
    # `foldoutcasts`, because narrowing a producer's result can turn a cast that
    # was a no-op into a widening one this pass can then remove.
    graphs, nincast = foldincasts(graphs)
    # Before `fuseops`, which would collapse the norm's `add`+`rsqrt` into a
    # `FusedOp` and hide the epsilon this needs to read.
    graphs, nswi = fuseswiglu(graphs)
    nswi > 0 && @info "fuseswiglu: $nswi SwiGLU(s) -> one op each"
    graphs, nrms = fusegroupedrms(graphs)
    nrms > 0 && @info "fusegroupedrms: $nrms grouped RMS norm(s) -> one op each"
    # Before `dropdead`, which would otherwise see the `cat` as live and keep the
    # copies it feeds. After the weight passes, because it only looks at inputs
    # and outputs and nothing above changes those.
    graphs, ncache = foldcacheupdate(graphs)
    ncache > 0 && @info "foldcacheupdate: $ncache KV cache(s) updated in place"
    # After `foldcacheupdate`: a rotary embedding whose only consumer is an
    # in-place cache write can store into the cache itself, and that is only
    # visible once the write has been marked in place.
    graphs, nrope = fuserope(graphs)
    nrope > 0 && @info "fuserope: $nrope rotary embedding(s) -> one op each"
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
    # Upload one tensor at a time, dropping each host copy as it lands. A
    # comprehension holds BOTH dicts alive at once, so peak is twice the
    # weights: fine at SAM 2's 943 MB, fatal at K2 Horizon 32B's
    # 64.78 GiB, which needs 129.6 GiB against a 103.9 GiB cgroup cap and is
    # killed by the OOM reaper with no Julia frame naming the cause.
    #
    # `host` is dead after this line, so emptying it as we go is safe. The
    # periodic `GC.gc()` is what actually returns the pages — dropping the
    # reference alone leaves them to the collector's own schedule, which for
    # 580 tensors of ~100 MB is far too slow to keep under the cap.
    weights = Dict{String,Any}()
    thost = (time_ns() - t0) / 1e9
    qk = quantize ? quantkeys(graphs) : Set{String}()
    let ks = collect(keys(host)), n = 0, nq = 0
        for k in ks
            d = toback(backend, host[k])
            # Quantise ON THE DEVICE, from the copy that just landed, and drop
            # the float immediately: the host never sees 33.5 G elements and the
            # peak is one weight above the int8 total rather than both sets.
            if k in qk && ndims(d) == 2 && eltype(d) <: AbstractFloat
                weights[k] = quantizeint8(backend, d)
                nq += 1
            else
                weights[k] = d
            end
            d = nothing
            delete!(host, k)
            n += 1
            n % 32 == 0 && GC.gc(false)
        end
        empty!(host)
        GC.gc()
        quantize && @info "Model: $nq of $(length(ks)) weights stored as int8"
    end
    tupload = (time_ns() - t0) / 1e9 - thost
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
    # Last, and that ordering is the whole reason it finds anything. Every pass
    # above changes what is adjacent to what: `hoistcasts` and friends take
    # SAM 2's `_to_copy` count from 603 to 6, and those casts are what the
    # elementwise chains were mostly made of. Fusing first would fuse casts that
    # were about to be deleted, and leave the chains they were sitting between
    # unfused because a dead op stood in the way.
    nfused, nepi, npre = 0, 0, 0
    if fuse
        # Before `fuseops`: the attention it collapses contains a `clone` and a
        # `softmax` that the elementwise fuser would otherwise absorb into a
        # group, and a fused group is no longer recognisable as attention. This
        # rewrite is worth 9.88x per layer where it fires, so it goes first.
        graphs, nattn = fuseattention(graphs)
        nattn > 0 && @info "fuseattention: $nattn attention block(s) -> fused.sdpa"
        graphs, nfused = fuseops(graphs)
        graphs, nmasked = fusemaskedattention(graphs)
        nmasked > 0 && @info "fusemaskedattention: $nmasked masked attention blocks"
        # After `fuseops`, so a chain it collapsed can be folded into the GEMM
        # whole rather than only its last link.
        graphs, nepi = foldepilogue(graphs)
        # And the mirror image: into the reduction that consumes it. After
        # `foldepilogue` because an op between an `addmm` and a reduction can be
        # folded either way and only once; the epilogue is the better home, since
        # the GEMM's store touches those elements whether or not anything is
        # folded into it, while a reduction's map step is work either way.
        graphs, npre = foldpremap(graphs)
    end
    @debug "DNNKernels: folded $nfold batch-norms, $nact relus, $noutcast output casts and $nincast input casts, hoisted $nhoist casts, $nperm permutes, $nconst constants and $nsub constant-subgraph ops, fused $nfused elementwise ops, folded $nepi epilogues and $npre premaps, dropped $ndead dead ops and $dropped orphaned weights"
    tfuse = (time_ns() - t0) / 1e9 - thost - tupload
    @info "Model: built in $(round(thost + tupload + tfuse, digits=1)) s" host_passes_s =
        round(thost, digits=1) upload_s = round(tupload, digits=1) fusion_s = round(tfuse, digits=1)
    Model(graphs, weights, backend, memevery, memframes, topk; record, record_maxpasses)
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

"""
    quantkeys(graphs) -> Set{String}

Weight keys that may be stored as int8: the MATRIX operand of an `mm`/`addmm`
and nothing else.

"Nothing else" is the load-bearing half. A weight that is also read by an
`embedding`, an elementwise op or a second matmul in the other position would
need a float view of itself, and there is no such thing once it is packed — so a
key used anywhere outside these two positions is left alone rather than
quantised and then silently mis-read.
"""
function quantkeys(graphs)
    cand = Set{String}()
    other = Set{String}()
    for g in values(graphs)
        matrixin = Dict{String,Int}("mm.default" => 2, "addmm.default" => 3)
        for op in g.ops
            mi = get(matrixin, op.aten, 0)
            for (i, id) in enumerate(op.ins)
                b = get(g.buffers, id, nothing)
                b === nothing && continue
                b.kind === :weight && !isempty(b.key) || continue
                if i == mi && length(b.shape) == 2
                    push!(cand, b.key)
                else
                    push!(other, b.key)
                end
            end
        end
        # A declared output that is a weight escapes the graph as a float.
        for id in g.outputs
            b = get(g.buffers, id, nothing)
            b !== nothing && b.kind === :weight && !isempty(b.key) && push!(other, b.key)
        end
    end
    setdiff(cand, other)
end


"""
    record

**Not a mode any more.** `call` declares each graph into a Mantle plan once per
`(name, dims, clampattn, noise)` and replays that plan on every later call; there
is no immediate path left for it to be turned off in favour of. The keyword is
accepted and ignored. What follows is why the recorded form is the one there is,
and what a model has to satisfy for it to be correct — which is still true and
is now a precondition rather than a caveat.

The backend's immediate launch is one command buffer and one queue submit per
dispatch, which is right for an ad hoc kernel and wrong for a step that issues a
thousand. Recorded into a graph, the whole step is one command buffer whose
barriers Mantle derives from what each pass declares it touches, and the host
does nothing per step but write the inputs and submit. On K2 Horizon 32B decode,
1804 dispatches a step, on a Radeon 8060S:

    immediate, one-shot per dispatch   190.9 ms   host 197, GPU 151, overlapped
    one batched submit, barriers       211.6 ms   host serialised ahead of the GPU
    recorded plan                      153.1 ms   host ~0, GPU-bound

against llama.cpp's Vulkan coopmat backend at 154.8 ms for the same model on the
same device. The step moves 33.5 GB, so 153.1 ms is 219 GB/s of the 229 GB/s this
device reads at, and the logits are bit-identical to the immediate path.

`Mantle.record_into(graph, name)` captures every KA launch as a pass of its own,
with the usage keyed to the STORAGE so views of one slab order against each
other, and a device-to-device `copyto!` as a copy kernel so that it is part of
the recording rather than something that happened once while it was made.

# What a model has to satisfy

These are REFUSALS: a host read during a recording cannot be papered over, so a
graph that cannot be declared says so instead of quietly differing from the
immediate path.

  * Nothing between the graph's first op and its last may read device memory on
    the HOST. A recording defers every launch, so a host read during it sees a
    buffer that has not been written — `index.Tensor` did this through
    `checkbounds` on a device index array and threw a `BoundsError` for an index
    that was plainly in range.
  * Everything the plan closed over has to be the same object at every call: the
    slab, the weights and the input buffers. `call` enforces the last of those by
    copying into the arrays it recorded against; the first two are fixed for the
    life of a `Model`.
  * Host noise draws cannot be replayed. `rand.default` has no `emitop!` at all,
    so a graph containing one is refused by name whichever `NoiseSource` it was
    given; the declared form would be a device RNG, which is a kernel and not a
    capture.

Whisper exposed two recording defects: dtype conversions outside the capture
scope, and missing submission tracking for captured dispatch buffers. Operation
results and output materialisation are now captured; Mantle registers the
buffers so host writes and work on other queues wait for pending replay reads.

`record_maxpasses = Dict(graph_name => N)` selects explicit submission
partitions on Vulkan. The full graph is still compiled once, with the same
dependency analysis. Horizon prefill uses 64 passes per submission because its
single submission timed out; decode keeps the default single submission.
"""

# NOT `MantlePlan` — `mantle.jl` already has one, and it is a slab placement.
# Recorded plan plus the arrays it closed over: the inputs the caller's arguments
# must be copied into, and the outputs it returns.
struct RecordedPlan{P,I,O}
    plan::P
    inputs::I
    outputs::O
    # The emit's owned buffers, which `inputs` and `outputs` are storage views
    # into. They have to outlive the plan and nothing finalises them, so the
    # plan is what owns them and `Mantle.free!` below is what returns them.
    owned::Vector{Any}
    RecordedPlan(plan::P, inputs::I, outputs::O, owned::Vector{Any}) where {P,I,O} =
        new{P,I,O}(plan, inputs, outputs, owned)
end

"""
Release a recorded plan: its Mantle regions and the buffers the emit owns.

A `Model` caches one of these per `(name, dims)` and keeps it for the process,
which is the point of recording. A caller that builds plans it does not keep --
a test over every exported graph, a tool that sweeps resolutions -- has to
release them: `Mantle.free!(plan)` returns the transients and the argument
memory, and the escaping buffers belong to nobody else. Idempotent through
`freeowned!`'s own emptying.
"""
function M.free!(mp::RecordedPlan)
    M.free!(mp.plan)
    for b in mp.owned
        M.free!(b)
    end
    empty!(mp.owned)
    return nothing
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
    # Against the DECLARED shape, and this is the only place that can be.
    # Without it a wrong-shaped input ran: the graph is built around its own
    # declaration, so it produced a result shaped like the declaration and threw
    # nothing away loudly. Worse, the first call RECORDS, so the plan closed over
    # the mistake, and the replay check below — which compares against whatever
    # that first call passed — then rejected the *correct* shape. Measured on
    # `horizon32b_decode_bucket` handed a batch-2 `input_ids`: it returned one
    # batch element's logits twice, and the batch-1 call after it was rejected by
    # the replay check.
    for (id, a) in zip(g.inputs, args)
        b = get(g.buffers, id, nothing)
        (b === nothing || isempty(b.shape)) && continue
        want = evalshape(b.shape, dims)
        size(a) == want || throw(ArgumentError(
            "$name input `$id` is declared $want but got $(size(a))"))
    end
    # One plan per (graph, resolution, kernel selection), built on first call and
    # replayed after. `clampattn` is in the key because it changes which kernels
    # the graph dispatches, and `noise` because ZeroNoise and RandomNoise are
    # different computations.
    #
    # No run precedes the plan: `emitgraph` declares instead of running, so
    # `Plan` sees the whole graph before a byte is touched and places every
    # intermediate itself.
    key = (:plan, name, dims, clampattn, typeof(noise))
    mp = get!(m.scratch, key) do
        planfor(m, g, name, dims, clampattn, noise)
    end
    # The plan reads the buffers it was declared against, so the call's arguments
    # have to land in those. Identical objects are the common case and cost
    # nothing.
    #
    # SHAPE only, in `replay!`. The dtype is allowed to differ and `copyto!`
    # converts, because the declared dtype is the one the graph reads and
    # converting into it is what the graph asked for -- not a guess. Two graphs
    # chained through a step disagree about it legitimately: under autocast
    # MatAnyone's `encode_image` hands back `f16` as `Float16` while
    # `transform_key` declares its input `Float32`, and refusing that stopped
    # the model at its second graph. The interpreted run converted in the same
    # place, through the same `copyto!`.
    return replay!(mp, name, args)
end

"""
    planfor(m, g, name, dims, clampattn, noise) -> RecordedPlan

Declare one graph into a Mantle plan and record it. Runs no op.

`emitgraph` walks the ops calling `emitop!`, which declares a dispatch per
launch and nothing else -- what each one reads and writes is inferred from the
kernel body. `Plan` then runs all seven of Mantle's phases over the whole graph
(`Dag`, `Schedule`, `Liveness`, `Place`, `Aliasing`, `Barriers`, `Pipelines`),
so placement, aliasing and barriers are decided before anything executes.

The inputs and outputs are the STORAGE of the resources the graph declared, not
fresh arrays: an output that escapes is a `Buffer` (see `escaping`), so reading
it needs no copy, and an input is the array the plan's dispatches were packed
with, which is why `call` copies into it rather than rebinding.
"""
planfor(m::Model, g::Graph, name::AbstractString, dims,
        clampattn::Bool, noise::NoiseSource) =
    planfor(Mantle.Device(m.backend), g, m.weights, dims;
            maxpasses = get(m.record_maxpasses, name, 0))

function planfor(dev, g::Graph, weights::AbstractDict, dims; maxpasses::Int = 0)
    mantlegraph, emitctx = emitgraph(dev, g, residentweights(dev, g, weights), dims)
    plan = Mantle.Plan(mantlegraph)
    Mantle.record!(plan; maxpasses)
    ins  = Tuple(Mantle.storage(emitctx.res[id]) for id in g.inputs)
    outs = Tuple(Mantle.storage(emitctx.res[id]) for id in g.outputs)
    # Handed over, not copied: the plan owns them from here and the context is
    # emptied so a later `freeowned!(emitctx)` cannot retire them a second time.
    owned = copy(emitctx.owned)
    empty!(emitctx.owned)
    return RecordedPlan(plan, ins, outs, owned)
end

"""
    replay!(mp, name, args) -> outputs

Write `args` into the buffers the plan was declared against, submit it, and hand
back its outputs.

Split from `call` because `wan.jl` needs the same two steps and had its own third
copy of them — one that went through `execute!`, a `planslab` slab and a
`Workspace`, all three of which the declared path replaced. There is one way to
replay a plan and this is it.

SHAPE has to match and the DTYPE does not: see the note in `call`.
"""
function replay!(mp::RecordedPlan, name::AbstractString, args)
    for (dst, src) in zip(mp.inputs, args)
        size(dst) == size(src) || throw(ArgumentError(
            "$name input is declared $(size(dst)) and this call passed " *
            "$(size(src)). A plan is built per `dims`, so a shape that does not " *
            "follow from them cannot be replayed."))
    end
    for (dst, src) in zip(mp.inputs, args)
        dst === src || copyto!(dst, src)
    end
    Mantle.run!(mp.plan)
    return mp.outputs
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
    # A `Recycler` flip was here, one per graph plus one for `step!` itself: it
    # alternated two banks of addresses so this step's outputs did not land on
    # bytes the last step still had to read. Mantle's placer decides that from
    # declared liveness now -- a value that outlives its graph is not a
    # transient, so nothing it might alias with is placed on it.

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

    # `materialize(v)` and not `materialize(rec, backend, v)`: `steprec` was
    # `step!`'s own `Recycler`, two banks of addresses alternated so this step's
    # outputs did not land on bytes the last step still had to read. Mantle's
    # placer decides that from declared liveness (see the note at the top of
    # this function), and `s.lastmask` outlives the step, so it is an ordinary
    # allocation. The `Recycler` went and these two calls still named it, which
    # is an `UndefVarError` on the first frame of every clip.
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
