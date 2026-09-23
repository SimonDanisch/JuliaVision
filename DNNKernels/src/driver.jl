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
struct Model{D,B}
    graphs::Dict{String,Graph}
    weights::Dict{String,Any}
    # The owner of the stream, pool, capability record and recording state.
    # Keep it for the lifetime of the model; a KA backend is only a launch
    # descriptor and, notably on ROCm, contains no device identity at all.
    device::D
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
function Model(graphs, weights, target, memevery, memframes, topk;
               record::Bool = true, record_maxpasses = Dict{String,Int}())
    Model(graphs, weights, target, memevery, memframes, topk, Dict{Any,Any}();
          record, record_maxpasses)
end
function Model(graphs, weights, target, memevery, memframes, topk, scratch;
               record::Bool = true, record_maxpasses = Dict{String,Int}())
    dev = M.todevice(target)
    Model(graphs, weights, dev, M.backend(dev), memevery, memframes, topk, scratch,
          Diagnostics(), Dict{String,Int}(record_maxpasses))
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

# A Mantle resource is already resident by construction — it IS a region of the
# device's pool — so residency has nothing to decide. It also has no `eltype` or
# `size(a)...` to copy through: the branch below is about host arrays.
toback(backend, r::Union{M.Buffer,M.GPURef}) = r

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
    # `GC.@preserve`, because `uploadsource` may hand back a wrapper that
    # points INTO `a` without referencing it, and `a` is dead to the compiler
    # the moment that call returns.
    GC.@preserve a copyto!(d, uploadsource(a))
    d
end

"""
The bytes `toback` hands to `copyto!`: the array itself wherever its elements
already lie the way a dense upload wants them.

A `DenseArray` obviously does. So does a CONTIGUOUS view of one, and that is
not a corner case: a compact Qwen-Image 2.1 checkpoint arrives as 224 column
slices of mmap'd `Matrix{Int8}`, 6.5 GiB of them, and a `SubArray` is not a
`DenseArray`, so every one used to be `collect`ed into anonymous memory first —
exactly the mmap defeat the comment above is about, reintroduced through the
other branch.

Contiguity is checked against the strides a dense array of that size would
have, not inferred from the index types: `view(A, 1:3, 1:5)` of a 10x10 is a
`FastContiguousSubArray` and its columns are seven elements apart.

This is about MEMORY, not time — `upload_s` does not move, because what it is
spent on is the device allocation and not the host copy. What it buys is that
the checkpoint stays a mapping: 6.5 GiB that used to be faulted into anonymous
memory, on a unified-memory machine where the host and the device share one
pool and the 64 GiB model in the comment above is already at the edge of it.
"""
@inline function uploadsource(a::AbstractArray)
    a isa DenseArray && return a
    if a isa SubArray && parent(a) isa DenseArray && isbitstype(eltype(a)) &&
       strides(a) == Base.size_to_strides(1, size(a)...)
        return unsafe_wrap(Array, pointer(a), size(a))
    end
    collect(a)
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

# A stack of packed quantised parts has no dense rows to copy into: four output
# rows share a word and each carries its own scale. `stackrows` assembles it in
# the checkpoint layout and the result packs exactly like a single weight, so
# the host peak is one fused matrix rather than the whole stacked model.
toback(backend, A::RowCat{T,<:AbstractVector{<:ConvRotQInt8HostMatrix}}) where {T} =
    toback(backend, stackrows(A.parts))

# W4A8 cannot stack in the checkpoint layout the way ConvRot INT8 does: the
# codebook is per tensor, so the parts only agree once decoded. `w4a8convrot`
# decodes each into its own row range of one pack.
toback(backend, A::RowCat{T,<:AbstractVector{<:W4A8ConvRotHostMatrix}}) where {T} =
    w4a8convrot(backend, A.parts)

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
    KI.Kernel(backend, k)(reshape(d, L, E, 1, 1), reshape(src, length(src)),
        Int32(1), st[1], st[2], Int32(0), Int32(0), Int32(E), Int32(L), Int32(1);
        ndrange = (32 * cld(E, 32), 4 * cld(L, 32), 1), workgroupsize = (32, 4, 1))
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

"""
    handoff!(old, new) -> new

`new`, with `old` emptied unless the pass handed back the dictionary it was given.

Every host weight pass below takes a dictionary and returns a new one holding
whatever it kept, so the one it was handed is dead the moment it returns. It is
not COLLECTED, though: it sits in this frame's slot until `Model` returns, and
it names every tensor in the checkpoint. One such reference is all it takes to
defeat the upload loop, which drops its own host copy tensor by tensor
specifically so a checkpoint never has to be resident twice.

Measured on Qwen-Image 2.1's denoiser, which is 6.9 GB of host weights against
8.3 GB on the device: `gc_live_bytes` came out of the upload at 7.11 GB with
`host` already emptied and a full `GC.gc(true)` just run, and the process peaked
at 9.95 GB resident against 7.88 GB of GTT — 17.8 GB of one APU's shared memory
for a 6.9 GB model. The same `GC.gc(true)` one frame later, with `Model` gone,
reported 176 MB.

So the hand-off is explicit. `old === new` because a pass that had nothing to do
is free to return its argument, and emptying that would delete the weights.
"""
function handoff!(old::AbstractDict, new::AbstractDict)
    old === new || empty!(old)
    return new
end

function Model(graphs::Dict{String,Graph}, weights::AbstractDict;
               backend=nothing, device=nothing, memevery=5, memframes=5, topk=30,
               # Off is how the fusion passes get checked: build the model twice
               # and compare the numbers. Every other pass here is verified
               # against an invariant or a PyTorch reference, but a fusion is
               # verified against *the same model unfused*, and there has to be a
               # way to ask for that. It is also the switch to reach for first if
               # a model ever comes out wrong — see `2026-08-08-elementwise-fusion.md`
               # for why a bad fusion looks like a precision bug.
               fuse::Bool = true,
               # Just the attention fusion, independently of the rest. A fused
               # attention is only recordable if its plan has a declared form,
               # and `CoopMatSDPAPlan` has none, so a model whose attention it
               # claims can be recorded OR fused and not both. Qwen-Image 2.1's
               # VAE is the case: one mid-block head 1152 wide, which
               # `flashcm_plan` declines and `coopmat_sdpa_plan` takes.
               fuseattn::Bool = true,
               # Store the matmul weights as int8 with a per-output-channel
               # scale. Decode reads every weight once per token and is
               # bandwidth-bound outright, so this is a straight halving of the
               # floor; see `quant.jl` for the scheme and what it costs.
               quantize::Bool = false,
               # Record each graph into a Mantle plan and replay it. See
               # [`record`](@ref) — it is off by default because it is not free
               # of preconditions, and a model that does not meet them comes out
               # WRONG rather than slow.
               record::Bool = false, record_maxpasses = Dict{String,Int}(),
               # Take ownership of `weights`: empty the caller's dictionary once
               # this one has its own, so the upload below holds the only
               # reference to each host tensor and dropping it actually frees it.
               #
               # Without this the streaming upload frees NOTHING, which is the
               # failure its own comment says it exists to prevent. Measured on
               # Qwen-Image 2.1's denoiser: the caller's dict pins all 6.9 GB
               # through the upload, so the peak is host plus device — 9.95 GB
               # resident against 7.88 GB of GTT, 17.8 GB for a model that is
               # 6.9. With it the host copy is gone tensor by tensor.
               #
               # Default `true` because every caller in this tree builds a fresh
               # dictionary and hands it over: `compact_transformer_weights`,
               # `rifeweights`, `readsafetensors` at a call site. A caller that
               # wants to build two models from one checkpoint passes `false`
               # and pays the peak it asked for.
               consume::Bool = true)
    backend !== nothing && device !== nothing &&
        throw(ArgumentError("pass either `device` or `backend`, not both"))
    # Resolve the convenience `backend` spelling exactly once, at this public
    # boundary. Everything below receives and retains the actual owner.
    dev = M.todevice(device === nothing ?
                     (backend === nothing ? KernelAbstractions.CPU() : backend) : device)
    backend = M.backend(dev)
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
    src = Dict{String,Any}(weights)
    # `src` now names every tensor, so the caller's dictionary is the only
    # thing keeping a second reference. See `consume` above.
    consume && empty!(weights)
    graphs, host, nfold = foldbatchnorm(graphs, src); host = handoff!(src, host)

    graphs, nact = foldrelu(graphs)
    # Before anything that reads a convolution's operands, and before the
    # weight passes, because it only rewrites one attribute and one input and
    # every pass after it sees a shorter graph.
    graphs, npad = foldconvpad(graphs)
    npad > 0 && @info "foldconvpad: $npad explicit pad(s) -> the convolution's own"
    graphs, next, nhoist = hoistcasts(graphs, host); host = handoff!(host, next)

    # After the casts: under autocast a weight's transposed view sits on top of
    # its fp16 cast, and hoisting the cast first turns that into a plain weight
    # this pass can then permute.
    graphs, next, nperm = hoistpermutes(graphs, host); host = handoff!(host, next)

    graphs, next, nconst = hoistconstants(graphs, host); host = handoff!(host, next)

    # After the permutes are materialised, because the stack is over the weights
    # in the layout the GEMM reads, and before the live-key sweep so the parts
    # that are now only reachable through the stack get dropped.
    graphs, next, nqkv = fuseqkv(graphs, host); host = handoff!(host, next)

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
    # After both cast folds, because a clone is only an alias when its input and
    # its output declare the same dtype and `foldoutcasts` is what narrows the
    # output of the other half of them.
    graphs, nclone = dropclones(graphs)
    nclone > 0 && @info "dropclones: $nclone clone(s) -> alias"
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
    # The other rotary spelling, which rotates adjacent PAIRS rather than
    # halves. Same place in the order and for the same reasons.
    graphs, npair = fusepairrope(graphs)
    npair > 0 && @info "fusepairrope: $npair interleaved rotary embedding(s) -> one op each"
    graphs, ndead = dropdead(graphs)
    # Upload only the weights the surviving graphs still name. `dropdead` prunes
    # dead *ops*; without this the host dict keeps every orphan those passes
    # created — above all the fp32 masters whose `_to_copy` `hoistcasts` turned
    # into a plain fp16 weight. Uploading them anyway cost 849 MB of VRAM on
    # SAM 2 (1852 MB of weights resident against 1003 MB of parameters), for
    # tensors no op reads.
    live = livekeys(graphs)
    dropped = length(host) - count(k -> k in live, keys(host))
    host = handoff!(host, Dict{String,Any}(k => v for (k, v) in host if k in live))
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
            # FULL, not `GC.gc(false)`. A checkpoint's tensors are read and
            # converted before the first one is uploaded, so by the time this
            # loop runs they have survived several collections and are all in
            # the OLD generation, which an incremental pass does not visit. The
            # incremental call was therefore free and did nothing: measured on
            # Qwen-Image 2.1's denoiser, `gc_live_bytes` came out of this loop
            # at 7.1 GB and one `GC.gc(true)` afterwards took it to 176 MB.
            #
            # Every 32 and not every tensor because a full collection walks the
            # whole heap: at 298 tensors that is nine of them, and the load goes
            # from 2.4 s to 2.7 s for 6.9 GB of peak that is no longer paid.
            n % 32 == 0 && GC.gc(true)
        end
        empty!(host)
        GC.gc(true)
        quantize && @info "Model: $nq of $(length(ks)) weights stored as int8"
    end
    tupload = (time_ns() - t0) / 1e9 - thost
    # The one pass that has to *run* the ops it folds, so it comes after the
    # upload and works on the device weights: constant subgraphs, not just the
    # nullary constants `hoistconstants` took above. Then the same sweep again,
    # because folding a subgraph orphans whatever only it read.
    graphs, weights, nsub = hoistconstants(graphs, weights, dev)
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
        if fuseattn
            graphs, nattn = fuseattention(graphs)
            if nattn > 0
                @info "fuseattention: $nattn attention block(s) -> fused.sdpa"
                # The fusion reads q, k and v from further back than the `bmm`
                # did, so the casts and scalings in between are left with no
                # consumer. They are not free: Qwen-Image's are 64 MB apiece,
                # 6 per layer.
                graphs, nattndead = dropdead(graphs)
                ndead += nattndead
            end
        end
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
    Model(graphs, weights, dev, memevery, memframes, topk; record, record_maxpasses)
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

`emitgraph` declares one pass per dependent stage and one dispatch per launch,
with each argument's usage read off the kernel body, so two ops that only read
the same weight do not order against each other and a device-to-device copy is a
pass like any other.

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
  * Host noise draws cannot be replayed. `RandomNoise` therefore declares a
    persistent counter advanced by the recorded device graph and expands it in
    parallel; `ZeroNoise` is the separate deterministic device fill.

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
    releasedevice!(x)

Free every device array reachable from `x`.

**A dropped device array does not return its memory.** Mantle frees on an
explicit verb and never from a finalizer — `Mantle.trim!` says why: a pool freed
from the GC finalizer thread gave Lava a `ConcurrencyViolationError` and then a
SIGSEGV. So `empty!(weights)` drops the last Julia reference and the pool's
ledger still records every region as on loan, `trim!` finds no empty block, and
nothing goes back to the driver.

Structural rather than a list of types, because a weight is not always an array:
Qwen-Image's conditioner is 144 `ConvRotQInt8Matrix`, each a `q` and a `scale`,
and that is where its 6.9 GB lives. Walking fields means a new quantised layout
needs no entry here.

Only for teardown, and only after whatever recorded it is freed: a plan packs
device ADDRESSES, so freeing a weight a live plan still replays is a
use-after-free.
"""
function releasedevice!(a::AbstractArray)
    # `Mantle.isdevicearray`, and NOT `KernelAbstractions.get_backend`: the walk
    # reaches a `Dict`'s internal `Memory{UInt8}` and `get_backend` THROWS on it
    # rather than answering, so asking is not safe. `isdevicearray` is the
    # backend's own declaration, defaults to `false`, and cannot throw.
    #
    # `!(a isa Array)` because the host backend declares a plain `Array` to be
    # device memory, which is true there and would stop the walk here.
    if M.isdevicearray(a) && !(a isa Array)
        KernelAbstractions.unsafe_free!(a)
    elseif !isbitstype(eltype(a))
        # A host array of WRAPPERS still holds device arrays; one of scalars
        # does not, and walking it would be a walk per element.
        #
        # `isassigned`, because the walk reaches a `Dict`'s key slots — a
        # `Memory{Symbol}` whose unused entries are undefined — and iterating it
        # is an `UndefRefError`.
        for i in eachindex(a)
            isassigned(a, i) && releasedevice!(a[i])
        end
    end
    return nothing
end

# A Mantle resource has its OWN verb, and the field walk below must not reach
# it: a `Buffer` holds its region in a `store` field that answers
# `isdevicearray`, so the generic walk would hand a pool region to
# `KernelAbstractions.unsafe_free!` — Lava's free, for memory Lava did not
# allocate. `Mantle.free!` retires it to the pool it came from.
releasedevice!(r::Union{M.Buffer,M.GPURef}) = (M.free!(r); nothing)

function releasedevice!(x)
    T = typeof(x)
    isconcretetype(T) || return nothing
    for i in 1:fieldcount(T)
        isdefined(x, i) && releasedevice!(getfield(x, i))
    end
    return nothing
end

"""
    releaseweights!(weights) -> weights

Free the device memory a weight dict holds, then empty it.

What `empty!` alone was meant to do and never did. See [`releasedevice!`](@ref).
"""
function releaseweights!(d::AbstractDict)
    for (_, v) in d
        releasedevice!(v)
    end
    empty!(d)
    return d
end

"""
Return one plan-owned allocation, whichever kind it is.

`emitctx.owned` holds Mantle `Buffer`s and `planfor` adds the weight uploads,
which are backend arrays. Two verbs, because they are two things: a `Buffer` is
a pool region with a `retire!`, and a device array is GPUArrays' to free.
"""
releaseowned!(b) = M.free!(b)
releaseowned!(a::AbstractArray) = KernelAbstractions.unsafe_free!(a)

"""
Release a recorded plan: its Mantle regions, the buffers the emit owns and the
weight uploads `planfor` handed it.

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
        releaseowned!(b)
    end
    empty!(mp.owned)
    return nothing
end

"""
What `call` caches a plan under, in `Model.scratch`.

`clampattn` is in the key because it changes which kernels the graph dispatches
and `noise` because `ZeroNoise` and `RandomNoise` are different computations.

Its own function so [`planahead!`](@ref) cannot key differently from `call` and
silently build a second plan that the first call then ignores.
"""
plankey(name::AbstractString, dims, clampattn::Bool, noise::NoiseSource) =
    (:plan, name, dims, clampattn, typeof(noise))

"""
    planahead!(model, name; dims = (;), clampattn = false, noise = RandomNoise())

Build the plan `call` would build, now, and cache it.

`call` plans on first use, which is right for a driver and wrong for a runner
that must not compile a pipeline inside its own latency measurement: RIFE and
Depth Anything each assert, in a fresh process, that the first frame refuses
zero pipeline compiles. Their load is where the compiling belongs.

Returns the model, so it composes with a constructor.
"""
function planahead!(m::Model, name::AbstractString; dims = (;),
                    clampattn::Bool = false, noise::NoiseSource = RandomNoise())
    g = m.graphs[name]
    get!(m.scratch, plankey(name, dims, clampattn, noise)) do
        planfor(m, g, name, dims, clampattn, noise)
    end
    return m
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
    mp = get!(m.scratch, plankey(name, dims, clampattn, noise)) do
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
    planfor(m.device, g, m.weights, dims;
            maxpasses = get(m.record_maxpasses, name, 0), noise)

# `profile = true` builds the plan with a timestamp query pool, so
# `Mantle.timings(plan.plan)` reports per-pass GPU milliseconds after a replay.
# It is not free — a query pair around every pass — so it is off by default and
# a plan asked for it is a plan being measured.
function planfor(dev, g::Graph, weights::AbstractDict, dims;
                 maxpasses::Int = 0, noise::NoiseSource = RandomNoise(),
                 profile::Bool = false)
    resident = residentweights(dev, g, weights)
    mantlegraph, emitctx = emitgraph(dev, g, resident, dims; noise)
    plan = Mantle.Plan(mantlegraph; profile)
    Mantle.record!(plan; maxpasses)
    ins  = Tuple(Mantle.storage(emitctx.res[id]) for id in g.inputs)
    outs = Tuple(Mantle.storage(emitctx.res[id]) for id in g.outputs)
    # Handed over, not copied: the plan owns them from here and the context is
    # emptied so a later `freeowned!(emitctx)` cannot retire them a second time.
    owned = copy(emitctx.owned)
    empty!(emitctx.owned)
    # …and the WEIGHT uploads, which nothing else was returning. `emitctx.owned`
    # holds what the emit allocated; `residentweights` allocated separately and
    # handed the arrays to the graph, so they were reachable from the plan and
    # owned by nobody. Measured on Qwen-Image's text encoder: `release!` gave
    # back 440 MiB of 7.5 GB and the ledger did not move, so the denoiser loaded
    # on top of a conditioner that was supposed to be gone.
    #
    # Only what WE uploaded. `toback` returns its argument unchanged when it is
    # already on this backend, so an entry that is the SAME OBJECT as the host
    # weight belongs to the caller's dict and freeing it would pull a model's
    # weights out from under it. Identity is the whole test, and it is `toback`'s
    # own documented contract.
    for (k, v) in resident
        v === get(weights, k, nothing) || push!(owned, v)
    end
    return RecordedPlan(plan, ins, outs, owned)
end

"""
    releaseplans!(m::Model) -> m

Free every plan `call` has built on `m` and drop the cache.

`call` keys its plans on `(:plan, name, dims, clampattn, noisetype)` and keeps
them in `m.scratch` for the model's life, which is the point — one build, many
replays. A caller that is finished with a model wants the device memory back,
and the key shape is this file's business, not a runner's: `release!` in
QwenImageRunner used to reach for `component.plan.plan` because a component held
exactly one plan and nothing else did.

The model stays usable. `call` rebuilds on the next call to it, at the cost of
another build.
"""
function releaseplans!(m::Model)
    for (_, v) in m.scratch
        # `free!(v)`, NOT `free!(v.plan)`: the second frees the Mantle `Plan` and
        # leaves `v.owned` — the resident WEIGHTS, which `residentweights`
        # uploaded and `planfor` handed to the plan to own. That is most of the
        # memory. See the method above.
        v isa RecordedPlan && Mantle.free!(v)
    end
    empty!(m.scratch)
    return m
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

"""
    runonce!(g) -> nothing

Plan, record and run one graph, then free the plan.

For work that happens ONCE and whose result is afterwards only read: packing a
checkpoint's weights and quantising them. That is a kernel like any other, so it
belongs in a `Mantle.Graph` rather than in a bare `KI.Kernel` launch over
`KernelAbstractions.allocate`d memory, and the difference is not only tidiness:

  * **`Plan` puts the barrier in.** `quantizeint8` runs a scale pass and then a
    pack pass that READS what the scale pass wrote. Launched back to back there
    is nothing between them but the queue's own ordering; declared, the
    dependency is inferred from what each kernel touches and `Barriers` emits
    one.
  * **The memory is asked for as what it is.** Not a different pool —
    `KA.allocate` on a `LavaBackend` reaches Mantle's pool too, and the ledger
    says so. What differs is what each asks it for: `Mantle.Buffer` takes
    `Persistent()` with the usage bits `bufferusage(dev, T)` names, in one
    ledger entry; the `KA.allocate` path takes the ordinary bits and two. The
    usage bits are not cosmetic — `persistentarray`'s docstring records a
    predicate read from a buffer that lacked its own bit, which RADV answered
    correctly and NVIDIA hung on.
  * **One verb owns it.** A `Buffer` is freed by `Mantle.free!`, the same verb
    the emit's own owned buffers take, so `releaseweights!` has one rule and
    not two.

**No wait.** `free!` retires the plan's regions rather than freeing them:
`retire!` stamps each with the last submission that named it and `reclaim!`
runs the destroy only once `passed` says the device is through with it, which
is precisely the in-flight case. The caller dropping the float it quantised
from is safe for the same reason from the other side — the submission holds its
own arguments, which is what `holdleaves!` is for. Waiting here changed nothing
measurable either way; it is gone because it is unnecessary, not because it was
expensive.

**It is not free.** Declaring costs about 0.9 ms a weight more than the bare
`KI.Kernel` launches it replaces — 64 weights of 2048x2048 in 0.085 s against
0.030 s, both fully synchronised — and that is the `Plan` and `record!` per
weight, not the submission. On a checkpoint of a few hundred quantised tensors
it is a fraction of a second against a load measured in tens, which is the
trade being made: a barrier the graph derives, one ownership verb, and the
usage bits the element type asks for.

The OUTPUT buffers are not freed — they are the weight. `free!` gives back only
what the plan owns: its recording, argument memory and arenas.
"""
function runonce!(g)
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    Mantle.run!(plan)
    Mantle.free!(plan)
    return nothing
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
    # The Mantle buffers `sensory` and `lastmask` are the storage of, for
    # `release!` to return. See the bank's own `buffers` field.
    owned::Tuple
end

"""
    initstate(model, W, H; sensory_dim, ...) -> State

`W`, `H` are the padded full-resolution extents (multiples of 16).
"""
function initstate(m::Model, W::Int, H::Int; ck=64, cv=256, sensory=256,
                   nobj=1, bs=1, q=16, embed=256, T=Float32)
    w, h = W ÷ 16, H ÷ 16
    # `M.Buffer`, for the same two reasons as the bank's own storage: these
    # outlive every frame's transients, and a dropped device array frees
    # nothing. `release!` below returns all of it.
    dev = M.todevice(m.backend)
    sens = M.Buffer(dev, T, (w, h, sensory, nobj, bs))
    last = M.Buffer(dev, T, (W, H, nobj, bs))
    State(MemoryBank(m.backend, T, w * h, m.memframes, ck, cv, nobj, bs, q, embed),
          fill!(M.storage(sens), 0), fill!(M.storage(last), 0),
          nothing, nothing, -1, 0, (h=h, w=w), (sens, last))
end

"""
    release!(state)

Return everything a tracking state holds: its memory bank, its sensory and
last-mask buffers, and whatever the last step left in it.

There was no way to do this. A `State` is per clip and holds the bank, which at
512x288 is the clip's whole memory, and none of it came back when the state went
out of scope — Mantle frees on a verb and never from a finalizer.

`lastpixfeat` and `lastmskvalue` are outputs of a recorded plan, so they belong
to that plan and are NOT freed here; dropping the references is all this can
correctly do with them.
"""
function M.release!(s::State)
    M.release!(s.bank)
    foreach(M.free!, s.owned)
    s.lastpixfeat = nothing
    s.lastmskvalue = nothing
    return s
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
            visual = readmemory(Ctx(m.device; diag=m.diag), s.bank, key, selection,
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
