"""
K2 Horizon 32B — local language model.

A 32B dense decoder from the K2 Horizon fleet (IFM / MBZUAI, 3 Sep 2026), Apache-2.0, with a 524288-token context.

Ported FIRST of the family on purpose. The 36B-A4B sibling is the interesting one, but it carries two independent routers per layer — 100 FFN experts at top-8 plus a shared one, and MoVA's 64 value-experts at top-4 inside attention — and neither has an equivalent in DNNKernels. This model has `num_experts: 0`, `mova_num_experts: 0` and `attention_gate_func: None`: 64 layers of plain GQA (64 heads over 8 KV heads, head_dim 128), RMSNorm, SiLU and RoPE at theta 1e7. That is the bring-up vehicle — it exercises the autoregressive path, the KV cache and the 250624-entry vocabulary without also needing routing.

Supports greedy generation with int8 weights, reusable recorded dispatches,
and optional prompt/attention buckets. Supply a local export with `dir`;
the model assets are not yet distributed as a bound artifact.

Upstream: https://huggingface.co/IFM/K2-Horizon-32B
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * none — the runtime already covers it

See `models-to-port.md` for the state of this one, and `tools/export_horizon32b.py`
for the export that feeds it.
"""
module HorizonRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors, Model, call, toback

export horizon32bgraphs, horizon32bweights, horizon32b, generate, Horizon32B, newcache, causalmask

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    GRAPHS

The two graphs, and why there are two. `horizon32b_decode` is one token against
the cache and is genuinely static — `T = 1` forever. `horizon32b_prefill` runs a
whole prompt at a fixed bucket. They are the same forward exported at two
shapes, so they cannot drift; see `tools/export_horizon32b.py`.
"""
const GRAPHS = ("horizon32b_decode", "horizon32b_prefill")

"""
    assetdir() -> String

Throws for now. The export exists but is 64.78 GiB, which is ~33 GitHub release
assets at the 2 GiB cap, so it is not bound yet. Pass `dir` explicitly meanwhile
— the same interim shape `Hunyuan3DRunner` uses while its artifact is unbound.
"""
assetdir() = error(
    "HorizonRunner: the export exists but no artifact is bound to its 64.78 GiB. " *
    "Pass `dir = \"<gen>/graphs/horizon32b\"` meanwhile, or shard and bind it with " *
    "`tools/shard_safetensors.jl` + `tools/make_artifacts.jl`.")

function horizon32bgraphs(; dir::AbstractString = assetdir(), bucketdir::AbstractString = dir)
    gs = Dict(n => loadgraph(joinpath(dir, "$n.json")) for n in GRAPHS)
    if isdir(bucketdir)
        for file in readdir(bucketdir)
            occursin(r"^horizon32b_(decode_bucket|prefill_bucket_[0-9]+)\.json$", file) || continue
            name = splitext(file)[1]
            g = loadgraph(joinpath(bucketdir, file))
            g.symbols == ["kv"] || error("$file must be symbolic only in kv")
            for input in ("k_cache", "v_cache")
                b, base = g.buffers[input], gs["horizon32b_decode"].buffers[input]
                b.shape == base.shape && b.dtype === base.dtype ||
                    error("$file has a different cache layout from the base export")
            end
            gs[name] = g
        end
    end
    gs
end

horizon32bweights(; dir::AbstractString = assetdir()) =
    readsafetensors(joinpath(dir, "weights.safetensors"))

ready(; dir::AbstractString = "") =
    !isempty(dir) && all(isfile(joinpath(dir, "$n.json")) for n in GRAPHS) &&
    isfile(joinpath(dir, "weights.safetensors"))

"""
    Horizon32B

A loaded model plus the shapes the graphs were exported at. `maxlen` and
`prefill` are baked into the export, not arguments — a different bucket is a
re-export.
"""
struct Horizon32B{B,M}
    backend::B
    model::M
    maxlen::Int
    prefill::Int
    vocab::Int
end

"""
    horizon32b(; backend, dir) -> Horizon32B

Load the base graphs, weights, and any bucket graphs in `bucketdir` (default `dir`).
Bucket graphs share the weights and full-capacity KV cache with the base graphs.

`quantize` stores every matmul weight as int8 with a per-output-channel scale,
which is on by default because decode is bandwidth-bound on exactly those
weights: 64.78 GiB becomes 32.4 GiB and the per-token floor halves. Pass
`quantize = false` for the fp16 reference.

`record` replays each graph from a compiled Mantle plan instead of submitting a
command buffer per dispatch. Prefill recordings use bounded submission sizes.
On by default HERE and not in `Model`: the preconditions are real
(`DNNKernels.record`), and this is the model they have been checked against.
"""
function horizon32b(; backend = Mantle.LavaBackend(), dir::AbstractString = assetdir(),
                    quantize::Bool = true, record::Bool = true, bucketdir::AbstractString = dir)
    gs = horizon32bgraphs(; dir, bucketdir)
    # A full 512-token prefill takes over 20 seconds on the 8060S. Give the
    # driver completion points between baked pieces; decode remains one submit.
    m = Model(gs, horizon32bweights(; dir); backend, quantize, record,
              record_maxpasses = Dict(n => 64 for n in keys(gs) if startswith(n, "horizon32b_prefill")))
    dec = m.graphs["horizon32b_decode"]
    # Julia sees safetensors dims REVERSED, so torch (L,B,KVH,maxlen,hd) is
    # (hd,maxlen,KVH,B,L) here. Read the extents off the graph rather than the
    # config: the export is the authority on what it was built at.
    kb = dec.buffers["k_cache"].shape          # torch order, from the JSON
    maxlen = kb[4]
    vocab = dec.buffers[dec.outputs[1]].shape[3]
    pre = m.graphs["horizon32b_prefill"].buffers["input_ids"].shape[2]
    return Horizon32B(backend, m, Int(maxlen), Int(pre), Int(vocab))
end

"""
    newcache(m) -> (k, v)

Zeroed KV, on the device, in Julia's reversed layout.
"""
function newcache(m::Horizon32B)
    dec = m.model.graphs["horizon32b_decode"]
    b = dec.buffers["k_cache"]
    L, _, KVH, maxlen, hd = b.shape
    z = () -> fill!(KA.allocate(m.backend, b.dtype, hd, maxlen, KVH, 1, L), zero(b.dtype))
    return z(), z()
end

"""
    causalmask(m, positions) -> device array

The additive mask, `(1, 1, T, maxlen)` in torch order and so
`(maxlen, n_rep*T, 1, 1)` here — grouped, because the exporter regroups Q by
KV head instead of expanding the cache. Slot `s` is visible to query `i` iff
`s <= positions[i]`; everything else gets `-floatmax`, which is exactly
`torch.finfo(dtype).min` and what the export was traced with.

NOT `typemin`, which is `-Inf` for a float in Julia. `-Inf` is wrong twice over:
the trace's own safe-softmax guard tests `scores == -inf` and would start firing,
and `-Inf` does not survive the fp16 cast that the finite bound does.

An ARGUMENT rather than something the graph builds. Built in-graph, SDPA
broadcasts it to `(B, H, T, maxlen)` and the trace keeps one per layer: 64
tensors of `[1, 64, 512, 4096]` for a single prompt bucket. Here it is 4 MiB,
uploaded once per call.
"""
function causalmask(m::Horizon32B, positions::AbstractVector{<:Integer})
    h = maskbuffer(m, length(positions))
    causalmask!(h, positions)
    return toback(m.backend, h)
end

"""
    maskbuffer(m, T) -> host array

The host side of [`causalmask`](@ref), uninitialised. Separate so a decode loop
can allocate it ONCE and refill it, which is what `generate` does.
"""
function maskbuffer(m::Horizon32B, T::Integer; kv::Integer=m.maxlen)
    b = m.model.graphs["horizon32b_decode"].buffers["mask"]
    # The DECODE graph declares `(1, 1, n_rep*1, maxlen)`, so its row count is
    # `n_rep` outright. Off the export rather than the config: the exporter is
    # the authority on the shape it grouped to.
    Array{b.dtype}(undef, kv, Int(b.shape[3]) * T, 1, 1)
end

# Each specialization owns only its graph's scratch. Planning all prompt graphs
# for every decode bucket would retain a full-prefill slab per bucket.
function bucketmodel(m::Horizon32B, name::String)
    get!(m.model.scratch, (:horizon_bucket_model, name)) do
        base = m.model
        Model(Dict(name => base.graphs[name]), base.weights, base.device,
              base.memevery, base.memframes, base.topk; record=base.record,
              record_maxpasses=base.record_maxpasses)
    end
end

attentionbucket(m::Horizon32B, occupied::Int) = min(m.maxlen, nextpow(2, max(16, occupied)))

"""
    chunkkv(m, occupied) -> Int

Attention extent for a prefill chunk that leaves `occupied` slots filled.

Two ladders, because they answer different questions. Up to the largest prompt
bucket the power-of-two one is right: it is what makes a 16-token prompt cost
0.215 s instead of a 512-token one, and the extents it produces are exactly the
prompt sizes. Above it, rounding up to a power of two makes a chunk with 2560
live slots sweep 4096 — measured on a 4096-token prompt, 27.67 s for the
power-of-two ladder against 23.58 s for this one (llama.cpp: 23.60 s). What it
costs is one recorded plan and one slab per distinct extent: +2.7 GiB of pool
reservation at 4096 slots, and 0.91 s on the first such prompt for the four
extents it adds. Against 4.1 s a prompt, that pays immediately.

**Decode deliberately does NOT use this**, and the numbers are why. Its per-step
win is real but tiny — 159.10 against 159.86 ms at depth 1088, 160.77 against
163.58 at 2080, 161.79 against 164.10 at 3040, so 0.5% to 1.7% — because a step
reads 32 GB of weights and only a few hundred MB of cache. Each new extent costs
0.7 s to record, five of them across a full context, and it lands on the first
token at that depth. A saving under the pipeline's own ±3% replay spread is not
worth making the first token slower; the prompt case is the opposite trade.
"""
chunkkv(m::Horizon32B, occupied::Int) =
    occupied <= m.prefill ? attentionbucket(m, occupied) :
                            min(m.maxlen, cld(occupied, m.prefill) * m.prefill)

function prefillbucket(m::Horizon32B, n::Int)
    sizes = [Int(g.buffers["input_ids"].shape[2]) for (name, g) in m.model.graphs
             if startswith(name, "horizon32b_prefill_bucket_")]
    minimum(filter(t -> n <= t <= m.prefill, sizes); init=m.prefill)
end

function bucketcall(m::Horizon32B, name::String, args...; kv::Int)
    call(bucketmodel(m, name), name, args...; dims=(; kv))
end

"""
    causalmask!(h, positions) -> h

Fill a [`maskbuffer`](@ref) for `positions`.
"""
function causalmask!(h::AbstractArray{dt,4}, positions::AbstractVector{<:Integer}) where {dt}
    maxlen, rows = size(h, 1), size(h, 2)
    T = length(positions)
    neg = -floatmax(dt)
    for r in 1:rows
        # Row r of the grouped scores is query head (r-1) ÷ T, position (r-1) % T.
        i = (r - 1) % T + 1
        for s in 1:maxlen
            # `s - 1` and `positions` are both the model's 0-based absolute slots.
            h[s, r, 1, 1] = (s - 1) <= positions[i] ? zero(dt) : neg
        end
    end
    return h
end

"""
    generate(m, ids; maxnew = 32) -> Vector{Int}

Greedy continuation of `ids` (1-based token ids are NOT used here: these are the
model's own 0-based ids, as the tokenizer emits them). Stops early on any id in
`stop`, which is where the tokenizer's EOS goes.

Prefill pads to the smallest available prompt bucket, and a prompt longer than
the largest one runs as consecutive bucket prefills against the growing cache —
so the ceiling is `maxlen`, the cache capacity, not `prefill`. Attention visits
a prefix covering the occupied slots; see [`chunkkv`](@ref) for how wide.
New bucket exports project only the requested prompt position into logits.
Without bucket exports (or with `bucketed=false`), use the original fixed graphs.
The generation cache is reused and reset between calls; concurrent calls on the
same model are not supported.
"""
function generate(m::Horizon32B, ids::AbstractVector{<:Integer}; maxnew::Int = 32,
                  stop = (), verbose::Bool = false, bucketed::Bool = true)
    t0 = time()
    n = length(ids)
    n > 0 || throw(ArgumentError("prompt must not be empty"))
    maxnew >= 0 || throw(ArgumentError("maxnew must be nonnegative"))
    maxnew == 0 && return Int[]
    n <= m.maxlen || error("prompt exceeds the cache capacity $(m.maxlen)")
    chunked = n > m.prefill
    chunked && !bucketed &&
        error("prompt of $n exceeds the exported bucket $(m.prefill); the fixed " *
              "graphs run one prompt pass and cannot chunk it")
    k, v = get!(m.model.scratch, (:horizon_generation_cache,)) do
        newcache(m)
    end
    fill!(k, 0); fill!(v, 0)

    # ---- prefill: the prompt, padded up to the bucket. Padding sits at
    # positions >= n and is never read back, because only slot n-1's logits are
    # taken and later decode positions overwrite those cache slots.
    #
    # A prompt longer than the largest export runs as CONSECUTIVE bucket
    # prefills against the growing cache. Each chunk attends over the slots the
    # ones before it wrote, which is what a single pass would have done, so the
    # engine's limit is the cache capacity and not the prompt bucket. Measured
    # against llama.cpp on one 4096-token prompt: 23.58 s against 23.60 s.
    logits, T, kv, selectedlogit = nothing, 0, 0, false
    for start in 0:m.prefill:(n - 1)
        valid = min(m.prefill, n - start)
        T = bucketed ? prefillbucket(m, valid) : m.prefill
        kv = bucketed ? chunkkv(m, start + T) : m.maxlen
        prefillname = "horizon32b_prefill_bucket_$T"
        usebucket = bucketed && kv <= m.maxlen && haskey(m.model.graphs, prefillname)
        if !usebucket
            T, kv = m.prefill, m.maxlen
        end
        padded = zeros(Int64, T, 1)
        padded[1:valid, 1] .= @view ids[(start + 1):(start + valid)]
        slots = collect(Int64, start:(start + T - 1))
        pos = toback(m.backend, slots)
        premask = maskbuffer(m, T; kv)
        causalmask!(premask, slots)
        preargs = (toback(m.backend, padded), k, v, pos, toback(m.backend, premask))
        selectedlogit = usebucket && "logits_indices" in m.model.graphs[prefillname].inputs
        if selectedlogit
            preargs = (preargs..., toback(m.backend, Int64[valid - 1]))
        end
        logits, k, v = usebucket ? bucketcall(m, prefillname, preargs...; kv) :
            call(m.model, "horizon32b_prefill", preargs...; dims=(;))
    end
    out = Int[]
    lastpos = chunked ? mod(n - 1, m.prefill) + 1 : n
    last = selectedlogit ? vec(Array(logits)) : Array(copy(view(logits, :, lastpos, 1)))
    nxt = argmax(last) - 1                 # back to the model's 0-based ids
    push!(out, nxt)
    verbose && println(stderr, "prefill($T, kv=$kv",
        chunked ? ", $(cld(n, m.prefill)) chunks" : "", ") ",
        round(time() - t0, digits=2), " s")
    nxt in stop && return out
    td = time()

    # ---- decode
    #
    # Every buffer the loop needs, allocated ONCE. A step is ~153 ms of GPU work
    # and three fresh device arrays plus a 500 kB logits download per iteration
    # cost 4-73 ms of host time on top of it — allocation, not transfer, and it
    # landed as a spike every third step rather than as a constant. Refilling
    # them in place is the same arithmetic with none of that.
    tokh = zeros(Int64, 1, 1);   tokd = toback(m.backend, tokh)
    posh = zeros(Int64, 1);      posd = toback(m.backend, posh)
    usebucket = bucketed && haskey(m.model.graphs, "horizon32b_decode_bucket")
    kv = usebucket ? attentionbucket(m, n + 1) : m.maxlen
    maskh = maskbuffer(m, 1; kv); maskd = toback(m.backend, maskh)
    # (vocab, 1, 1) — the DECODE shape, not the prefill bucket `logits` still holds.
    logith = Array{eltype(logits)}(undef, m.vocab, 1, 1)
    for step in 1:(maxnew - 1)
        p = n + step - 1
        p < m.maxlen || break
        nextkv = usebucket ? attentionbucket(m, p + 1) : m.maxlen
        if nextkv != kv
            kv = nextkv
            maskh = maskbuffer(m, 1; kv)
            maskd = toback(m.backend, maskh)
        end
        tokh[1, 1] = nxt;  copyto!(tokd, tokh)
        posh[1] = p;       copyto!(posd, posh)
        causalmask!(maskh, posh); copyto!(maskd, maskh)
        logits, k, v = usebucket && kv < m.maxlen ?
            bucketcall(m, "horizon32b_decode_bucket", tokd, k, v, posd, maskd; kv) :
            call(m.model, "horizon32b_decode", tokd, k, v, posd, maskd; dims=(;))
        # Downloaded and reduced on the HOST, which is the faster of the two and
        # was worth checking: 500 kB is 0.21 ms and the host `argmax` 0.64, while
        # `argmax` on the device array is 0.21 in isolation — but it is several
        # one-shot dispatches, and in the loop it costs MORE than the download it
        # replaces. Interleaved, 12 steps each: 154.37 ms host, 154.95 device.
        copyto!(logith, logits)
        nxt = argmax(view(logith, :, 1, 1)) - 1
        push!(out, nxt)
        nxt in stop && break
    end
    if verbose && length(out) > 1
        println(stderr, "decode ", round((time() - td) / (length(out) - 1) * 1000, digits=1),
                " ms/token over ", length(out) - 1, " steps")
    end
    return out
end


end # module
