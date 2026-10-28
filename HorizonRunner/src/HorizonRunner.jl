"""
K2 Horizon 32B — local language model.

A 32B dense decoder from the K2 Horizon fleet (IFM / MBZUAI, 3 Sep 2026), Apache-2.0, with a 524288-token context.

Ported FIRST of the family on purpose. The 36B-A4B sibling is the interesting one, but it carries two independent routers per layer — 100 FFN experts at top-8 plus a shared one, and MoVA's 64 value-experts at top-4 inside attention — and neither has an equivalent in DNNKernels. This model has `num_experts: 0`, `mova_num_experts: 0` and `attention_gate_func: None`: 64 layers of plain GQA (64 heads over 8 KV heads, head_dim 128), RMSNorm, SiLU and RoPE at theta 1e7. That is the bring-up vehicle — it exercises the autoregressive path, the KV cache and the 250624-entry vocabulary without also needing routing.

Supports greedy generation with quantized weights, reusable recorded dispatches,
and optional prompt/attention buckets. Graphs, tokenizer and fp16 weight shards
are lazy Julia artifacts, shared across environments and downloaded on first use.

Upstream: https://huggingface.co/IFM/K2-Horizon-32B
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * none — the runtime already covers it

See `models-to-port.md` for the state of this one, and `tools/export_horizon32b.py`
for the export that feeds it.
"""
module HorizonRunner

using Lava, DNNKernels, KernelAbstractions
using Artifacts, LazyArtifacts, JSON3
import Mantle
using DNNKernels: loadgraph, readsafetensors, Model, call, toback

export horizon32bgraphs, horizon32bweights, horizon32b, generate, Horizon32B, newcache, causalmask

const KA = KernelAbstractions
const ARTIFACTS_TOML = joinpath(@__DIR__, "..", "Artifacts.toml")

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

Resolve the lazy artifact containing the graphs, tokenizer and weight index.
The fp16 weights are separate artifacts so every release download fits GitHub's
per-asset limit. No generated-directory fallback is consulted.
"""
assetdir() = artifact"horizon32b"
tinydir() = artifact"horizon32b-tiny"

function horizon32bgraphs(; dir::AbstractString = assetdir(), bucketdir::AbstractString = dir)
    gs = Dict(n => loadgraph(joinpath(dir, "$n.json")) for n in GRAPHS)
    if isdir(bucketdir)
        for file in readdir(bucketdir)
            occursin(r"^horizon32b_(decode_bucket|prefill_bucket_[0-9]+|verify_bucket_[0-9]+)\.json$", file) || continue
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

function horizon32bweights(; dir::AbstractString = assetdir())
    index = joinpath(dir, "weights.index.json")
    # An explicitly supplied developer export retains the original file layout.
    isfile(index) || return readsafetensors(joinpath(dir, "weights.safetensors"))
    manifest = JSON3.read(read(index, String))
    manifest.format == 1 || error("unsupported Horizon weight index format")
    weights = Dict{String,Any}()
    for part in manifest.shards
        name = String(part.artifact)
        path = joinpath(@artifact_str(name), String(part.file))
        shard = readsafetensors(path; mmap=true)
        isempty(intersect(keys(weights), keys(shard))) || error("duplicate Horizon weight in $name")
        merge!(weights, shard)
    end
    Set(keys(weights)) == Set(String.(keys(manifest.tensor_sha256))) ||
        error("Horizon weight shards do not match the index")
    return weights
end

"""Whether runnable assets are already installed; never downloads anything."""
function ready(; dir::Union{Nothing,AbstractString}=nothing)
    if dir === nothing
        h = artifact_hash("horizon32b", ARTIFACTS_TOML)
        h !== nothing && artifact_exists(h) || return false
        dir = artifact_path(h)
    end
    all(isfile(joinpath(dir, "$n.json")) for n in GRAPHS) || return false
    isfile(joinpath(dir, "weights.safetensors")) && return true
    index = joinpath(dir, "weights.index.json")
    isfile(index) || return false
    manifest = JSON3.read(read(index, String))
    all(manifest.shards) do part
        hash = artifact_hash(String(part.artifact), ARTIFACTS_TOML)
        hash !== nothing && artifact_exists(hash)
    end
end

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

`quantize` is how the matmul weights are stored, and they are quantized by
default because decode is bandwidth-bound on exactly those weights. The default
`:q4k_m` is llama.cpp's Q4_K_M: Q4_K, with Q6_K where it gives more bits (see
[`q4km`](@ref)), 18.9 GiB against the export's 64.78. `:q4k` is Q4_K throughout,
17.5 GiB; `true` or `:int8` is int8 with a per-output-channel scale, 32.4 GiB,
slower by half and a little more exact (see the README). Pass `quantize = false`
for the fp16 reference.

`cache` is a directory for the quantized weights: the first load writes them
there, and later loads read them instead of reading the fp16 export and
quantizing it (see DNNKernels' `quantcache.jl`).

`record` replays each graph from a compiled Mantle plan instead of submitting a
command buffer per dispatch. Mantle cuts a long prefill into submissions itself.
On by default HERE and not in `Model`: the preconditions are real
(`DNNKernels.record`), and this is the model they have been checked against.
"""
function horizon32b(; backend = Mantle.defaultbackend(), dir::AbstractString = assetdir(),
                    quantize::Union{Bool,Symbol} = :q4k_m, record::Bool = true, bucketdir::AbstractString = dir,
                    cache::Union{Nothing,AbstractString} = nothing)
    gs = horizon32bgraphs(; dir, bucketdir)
    # (L, B, KVH, maxlen, hd) in torch order: the layer count is the export's.
    policy = quantize === :q4k_m ? q4km(Int(gs["horizon32b_decode"].buffers["k_cache"].shape[1])) : quantize
    # A full 512-token prefill takes over 20 seconds on the 8060S; Mantle cuts
    # its recording by the device's submission budget.
    m = Model(gs, horizon32bweights(; dir); backend, quantize = policy, record, quantcache = cache)
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
    q4km(nlayers) -> QuantPolicy

Q4_K, with Q6_K where llama.cpp's Q4_K_M puts it: `lm_head`, and the down
projection of the first and last eighth of the layers and of every third layer
between (its `use_more_bits`).
Its other choice, the value projections, is not available: `fuseqkv` has stacked
them with the queries and keys into one weight.
"""
function q4km(nlayers::Int)
    morebits(i) = i < nlayers ÷ 8 || i >= 7nlayers ÷ 8 || (i - nlayers ÷ 8) % 3 == 2
    pick(key, _) = occursin("lm_head", key) ? :q6k :
        (m = match(r"layers\.(\d+)\.mlp\.down_proj", key);
         m !== nothing && morebits(parse(Int, m[1])) ? :q6k : :q4k)
    return DNNKernels.QuantPolicy(:q4k_m, pick)
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
              base.memevery, base.memframes, base.topk)
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

"""
    verifybucket(m, T) -> Int

The smallest exported draft-checking bucket of at least `T` tokens, 0 if none.
"""
function verifybucket(m::Horizon32B, T::Int)
    fits = [Int(g.buffers["input_ids"].shape[2]) for (name, g) in m.model.graphs
            if startswith(name, "horizon32b_verify_bucket_") && g.buffers["input_ids"].shape[2] >= T]
    return isempty(fits) ? 0 : minimum(fits)
end

"""
    lookupdraft(tokens, k; ngram = 3) -> Vector{Int}

Up to `k` tokens guessed by looking the context up in itself ("prompt lookup"):
the tokens that followed the most recent earlier occurrence of its last `n`
tokens, for the longest `n <= ngram` that has one. Free to compute, and right
where the text repeats itself: a reasoning model quoting its prompt, code being
edited, a list continuing.
"""
function lookupdraft(tokens::AbstractVector{<:Integer}, k::Int; ngram::Int = 3)
    L = length(tokens)
    k > 0 || return Int[]
    for n in min(ngram, L - 1):-1:1
        tail = @view tokens[(L - n + 1):L]
        for s in (L - n):-1:1
            if @view(tokens[s:(s + n - 1)]) == tail
                follow = tokens[(s + n):min(s + n + k - 1, L)]
                isempty(follow) || return collect(Int, follow)
            end
        end
    end
    return Int[]
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

`draft = k` checks up to `k` guessed tokens per pass (see [`lookupdraft`](@ref))
through the smallest verify bucket of at least `k + 1` tokens, and takes a plain
decode step where the context offers no guess. The tokens are the same as without:
a guess is kept only where it is the greedy choice.
The generation cache is reused and reset between calls; concurrent calls on the
same model are not supported.
"""
function generate(m::Horizon32B, ids::AbstractVector{<:Integer}; maxnew::Int = 32,
                  stop = (), verbose::Bool = false, bucketed::Bool = true, draft::Int = 0)
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
    # Drafts: up to `draft` tokens guessed by `lookupdraft`, checked together with
    # the token they follow in one pass of a verify bucket, which reads every
    # weight once for all of them. Greedy, so the output is the same as without.
    V = draft > 0 && bucketed ? verifybucket(m, draft + 1) : 0
    if V > 0
        vtokh = zeros(Int64, V, 1);  vtokd = toback(m.backend, vtokh)
        vposh = zeros(Int64, V);     vposd = toback(m.backend, vposh)
        vkv = attentionbucket(m, n + V)
        vmaskh = maskbuffer(m, V; kv = vkv); vmaskd = toback(m.backend, vmaskh)
        vlogith = Array{eltype(logits)}(undef, m.vocab, V, 1)
    end
    history = Int[collect(Int, ids); nxt]
    passes = drafted = accepted = 0
    while length(out) < maxnew
        p = n + length(out) - 1             # the slot `nxt` goes to
        p < m.maxlen || break
        d = V > 0 && p + V <= m.maxlen ?
            lookupdraft(history, min(draft, V - 1, maxnew - length(out) - 1)) : Int[]
        passes += 1
        if isempty(d)
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
            push!(out, nxt); push!(history, nxt)
            nxt in stop && break
            continue
        end
        # `nxt` at slot `p`, the drafts after it, and padding that is never read:
        # its slots are past every position a later query can see until they are
        # written again.
        nextkv = attentionbucket(m, p + V)
        if nextkv != vkv
            vkv = nextkv
            vmaskh = maskbuffer(m, V; kv = vkv)
            vmaskd = toback(m.backend, vmaskh)
        end
        for i in 1:V
            vtokh[i, 1] = i == 1 ? nxt : d[min(i - 1, length(d))]
            vposh[i] = p + i - 1
        end
        copyto!(vtokd, vtokh); copyto!(vposd, vposh)
        causalmask!(vmaskh, vposh); copyto!(vmaskd, vmaskh)
        logits, k, v = bucketcall(m, "horizon32b_verify_bucket_$V", vtokd, k, v, vposd, vmaskd; kv = vkv)
        copyto!(vlogith, logits)
        drafted += length(d)
        # Row `i` predicts the token after input `i`: keep drafts while they are
        # what the model would have said, then its own token where they stop.
        for i in 1:(length(d) + 1)
            nxt = argmax(view(vlogith, :, i, 1)) - 1
            push!(out, nxt); push!(history, nxt)
            (nxt in stop || length(out) >= maxnew) && break
            i <= length(d) && nxt == d[i] || break
            accepted += 1
        end
        nxt in stop && break
    end
    if verbose && length(out) > 1
        println(stderr, "decode ", round((time() - td) / (length(out) - 1) * 1000, digits=1),
                " ms/token, ", length(out) - 1, " tokens in ", passes, " passes",
                V > 0 ? ", $accepted of $drafted drafted tokens kept" : "")
    end
    return out
end


end # module
