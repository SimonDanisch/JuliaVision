# The GPT's decode step, written out: one token per beam against a persistent
# cache, as one recorded plan replayed per token.
#
# The exported `gpt_decode` graph is the reference this replaces. It reorders
# the beams by copying the whole cache (`k_cache[:, beam_idx]`, HF's
# `_reorder_cache`), attends over a fixed bucket of rows behind a mask, and every
# attention width is a plan with caches of its own. Here:
#
#   * The cache is written in place. A beam's rows are found through `src`, the
#     beam each of its positions was written by: a reorder copies that table
#     (positions × beams integers), never the cache. Rows are written once, at
#     the beam's own column, and never overwritten.
#   * The prefix is shared: the prefill's keys and values are stored once, in
#     column 0, and every beam's table points there.
#   * Attention reads only filled rows, in chunks of 64 positions per workgroup
#     (flash decoding: partial maxima, sums and weighted values, then one combine),
#     so the step's cost follows the utterance, not the cache's size.
#   * The weights are stored k-contiguous, `(K, N)`, each output's inputs
#     adjacent, so a GEMV reads them in order (llama.cpp's `mul_mat_vec`
#     layout); a step reads each weight once for all beams.

const CHUNK = 64          # positions per attention workgroup; also the head width
const HEADS = 20
const DIM = 1280
const FFN = 5120
const LAYERS = 24
const LN_EPS = 1f-5

# ── epilogues: what a GEMV does with each output ─────────────────────────────

"""`y[n, b] = v + bias[n]`."""
struct Plain end
"""`y[n, b] += v + bias[n]`: a residual added in place."""
struct Residual end
"""`y[n, b] = gelu_new(v + bias[n])`, GPT-2's tanh approximation."""
struct GeluNew end
"""
The fused projection `c_attn`: queries to `q`, keys and values into the cache at
`slot`. A tag only: the buffers are the kernel's own trailing arguments, since a
dispatch converts top-level arguments to device arrays and not fields of a struct.
"""
struct QKVStore end

@inline geluNew(x::Float32) = 0.5f0 * x * (1f0 + tanh(0.7978845608028654f0 * (x + 0.044715f0 * x * x * x)))

@inline store!(::Plain, y, n, b, v, aux...) = (@inbounds y[n, b] = v; nothing)
@inline store!(::Residual, y, n, b, v, aux...) = (@inbounds y[n, b] += v; nothing)
@inline store!(::GeluNew, y, n, b, v, aux...) = (@inbounds y[n, b] = geluNew(v); nothing)
@inline function store!(::QKVStore, y, n, b, v, q, kc, vc, slot)
    part, i = divrem(n - Int32(1), Int32(DIM))
    h, d = divrem(i, Int32(CHUNK))
    @inbounds if part == Int32(0)
        q[d + Int32(1), h + Int32(1), b] = v
    elseif part == Int32(1)
        kc[d + Int32(1), Int32(slot[1]) + Int32(1), h + Int32(1), b] = v
    else
        vc[d + Int32(1), Int32(slot[1]) + Int32(1), h + Int32(1), b] = v
    end
    return nothing
end

# ── kernels ──────────────────────────────────────────────────────────────────

struct BeamGEMV{NR,BLOCK,NB,SUB} <: Function end

"""
    BeamGEMV{NR,BLOCK,NB,SUB}()(y, W, x, bias, epi, K, N, a1, a2, a3, a4)

`epi(Σ_k W[k, n] x[k, b] + bias[n])` for `NB` vectors `x[:, b]` at once (`a1..a4`
are the epilogue's own buffers, `nothing` where it has none): each
workgroup of `BLOCK` items owns `NR` outputs, items walk `K` in steps of four, a
weight is loaded once for all `NB` vectors, partials reduce over the subgroup
(`SUB` wide) and then across subgroups in workgroup memory.
"""
@generated function (::BeamGEMV{NR,BLOCK,NB,SUB})(y, W, x, bias, epi, K::Int32, N::Int32,
                                                   a1, a2, a3, a4) where {NR,BLOCK,NB,SUB}
    nsub = cld(BLOCK, SUB)
    quote
        parts = KI.localmemory(Float32, Val(($(NR * NB * nsub),)), Val(1))
        tid = Int32(KI.get_local_id().x) - Int32(1)
        grp = Int32(KI.get_group_id().x) - Int32(1)
        n0 = grp * Int32($NR)
        lane = tid % Int32($SUB)
        sub = tid ÷ Int32($SUB)
        Base.Cartesian.@nexprs $NR r -> Base.Cartesian.@nexprs $NB b -> acc_r_b = 0f0
        k = tid * Int32(4)
        @inbounds while k < K
            Base.Cartesian.@nexprs $NB b -> begin
                x1_b = x[k + Int32(1), b]
                x2_b = x[k + Int32(2), b]
                x3_b = x[k + Int32(3), b]
                x4_b = x[k + Int32(4), b]
            end
            Base.Cartesian.@nexprs $NR r -> begin
                n_r = n0 + Int32(r)
                if n_r <= N
                    w1 = Float32(W[k + Int32(1), n_r])
                    w2 = Float32(W[k + Int32(2), n_r])
                    w3 = Float32(W[k + Int32(3), n_r])
                    w4 = Float32(W[k + Int32(4), n_r])
                    Base.Cartesian.@nexprs $NB b -> begin
                        acc_r_b += w1 * x1_b + w2 * x2_b + w3 * x3_b + w4 * x4_b
                    end
                end
            end
            k += Int32($(4 * BLOCK))
        end
        Base.Cartesian.@nexprs $NR r -> Base.Cartesian.@nexprs $NB b -> red_r_b = KI.sub_group_reduce_add(acc_r_b)
        @inbounds if lane == Int32(0)
            Base.Cartesian.@nexprs $NR r -> Base.Cartesian.@nexprs $NB b -> begin
                parts[sub * Int32($(NR * NB)) + Int32((r - 1) * $NB + b)] = red_r_b
            end
        end
        KI.barrier()
        @inbounds if tid < Int32($(NR * NB))
            r = tid ÷ Int32($NB)
            b = tid % Int32($NB) + Int32(1)
            n = n0 + r + Int32(1)
            if n <= N
                t = 0f0
                for s in Int32(0):Int32($(nsub - 1))
                    t += parts[s * Int32($(NR * NB)) + tid + Int32(1)]
                end
                store!(epi, y, n, b, t + bias[n], a1, a2, a3, a4)
            end
        end
        return nothing
    end
end

"""`h[:, b] = embedding[:, token[b] + 1] + position[:, pos + 1]`."""
function embed!(h, emb, posemb, tokens, mpos, nb::Int32)
    g = Int32(KI.get_global_id().x)
    g <= Int32(DIM) * nb || return nothing
    d, b = (g - Int32(1)) % Int32(DIM) + Int32(1), (g - Int32(1)) ÷ Int32(DIM) + Int32(1)
    @inbounds h[d, b] = emb[d, tokens[b] + Int32(1)] + posemb[d, Int32(mpos[1]) + Int32(1)]
    return nothing
end

"""`LayerNorm` over each column of `h` (one workgroup a column), eps 1e-5, with
weight and bias the columns of `wb`."""
function layernorm!(xn, h, wb, ::Val{G}) where {G}
    acc = KI.localmemory(Float32, Val((G,)), Val(1))
    l = Int32(KI.get_local_id().x)
    b = Int32(KI.get_group_id().x)
    s = 0f0
    i = l
    @inbounds while i <= Int32(DIM)
        s += h[i, b]
        i += Int32(G)
    end
    @inbounds acc[l] = s
    KI.barrier()
    half = Int32(G ÷ 2)
    while half >= Int32(1)
        @inbounds l <= half && (acc[l] += acc[l + half])
        KI.barrier()
        half ÷= Int32(2)
    end
    @inbounds mu = acc[1] / Float32(DIM)
    KI.barrier()
    s = 0f0
    i = l
    @inbounds while i <= Int32(DIM)
        d = h[i, b] - mu
        s += d * d
        i += Int32(G)
    end
    @inbounds acc[l] = s
    KI.barrier()
    half = Int32(G ÷ 2)
    while half >= Int32(1)
        @inbounds l <= half && (acc[l] += acc[l + half])
        KI.barrier()
        half ÷= Int32(2)
    end
    @inbounds rstd = 1f0 / sqrt(acc[1] / Float32(DIM) + LN_EPS)
    i = l
    @inbounds while i <= Int32(DIM)
        xn[i, b] = (h[i, b] - mu) * rstd * wb[i, 1] + wb[i, 2]
        i += Int32(G)
    end
    return nothing
end

"""
Partial attention over one chunk of 64 positions for one head of one beam:
item `t` scores position `chunk·64 + t`, then, as dimension `t`, sums the chunk's
values weighted by `exp(score - max)`. Chunks past the slot do nothing.
"""
function attnpartial!(pmax, psum, pacc, q, kc, vc, src, slot, ::Val{MAXCHUNKS}) where {MAXCHUNKS}
    w = KI.localmemory(Float32, Val((CHUNK,)), Val(1))
    qs = KI.localmemory(Float32, Val((CHUNK,)), Val(2))
    red = KI.localmemory(Float32, Val((CHUNK,)), Val(3))
    t = Int32(KI.get_local_id().x) - Int32(1)
    wg = Int32(KI.get_group_id().x) - Int32(1)
    c = wg % Int32(MAXCHUNKS)
    hb = wg ÷ Int32(MAXCHUNKS)
    h = hb % Int32(HEADS) + Int32(1)
    b = hb ÷ Int32(HEADS) + Int32(1)
    last = Int32(slot[1])
    s0 = c * Int32(CHUNK)
    s0 > last && return nothing                 # uniform across the workgroup
    @inbounds qs[t + Int32(1)] = q[t + Int32(1), h, b]
    KI.barrier()
    s = s0 + t
    score = -Inf32
    @inbounds if s <= last
        src_b = src[s + Int32(1), b] + Int32(1)
        acc = 0f0
        for d in Int32(1):Int32(CHUNK)
            acc += qs[d] * Float32(kc[d, s + Int32(1), h, src_b])
        end
        score = acc * 0.125f0
    end
    @inbounds red[t + Int32(1)] = score
    KI.barrier()
    half = Int32(CHUNK ÷ 2)
    while half >= Int32(1)
        @inbounds t < half && (red[t + Int32(1)] = max(red[t + Int32(1)], red[t + half + Int32(1)]))
        KI.barrier()
        half ÷= Int32(2)
    end
    @inbounds m = red[1]
    KI.barrier()
    p = s <= last ? exp(score - m) : 0f0
    @inbounds w[t + Int32(1)] = p
    @inbounds red[t + Int32(1)] = p
    KI.barrier()
    half = Int32(CHUNK ÷ 2)
    while half >= Int32(1)
        @inbounds t < half && (red[t + Int32(1)] += red[t + half + Int32(1)])
        KI.barrier()
        half ÷= Int32(2)
    end
    acc = 0f0
    n = min(Int32(CHUNK), last - s0 + Int32(1))
    @inbounds for j in Int32(1):n
        sj = s0 + j
        acc += w[j] * Float32(vc[t + Int32(1), sj, h, src[sj, b] + Int32(1)])
    end
    @inbounds begin
        pacc[t + Int32(1), c + Int32(1), h, b] = acc
        if t == Int32(0)
            pmax[c + Int32(1), h, b] = m
            psum[c + Int32(1), h, b] = red[1]
        end
    end
    return nothing
end

"""The chunks' partials of one head of one beam combined; item `t` is dimension `t`."""
function attncombine!(out, pmax, psum, pacc, slot)
    t = Int32(KI.get_local_id().x)
    wg = Int32(KI.get_group_id().x) - Int32(1)
    h = wg % Int32(HEADS) + Int32(1)
    b = wg ÷ Int32(HEADS) + Int32(1)
    nchunks = Int32(slot[1]) ÷ Int32(CHUNK) + Int32(1)
    m = -Inf32
    @inbounds for c in Int32(1):nchunks
        m = max(m, pmax[c, h, b])
    end
    l = 0f0
    acc = 0f0
    @inbounds for c in Int32(1):nchunks
        e = exp(pmax[c, h, b] - m)
        l += psum[c, h, b] * e
        acc += pacc[t, c, h, b] * e
    end
    @inbounds out[(h - Int32(1)) * Int32(CHUNK) + t, b] = acc / l
    return nothing
end

"""This step's table: each beam takes the rows of the beam it continues, and its
own column at `slot`."""
function reordertable!(dst, src, beamidx, slot, rows::Int32, nb::Int32)
    g = Int32(KI.get_global_id().x)
    g <= rows * nb || return nothing
    s, b = (g - Int32(1)) % rows + Int32(1), (g - Int32(1)) ÷ rows + Int32(1)
    last = Int32(slot[1]) + Int32(1)
    @inbounds if s < last
        dst[s, b] = src[s, beamidx[b] + Int32(1)]
    elseif s == last
        dst[s, b] = b - Int32(1)
    end
    return nothing
end

"""Copy `src` into `dst`, element by element."""
function copyall!(dst, src, n::Int32)
    g = Int32(KI.get_global_id().x)
    g <= n || return nothing
    @inbounds dst[g] = src[g]
    return nothing
end

"""The prefill's keys or values, `(64, P, heads, 1)` of one layer, into column 0 of the cache."""
function seedrows!(c, kv, layer::Int32, P::Int32)
    g = Int32(KI.get_global_id().x)
    g <= Int32(CHUNK) * P * Int32(HEADS) || return nothing
    d = (g - Int32(1)) % Int32(CHUNK) + Int32(1)
    r = (g - Int32(1)) ÷ Int32(CHUNK)
    s, h = r % P + Int32(1), r ÷ P + Int32(1)
    @inbounds c[d, s, h, 1] = kv[d, s, h, 1, layer]
    return nothing
end

# ── the decoder ──────────────────────────────────────────────────────────────

"""
    LayerWeights

One GPT-2 block on the device: layer norms as `(DIM, 2)` tables (weight, bias),
the projections and the MLP k-contiguous in the weight type, biases in float32.
"""
struct LayerWeights{N,W,V}
    ln1::N
    qkv::W
    qkvb::V
    o::W
    ob::V
    ln2::N
    fc::W
    fcb::V
    proj::W
    projb::V
end

"""
    GPTWeights

The decode step's weights on the device, shared by every [`Decoder`](@ref).
"""
struct GPTWeights{T,L,N,W,V,E}
    layers::Vector{L}
    lnf::N
    fnorm::N
    head::W
    headb::V
    emb::E
    posemb::E
end

ktranspose(w::AbstractMatrix, ::Type{T}) where {T} = Matrix{T}(permutedims(w))

"""Pack (weight, bias) as a `(DIM, 2)` layer-norm table."""
lnpair(w, b) = hcat(Float32.(w), Float32.(b))

"""
    GPTWeights(dev, w, T)

Upload the GPT from its export's weights (`b.gpt.h.<i>...`). Torch shapes read
reversed, so a `Conv1D` weight `(in, out)` arrives `(out, in)` and is transposed
to `(in, out)`, k-contiguous; `nn.Linear`'s `(out, in)` already arrives so.
Matrices in `T`, everything else float32.
"""
function GPTWeights(dev, w::AbstractDict, ::Type{T}) where {T}
    up(x) = Mantle.Buffer(dev, x)
    layers = map(0:(LAYERS - 1)) do i
        p = "b.gpt.h.$i."
        LayerWeights(up(lnpair(w[p * "ln_1.weight"], w[p * "ln_1.bias"])),
                     up(ktranspose(w[p * "attn.c_attn.weight"], T)), up(Float32.(w[p * "attn.c_attn.bias"])),
                     up(ktranspose(w[p * "attn.c_proj.weight"], T)), up(Float32.(w[p * "attn.c_proj.bias"])),
                     up(lnpair(w[p * "ln_2.weight"], w[p * "ln_2.bias"])),
                     up(ktranspose(w[p * "mlp.c_fc.weight"], T)), up(Float32.(w[p * "mlp.c_fc.bias"])),
                     up(ktranspose(w[p * "mlp.c_proj.weight"], T)), up(Float32.(w[p * "mlp.c_proj.bias"])))
    end
    emb = up(Matrix{Float32}(w["b.mel_embedding.weight"]))
    posemb = up(Matrix{Float32}(w["b.mel_pos_embedding.emb.weight"]))
    return GPTWeights{T,eltype(layers),typeof(layers[1].ln1),typeof(layers[1].qkv),typeof(layers[1].qkvb),
                      typeof(emb)}(
        layers, up(lnpair(w["b.gpt.ln_f.weight"], w["b.gpt.ln_f.bias"])),
        up(lnpair(w["b.final_norm.weight"], w["b.final_norm.bias"])),
        up(Matrix{T}(w["b.mel_head.weight"])), up(Float32.(w["b.mel_head.bias"])), emb, posemb)
end

"""
    Decoder(dev, weights; beams, rows, subgroup)

The cache for `beams` beams and `rows` positions, the step's small inputs and
its logits, and the step recorded once.
"""
mutable struct Decoder{T}
    weights::GPTWeights{T}
    k::Vector{Any}
    v::Vector{Any}
    src::Any
    beams::Int
    rows::Int
    tokens::Any
    beamidx::Any
    mpos::Any
    slot::Any
    logits::Any
    plan::Any
end

function Decoder(dev, w::GPTWeights{T}; beams::Int, rows::Int = CACHE_ROWS,
                 subgroup::Int = Mantle.caps(Mantle.backend(dev)).subgroup, profile::Bool = false) where {T}
    cache() = fill!(Mantle.Buffer(dev, T, (CHUNK, rows, HEADS, beams)), zero(T))
    d = Decoder{T}(w, Any[cache() for _ in 1:LAYERS], Any[cache() for _ in 1:LAYERS],
                   fill!(Mantle.Buffer(dev, Int32, (rows, beams)), Int32(0)), beams, rows,
                   fill!(Mantle.Buffer(dev, Int32, beams), Int32(0)),
                   fill!(Mantle.Buffer(dev, Int32, beams), Int32(0)),
                   Mantle.GPURef(dev, Int32(0)), Mantle.GPURef(dev, Int32(0)),
                   Mantle.Buffer(dev, Float32, (VOCAB_MEL, beams)), nothing)
    d.plan = recordstep(d, dev, subgroup; profile)
    return d
end

gemvkernel(nb::Int, sub::Int; nr::Int = 4, block::Int = 128) = BeamGEMV{nr,block,nb,sub}()

function gemv!(g, kern::BeamGEMV{NR,BLOCK}, y, W, x, bias, epi, K::Int, N::Int,
               aux = (nothing, nothing, nothing, nothing); name) where {NR,BLOCK}
    Mantle.dispatch!(g, kern, (y, W, x, bias, epi, Int32(K), Int32(N), aux...), cld(N, NR) * BLOCK;
                     group = BLOCK, name)
end

"""Declare one decode step into a graph and record it; `profile` times every pass."""
function recordstep(d::Decoder, dev, sub::Int; profile::Bool = false)
    B, W = d.beams, d.weights
    g = Mantle.Graph(dev)
    kern = gemvkernel(B, sub)
    maxchunks = cld(d.rows, CHUNK)
    h = Mantle.Transient.Buffer(g, Float32, (DIM, B))
    xn = Mantle.Transient.Buffer(g, Float32, (DIM, B))
    q = Mantle.Transient.Buffer(g, Float32, (CHUNK, HEADS, B))
    att = Mantle.Transient.Buffer(g, Float32, (DIM, B))
    ff = Mantle.Transient.Buffer(g, Float32, (FFN, B))
    pmax = Mantle.Transient.Buffer(g, Float32, (maxchunks, HEADS, B))
    psum = Mantle.Transient.Buffer(g, Float32, (maxchunks, HEADS, B))
    pacc = Mantle.Transient.Buffer(g, Float32, (CHUNK, maxchunks, HEADS, B))
    old = Mantle.Transient.Buffer(g, Int32, (d.rows, B))
    Mantle.dispatch!(g, copyall!, (old, d.src, Int32(d.rows * B)), d.rows * B; name = "decode/table copy")
    Mantle.dispatch!(g, reordertable!, (d.src, old, d.beamidx, d.slot, Int32(d.rows), Int32(B)), d.rows * B;
                     name = "decode/table")
    Mantle.dispatch!(g, embed!, (h, W.emb, W.posemb, d.tokens, d.mpos, Int32(B)), DIM * B; name = "decode/embed")
    for (i, L) in enumerate(W.layers)
        Mantle.dispatch!(g, layernorm!, (xn, h, L.ln1, Val(256)), 256B; group = 256, name = "decode/$i/ln1")
        gemv!(g, kern, q, L.qkv, xn, L.qkvb, QKVStore(), DIM, 3DIM, (q, d.k[i], d.v[i], d.slot);
              name = "decode/$i/qkv")
        Mantle.dispatch!(g, attnpartial!, (pmax, psum, pacc, q, d.k[i], d.v[i], d.src, d.slot, Val(maxchunks)),
                         maxchunks * HEADS * B * CHUNK; group = CHUNK, name = "decode/$i/attention")
        Mantle.dispatch!(g, attncombine!, (att, pmax, psum, pacc, d.slot), HEADS * B * CHUNK;
                         group = CHUNK, name = "decode/$i/combine")
        gemv!(g, kern, h, L.o, att, L.ob, Residual(), DIM, DIM; name = "decode/$i/o")
        Mantle.dispatch!(g, layernorm!, (xn, h, L.ln2, Val(256)), 256B; group = 256, name = "decode/$i/ln2")
        gemv!(g, kern, ff, L.fc, xn, L.fcb, GeluNew(), DIM, FFN; name = "decode/$i/fc")
        gemv!(g, kern, h, L.proj, ff, L.projb, Residual(), FFN, DIM; name = "decode/$i/proj")
    end
    Mantle.dispatch!(g, layernorm!, (xn, h, W.lnf, Val(256)), 256B; group = 256, name = "decode/ln_f")
    Mantle.dispatch!(g, layernorm!, (att, xn, W.fnorm, Val(256)), 256B; group = 256, name = "decode/final_norm")
    gemv!(g, kern, d.logits, W.head, att, W.headb, Plain(), DIM, VOCAB_MEL; name = "decode/head")
    Mantle.Readback(g, d.logits)
    plan = Mantle.Plan(g; profile, budget=Inf)
    Mantle.record!(plan)
    return plan
end


function resettable!(table, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n || return nothing
    @inbounds table[i] = Int32(0)
    return nothing
end

"""Declare prefix-cache seeding and beam-table reset into the prefill graph."""
function seedcache!(g, d::Decoder, K, V)
    P = size(K, 2)
    n = CHUNK * P * HEADS
    for l in 1:LAYERS
        Mantle.dispatch!(g, seedrows!, (d.k[l], K, Int32(l), Int32(P)), n; name="seed/k$l")
        Mantle.dispatch!(g, seedrows!, (d.v[l], V, Int32(l), Int32(P)), n; name="seed/v$l")
    end
    Mantle.dispatch!(g, resettable!, (d.src, Int32(length(d.src))), length(d.src); name="seed/table")
    return P
end

"""
    seed!(d, K, V) -> P

The prefill's keys and values `(64, P, heads, 1, layers)` into column 0 of the
cache, and every beam's table pointed at it.
"""
function seed!(d::Decoder, K, V, dev)
    P = size(K, 2)
    g = Mantle.Graph(dev)
    n = CHUNK * P * HEADS
    for l in 1:LAYERS
        Mantle.dispatch!(g, seedrows!, (d.k[l], K, Int32(l), Int32(P)), n; name = "seed/k$l")
        Mantle.dispatch!(g, seedrows!, (d.v[l], V, Int32(l), Int32(P)), n; name = "seed/v$l")
    end
    Mantle.runonce!(g)
    fill!(d.src, Int32(0))
    return P
end

"""
    step!(d, tokens, beamidx, melpos, slot) -> device logits `(8194, beams)`

One decode step: beam `b` continues beam `beamidx[b]` (zero-based) with
`tokens[b]` at mel position `melpos`, written to cache row `slot`.
"""
function step!(d::Decoder, tokens::AbstractVector{<:Integer}, beamidx::AbstractVector{<:Integer},
               melpos::Integer, slot::Integer; cpu::Bool=false)
    d.tokens[:] = Int32.(tokens)
    d.beamidx[:] = Int32.(beamidx)
    d.mpos[] = Int32(melpos)
    d.slot[] = Int32(slot)
    future = cpu ? Mantle.Readback(d.plan.graph, d.logits) : nothing
    Mantle.run!(d.plan)
    return cpu ? fetch(future) : d.logits
end

function Base.close(d::Decoder)
    d.plan === nothing || (Mantle.free!(d.plan); d.plan = nothing)
    foreach(Mantle.free!, d.k); empty!(d.k)
    foreach(Mantle.free!, d.v); empty!(d.v)
    foreach(Mantle.free!, (d.src, d.tokens, d.beamidx, d.mpos, d.slot, d.logits))
    return nothing
end

function Base.close(w::GPTWeights)
    for layer in w.layers, name in fieldnames(typeof(layer))
        Mantle.free!(getfield(layer, name))
    end
    foreach(Mantle.free!, (w.lnf, w.fnorm, w.head, w.headb, w.emb, w.posemb))
    return nothing
end
