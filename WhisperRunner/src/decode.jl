# The autoregressive half: cache, loop, sampling.
#
# The decoder graph is ONE step — `tools/export_whisper_decoder.py` exports
#
#     (input_ids, self_k, self_v, cross_k, cross_v, cache_position)
#         -> (logits, new_self_k, new_self_v)
#
# and everything that makes it a transcript lives here, on the host. That split
# is llama.cpp's and whisper.cpp's, and it is what makes the decoder exportable
# at all: the *loop* is not a static graph, a *step* is.
#
# The cache goes in and comes back out. That is not decoration: a graph that
# takes the cache and does not return it compiles, runs, and decodes against a
# cache that never advances — the first export did exactly that, and the symptom
# is a fluent transcript of the wrong words. The exporter's docstring has the
# details.
#
# ## Where the time goes
#
# 4 decoder layers (large-v3-turbo is distilled from 32), so per token the model
# reads ~158M parameters — 92M of decoder weights plus the 66M output projection
# over 51866 tokens. At 634 MB in fp32 and the 307 GB/s this card sustains, a
# token cannot cost less than **2.07 ms** and essentially all of it is weight
# traffic. Every matmul has `M = 1`.
#
# That is why `Lava.gemv!` exists: `mul!` measured 8.7 ms/token on these shapes
# because `coopmat_gemm!` needs `M >= 16` and pads away fifteen sixteenths of
# every tile.
#
# On top of the weights, the cache costs: reading it is 18 MB of self plus 61 MB
# of cross per token, and the functional write copies the self cache twice more
# (`index_put` then `cat`) plus once again in [`decodestep!`](@ref). Roughly 0.35
# ms of the budget is cache traffic that an in-place write would remove.
#
# ## The cross-attention cache is computed once
#
# Cross-attention reads the *encoder* output, which does not change while
# decoding a window. So its K and V are projected once per window and reused for
# every token — `(4, 1, 20, 1500, 64)` each. Recomputing them per step would
# multiply the decoder's cost by roughly the ratio of 1500 to the token count,
# which is most of why the turbo decoder is cheap at all. It is also why
# `encoder_hidden` is not an input to the step at all.

"""
    KVCache

The decoder's state between steps: self-attention K/V grown one slot per token,
and cross-attention K/V fixed for the window.

Fixed-capacity, not grown. `max_target` slots are allocated once and written at
`position`, because a reallocation per token would dominate a step whose whole
budget is 2 ms — and because the exported graph has the capacity baked into its
shapes.
"""
mutable struct KVCache{A}
    self_k::A          # (layers, 1, heads, max_target, head_dim), torch order
    self_v::A
    cross_k::A         # (layers, 1, heads, src_len, head_dim)
    cross_v::A
    position::Int      # how many slots of the self cache are live
end

"""
    kvcache(backend, layers, heads, headdim; max_target = 448, src_len = 1500)

Allocate the caches for one decoding window, zeroed.

Zeroed matters less than it looks — the graph masks every slot past
`cache_position`, so stale contents are unattended — but a NaN left in an
unattended slot still reaches the softmax on some paths, and zero is the value
that is inert if the mask is ever wrong. 4 MB, once per model.

Shapes are stored in torch's order and reversed by the runtime, so `self_k` is
`(head_dim, max_target, heads, 1, layers)` in Julia. Written that way here so it
lines up with the graph's declared `[4, 1, 20, 448, 64]` at a glance.
"""
function kvcache(backend, layers::Int, heads::Int, headdim::Int;
                 max_target::Int = 448, src_len::Int = 1500)
    z(n) = (a = KA.allocate(backend, Float32, headdim, n, heads, 1, layers);
            fill!(a, 0.0f0); a)
    KVCache(z(max_target), z(max_target), z(src_len), z(src_len), 0)
end

"""
    reset!(c::KVCache)

Rewind to the start of a window. Does not clear the buffers — `position = 0`
already makes every slot unattended. The cross cache is *not* touched; it belongs
to the window and [`fillcross!`](@ref) replaces it.
"""
reset!(c::KVCache) = (c.position = 0; c)

"""
    Whisper

Encoder and decoder together, with the weights resident. Built by
[`whisper`](@ref).

Two graphs, not one: the encoder runs once per 30 s window and the decoder once
per token, so they are separate `Model`s with separate scratch plans. Fusing them
would mean planning a slab for the union of two workloads that never run at the
same time.

`ids` and `pos` are the step's two scalar inputs, allocated once. They are two
device buffers of one element each and re-allocating them per token would put a
pool round-trip inside the hot loop.

`melfilters` is the Slaney filterbank, built once — it is a closed-form table and
depends only on `(nmels, nfreq, rate)`. `suppress`/`beginsuppress` are the
decoding policy's two token lists, read from `generation_config.json` at load
rather than per window.
"""
struct Whisper{B,ME,MD,C,I2,I1,F}
    backend::B
    encoder::ME
    decoder::MD
    cache::C
    ids::I2            # (1, 1) Int64, the token to condition on
    pos::I1            # (1,)   Int64, the cache slot to write
    melfilters::F
    assets::String
    suppress::Vector{Int}          # `suppress_tokens`, banned at every step
    beginsuppress::Vector{Int}     # ...and these only as the first generated token
    layers::Int
    heads::Int
    headdim::Int
    vocab::Int
end

"""
    whisper(; backend = LavaBackend(), dir = assetdir(), decdir = decoderdir())
        -> Whisper

Load both halves. The decoder artifact is separate from the encoder's — it is a
different export with different weights, and a caller who only wants embeddings
should not pay for the autoregressive half.
"""
function whisper(; backend = LavaBackend(),
                   precision::Symbol = :fp16,
                   dir::AbstractString = assetdir(precision),
                   decdir::AbstractString = decoderdir(),
                   maxtarget::Int = 448, srclen::Int = 1500)
    enc = Model(dir, joinpath(dir, "weights.safetensors");
                names = ["whisper"], backend)
    dec = Model(decdir, joinpath(decdir, "weights.safetensors");
                names = ["whisperdec", "whispercross"], backend)
    layers, heads, hd = 4, 20, 64            # large-v3-turbo
    cache = kvcache(backend, layers, heads, hd; max_target = maxtarget, src_len = srclen)
    # Int64, because the graph declares `input_ids` and `cache_position` int64 —
    # an Int32 buffer under an int64 binding reads two tokens as one.
    #
    # 201 = nfft/2 + 1 for Whisper's 400-sample window; 128 mel bins for
    # large-v3 (80 for every earlier checkpoint, which is why it is not implicit).
    # Read once. A streaming caller runs a window every few hundred ms and
    # `suppresslists` opens and parses a JSON file; that belongs at load time.
    sup, bsup = suppresslists(decdir)
    Whisper(backend, enc, dec, cache,
            KA.allocate(backend, Int64, 1, 1), KA.allocate(backend, Int64, 1),
            melfilters(128, 201, 16000), String(decdir), sup, bsup,
            layers, heads, hd, 51866)
end

"""
    fillcross!(w::Whisper, enc_hidden) -> w

Project the encoder output into the cross-attention K/V caches, once per window.

This is the `whispercross` graph — 8 matmuls of `(1500,1280)@(1280,1280)` for the
4 turbo layers. Per *token* cross-attention then only projects the query, which
is most of why the turbo decoder is cheap; recomputing these per step would
multiply its cost by roughly the token count.
"""
function fillcross!(w::Whisper, enc_hidden)
    k, v = call(w.decoder, "whispercross", enc_hidden; dims = (;))
    copyto!(w.cache.cross_k, k)
    copyto!(w.cache.cross_v, v)
    return w
end

"""
    decodestep!(w::Whisper, token::Integer) -> logits

One token through the decoder. `token` is the id to condition on; the returned
logits are over the full vocabulary for the *next* token.

The cache advances by one slot as a side effect. The graph returns the updated
cache as an ordinary output and this copies it back, which is ~18 MB per token —
the price of the cache being data rather than mutable state. Keeping the returned
buffers instead of copying is not safe: they are slots in the model's planned
scratch slab and the next `call` overwrites them.
"""
function decodestep!(w::Whisper, token::Integer)
    c = w.cache
    copyto!(w.ids, reshape(Int64[token], 1, 1))
    copyto!(w.pos, Int64[c.position])
    logits, newk, newv = call(w.decoder, "whisperdec",
                              w.ids, c.self_k, c.self_v, c.cross_k, c.cross_v, w.pos;
                              dims = (;))
    copyto!(c.self_k, newk)
    copyto!(c.self_v, newv)
    c.position += 1
    return logits
end

"""
    greedy(w::Whisper, prompt; maxtokens = 224, eot = 50257) -> Vector{Int}

Greedy decode: at every step take the argmax and feed it back.

Greedy first on purpose. It is the policy that is *falsifiable* — the token
sequence must match `WhisperForConditionalGeneration.generate` with
`do_sample=false, num_beams=1` exactly, token for token — and every richer policy
(temperature fallback, beam search, the compression-ratio and no-speech
thresholds) is a change to which token is chosen, not to any of the machinery
underneath. Getting those right before this one is checkable would be building on
sand.

`prompt` is the forced prefix — `<|startoftranscript|><|en|><|transcribe|>` and
friends — fed through the decoder without sampling, because those tokens are
decided by the caller and not by the model.

The argmax is host-side, which costs one 207 KB download and one sync per token.
The sync is not avoidable in a greedy loop — the next token *is* the next input —
and the download is ~20 us against a 2 ms step, so it is not worth a device-side
argmax yet.

[`fillcross!`](@ref) must have run for this window first; the step never sees the
encoder output.
"""
function greedy(w::Whisper, prompt::AbstractVector{<:Integer};
                maxtokens::Int = 224, eot::Int = 50257)
    reset!(w.cache)
    out = Int[]
    logits = nothing
    for t in prompt
        logits = decodestep!(w, t)
    end
    for _ in 1:maxtokens
        nxt = argmax(vec(Array(logits))) - 1          # 0-based token id
        nxt == eot && break
        push!(out, nxt)
        w.cache.position >= size(w.cache.self_k, 2) && break
        logits = decodestep!(w, nxt)
    end
    return out
end

# `transcribe` used to live here as a single-window, fixed-prompt helper. It is
# now `window.jl`: real audio needs the 30 s loop, the temperature fallback and
# the no-speech test, and a second entry point that skipped all three was a trap
# rather than a convenience.
