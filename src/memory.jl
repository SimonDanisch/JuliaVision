"""
The cross-frame memory bank and its read.

This is the one graph that is authored rather than exported. `torch.export`
traces whatever length the bank happened to hold, but the bank grows with the
frame index, so the shape it recorded is not a fact about the model. What *is* a
fact - measured in tools/enumerate.py across two clips at different resolutions -
is that the bank saturates at `capacity = mem_frames * HW` tokens and never
exceeds it. So it is allocated once at capacity, a validity count says how much
is live, and the consumer masks (lava-dnn.md, `enumerate.py`).

The mask is per *frame slot*, not per token: the bank only ever holds whole
frames, so `mem_frames` booleans describe it completely.

Slot 1 is permanent (the first frame, added `as_permanent='first'` in
kv_memory_store.py:122) and is never evicted; the rest are a FIFO ring.
Ordering does not matter to the reader - the affinity softmax and its top-k are
permutation-invariant over memory tokens, and nothing indexes the bank by
position - so eviction is a pointer bump, not a shift.
"""

mutable struct MemoryBank{K,V,B}
    key::K              # (CAP, CK, B)
    shrinkage::K        # (CAP, 1, B)
    value::V            # (CAP, CV, NOBJ, B)
    objmem::B           # (E+1, Q, NOBJ, B) streaming sum; last row of dim 1 is the count
    hw::Int             # tokens per frame
    frames::Int         # capacity in frames
    nvalid::Int         # live frames, 1..frames
    next::Int           # next non-permanent slot to overwrite, 2..frames
    engaged::Bool
end

function MemoryBank(backend, T, hw, frames, ck, cv, nobj, bs, q, embed)
    cap = hw * frames
    MemoryBank(KernelAbstractions.allocate(backend, T, cap, ck, bs),
               KernelAbstractions.allocate(backend, T, cap, 1, bs),
               KernelAbstractions.allocate(backend, T, cap, cv, nobj, bs),
               KernelAbstractions.allocate(backend, T, embed + 1, q, nobj, bs),
               hw, frames, 0, 2, false)
end

slotrange(m::MemoryBank, s::Int) = ((s - 1) * m.hw + 1):(s * m.hw)

"""
    reset!(bank)

`clear_temp_mem` (inference_core.py:92): the working memory and the object
memory are dropped, which the warm-up does on every frame.
"""
function reset!(m::MemoryBank)
    m.nvalid = 0
    m.next = 2
    m.engaged = false
    fill!(m.objmem, zero(eltype(m.objmem)))
    m
end

"""
    add!(bank, key, shrinkage, value, summaries)

`key`/`shrinkage` are `(w, h, C, B)`, `value` is `(w, h, CV, NOBJ, B)`, all at
stride 16. The first add fills the permanent slot; later ones cycle the rest.
"""
function add!(m::MemoryBank, key, shrinkage, value, summaries)
    hw = m.hw
    slot = if !m.engaged
        m.engaged = true
        m.nvalid = 1
        1
    else
        s = m.next
        m.next = m.next == m.frames ? 2 : m.next + 1
        m.nvalid = min(m.nvalid + 1, m.frames)
        s
    end
    r = slotrange(m, slot)
    ck, cv = size(m.key, 2), size(m.value, 2)
    copyto!(view(m.key, r, :, :), reshape(key, hw, ck, :))
    copyto!(view(m.shrinkage, r, :, :), reshape(shrinkage, hw, 1, :))
    copyto!(view(m.value, r, :, :, :), reshape(value, hw, cv, size(m.value, 3), :))

    # streaming average (memory_manager.py:318): the last row accumulates the
    # count and the transformer divides by it
    if slot == 1 && m.nvalid == 1
        copyto!(m.objmem, summaries)
    else
        m.objmem .+= summaries
    end
    m
end

live(m::MemoryBank) = 1:(m.nvalid * m.hw)

# ---------------------------------------------------------------- the read

"""
Anisotropic L2 similarity, `get_similarity` (memory_utils.py:7) with the
selection term present.

    sim[q, n] = (-Σ_c mk[n,c]² qe[q,c] + 2 Σ_c mk[n,c] qk[q,c] qe[q,c]
                 - Σ_c qe[q,c] qk[q,c]²) * ms[n] / √CK

One thread per (query, memory-token) pair; the reduction over channels is
sequential in-thread. Padded slots are not reached because the launch covers
only the live range.
"""
@inline function similarity_body(I, mk, ms, qk, qe, ::Val{CK}, scale) where {CK}
    q, n, b = I
    @inbounds begin
        T = accum(eltype(mk))
        asq = zero(T); ab = zero(T); bsq = zero(T)
        for c in 1:CK
            m = T(mk[n, c, b])
            k = T(qk[q, c, b])
            e = T(qe[q, c, b])
            asq = muladd(m * m, e, asq)
            ab = muladd(m, k * e, ab)
            bsq = muladd(e, k * k, bsq)
        end
        (-asq + 2ab - bsq) * T(ms[n, 1, b]) * T(scale)
    end
end

"""
    topk_softmax_kernel!(sums, sim, Val(K), Val(ACC), Val(WG), N)

One **workgroup** per query, not one thread.

The original body ran one thread per query — 120 threads for the whole launch,
about 0.2% occupancy on a 48-SM device — and kept its running top-K in a
`K`-element tuple indexed by a runtime position. A dynamically indexed tuple
cannot stay in registers, so the array spilled to local memory and every
insertion copied the whole thing. It measured 3.2 ms of GPU time to produce 120
numbers.

Here the top-K lives in shared memory (indexed, never copied wholesale), the row
scan and the write-back are spread across the workgroup, and the launch has one
workgroup per query so the device is actually filled. The selection itself is
still serial on lane 1: with K=30 out of N≈360 a parallel selection needs either
K rounds of masked argmax or a full sort, both of which cost more barriers than
the serial scan costs shared-memory operations.

Numerics are unchanged, including upstream's exponentiation *without* subtracting
the row max (memory_utils.py:59) — the values are negated distances and so are
bounded above.
"""
@kernel function topk_softmax_kernel!(sums, sim, ::Val{K}, ::Val{ACC}, ::Val{WG},
                                      N) where {K,ACC,WG}
    @uniform T = eltype(sim)
    gq, gb = @index(Group, NTuple)
    tid, = @index(Local, NTuple)

    vals = @localmem ACC (K,)
    idxs = @localmem Int32 (K,)
    tot = @localmem ACC (1,)

    @inbounds if tid == 1
        for j in 1:K
            vals[j] = typemin(ACC)
            idxs[j] = Int32(0)
        end
        for n in 1:N
            v = ACC(sim[gq, n, gb])
            if v > vals[K]
                pos = K
                while pos > 1 && v > vals[pos - 1]
                    vals[pos] = vals[pos - 1]
                    idxs[pos] = idxs[pos - 1]
                    pos -= 1
                end
                vals[pos] = v
                idxs[pos] = Int32(n)
            end
        end
        s = zero(ACC)
        for j in 1:K
            s += exp(vals[j])
        end
        tot[1] = s
        sums[gq, gb] = T(s)
    end

    @synchronize

    # the row is zeroed cooperatively, then only the K survivors are written
    @inbounds begin
        n = tid
        while n <= N
            sim[gq, n, gb] = zero(T)
            n += WG
        end
    end

    @synchronize

    @inbounds if tid == 1
        s = tot[1]
        for j in 1:K
            idxs[j] > 0 && (sim[gq, idxs[j], gb] = T(exp(vals[j]) / s))
        end
    end
end

"""
`do_softmax` with top-k (memory_utils.py:59): only the `k` largest similarities
per query survive, and note that upstream exponentiates them *without*
subtracting the row max - the values are negated distances, so they are bounded
above, and reproducing the reference means reproducing that.

One thread per query. The k-selection is a running insertion into a fixed-size
register array, so nothing is allocated and `k` is a type parameter.
"""
@inline function topk_softmax_body(I, sim, ::Val{K}) where {K}
    q, b = I
    @inbounds begin
        T = eltype(sim)
        N = size(sim, 2)
        vals = ntuple(_ -> typemin(T), Val(K))
        idxs = ntuple(_ -> 0, Val(K))
        for n in 1:N
            v = sim[q, n, b]
            if v > vals[K]
                # insertion sort into the running top-K
                nv = vals; ni = idxs
                pos = K
                while pos > 1 && v > nv[pos - 1]
                    nv = Base.setindex(nv, nv[pos - 1], pos)
                    ni = Base.setindex(ni, ni[pos - 1], pos)
                    pos -= 1
                end
                vals = Base.setindex(nv, v, pos)
                idxs = Base.setindex(ni, n, pos)
            end
        end
        s = zero(T)
        for j in 1:K
            s += exp(vals[j])
        end
        for n in 1:N
            sim[q, n, b] = zero(T)
        end
        for j in 1:K
            idxs[j] > 0 && (sim[q, idxs[j], b] = exp(vals[j]) / s)
        end
        s
    end
end

"""`_readout` (memory_manager.py:77): `out[q, c, o] = Σ_n aff[q, n] v[n, c, o]`."""
@inline function memreadout_body(I, aff, v)
    q, c, o, b = I
    @inbounds begin
        T = accum(eltype(v))
        acc = zero(T)
        for n in axes(aff, 2)
            acc = muladd(T(aff[q, n, b]), T(v[n, c, o, b]), acc)
        end
        acc
    end
end

"""
    readmemory(bank, key, shrinkage_unused, query_key, selection; topk, backend)

Returns the visual readout as `(w, h, CV, NOBJ, B)`.
"""
function readmemory(m::MemoryBank, qk, qe, w, h; topk::Int=30,
                    backend=KernelAbstractions.get_backend(qk))
    hw = w * h
    ck = size(m.key, 2)
    bs = size(m.key, 3)
    n = m.nvalid * m.hw
    T = accum(eltype(m.key))

    qk2 = reshape(qk, hw, ck, bs)
    qe2 = reshape(qe, hw, ck, bs)
    mk = view(m.key, 1:n, :, :)
    ms = view(m.shrinkage, 1:n, :, :)

    sim = KernelAbstractions.allocate(backend, T, hw, n, bs)
    launch!(similarity_body, sim, mk, ms, qk2, qe2, Val(ck), inv(sqrt(ck)); backend)

    sums = KernelAbstractions.allocate(backend, T, hw, bs)
    let WG = 64
        topk_softmax_kernel!(backend, (WG, 1))(sums, sim, Val(topk), Val(T), Val(WG),
                                               size(sim, 2); ndrange = (WG * hw, bs))
    end

    cv, nobj = size(m.value, 2), size(m.value, 3)
    out = KernelAbstractions.allocate(backend, T, hw, cv, nobj, bs)
    launch!(memreadout_body, out, sim, view(m.value, 1:n, :, :, :); backend)
    reshape(out, w, h, cv, nobj, bs)
end
