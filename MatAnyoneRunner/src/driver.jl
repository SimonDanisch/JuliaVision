mutable struct State
    bank::MemoryBank
    sensory::Any            # (w, h, S, NOBJ, B)
    lastmask::Any           # (W, H, NOBJ, B) full resolution
    lastpixfeat::Any
    lastmskvalue::Any       # (w, h, CV, NOBJ, B)
    # `recordmemread`'s output. Its shape is `(hw, CV, NOBJ, B)` and every one of
    # those is fixed for the clip, so it is allocated once here rather than per
    # frame, and freed with the rest of `owned`. Named for what it holds: `step!`
    # has a local `readout` that is the pixel fusion's, which is a different
    # thing.
    visualreadout::Any
    # The memory read, recorded ONCE for the clip and replayed every frame.
    # Everything it names is stable — `qk`/`qe` are a recorded plan's outputs and
    # the bank's arrays are persistent — and the one thing that is not, how much
    # of the bank is live, lives in `memreadn` instead of in the recording. A
    # `GPURef` is the storage a per-run value lives in; passing the count by
    # value would bake it in and force a rebuild per frame.
    memreadplan::Any
    memreadn::Any
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
    visualreadout = M.Buffer(dev, T, (w * h, cv, nobj, bs))
    State(MemoryBank(m.backend, T, w * h, m.memframes, ck, cv, nobj, bs, q, embed),
          fill!(M.storage(sens), 0), fill!(M.storage(last), 0),
          nothing, nothing, visualreadout, nothing, M.GPURef(dev, Int32(0)),
          -1, 0, (h=h, w=w), (sens, last, visualreadout))
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
    s.memreadplan === nothing || (M.free!(s.memreadplan); s.memreadplan = nothing)
    M.free!(s.memreadn)
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
            # Recorded on the first frame that reads memory and replayed on
            # every one after. How much of the bank is live goes in the ref, so
            # the recording never has to change.
            s.memreadn[] = Int32(s.bank.nvalid * s.bank.hw)
            if s.memreadplan === nothing
                s.memreadplan = recordmemread(m.device, s.bank, key, selection,
                                              dims.w, dims.h, s.visualreadout,
                                              s.memreadn; topk = m.topk)
            end
            M.run!(s.memreadplan)
            visual = reshape(M.storage(s.visualreadout), dims.w, dims.h,
                             size(s.bank.value, 2), size(s.bank.value, 3),
                             size(s.bank.key, 3))
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
