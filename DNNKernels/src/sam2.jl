"""
SAM 2.1's single-image path: one encode per frame, one decode per click.

MatAnyone propagates a mask it is given; it does not find one. That is the whole
reason this is here — measured on both implementations, MatAnyone's matte tracks
its seed (IoU 0.95 falling to 0.82 over a clip) rather than segmenting the
subject, so the seed has to come from somewhere. SAM 2.1 is that somewhere, and
it is Apache 2.0, unlike SAM 3.

The split into two graphs is the interaction model, not an optimisation:

  * `encode` embeds a frame. It is the expensive half and depends only on the
    frame, so it runs once when the user starts marking.
  * `decode` turns points into masks against a cached embedding. It runs on every
    click, every drag, every added point — so it has to stay cheap enough that
    the overlay updates while the mouse moves.

Fusing them would re-embed the frame on every point, which is the difference
between a live preview and a progress bar.
"""

"""
    SAM2(graphdir, weights; backend, res = 1024, maxpoints = 16)

`res` and `maxpoints` describe the graphs as exported: the encoder takes a
`res`-square image, the decoder has `maxpoints` point slots. They are checked
against the graph on construction rather than trusted, because getting either
wrong produces a shape error hundreds of ops deep instead of here.
"""
struct SAM2
    model::Model
    res::Int
    maxpoints::Int
    dims::NamedTuple
    # The decoder's inputs in ITS dtype, and the `feats` they were made from.
    #
    # The two graphs are exported under different precision policies, so three of
    # the encoder's outputs need converting before the decoder can read them —
    # 12.6 MB of device copies. That is per *frame*, not per click: one embedding
    # serves every click on it, and the editor's whole shape is encode once,
    # decode many. Converting inside `decode` made it per click, which is 12.6 of
    # the 22.3 MB each decode allocated and a full pass over the features on
    # every one.
    #
    # Invalidated by `encode`, which clears the key. `===` on the feats tuple is
    # NOT enough on its own and the comment here used to say it was: the encoder
    # writes into the same slab buffers every frame, so the tuple `decode` gets
    # back is identical to the one it converted from and holds other numbers.
    cachekey::Base.RefValue{Any}
    cacheval::Base.RefValue{Any}
    # The two clicks-per-embedding fields. `prompts` is the one `(point, label)`
    # pair every click writes into, because a replay reads whatever those bytes
    # hold *now* — a fresh pair per click would leave the captured commands
    # pointing at freed storage. `replay` is `(feats, point, label, seq, out)`,
    # and holds those four alive for exactly that reason. It survives a new
    # embedding, because the addresses it recorded do not move; what retires it
    # is a converted input having to be reallocated.
    prompts::Base.RefValue{Any}
    replay::Base.RefValue{Any}

    # ── Policy. Fields rather than module-level `Ref`s (review finding 3): they
    # describe *this* model, so two of them can be configured differently and be
    # alive at once, and a test that changes one cannot leak into the next.

    # ── Reuse the decoder's dtype-converted inputs across clicks on one embedding.
    #
    # **On, and it is a precondition rather than a saving.** The saving is real and measured — the
    # conversion is 12.6 MB of the 22.3 MB each `decode` allocates, and it is per
    # click where it should be per frame — but the numerical check needs a decode
    # result read back, and every attempt at that hit the flush hang (see
    # `perf-plan.md`; it fired about ten times in one afternoon). Shipping a cache on
    # the path that produces masks, without having compared one against the uncached
    # result, is not a trade worth making for 12.6 MB.
    #
    # **The open question is answered.** It asked whether any decoder op writes into an
    # *input* buffer, since the planner is free to alias. Read back on 2026-08-02: the
    # cached first call and the cached *second* call both differ from the uncached
    # result by `0.000e+00`. It agrees with the static argument — no op in either graph
    # names its own output among its inputs, and `escaping` gives externals no slab
    # space — so neither structural route to aliasing exists.
    #
    # **And it buys no time on its own: +0.008 ms.** 12.6 MB at 250 GB/s is 0.05 ms;
    # the "12.6 of the 22.3 MB each decode allocates" above was about allocation churn
    # and had been read as though it were about time. What it *is* worth is
    # `replaydecode`, which needs every device address to be identical next
    # call — "an input buffer written in place rather than reallocated" — and that is
    # 1 ms. Turning this off turns that off with it.
    cacheinputs::Bool

    # ── Capture the decoder's command buffers on the first click and re-submit
    # them on every click after.
    #
    # **Once, not once per frame.** The recorded addresses are the slab's, the
    # weights', and the persistent pair `prompt` writes into, and none of those move
    # when a new image is encoded — the encoder overwrites its outputs in place. So a
    # capture keeps serving after `encode`, and it is checked that way rather than
    # assumed: `test_replay_decode.jl` encodes a second image and compares the
    # replayed masks against the recorded path bit for bit.
    #
    # **40% of a 3.9 ms decode is not any graph op** — it is the host rebuilding a
    # 149-dispatch launch sequence that is identical every time. `replay!` re-submits
    # it with one `vkQueueSubmit2` and no recording: **4.211 -> 3.208 ms**, bit-exact,
    # 50% of PyTorch to 65%.
    #
    # The preconditions are `capture`'s, and all three are things this file already
    # does for other reasons: a statically planned slab, fixed weights, and inputs
    # written in place. That last one is why `cacheinputs` matters — measured
    # alone it is worth 0.008 ms and looks pointless, and it is the thing that makes a
    # replay legal. `prompt` writes into one persistent pair for the same reason.
    #
    # The masks come back in the captured buffer, so they are valid until the next
    # decode. That was already true of the slab-backed result and is worth saying
    # twice: materialising two decodes' outputs and then comparing them compares one
    # array with itself.
    replaydecode::Bool

    # ── How close two predicted IoUs have to be for `pick = :confident` to treat
    # them as a tie. 0.1 covers the cases measured; see [`segment`](@ref).
    segmenttie::Float32
end

function SAM2(graphdir::AbstractString, weightpath::AbstractString;
              backend=KernelAbstractions.CPU(), res::Int=1024, maxpoints::Int=16,
              cacheinputs::Bool=true, replaydecode::Bool=true,
              segmenttie::Real=0.1f0)
    m = Model(graphdir, weightpath; backend, names=["sam2_encoder", "sam2_decoder"])
    enc, dec = m.graphs["sam2_encoder"], m.graphs["sam2_decoder"]
    # torch order, so the image is (n, c, y, x) and the point list (n, k, 2).
    img = enc.buffers[enc.inputs[1]].shape
    img[end] == res || error("encoder graph takes a $(img[end])-square image, not $res")
    pts = dec.buffers[dec.inputs[4]].shape
    pts[2] == maxpoints || error("decoder graph has $(pts[2]) point slots, not $maxpoints")
    SAM2(m, res, maxpoints, (res=res,), Ref{Any}(nothing), Ref{Any}(nothing),
         Ref{Any}(nothing), Ref{Any}(nothing),
         cacheinputs, replaydecode, Float32(segmenttie))
end

KernelAbstractions.get_backend(s::SAM2) = s.model.backend

"""
    encode(s, image) -> features

`image` is `(res, res, 3, 1)` in 0..1 RGB — the graph carries the ImageNet
normalisation, so this is a plain picture, not a pre-whitened tensor.

Returns all six encoder outputs; `decode` reads the first three and the rest are
the positional encodings the graph produced alongside them. Keep the whole tuple
between clicks: that *is* the cached embedding.
"""
function encode(s::SAM2, image)
    # A new embedding has to invalidate the converted decoder inputs here,
    # because `decode` cannot detect it: `call` writes the encoder's outputs into
    # the same statically planned slab buffers every time, so it returns a tuple
    # that is `===` to the one `decode` last converted from while holding
    # different numbers. The cache's own comment used to claim identity was
    # enough; it never was, and it only went unnoticed because the shipped
    # autocast decoder declares the dtypes the encoder already produces, so
    # nothing is converted and the "cache" holds the live features themselves.
    s.cachekey[] = nothing
    call(s.model, "sam2_encoder", image; dims=s.dims)
end


"""
    decode(s, feats, point, label) -> (masks, iou)

`point` is `(2, k, 1)` in the model's own `res`-square pixel coordinates and
`label` is `(k, 1)` of `Int32`. Both are reversed-layout, like everything else
here.

`masks` is `(256, 256, 3, 1)` of **logits**, not probabilities: SAM thresholds at
0, so `mask > 0` is the mask and the magnitude is a soft edge worth keeping when
the result is about to be resampled. `iou` is `(3, 1)`, the model's own estimate
of how good each of the three is.
"""
function decode(s::SAM2, feats, point, label; replay::Bool = s.replaydecode)
    g = s.model.graphs["sam2_decoder"]
    # The two graphs are exported under different precision policies — the
    # encoder under autocast, so its matmuls land on fp16 tensor cores, and the
    # decoder in fp32, where 205 ops on a 64x64 embedding cost nothing and the
    # mask logits keep their precision through the threshold. That makes the
    # handover a dtype boundary, and the graph's own declaration is what decides
    # it, not an assumption about which policy either side was built with.
    # Converted once per embedding, not once per click — see the cache fields on
    # `SAM2`. `encode` clears the key, which is what makes a new embedding
    # rebuild this; `===` alone cannot, since the tuple is the same one.
    args = if s.cacheinputs && s.cachekey[] === feats
        s.cacheval[]::NTuple{3,Any}
    else
        prev = s.cacheval[]
        fresh = false
        a = ntuple(3) do i
            want = g.buffers[g.inputs[i]].dtype
            f = feats[i]
            eltype(f) === want && return f
            # Converted INTO the previous destination whenever there is a usable
            # one, rather than allocated afresh. A `replaydecode` capture
            # records these addresses in its push constants, and a replay over
            # moved addresses does not fail — it re-runs against whatever now
            # lives there and returns a plausible mask. When the destination
            # genuinely has to move, drop the capture instead.
            d = prev === nothing ? nothing : prev[i]
            if !(d isa AbstractArray) || eltype(d) !== want || size(d) != size(f)
                d = similar(f, want)
                fresh = true
            end
            d .= f
            d
        end
        fresh && (s.replay[] = nothing)
        s.cachekey[] = feats
        s.cacheval[] = a
        a
    end
    # Padded attention tiles are the decoder's business, not the encoder's.
    # `clampattn` lets an attention whose extents do not divide the tile take the
    # cooperative-matrix path anyway, padded and masked. The decoder's are 23
    # tokens and want exactly that; the encoder has six `Lq = 16` calls that would
    # go along at 50% waste, and they cost **+2.12 ms of encode** for nothing.
    # Measured interleaved in one process on the autocast export, which is the only
    # form that means anything for a 4 ms call on a shared card: decode 8.24 ->
    # 4.22 ms with the clamp, encode 118.63 -> 120.75.
    #
    # It was a `Ref` set here and restored in a `finally`, on the grounds that
    # `flashcm_tiling` reads it six frames down inside `runop!`. It is now an
    # argument to the one call that wants it: the context carries it those six
    # frames, which is what the context is for. Nothing to restore, so nothing an
    # error can leave switched on for the next encode.
    run() = call(s.model, "sam2_decoder", args[1], args[2], args[3], point, label;
                 dims=s.dims, clampattn=true)
    replay && replayable(s, feats, point, label) || return run()

    r = s.replay[]
    if r !== nothing && r[1] === feats && r[2] === point && r[3] === label
        Lava.replay!(r[4])
        return r[5]
    end
    # First click on this embedding: `capture` RUNS the body, so this costs one
    # decode and not two. Everything it records must keep its addresses — the slab
    # is statically planned, the weights are fixed, `args` comes from the cache
    # above and `point`/`label` are the persistent pair `prompt` writes into.
    bq = Lava.VK_CONTEXT_REF[].default_bq
    out = nothing
    seq = Lava.capture(bq) do
        out = run()
    end
    s.replay[] = (feats, point, label, seq, out)
    return out
end


"""Whether this call can be captured and replayed — see `SAM2.replaydecode`."""
function replayable(s::SAM2, feats, point, label)
    s.cacheinputs || return false
    s.model.backend isa Lava.LavaBackend || return false
    pb = s.prompts[]
    pb !== nothing && point === pb[1] && label === pb[2]
end

"""
    prompt(s, points, labels) -> (point, label)

Build the decoder's two prompt tensors from `k <= maxpoints` points.

`points` are `(x, y)` in **normalized frame coordinates**, 0..1 with the origin
top-left, which is what a click on the preview gives after the canvas matrices
have had their say. `labels` are `true` for foreground and `false` for
background.

Unused slots are filled with label -1, SAM's "not a point": the prompt encoder
substitutes a learned embedding for those, so a graph exported for 16 points
answers a one-point question correctly.
"""
function prompt(s::SAM2, points, labels)
    k = length(points)
    k == length(labels) || error("$(k) points but $(length(labels)) labels")
    k <= s.maxpoints || error("$k points, but the decoder graph has $(s.maxpoints) slots")
    xy = zeros(Float32, 2, s.maxpoints, 1)
    lb = fill(Int32(-1), s.maxpoints, 1)
    for (i, p) in enumerate(points)
        xy[1, i, 1] = Float32(p[1] * s.res)
        xy[2, i, 1] = Float32(p[2] * s.res)
        lb[i, 1] = labels[i] ? Int32(1) : Int32(0)
    end
    b = s.model.backend
    # Written into ONE persistent pair, not a fresh one per click. `replay!`
    # re-submits command buffers whose push constants point at these exact
    # addresses, so the buffers have to outlive the capture — see `decode`.
    pb = s.prompts[]
    if pb === nothing
        pb = (toback(b, xy), toback(b, lb))
        s.prompts[] = pb
    else
        copyto!(pb[1], xy)
        copyto!(pb[2], lb)
    end
    pb
end


"""
    segment(s, feats, points, labels; pick = :best) -> (mask, score)

One decode, resolved to a single 256x256 logit plane.

`pick` selects among SAM's three proposals:

  * `:best` — highest predicted IoU. **SAM's own convention**, and what
    `SAM2ImagePredictor` does; kept exactly so, so that parity against PyTorch
    means what it says.
  * `:confident` — highest predicted IoU, but a near-tie is broken by the
    strongest logits. See below.
  * `1`, `2`, `3` — that proposal outright, which is what a "cycle through the
    alternatives" control in the UI would do. The ambiguity of a single click (a
    stripe, the jacket, the person) is real, and resolving it by argmax forever
    would throw away the reason the model returns three.

**Why `:confident` exists.** Predicted IoU and logit magnitude disagree, and when
they do, argmax picks a mask that is barely above threshold everywhere and so
comes out speckled. Measured on a real editor frame, three clicks, and confirmed
identical in PyTorch (so this is SAM's behaviour, not a porting artifact):

    click         proposal  predicted IoU  max logit  boundary/compact
    (0.55, 0.50)     2          0.478         5.09         3.6x
                     3          0.516         1.55        13.6x   <- :best takes this
    (0.50, 0.30)     2          0.913         1.92         8.4x   <- :best takes this
                     3          0.904         5.98         6.2x

A 0.04 and a 0.009 difference in a *predicted* score, against 3x the logit
magnitude and half the boundary. For a matte seed that is the wrong trade: the
seed is what MatAnyone propagates for the length of the clip, and it is thresholded
at zero, so a mask that hovers at zero fragments the moment it is resampled to
frame resolution.
"""
function segment(s::SAM2, feats, points, labels; pick=:best)
    pt, lb = prompt(s, points, labels)
    masks, iou = decode(s, feats, pt, lb)
    scores = Array(iou)[:, 1]
    i = if pick === :best
        argmax(scores)
    elseif pick === :confident
        best = maximum(scores)
        near = findall(>=(best - s.segmenttie), scores)
        # ONE download of all three planes (768 KB), not one per candidate. Each
        # download drains the batch this decode was recorded into, and repeatedly
        # draining a queue that is still being written is how the decode path
        # reaches `vkWaitSemaphores` timeouts.
        h = Array(masks)
        near[argmax(maximum(view(h, :, :, k, 1)) for k in near)]
    else
        Int(pick)
    end
    (view(masks, :, :, i, 1), scores[i])
end
