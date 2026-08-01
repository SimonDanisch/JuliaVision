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
    # Keyed by `===` on the feats tuple, so a new embedding invalidates it and
    # nothing has to remember to.
    cachekey::Base.RefValue{Any}
    cacheval::Base.RefValue{Any}
end

function SAM2(graphdir::AbstractString, weightpath::AbstractString;
              backend=KernelAbstractions.CPU(), res::Int=1024, maxpoints::Int=16)
    m = Model(graphdir, weightpath; backend, names=["sam2_encoder", "sam2_decoder"])
    enc, dec = m.graphs["sam2_encoder"], m.graphs["sam2_decoder"]
    # torch order, so the image is (n, c, y, x) and the point list (n, k, 2).
    img = enc.buffers[enc.inputs[1]].shape
    img[end] == res || error("encoder graph takes a $(img[end])-square image, not $res")
    pts = dec.buffers[dec.inputs[4]].shape
    pts[2] == maxpoints || error("decoder graph has $(pts[2]) point slots, not $maxpoints")
    SAM2(m, res, maxpoints, (res=res,), Ref{Any}(nothing), Ref{Any}(nothing))
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
encode(s::SAM2, image) = call(s.model, "sam2_encoder", image; dims=s.dims)

"""
    CACHE_DECODER_INPUTS[] :: Bool

Reuse the decoder's dtype-converted inputs across clicks on one embedding.

**Off, because it is unverified.** The saving is real and measured — the
conversion is 12.6 MB of the 22.3 MB each `decode` allocates, and it is per
click where it should be per frame — but the numerical check needs a decode
result read back, and every attempt at that hit the flush hang (see
`perf-plan.md`; it fired about ten times in one afternoon). Shipping a cache on
the path that produces masks, without having compared one against the uncached
result, is not a trade worth making for 12.6 MB.

To verify: decode twice on the same `feats` with this on, once more with
`s.cachekey[] = nothing` in between, and compare. The open question it settles is
whether any decoder op writes into an *input* buffer — the planner is free to
alias, and if it does, the second click on an embedding reads corrupted features.
Turn this on when that comparison passes.
"""
const CACHE_DECODER_INPUTS = Ref(false)

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
function decode(s::SAM2, feats, point, label)
    g = s.model.graphs["sam2_decoder"]
    # The two graphs are exported under different precision policies — the
    # encoder under autocast, so its matmuls land on fp16 tensor cores, and the
    # decoder in fp32, where 205 ops on a 64x64 embedding cost nothing and the
    # mask logits keep their precision through the threshold. That makes the
    # handover a dtype boundary, and the graph's own declaration is what decides
    # it, not an assumption about which policy either side was built with.
    # Converted once per embedding, not once per click — see the cache fields on
    # `SAM2`. `===` on the tuple, so a fresh `encode` invalidates it by identity.
    args = if CACHE_DECODER_INPUTS[] && s.cachekey[] === feats
        s.cacheval[]::NTuple{3,Any}
    else
        a = ntuple(3) do i
            want = g.buffers[g.inputs[i]].dtype
            f = feats[i]
            eltype(f) === want ? f : want.(f)
        end
        s.cachekey[] = feats
        s.cacheval[] = a
        a
    end
    call(s.model, "sam2_decoder", args[1], args[2], args[3], point, label; dims=s.dims)
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
    (toback(b, xy), toback(b, lb))
end

"""
    SEGMENT_TIE[] :: Float32

How close two predicted IoUs have to be for `pick = :confident` to treat them as
a tie. 0.1 covers the cases measured; see [`segment`](@ref).
"""
const SEGMENT_TIE = Ref(0.1f0)

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
        near = findall(>=(best - SEGMENT_TIE[]), scores)
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
