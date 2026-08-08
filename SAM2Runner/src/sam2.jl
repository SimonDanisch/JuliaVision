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
    # The `feats` tuple the plan's feature buffers were last filled from.
    # `encode` sets it in the same call that fills them, so the key and the
    # contents cannot disagree, and `===` is a complete test: the encoder plan
    # returns its own output buffers, whose identity is stable across frames.
    # `decode` re-stages by copy on a miss.
    cachekey::Base.RefValue{Any}
    # The one `(point, label)` pair every click writes into, because the baked
    # decoder plan was built against these exact arrays and reads whatever they
    # hold *now* — a fresh pair per click would leave the plan reading the old
    # one, which is not an error, it is a plausible mask for the previous click.
    prompts::Base.RefValue{Any}

    # ── The two Mantle plans, built on first `encode` and then baked.
    #
    # **Two, not one.** `buildchain` can put both graphs in a single plan and make
    # a whole step one queue submission, and that is the right shape for "here is
    # a picture and a click". It is the wrong shape for the editor, whose whole
    # pattern is encode once and decode many: a chain re-encodes per click, which
    # is 300 ms where a decode is 4.
    #
    # So the encoder is one baked plan and the decoder another, with the dtype
    # conversion between them done in `encode` — per frame, which is where it
    # belongs, and which is the whole of what `cacheinputs` used to arrange.
    #
    # **This is what replaced `replay`.** That field held a raw
    # `Lava.CapturedSequence` plus the four objects it needed kept alive, because
    # a capture holds no references to the buffers its push constants point at.
    # Under GC pressure they went anyway, and the replay dispatched against
    # address 0 — which is what made `@setup_workload` lose the device on every
    # fresh precompile and silently freeze nothing. A Mantle plan holds
    # references to everything it names, so the failure is not guarded against
    # here; it cannot be expressed.
    plans::Base.RefValue{Any}

    # ── Policy. Fields rather than module-level `Ref`s (review finding 3): they
    # describe *this* model, so two of them can be configured differently and be
    # alive at once, and a test that changes one cannot leak into the next.

    # ── Reuse the decoder's dtype-converted inputs across clicks on one embedding.
    #
    # **Inert: the conversion moved into `encode`.** This used to gate a cache in
    # `decode` of the three encoder outputs converted to the decoder's dtypes.
    # The baked plans made that moot — `plansfor` owns persistent `feat` buffers
    # and `encode` fills them per frame, which is where the conversion belongs —
    # so nothing reads this flag any more. It stays in the signature for one
    # release, like `replaydecode`, so existing calls keep working. The
    # measurements that justified the cache still stand: 12.6 MB of device copies
    # per click if it were done in `decode`, +0.008 ms of actual time, and a
    # read-back on 2026-08-02 showing no decoder op writes into an input buffer
    # (cached and uncached decodes agree to `0.000e+00`).
    cacheinputs::Bool

    # ── How close two predicted IoUs have to be for `pick = :confident` to treat
    # them as a tie. 0.1 covers the cases measured; see [`segment`](@ref).
    segmenttie::Float32
end

function SAM2(graphdir::AbstractString, weightpath::AbstractString;
              backend=KernelAbstractions.CPU(), res::Int=1024, maxpoints::Int=16,
              cacheinputs::Bool=true, replaydecode::Bool=true,
              segmenttie::Real=0.1f0)
    # Accepted and ignored: a plan replays by construction, so there is nothing
    # to switch off. Kept in the signature for one release so that callers
    # passing `replaydecode = false` — which was the documented way around the
    # capture fault — keep working instead of erroring.
    replaydecode || @warn "SAM2: `replaydecode = false` no longer does anything. " *
        "The decoder is a baked Mantle plan, which holds references to everything " *
        "it names, so the capture/replay fault it worked around cannot occur." maxlog = 1
    # Same story: the conversion this gated now happens in `encode`, per frame,
    # into the plan's own feature buffers.
    cacheinputs || @warn "SAM2: `cacheinputs = false` no longer does anything. " *
        "The dtype conversion it gated moved into `encode`, which writes the " *
        "plan's feature buffers per frame." maxlog = 1
    m = Model(graphdir, weightpath; backend, names=["sam2_encoder", "sam2_decoder"])
    enc, dec = m.graphs["sam2_encoder"], m.graphs["sam2_decoder"]
    # torch order, so the image is (n, c, y, x) and the point list (n, k, 2).
    img = enc.buffers[enc.inputs[1]].shape
    img[end] == res || error("encoder graph takes a $(img[end])-square image, not $res")
    pts = dec.buffers[dec.inputs[4]].shape
    pts[2] == maxpoints || error("decoder graph has $(pts[2]) point slots, not $maxpoints")
    SAM2(m, res, maxpoints, (res=res,), Ref{Any}(nothing), Ref{Any}(nothing),
         Ref{Any}(nothing),
         cacheinputs, Float32(segmenttie))
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
    p = plansfor(s)
    # In place. The baked plan names this address, and a fresh buffer per frame
    # would leave the recording reading the old one — which is not an error, it
    # is a plausible mask computed from the previous picture.
    copyto!(p.image, image)
    out = DNNKernels.runplan(p.enc)
    # The dtype handover, here rather than in `decode`: it is per *embedding*,
    # and the editor's shape is encode once and decode many. This is what the
    # `cacheinputs` machinery used to arrange by caching a conversion; doing it
    # where the conversion belongs needs no cache to be right.
    for i in 1:3
        p.feat[i] .= out[i]
    end
    s.cachekey[] = out
    out
end

"""
    plansfor(s) -> (; dev, image, feat, enc, dec)

The two baked plans and the buffers they read, built on first use and kept.

**Two plans, not one.** `DNNKernels.buildchain` can put both graphs into a single
plan and make a whole step one queue submission, and that is right for "a picture
and a click". It is wrong here: a chain re-encodes on every click, and this
package exists for an editor that encodes once and decodes many.

Both are built before either is baked, and that order is load-bearing — plans
share the device's arena, so a second plan can need it grown, and growing it
under a baked plan is refused rather than silently invalidating its recording.
"""
function plansfor(s::SAM2)
    p = s.plans[]
    p === nothing || return p
    s.model.backend isa Lava.LavaBackend || error(
        "SAM2 runs its graphs as Mantle plans, which need a GPU backend; got " *
        "$(typeof(s.model.backend)). Build the model with `backend = LavaBackend()`.")
    m = s.model
    enc_g, dec_g = m.graphs["sam2_encoder"], m.graphs["sam2_decoder"]
    dev = M.Device(Lava)

    image = KA.allocate(m.backend, Float32, s.res, s.res, 3, 1)
    fill!(image, 0f0)
    encp = DNNKernels.build(dev, enc_g, Dict{String,Any}(enc_g.inputs[1] => image),
                            m.weights; dims = s.dims)

    # The prompt pair is created here, not in `prompt`, because the decoder plan
    # is built against these exact arrays and every click writes into them.
    pb = s.prompts[]
    if pb === nothing
        pb = (KA.allocate(m.backend, Float32, 2, s.maxpoints, 1),
              KA.allocate(m.backend, Int32, s.maxpoints, 1))
        fill!(pb[1], 0f0)
        fill!(pb[2], Int32(-1))
        s.prompts[] = pb
    end

    # The decoder's three feature inputs in ITS dtypes — the handover buffers.
    feat = ntuple(3) do i
        b = dec_g.buffers[dec_g.inputs[i]]
        a = KA.allocate(m.backend, b.dtype, DNNKernels.evalshape(b.shape, s.dims)...)
        fill!(a, zero(b.dtype))
        a
    end
    decp = DNNKernels.build(dev, dec_g,
                            Dict{String,Any}(dec_g.inputs[1] => feat[1],
                                             dec_g.inputs[2] => feat[2],
                                             dec_g.inputs[3] => feat[3],
                                             dec_g.inputs[4] => pb[1],
                                             dec_g.inputs[5] => pb[2]),
                            m.weights; dims = s.dims, clampattn = true)

    # Run each once before baking it, and this is a precondition rather than a
    # warm-up. `capture` records command buffers and keeps them; compiling a
    # kernel *during* a capture shells out to `spirv-opt` and does device work of
    # its own inside the recording, and the result is a lost device — reliably,
    # on a cold kernel cache, which is exactly what a fresh precompile is.
    #
    # It cost a day to see because every interactive test ran the plan before
    # baking it without meaning to, so the capture was always warm; the workload
    # is the one caller that bakes cold.
    DNNKernels.runplan(encp)
    KA.synchronize(m.backend)
    DNNKernels.bakeplan!(encp)
    DNNKernels.runplan(decp)
    KA.synchronize(m.backend)
    DNNKernels.bakeplan!(decp)
    s.plans[] = (; dev, image, feat, enc = encp, dec = decp)
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
function decode(s::SAM2, feats, point, label; replay::Bool = true)
    p = plansfor(s)
    # The plan reads fixed buffers, so anything that is not already in them has
    # to be staged. `encode` leaves the current embedding there and records it as
    # the key, so the editor's path — click again on the frame just encoded —
    # stages nothing. A decode against some *other* feats tuple copies, which is
    # what the old per-click conversion did unconditionally.
    #
    # `===` is enough here and was not before: it used to be compared against a
    # tuple the encoder had since overwritten in place, so identity held while
    # the numbers had changed. The key is now set by the same call that fills the
    # buffers, so the two cannot disagree.
    if feats !== nothing && s.cachekey[] !== feats
        for i in 1:3
            p.feat[i] .= feats[i]
        end
        s.cachekey[] = feats
    end
    # Likewise the prompt. `prompt` writes into the pair the plan was built
    # against, so this copies only when a caller passed something else.
    pb = s.prompts[]
    point === pb[1] || copyto!(pb[1], point)
    label === pb[2] || copyto!(pb[2], label)

    # `clampattn` is the decoder's business and was decided when the plan was
    # built: its attentions are 23 tokens and want the padded cooperative-matrix
    # path, while the encoder has six `Lq = 16` calls that would go along at 50%
    # waste and cost +2.12 ms of encode for nothing. Measured interleaved on the
    # autocast export: decode 8.24 -> 4.22 ms with the clamp, encode 118.63 ->
    # 120.75. (Those milliseconds are an RTX 4000 Ada's; the Radeon 8060S APU
    # runs the same encode in ~270 ms — the gap is the card, not a regression.)
    # Two plans is what lets the two graphs disagree about it.
    DNNKernels.runplan(p.dec)
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
    # Written into ONE persistent pair, not a fresh one per click, because the
    # baked decoder plan was built against these exact arrays and reads whatever
    # they hold now. `plansfor` creates them for that reason, so asking it here
    # is also what guarantees they exist.
    plansfor(s)                 # creates the pair if this is the first prompt
    pb = s.prompts[]
    copyto!(pb[1], xy)
    copyto!(pb[2], lb)
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
