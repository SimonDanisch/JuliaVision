"""
Fold inference batch-norm into the preceding convolution.

At inference a `_native_batch_norm_legit_no_training` is an affine map with
constant coefficients:

    s = γ / √(v + ε)
    z = s * (W ⊛ x + b) + (β - μ*s)
      = (s*W) ⊛ x + (s*b + β - μ*s)

so it can be absorbed into the convolution's weights and bias and the op
deleted outright. This is what `eval()`-mode PyTorch and every inference runtime
does; it is exact up to float rounding, not an approximation.

Two things it buys us, both measured as significant:

  * 75 fewer kernel launches *and* 75 fewer allocations per step. The step is
    host-bound (the GPU idles ~60% of wall time), so removing launches is worth
    more than removing arithmetic.
  * the existing op recomputed `s` and `β - μ*s` from constant weights on every
    single step, 75 times a step, and each of those is its own launch and
    allocation.

Only folds when the convolution's output feeds nothing but the batch-norm, and
when only element 0 of the batch-norm's tuple output is ever read (elements 1
and 2 are the saved mean/variance, which inference never touches).
"""

"""Reshape a per-output-channel vector to broadcast over a weight's last axis."""
chanshape(v, W) = reshape(v, ntuple(i -> i == ndims(W) ? length(v) : 1, ndims(W)))

function foldbatchnorm(graphs::AbstractDict, weights::AbstractDict)
    w = Dict{String,Any}(weights)
    out = Dict{String,Graph}()
    folded = 0
    for (name, g) in graphs
        g2, n = foldbatchnorm(g, w)
        out[name] = g2
        folded += n
    end
    (out, w, folded)
end

function foldbatchnorm(g::Graph, weights::Dict{String,Any})
    producer = Dict(op.out => op for op in g.ops)
    uses = Dict{String,Int}()
    bump!(id) = (uses[id] = get(uses, id, 0) + 1)
    for op in g.ops, i in op.ins
        bump!(i)
    end
    for (_, b) in g.buffers
        isempty(b.of) || bump!(b.of)
    end
    for o in g.outputs
        bump!(o)
    end

    buffers = Dict{String,Buffer}(g.buffers)
    ops = copy(g.ops)
    drop = Set{String}()
    folded = 0

    for (k, bn) in enumerate(g.ops)
        bn.aten == "_native_batch_norm_legit_no_training.default" || continue
        conv = get(producer, bn.ins[1], nothing)
        conv === nothing && continue
        conv.aten == "convolution.default" || continue
        get(uses, conv.out, 0) == 1 || continue          # conv feeds only this bn

        # every reader of the bn must be a getitem on element 0
        readers = [b for (_, b) in buffers if b.of == bn.out]
        all(b -> occursin("getitem", b.viewop) && Int(get(b.attrs, "arg1", 0)) == 0,
            readers) || continue
        get(uses, bn.out, 0) == length(readers) || continue

        # Under autocast every operand arrives through a `_to_copy` cast, so the
        # buffer named by the op is a transient and the constant lives one hop
        # further back. `weightsource` walks that chain; anything genuinely
        # computed returns `nothing` and is skipped.
        srcs = map(i -> weightsource(g, i),
                   (bn.ins[2], bn.ins[3], bn.ins[4], bn.ins[5], conv.ins[2]))
        any(isnothing, srcs) && continue
        all(s -> haskey(weights, s.key), srcs) || continue
        γ, β, μ, v = weights[srcs[1].key], weights[srcs[2].key],
                     weights[srcs[3].key], weights[srcs[4].key]
        wsrc = weights[srcs[5].key]
        eps = Float32(bn.attrs["arg6"])

        s = γ ./ sqrt.(v .+ eps)
        # The *declared* dtype of the conv's operand, which under autocast is
        # fp16 even though the master weight is fp32. Scaling in fp32 and
        # rounding once at the end keeps this as exact as the cast already was;
        # folding into an already-rounded fp16 weight would round twice.
        wb = buffers[conv.ins[2]]
        hasbias = length(conv.ins) >= 3
        bsrc = hasbias ? weightsource(g, conv.ins[3]) : nothing
        hasbias && (bsrc === nothing || !haskey(weights, bsrc.key)) && continue
        b0 = hasbias ? weights[bsrc.key] : zero(s)
        newW = wb.dtype.(wsrc .* chanshape(s, wsrc))
        newb = b0 .* s .+ (β .- μ .* s)

        # Keys must be unique across *all* graphs, because they share one weights
        # dict. Op ids are only unique within a graph — "convolution" exists in
        # encode_image, encode_mask_deep and encode_mask_shallow alike — so an
        # unscoped key silently overwrites another graph's folded bias, and every
        # graph still verifies in isolation.
        #
        # Key on the *convolution*, not on the weight buffer: under autocast the
        # operand is a transient whose `key` is empty, so keying on it collapsed
        # every fold in a graph onto "<graph>||foldbn" and the last one won — the
        # 7x7x3x64 stem picked up a 1x1x256x1024 weight.
        wkey = g.name * "|" * conv.id * "|foldbn_weight"
        bkey = g.name * "|" * conv.id * "|foldbn_bias"
        weights[wkey] = newW
        weights[bkey] = newb

        wid = conv.id * "|foldbn_weight"
        bid = conv.id * "|foldbn_bias"
        buffers[wid] = Buffer(wid, :weight, wb.shape, wb.dtype, wkey, (0, 0), "", "",
                              Dict{String,Any}())
        bbuf = hasbias ? buffers[conv.ins[3]] : buffers[bn.ins[3]]
        buffers[bid] = Buffer(bid, :weight, bbuf.shape, bbuf.dtype, bkey, (0, 0), "", "",
                              Dict{String,Any}())

        ci = findfirst(o -> o.id == conv.id, ops)
        ops[ci] = Op(conv.id, conv.aten, [conv.ins[1], wid, bid], conv.out, conv.attrs)

        # the getitem that read bn[0] now aliases the convolution's own output
        for b in readers
            buffers[b.id] = Buffer(b.id, :view, b.shape, b.dtype, "", (0, 0),
                                   conv.out, "alias.default", b.attrs)
        end
        push!(drop, bn.id)
        folded += 1
    end

    folded == 0 && return (g, 0)
    ops = [o for o in ops if !(o.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    append!(order, [id for id in keys(buffers) if endswith(id, "|foldbn_weight") ||
                    endswith(id, "|foldbn_bias")])
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), folded)
end
