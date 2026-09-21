"""
Fold an explicit zero pad into the convolution that reads it.

`F.pad(x, (1,1,1,1))` followed by a `Conv2d(..., padding=0)` is a convolution
with `padding=1`, and the kernel has done its own zero padding all along — it
reads outside the input and substitutes zero. Exporting the two separately is
what `torch.nn.utils` does for reflection-padded blocks whose reflection was
`constant` after all, and the Qwen-Image 2.1 VAE has forty of them.

They are not cheap: the pad materialises the whole input one pixel wider on
each side. `constant_pad_nd_33` writes 607 MB from 604, and the forty together
are **661 ms of a 4.3 s decode** and forty transients the placer has to fit.
Folded, they are nothing at all.

Conditions, all of them checked because a convolution taking a pad it should
not have is a wrong picture with nothing in the numbers to point at it:

  * the pad value is zero, which is what the convolution substitutes;
  * the pad covers only the two spatial axes and is symmetric on each, because
    `arg4` is one number per axis and cannot say "one on the left, two on the
    right";
  * the convolution has no padding of its own — adding them would be right
    arithmetically, and there is no shape in this tree that does both, so the
    conservative form is what ships;
  * the convolution is the pad's ONLY reader, is not transposed, and reads it as
    its input rather than as a weight.
"""

"""
    foldconvpad(graph) -> (graph, nfolded)

Rewrite each foldable `constant_pad_nd` into its convolution's `padding` and
drop it.
"""
function foldconvpad(g::Graph)
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

    consumer = Dict{String,Op}()
    for op in g.ops, i in op.ins
        consumer[i] = op
    end

    ops = copy(g.ops)
    drop = Set{String}()
    folded = 0

    for pad in g.ops
        String(pad.aten) == "constant_pad_nd.default" || continue
        length(pad.ins) == 1 || continue
        get(uses, pad.out, 0) == 1 || continue
        Float64(something(get(pad.attrs, "arg2", 0), 0)) == 0.0 || continue
        amt = ints(pad.attrs["arg1"])
        # torch pads the LAST axis first, two numbers per axis. Four of them is
        # the two spatial axes of an NCHW tensor; two is the innermost alone.
        (length(amt) == 2 || length(amt) == 4) || continue
        pw = (amt[1], amt[2])
        ph = length(amt) == 4 ? (amt[3], amt[4]) : (0, 0)
        pw[1] == pw[2] && ph[1] == ph[2] || continue

        conv = get(consumer, pad.out, nothing)
        conv === nothing && continue
        String(conv.aten) == "convolution.default" || continue
        get(conv.attrs, "arg6", false) == true && continue     # transposed
        isempty(conv.ins) || conv.ins[1] == pad.out || continue
        cpad = ints(conv.attrs["arg4"])
        all(iszero, cpad) || continue
        length(cpad) == 2 || continue

        ci = findfirst(o -> o.id == conv.id, ops)
        attrs = Dict{String,Any}(conv.attrs)
        # torch order, outermost spatial axis first: `[pad_h, pad_w]`.
        attrs["arg4"] = Any[ph[1], pw[1]]
        ins = copy(conv.ins)
        ins[1] = pad.ins[1]
        ops[ci] = Op(conv.id, conv.aten, ins, conv.out, attrs)
        push!(drop, pad.id)
        folded += 1
    end

    folded == 0 && return (g, 0)
    ops = [o for o in ops if !(o.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    buffers = Dict{String,Buffer}(k => v for (k, v) in g.buffers if !(k in drop))
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), folded)
end

function foldconvpad(graphs::AbstractDict)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = foldconvpad(g)
        out[name] = g2
        n += k
    end
    (out, n)
end
