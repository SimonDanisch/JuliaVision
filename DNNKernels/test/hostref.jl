"""
A host reference for the declared ops, in plain Julia.

Not a second runtime: it walks an ATen graph with `Base` arrays and nested loops,
answering only the ops a declared graph is checked against. The point is that it
shares no code with the kernels — a convolution written as four nested loops
cannot have the same tiling bug as an implicit GEMM, and a `sum(a; dims)` cannot
have the same index bug as `sumdims!`.

The alternative would be an A/B against `runop!`, and there is no longer one to
run: `planslab`, `Workspace` and the lazy-broadcast path were the three
allocators the declaration refactor deleted. A reference that is the DEFINITION
also outlives the thing it was written to check, which an A/B does not.

Shapes are the reversed ones the graph carries: `x` is `(W, H, Cin, N)`, `w` is
`(KW, KH, Cin, Cout)`, `out` is `(OW, OH, Cout, N)`.
"""
module HostRef

import DNNKernels
const DK = DNNKernels

convsize(n, k, s, p, d) = (n + 2p - d * (k - 1) - 1) ÷ s + 1

"""`out[ow, oh, co, n]` from the gather that defines a convolution, one output
element per iteration and nothing shared between them."""
function conv2d(x, w, bias, stride, pad, dil)
    KW, KH, Cin, Cout = size(w)
    Wi, Hi, _, N = size(x)
    SX, SY = stride
    PX, PY = pad
    DX, DY = dil
    OW = convsize(Wi, KW, SX, PX, DX)
    OH = convsize(Hi, KH, SY, PY, DY)
    out = zeros(eltype(x), OW, OH, Cout, N)
    for n in 1:N, co in 1:Cout, oh in 1:OH, ow in 1:OW
        acc = bias === nothing ? zero(Float32) : Float32(bias[co])
        for ci in 1:Cin, kh in 1:KH, kw in 1:KW
            ix = (ow - 1) * SX - PX + (kw - 1) * DX + 1
            iy = (oh - 1) * SY - PY + (kh - 1) * DY + 1
            (1 <= ix <= Wi && 1 <= iy <= Hi) || continue
            acc = muladd(Float32(x[ix, iy, ci, n]), Float32(w[kw, kh, ci, co]), acc)
        end
        out[ow, oh, co, n] = acc
    end
    return out
end

"""Training-mode batch norm: statistics over every axis but the channel, which
is torch's dim 1 and so `ndims - 1` here."""
function batchnorm(x, gamma, beta, eps)
    c = ndims(x) - 1
    rd = Tuple(i for i in 1:ndims(x) if i != c)
    rs = ntuple(i -> i == c ? length(gamma) : 1, ndims(x))
    n = length(x) ÷ size(x, c)
    mu = sum(Float32, x; dims = rd) ./ n
    d = x .- mu
    v = sum(abs2, d; dims = rd) ./ n
    invstd = 1 ./ sqrt.(v .+ eps)
    return (d .* (invstd .* reshape(gamma, rs)) .+ reshape(beta, rs),
            vec(mu), vec(invstd))
end

"""
    forward(graph, weights, inputs) -> Dict

Every buffer's host value, by id. `weights` are host arrays keyed as the graph
names them; `inputs` likewise.
"""
function forward(graph, weights::AbstractDict, inputs::AbstractDict)
    v = Dict{String,Any}()
    for (id, b) in graph.buffers
        b.kind === :weight && (v[id] = weights[b.key])
        b.kind === :external && haskey(inputs, id) && (v[id] = inputs[id])
    end
    # A view is resolved when read, so that a `getitem` can name a tuple element
    # its producing op has not returned yet.
    function val(id)
        haskey(v, id) && return v[id]
        b = graph.buffers[id]
        b.kind === :view || error("hostref: $id of kind $(b.kind) has no value")
        p = val(b.of)
        occursin("getitem", b.viewop) && return (v[id] = p[Int(b.attrs["arg1"]) + 1])
        b.viewop in DK.SHAPEONLY_VIEWS ||
            error("hostref: view $id is a $(b.viewop), which this reference does not do")
        return (v[id] = reshape(p, DK.evalshape(b.shape, (;))))
    end
    for op in graph.ops
        a = op.attrs
        v[op.out] = if op.aten == "convolution.default"
            conv2d(val(op.ins[1]), val(op.ins[2]),
                   length(op.ins) >= 3 ? val(op.ins[3]) : nothing,
                   reverse(DK.ints(a["arg3"])), reverse(DK.ints(a["arg4"])),
                   reverse(DK.ints(a["arg5"])))
        elseif op.aten == "leaky_relu.default"
            s = Float32(something(get(a, "arg1", nothing), 0.01))
            x = val(op.ins[1])
            ifelse.(x .>= 0, x, s .* x)
        elseif op.aten == "repeat.default"
            x = val(op.ins[1])
            reps = reverse(DK.ints(a["arg1"]))
            repeat(x, ntuple(k -> k <= length(reps) ? reps[k] : 1, ndims(x))...)
        elseif op.aten == "_native_batch_norm_legit.no_stats"
            batchnorm(val(op.ins[1]), val(op.ins[2]), val(op.ins[3]),
                      Float32(a["arg5"]))
        elseif op.aten == "clone.default"
            copy(val(op.ins[1]))
        elseif op.aten == "mul.Tensor"
            val(op.ins[1]) .* val(op.ins[2])
        elseif op.aten == "sum.dim_IntList"
            x = val(op.ins[1])
            d = Tuple(DK.jdim(k, ndims(x)) for k in DK.ints(a["arg1"]))
            dropdims(sum(x; dims = d); dims = d)
        else
            error("hostref: no reference for $(op.aten) (op $(op.id))")
        end
    end
    return v
end

end
