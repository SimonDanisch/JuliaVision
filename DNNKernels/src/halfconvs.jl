"""
Run convolutions on fp16 operands and keep their results in fp32.

An fp32 export reaches the scalar implicit-GEMM kernel with every convolution,
because the tensor-core path takes fp16 operands only. A whole-model fp16 export
reaches the tensor cores and is wrong wherever an activation leaves fp16's
range: the Qwen-Image 2.1 VAE at fp16 decoded unscaled `randn` latents to 80.9%
NaN. So this keeps the graph fp32 and moves only the convolutions' operands to
fp16. Every lowering accumulates in fp32 and stores through the declared result
type, so the results keep fp32's range and only the operands are rounded.

**Whether an operand fits fp16 is decided from the graph, not from a sample.**
[`magnitudebounds`](@ref) propagates an upper bound on `|x|` through the ops
whose bound follows from their operands and the weights. A normalised
activation has one: `x / max(‖x‖, eps)` is at most 1, the scale and `γ` after it
are constants, and a SiLU never grows a value. The residual stream has none, and
it does leave the range. Measured on the VAE with `randn` latents at fp32, the
convolutions fed by norm, SiLU read at most 23, while the ones fed by the
residual stream read 106, 294, 7.6e3, 1.37e4 and 3.50e5, growing with depth; the
last is `up_blocks.4.resnets.0.conv_shortcut`, and a plain cast turned 0.37% of
its input into Inf. On the transparent-background latents the same tensors are
ten times smaller, so no sample says what the next input will do.

Three outcomes per convolution:

  * **bounded input**: a `_to_copy` to fp16 in front of the input and the
    weight, which is what an autocast export looks like. `hoistcasts` turns the
    weight cast into an fp16 weight at load and `foldoutcasts` folds the input
    cast into a producer that can store fp16.
  * **unbounded input, spatial kernel**: the input is divided by a power of two
    taken from its own `max |x|` at run time (`halfconvs.scale`), cast
    (`halfconvs.scaledcast`), convolved without its bias, and the fp32 result
    multiplied back and biased (`halfconvs.rescale`). A power of two makes both
    scalings exact, so the only rounding is the fp16 operand's, as in the first
    case. It costs a read of the input and a pass over the result, against the
    convolution moving to the tensor cores: the VAE's four such 3x3s went
    784.6 -> 400.9 ms.
  * **unbounded input, 1x1**: left in fp32. On the VAE's five these measured
    56.1 ms at fp32 against 78.0 at fp16, the widest of them 20.8 against 46.1,
    so there is nothing to scale for.

Forward 2-D convolutions only. The transposed and 1-D lowerings have not been
checked with mixed operand and result types, and a convolution this does not
recognise keeps its fp32 operands rather than taking a path nobody has run.
"""

"""The view kinds whose elements are all elements of their parent."""
const VALUEVIEWS = Set(["slice.Tensor", "permute.default", "unsqueeze.default",
                        "expand.default", "view.default", "squeeze.dims", "squeeze.dim",
                        "select.int", "alias.default", "detach.default", "t.default",
                        "transpose.int", "_unsafe_view.default", "reshape.default",
                        "split_with_sizes.default"])

"""The views a norm is broadcast back through before it divides."""
const BROADCASTVIEWS = Set(["expand.default", "view.default", "unsqueeze.default",
                            "squeeze.dims", "squeeze.dim"])

"""A numeric attribute, or `nothing` for a tensor operand or an absent one."""
numarg(op::Op, key::AbstractString) =
    (v = get(op.attrs, key, nothing); v isa Real ? Float64(v) : nothing)

"""Operand `pos` (0-based, as the export numbers them) as a buffer id or a number."""
function positional(op::Op, pos::Int)
    v = numarg(op, "arg$pos")
    v === nothing || return v
    idx = pos + 1 - count(p -> numarg(op, "arg$p") !== nothing, 0:(pos - 1))
    idx <= length(op.ins) ? op.ins[idx] : nothing
end

"""
    magnitudebounds(graph, weights) -> id -> bound

An upper bound on `|x|` over every element of a buffer, `Inf` where the graph
does not imply one. Computed on demand, so ask with `bound(id)`.
"""
function magnitudebounds(g::Graph, weights::AbstractDict)
    producer = Dict(op.out => op for op in g.ops)
    memo = Dict{String,Float64}()
    function throughviews(id, kinds)
        for _ in 1:32
            b = get(g.buffers, id, nothing)
            (b !== nothing && b.kind === :view && b.viewop in kinds) || return id
            id = b.of
        end
        return id
    end
    # `x / d` where `d` is a p-norm of `x` itself, taken with keepdim over the
    # axes it is broadcast back along, and at most clamped from below. Each
    # element is then at most 1: `|x_i| <= ‖x‖_p` for every `p >= 1`.
    function isnormof(num, den)
        d = throughviews(den, BROADCASTVIEWS)
        op = get(producer, d, nothing)
        op === nothing && return false
        if op.aten == "clamp.default"
            lo = numarg(op, "arg1")
            (lo !== nothing && lo >= 0 && get(op.attrs, "arg2", nothing) === nothing) || return false
            op = get(producer, throughviews(op.ins[1], BROADCASTVIEWS), nothing)
            op === nothing && return false
        end
        op.aten == "linalg_vector_norm.default" || return false
        ord = something(numarg(op, "arg1"), 2.0)
        ord >= 1 && get(op.attrs, "arg3", false) == true && op.ins[1] == num
    end
    bound(x::Real) = abs(Float64(x))
    bound(::Nothing) = Inf
    function bound(id::AbstractString)
        haskey(memo, id) && return memo[id]
        memo[id] = Inf                  # a cycle cannot occur, but must not recurse
        memo[id] = opbound(id)
    end
    function opbound(id)
        b = get(g.buffers, id, nothing)
        b === nothing && return Inf
        if b.kind === :weight
            w = get(weights, b.key, nothing)
            (w isa AbstractArray{<:AbstractFloat} && !isempty(w)) || return Inf
            return maximum(x -> abs(Float64(x)), w)
        end
        b.kind === :view && return b.viewop in VALUEVIEWS ? bound(b.of) : Inf
        op = get(producer, id, nothing)
        op === nothing && return Inf
        a = op.aten
        if a in ("clone.default", "_to_copy.default", "index.Tensor", "relu.default",
                 "silu.default", "gelu.default", "neg.default")
            return bound(op.ins[1])
        elseif a == "constant_pad_nd.default"
            return max(bound(op.ins[1]), abs(something(numarg(op, "arg2"), 0.0)))
        elseif a == "cat.default"
            return maximum(bound, op.ins)
        elseif a in ("sigmoid.default", "tanh.default")
            return 1.0
        elseif a == "clamp.default"
            lo, hi = numarg(op, "arg1"), numarg(op, "arg2")
            lo !== nothing && hi !== nothing && return max(abs(lo), abs(hi))
            return max(bound(op.ins[1]), abs(something(lo, hi, 0.0)))
        elseif a in ("mul.Tensor", "mul.Scalar")
            return bound(positional(op, 0)) * bound(positional(op, 1))
        elseif a in ("add.Tensor", "sub.Tensor")
            alpha = something(numarg(op, "alpha"), numarg(op, "arg2"), 1.0)
            return bound(positional(op, 0)) + abs(alpha) * bound(positional(op, 1))
        elseif a == "div.Tensor"
            num, den = positional(op, 0), positional(op, 1)
            den isa Real && den != 0 && return bound(num) / abs(den)
            num isa AbstractString && den isa AbstractString && isnormof(num, den) && return 1.0
            return Inf
        end
        return Inf
    end
    return bound
end

"""
    halfconvs(graph, weights) -> (graph, (; plain, scaled, kept))

Put each fp32 forward 2-D convolution on fp16 operands: a plain cast where its
input is bounded inside fp16's range, a run-time power-of-two scale where it is
not, and nothing at all for an unbounded 1x1.
"""
function halfconvs(g::Graph, weights::AbstractDict)
    bound = magnitudebounds(g, weights)
    fp16max = Float64(floatmax(Float16))
    buffers = Dict{String,Buffer}(g.buffers)
    order = copy(g.order)
    ops = Op[]
    transient(id, shape, T) =
        (buffers[id] = Buffer(id, :transient, shape, T, "", (0, 0), "", "", Dict{String,Any}());
         push!(order, id); id)
    # One cast per source buffer, shared by every convolution that reads it: a
    # residual block's shortcut and its first convolution read the same tensor.
    casts = Dict{String,String}()
    function half!(src::String)
        haskey(casts, src) && return casts[src]
        id = transient("halfconvs|" * src, buffers[src].shape, Float16)
        push!(ops, Op(id, "_to_copy.default", [src], id, Dict{String,Any}("dtype" => "float16")))
        casts[src] = id
    end
    scaled = Dict{String,Tuple{String,String}}()
    function scaledhalf!(src::String)
        haskey(scaled, src) && return scaled[src]
        s = transient("halfconvs|scale|" * src, Any[1], Float32)
        push!(ops, Op(s, "halfconvs.scale", [src], s, Dict{String,Any}()))
        xs = transient("halfconvs|scaled|" * src, buffers[src].shape, Float16)
        push!(ops, Op(xs, "halfconvs.scaledcast", [src, s], xs, Dict{String,Any}()))
        scaled[src] = (s, xs)
    end
    nplain = nscaled = nkept = 0
    for op in g.ops
        x = op.aten == "convolution.default" && get(op.attrs, "arg6", false) != true ?
            get(buffers, op.ins[1], nothing) : nothing
        w = x === nothing ? nothing : get(buffers, op.ins[2], nothing)
        if w === nothing || length(w.shape) != 4 || x.dtype !== Float32 ||
           w.dtype !== Float32 || weightsource(g, w.id) === nothing ||
           !(bound(w.id) < fp16max)
            push!(ops, op)
            continue
        end
        if bound(x.id) < fp16max
            ins = copy(op.ins)
            ins[1] = half!(op.ins[1])
            ins[2] = half!(op.ins[2])
            push!(ops, Op(op.id, op.aten, ins, op.out, op.attrs))
            nplain += 1
        elseif w.shape[3] == 1 && w.shape[4] == 1
            push!(ops, op)
            nkept += 1
        else
            s, xs = scaledhalf!(op.ins[1])
            y = transient("halfconvs|unscaled|" * op.out, buffers[op.out].shape, Float32)
            # The bias and a folded activation move to the rescale, which is where
            # the result is back at its own scale.
            attrs = Dict{String,Any}(k => v for (k, v) in op.attrs if k != "act")
            push!(ops, Op(op.id, op.aten, [xs, half!(op.ins[2])], y, attrs))
            rs = Dict{String,Any}()
            haskey(op.attrs, "act") && (rs["act"] = op.attrs["act"])
            push!(ops, Op(op.id * "|rescale", "halfconvs.rescale",
                          vcat([y, s], op.ins[3:end]), op.out, rs))
            nscaled += 1
        end
    end
    n = (; plain = nplain, scaled = nscaled, kept = nkept)
    nplain + nscaled == 0 && return (g, n)
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), n)
end

function halfconvs(graphs::AbstractDict, weights::AbstractDict)
    out = Dict{String,Graph}()
    n = (; plain = 0, scaled = 0, kept = 0)
    for (name, g) in graphs
        g2, k = halfconvs(g, weights)
        out[name] = g2
        n = map(+, n, k)
    end
    (out, n)
end

# ── the three ops `halfconvs` introduces ─────────────────────────────────────

"""
`max |x|` as the bits of an fp32, by atomic unsigned max: a non-negative float
orders as its bit pattern does. Each thread strides the whole array and makes
one atomic, so the contention is one per thread and not one per element. A NaN
has larger bits than `Inf` and wins, which the scale then passes on.
"""
function absmaxbits_kernel!(bits, x, n::I, nthreads::I) where {I<:Integer}
    i = I(KI.get_global_id().x)
    m = UInt32(0)
    @inbounds while i <= n
        m = max(m, reinterpret(UInt32, abs(Float32(x[i]))))
        i += nthreads
    end
    m == UInt32(0) || Atomix.@atomic max(bits[1], m)
    return nothing
end

"""
The smallest power of two `s >= 1` with `max |x| / s < 2^15`, from the bits of
`max |x|`: its unbiased exponent `e` gives `max |x| < 2^(e+1)`, so `s = 2^(e-14)`.
"""
function scalefrombits_kernel!(s, bits)
    if KI.get_global_id().x == 1
        @inbounds e = Int32((bits[1] >> 23) & 0xff) - Int32(127)
        k = max(Int32(0), e - Int32(14))
        @inbounds s[1] = reinterpret(Float32, UInt32(k + Int32(127)) << 23)
    end
    return nothing
end

"""How many threads `absmaxbits_kernel!` strides with."""
const ABSMAX_THREADS = 65536

function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("halfconvs.scale")})
    x = operand(emitctx, op.ins[1])
    s = dest(emitctx)
    bits = scratch(emitctx, UInt32, 1)
    M.dispatch!(emitctx.g, M.fill_kernel!, (bits, UInt32(0)), 1; name = "$(op.id).zero")
    n = length(x)
    I = n <= typemax(Int32) ? Int32 : Int64
    nt = min(n, ABSMAX_THREADS)
    M.dispatch!(emitctx.g, absmaxbits_kernel!, (bits, x, I(n), I(nt)), nt;
                name = "$(op.id).absmax")
    M.dispatch!(emitctx.g, scalefrombits_kernel!, (s, bits), 1; name = op.id)
    return s
end

function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("halfconvs.scaledcast")})
    od = size(dest(emitctx))
    sx = strideview(emitctx, op, 1, od)
    x = sx === nothing ? operand(emitctx, op.ins[1]) : sx
    # Division by a power of two is exact; the store into fp16 is the rounding.
    return elementwise!(emitctx, op, /, x, operand(emitctx, op.ins[2]))
end

function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("halfconvs.rescale")})
    out = dest(emitctx)
    od = size(out)
    y, s = operand(emitctx, op.ins[1]), operand(emitctx, op.ins[2])
    f = actfn(Symbol(get(op.attrs, "act", "none")))
    one_ = ntuple(_ -> 1, length(od))
    if length(op.ins) >= 3
        bias = operand(emitctx, op.ins[3])
        # Per output channel, which is axis 3 of `(OW, OH, Cout, N)`.
        bd = ntuple(k -> k == 3 ? length(bias) : 1, length(od))
        return ewdispatch!(emitctx, out, od, (y, s, bias),
                           (bcstrides(od, od), bcstrides(od, one_), bcstrides(od, bd)),
                           (v, k, b) -> f(v * k + b); name = op.id)
    end
    return ewdispatch!(emitctx, out, od, (y, s), (bcstrides(od, od), bcstrides(od, one_)),
                       (v, k) -> f(v * k); name = op.id)
end
