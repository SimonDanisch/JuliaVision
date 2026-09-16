"""
Lowering an ATen graph into a Mantle graph.

The exported graph is at torch's granularity: one op per `aten::` call, tensors
with shapes, views declared as buffers naming their parent. A Mantle graph is at
KERNEL granularity: one pass per dependent stage, a dispatch per launch, and
every byte a resource the placer owns. This file is the lowering between them,
and it is the reason none of the three allocators this package used to carry
exist any more.

Nothing here executes. `emitgraph` walks the ops declaring what each one will do,
`Plan` then runs all seven of Mantle's phases — `Dag`, `Schedule`, `Liveness`,
`Place`, `Aliasing`, `Barriers`, `Pipelines` — and `run!` walks or replays the
result. An op's body says `dispatch!(p, kernel, args, ndrange)` where it used to
say `kernel(backend, wg)(args...; ndrange)`, which is a one-line change per
launch and is the whole of what the 88 `emitop!` methods are.

**What this replaces, and why each one existed.** All three were mechanisms for
recovering something the graph had already stated:

  * `planslab` laid the intermediates out itself, because passes used to be
    discovered by CAPTURE and placement therefore could not run before the ops
    had run. Declared, `Liveness`/`Place`/`Aliasing` see the whole graph before
    a single byte is touched.
  * `Workspace` was a bump arena for op-internal scratch, reset per op with a
    retirement list, because that scratch appears in no ATen graph — the export
    is at torch granularity and a transposed copy of `q` is our kernel's
    business, not the model's. Lowering is exactly where it becomes a
    declaration, so scratch is a transient like any other and the placer aliases
    it against the rest of the graph rather than only within one op.
  * `makeview` rebuilt each declared view as a Julia wrapper over its parent's
    storage, so `stridedroot` then had to walk back down the stack to recover the
    parent, the offset and the strides. A view is a `Mantle.ResourceView` here,
    which is that descriptor and nothing else.
"""

"""
What an op declares into, and what it declares against.

`res` maps an ATen buffer id to its Mantle resource — a `Transient.Buffer` for an
intermediate, a `Buffer` for anything that outlives the plan, a plain value for a
host scalar. A multi-output op's results are keyed `"\$(id)#\$(i)"`, since the
graph gives the op one output id and the tuple elements need one resource each.
"""
struct EmitCtx{G,D}
    aten::Graph
    g::G                            # the Mantle graph being built
    dev::D
    dims::NamedTuple
    res::Dict{String,Any}
    esc::Set{String}
    # Which op is emitting, so `dest` can answer without being passed the op.
    outid::Base.RefValue{String}
end

"""
    emitgraph(dev, aten, weights, dims) -> (mantle_graph, ctx)

Declare `aten` into a fresh Mantle graph. Runs nothing.
"""
function emitgraph(dev, aten::Graph, weights::AbstractDict, dims::NamedTuple)
    g = M.Graph(dev)
    esc = escaping(aten)
    ec = EmitCtx(aten, g, dev, dims, Dict{String,Any}(), esc, Ref(""))
    for id in aten.order
        declare!(ec, aten.buffers[id], weights)
    end
    for op in aten.ops
        ec.outid[] = op.out
        emitop!(ec, op, op.tag)
    end
    return g, ec
end

"""
One buffer's resource.

The kinds are the export's, and the decision each one makes is Mantle's:

  * `:weight` and `:host` are values that already exist — the safetensor on the
    device, and a scalar evaluated from `dims`.
  * `:external` is written by the caller and `:view` is resolved on demand
    (`viewfor`), through the parent it names.
  * everything else is an intermediate, and an intermediate is a TRANSIENT unless
    it escapes. An escaping buffer is read after the plan has run — by the next
    graph, or by the caller reading an output — so it cannot be memory the placer
    may hand to something else at its last use.
"""
function declare!(ec::EmitCtx, b::Buffer, weights::AbstractDict)
    b.kind === :view && return                      # `viewfor`, on demand
    if b.kind === :weight
        haskey(weights, b.key) || error("missing weight $(b.key)")
        ec.res[b.id] = weights[b.key]
        return
    end
    if b.kind === :host
        ec.res[b.id] = evalexpr(String(b.attrs["expr"]), ec.dims)
        return
    end
    # A multi-output op declares `shapes`/`dtypes` instead of `shape`/`dtype`;
    # each element gets its own resource, and an element the runtime never reads
    # (sdpa's philox seed and offset) has `nothing` for a dtype and gets none.
    if haskey(b.attrs, "shapes")
        for (i, (shape, T)) in enumerate(zip(b.attrs["shapes"], b.attrs["dtypes"]))
            (shape === nothing || T === nothing) && continue
            ec.res["$(b.id)#$(i - 1)"] = make(ec, b.id, T, evalshape(shape, ec.dims))
        end
        return
    end
    ec.res[b.id] = make(ec, b.id, b.dtype, evalshape(b.shape, ec.dims))
end

"""A resource of this shape: transient unless the id escapes or is written from
outside."""
function make(ec::EmitCtx, id::AbstractString, ::Type{T}, dims::Dims) where {T}
    b = ec.aten.buffers[id]
    owned = b.kind === :external || id in ec.esc || id in ec.aten.outputs
    return owned ? M.Buffer(ec.dev, T, dims) : M.Transient.Buffer(ec.g, T, dims)
end

"""
    viewfor(ctx, id) -> ResourceView

A view buffer as a descriptor over the resource that owns its bytes.

Cached in `res`, because a `ResourceView` has to be ONE object per buffer: `use`
interns by identity, and two descriptors for the same window would be two
resources with no hazard between them.

Only the shape-only views are handled by construction — `view`, `_unsafe_view`,
`unsqueeze`, `squeeze` — which reinterpret extents over the parent's bytes
starting at its first element. Anything else (a `slice`, a `permute`, a `select`)
changes where the elements are, so it needs its own storage and a pass to fill
it; `materialisedview` in `plan.jl` is the predicate that says which, and those
are declared as ordinary transients by the op that produces them. Throwing here
rather than guessing is deliberate: a wrong offset is silent.
"""
function viewfor(ec::EmitCtx, id::AbstractString)
    haskey(ec.res, id) && return ec.res[id]
    b = ec.aten.buffers[id]
    b.kind === :view || error("buffer $id is not a view")
    # A `getitem` is not a window onto anything: it names one element of a
    # multi-output op's result, and `declare!` gave that element a resource of
    # its own under the `"$(id)#$(i)"` key. So this is a lookup, where
    # `makeview` had to index the tuple the op had already returned.
    occursin("getitem", b.viewop) &&
        return (ec.res[id] = ec.res["$(b.of)#$(Int(b.attrs["arg1"]))"])
    b.viewop in SHAPEONLY_VIEWS || error(
        "view $id is a `$(b.viewop)`, which moves its elements rather than only " *
        "reinterpreting the shape, so it cannot be a window onto its parent. It " *
        "needs a transient and a pass that fills it — see `materialisedview`.")
    parent = operand(ec, b.of)
    v = M.viewof(parent, evalshape(b.shape, ec.dims))
    ec.res[id] = v
    return v
end

"""
    operand(ctx, id)

What an op reads for input `id`: a resource, a view of one, or a host value.
"""
function operand(ec::EmitCtx, id::AbstractString)
    haskey(ec.res, id) && return ec.res[id]
    b = get(ec.aten.buffers, id, nothing)
    b === nothing && error("unknown buffer $id")
    b.kind === :view && return viewfor(ec, id)
    error("buffer $id of kind $(b.kind) was never declared")
end

"""
One operand of an op by POSITION, which is not the same as by index into `ins`.

torch folds a scalar operand into the schema, so it arrives in `attrs` under its
positional slot and `ins` holds only the tensors, consumed in order. Either side
can be the scalar: `1 - sigmoid(x)` exports as
`sub.Tensor(ins = [sigmoid], arg0 = 1)`.

`binary!` used to index `op.ins` directly, which works for as long as every op
reaching it has tensors on both sides — `mul`, `div` and `add` did — and is a
`BoundsError` the first time one does not. Same rule as `runop!`'s
`operand(ctx, op, pos)`, and `ARGKEY` is shared with it.
"""
function operand(ec::EmitCtx, op::Op, pos::Int)
    key = argkey(pos)
    haskey(op.attrs, key) && return numattr(ec.dims, op.attrs[key])
    idx = pos - count(p -> haskey(op.attrs, argkey(p)), 1:(pos - 1))
    idx <= length(op.ins) || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) has no operand at position $pos")
    return operand(ec, op.ins[idx])
end

"""    dest(ctx) -> resource

Where the op being emitted writes. `dest(ctx, i)` for element `i` of a
multi-output op, zero-based as the export numbers them.
"""
dest(ec::EmitCtx) = ec.res[ec.outid[]]
dest(ec::EmitCtx, i::Integer) = ec.res["$(ec.outid[])#$(i)"]

"""Every declared element of a multi-output op's result, in order."""
dests(ec::EmitCtx, n::Integer) = ntuple(i -> dest(ec, i - 1), n)

"""
    scratch(ctx, T, dims...) -> TransientBuffer

An op's own working buffer, declared into the graph.

This is `Workspace`'s replacement and it is smaller in every way. Scratch appears
in no ATen graph — the export is at torch granularity, and a split-K
accumulator or a transposed copy of `q` is a property of our kernels — so it was
bump-allocated from an arena that reset per op, which meant its bytes could only
ever be reused by the SAME op. Declared, its liveness is whatever its uses say,
so the placer aliases it against the whole graph like any other transient.
"""
scratch(ec::EmitCtx, ::Type{T}, dims::Integer...) where {T} =
    M.Transient.Buffer(ec.g, T, map(Int, dims))

"""The fallback, so an unported op says which one it is rather than failing four
frames down in `dispatch!`."""
emitop!(ec::EmitCtx, op::Op, ::Val{A}) where {A} = error(
    "DNNKernels.emitop!: no emit method for `$(op.aten)` (op $(op.id)). The op " *
    "declares its dispatches now instead of launching them; see `emit.jl` for " *
    "the patterns and `runop!` for what this one used to do.")

# ── elementwise, as a dispatch ────────────────────────────────────────────────
#
# `runop!` wrote these as `emit(ctx, Base.broadcasted(f, xs...))`, which returned
# the `Broadcasted` itself when `fuse.jl` said the value was consumed exactly once
# — so the fused kernel was generated at the CONSUMER's launch, pairwise, by
# GPUArrays' broadcast machinery. That is why SAM 2's encoder showed one
# `fused.elementwise` op and 1014 broadcast dispatches.
#
# The fusion DECISION was always static (`fuseops` at model build). What was at
# run time was the codegen. Declared, there is nothing to defer: the group is
# known, so it emits one dispatch.
#
# The kernel is `ew!` from `kernels/elementwise.jl`: macro-free, ONE method over
# a tuple of operands, taking the output shape and each operand's effective
# strides as plain arguments. So broadcast is no longer GPUArrays'
# either. It is `bcindex` over strides computed at emit time from shapes the
# graph already states, which is what lets NeuralLUT's `(1, 1, 1, 1, 3)` factor
# meet its `(33, 33, 33, 3, 3)` LUT with no materialised copy and no second
# dispatch.

"""
    elementwise!(ctx, op, f, ins...) -> resource

One pass, one dispatch: `dest(ctx) .= f.(ins...)`.

`f` is an argument rather than a type parameter of a wrapper, so a closure over
scalars, leaky ReLU's slope or an epsilon, needs no operand of its own.

Nothing here says what it touches. `ew!` stores through its first argument and
reads through the operand tuple, and that is read off the kernel body by
`Mantle.argument_usage` — so the destination is written, the operands are only
read, and two ops that read the same weight do not serialise against each other.
It was two `use` calls at this site, which is the same fact stated twice.
"""
function elementwise!(ec::EmitCtx, op::Op, f, ins...)
    out = dest(ec)
    od = size(out)
    ops, sts = operandtuples(od, ins)
    M.dispatch!(ec.g, ew!, (out, od, ops, sts, f), length(out); name = op.id)
    return out
end

"""
The operands and their effective strides, as the two tuples [`ew!`](@ref) walks
in step.

`ewkernel(Val(n))` was here, picking `ew1!`, `ew2!` or `ew3!`, and it refused a
fourth operand by name. The reason given was that `Mantle.resolve` is applied per
ELEMENT of a dispatch's argument tuple, so a nested tuple of resources arrived in
the kernel as unresolved handles. That was a one-line gap in core rather than a
reason for three kernels: `resolve` and `storage` take a `::Tuple` method now,
and `devicepointeroffsets` stopped counting a tuple as a level of nesting, which
is what a `resize!` under a recorded plan needs in order to find the operands'
addresses (`Mantle.nestinglevels`).

The second reason given was that each operand had to be a top-level argument so
that `use(p, x; read = true)` could name it. That was wrong when it was written
and is moot now: there is no `use`, and what the kernel reads is read off the
kernel.
"""
operandtuples(od::Dims, ::Tuple{}) = ((), ())
function operandtuples(od::Dims, ins::Tuple)
    x = first(ins)
    isresource(x) || error(
        "DNNKernels: elementwise operand of type $(typeof(x)) has no bytes to " *
        "index. A host scalar belongs in the function, as `Base.Fix2(f, x)`, " *
        "rather than in the operand list; `binary!` is where that is decided.")
    ops, sts = operandtuples(od, Base.tail(ins))
    return ((x, ops...), (bcstrides(od, size(x)), sts...))
end

"""
Whether an operand is MEMORY, as opposed to a host scalar.

Three things qualify and the third is easy to miss: a `Buffer` or a
`Transient.Buffer`, a `ResourceView` of one, and a plain device array, which is
what a WEIGHT is. `Model` uploads weights with `KA.allocate`, so they are not
Mantle resources; `use` interns them all the same, which is Mantle fencing bytes
it did not place.

Asking only about `Resource`/`ResourceView` said no to every weight, and the
consequence was not an error: `binary!` read that as "host scalar", bound a
3x3x33x33x33 LUT into a closure with `Fix2`, and the kernel failed to compile
with `unsupported call to jl_alloc_genericmemory_unchecked` — a broadcast inside
the kernel, four frames from anything naming a weight.
"""
isresource(x) = x isa M.Resource || x isa M.ResourceView || x isa AbstractArray

"""
    binary!(ctx, op, f) -> resource

A two-operand op, with a scalar side bound into `f` if it has one.

`mul.Tensor` can be given a tensor and a number, and a kernel argument has to
have bytes to index, so the number goes into the function rather than into the
operand list. `Fix1`/`Fix2` put it in the closure's TYPE, so it costs no
argument and no memory.
"""
function binary!(ec::EmitCtx, op::Op, f)
    a, b = operand(ec, op, 1), operand(ec, op, 2)
    isresource(a) && isresource(b) && return elementwise!(ec, op, f, a, b)
    isresource(a) && return elementwise!(ec, op, Base.Fix2(f, b), a)
    isresource(b) && return elementwise!(ec, op, Base.Fix1(f, a), b)
    error("DNNKernels: `$(op.aten)` (op $(op.id)) has a host scalar on both " *
          "sides, so it is a constant and `constfold` should have removed it.")
end

# ── the ops ──────────────────────────────────────────────────────────────────

emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("mul.Tensor")}) = binary!(ec, op, *)
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("div.Tensor")}) = binary!(ec, op, /)
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("add.Tensor")}) = binary!(ec, op, +)

# A copy is `identity` over one operand, which is the same one dispatch as any
# other elementwise op. `runop!` wrote `d .= a`, which is the same launch by
# another spelling.
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("clone.default")}) =
    elementwise!(ec, op, identity, operand(ec, op, 1))

function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("leaky_relu.default")})
    x = operand(ec, op, 1)
    s = eltype(dest(ec))(something(get(op.attrs, "arg1", nothing), 0.01))
    # The slope is captured, so it travels in the closure and not as an operand.
    elementwise!(ec, op, v -> v >= zero(v) ? v : s * v, x)
end

# ── one function of one operand ──────────────────────────────────────────────
#
# From `UNARY_FUSED`, the same table `runop!` generated its methods from and
# `fusedfunc` reads to build a `FusedOp`. Read a third time here rather than
# listed again: two lists for one fact took about an hour to diverge the first
# time, when a fusion emitting `x -> inv(sqrt(x))` was not the `rsqrt.default`
# anybody had tested.
for (name, f) in UNARY_FUSED
    @eval emitop!(ec::EmitCtx, op::Op, ::Val{Symbol($name)}) =
        elementwise!(ec, op, $f, operand(ec, op, 1))
end

"""
`aten::sub.Tensor(a, b, alpha)`, which is `a - alpha * b`.

`alpha` defaults to 1, and when it is 1 the multiply is not emitted at all --
the closure is `-` itself, so the kernel is the same one `add.Tensor` compiles.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("sub.Tensor")})
    k = alpha(op)
    return k == 1 ? binary!(ec, op, -) :
                    binary!(ec, op, (x, y) -> x - k * y)
end

emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("eq.Scalar")}) = binary!(ec, op, ==)

"""
`aten::_to_copy`, a dtype conversion as one elementwise pass.

torch's float-to-integer cast truncates toward zero and saturates: `+/-Inf`
become the integer extremes and `NaN` becomes 0, where Julia's `convert` throws
`InexactError`. `SafeTrunc` is that rule, and it matters on real graphs -- T5's
attention mask carries `-Inf` into exactly this cast, and the Wan VAE casts
index arithmetic where 0.25 is a legitimate input torch turns into 0.

Everything else is `convert`, which for a float-to-float narrowing is the single
store rounding once.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("_to_copy.default")})
    a = operand(ec, op, 1)
    T = eltype(dest(ec))
    f = (T <: Integer && !(eltype(a) <: Integer)) ? SafeTrunc{T}() : ToType{T}()
    return elementwise!(ec, op, f, a)
end

"""
`aten::clamp` with either bound optional, and either bound possibly symbolic --
the iSTFT clamps an index against the frame count, so a bound becomes a function
of the sequence length once the graph is length-generic.

The bounds travel in the CLOSURE and not as operands: they are host scalars, and
a kernel argument has to have bytes to index. `clampbounds` keeps them in the
operand's own type when that type is an integer, because a `Float32` bound
promotes the result and storing that into an integer destination is a `convert`
with an `InexactError` path -- a throw inside a kernel, whose exception
allocation is a hostcall on AMDGPU and dead code the compiler still emits
everywhere else.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("clamp.default")})
    x = operand(ec, op, 1)
    l, h = clampbounds(eltype(x), ec.dims, get(op.attrs, "arg1", nothing),
                       get(op.attrs, "arg2", nothing))
    return elementwise!(ec, op, v -> clamp(v, l, h), x)
end

"""
`aten::where.self(cond, a, b)`, as one three-operand pass.

A zero VALUE is captured and not the type: a closure capturing `T` has a
`Type{Float32}` field, which is not isbits, and a kernel cannot take a
non-bitstype argument. Either branch may also be a host scalar, which `binary!`
handles for two operands and this does for three.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("where.self")})
    c = operand(ec, op, 1)
    a = operand(ec, op, 2)
    b = operand(ec, op, 3)
    z = zero(eltype(dest(ec)))
    isresource(a) && isresource(b) &&
        return elementwise!(ec, op, (p, x, y) -> ifelse(p, oftype(z, x), oftype(z, y)),
                            c, a, b)
    isresource(a) && return elementwise!(ec, op,
        (p, x) -> ifelse(p, oftype(z, x), oftype(z, b)), c, a)
    isresource(b) && return elementwise!(ec, op,
        (p, y) -> ifelse(p, oftype(z, a), oftype(z, y)), c, b)
    return elementwise!(ec, op, p -> ifelse(p, oftype(z, a), oftype(z, b)), c)
end

"""
`full`, `full_like` and `empty.memory_format`: a constant into the declared
buffer.

`empty` leaves the contents undefined in torch, so zeroing is a legal
implementation and the only one under which a graph that wrongly reads the
result fails the same way twice instead of intermittently. SAM 2's decoder uses
it for the `(1, 0, 256)` tensor concatenated onto the sparse embeddings when
there are no boxes, where there is nothing to fill either way.

The VALUE can be symbolic, not just the shape: a graph that materialises its own
sequence length writes `full((1,), t)`.
"""
function emitfill!(ec::EmitCtx, op::Op, v)
    out = dest(ec)
    M.dispatch!(ec.g, M.fill_kernel!, (out, convert(eltype(out), v)), length(out);
                name = op.id)
    return out
end

emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("full.default")}) =
    emitfill!(ec, op, numattr(ec.dims, op.attrs["arg1"]))
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("full_like.default")}) =
    emitfill!(ec, op, numattr(ec.dims, op.attrs["arg1"]))
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("empty.memory_format")}) =
    emitfill!(ec, op, 0)

"""
`aten::pow.Tensor_Scalar`, with the small integer exponents written out.

`x^2` as a multiply rather than a call to `pow` is not a micro-optimisation on a
GPU: `pow` is a library call with a branchy implementation, and the exponent is
a host scalar so the specialisation is free. Anything else goes through `^`.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("pow.Tensor_Scalar")})
    a = operand(ec, op, 1)
    e = operand(ec, op, 2)
    if e isa Real && isinteger(e)
        n = Int(e)
        n == 1 && return elementwise!(ec, op, identity, a)
        n == 2 && return elementwise!(ec, op, x -> x * x, a)
        n == 3 && return elementwise!(ec, op, x -> x * x * x, a)
        n == -1 && return elementwise!(ec, op, inv, a)
        return elementwise!(ec, op, x -> intpow(x, n), a)
    end
    return elementwise!(ec, op, Base.Fix2(^, e), a)
end

"""
`aten::gelu`, in whichever formulation the export asked for.

`approximate = "tanh"` selects the cheap one and torch's default is exact. Read
through `atenarg` because `approximate` is a keyword in almost every PyTorch
source that writes it, and picking the wrong formulation is a silent accuracy
change rather than an error. Both evaluate in `accum(T)` and round once, which is
what PyTorch does for a half tensor.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("gelu.default")})
    f = String(atenarg(op, 1, "approximate", "none")) == "tanh" ? gelutanh : geluexact
    return elementwise!(ec, op, f, operand(ec, op, 1))
end

"""`aten::copy_`'s functional form: the SOURCE is what lands in the
destination, and the first argument is only there to give the shape."""
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("copy.default")}) =
    elementwise!(ec, op, identity, operand(ec, op, 2))

"""
`fused.elementwise` — the group [`fuseops`](@ref) collapses a chain of
elementwise ops into.

Its `FusedOp` is a plain callable, so it needs no kernel of its own: it is the
`f` of one `ew!` dispatch over the group's operands. Passing it as an argument is
the function barrier that keeps the per-element call static -- it comes out of a
`Dict{String,Any}`, so a body that read it inline would dispatch dynamically once
per element.
"""
emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("fused.elementwise")}) =
    elementwise!(ec, op, op.attrs["fused"], map(i -> operand(ec, i), op.ins)...)

# ── repeat, and a reduction ──────────────────────────────────────────────────

"""
`aten::repeat`, as one gather into the planned buffer.

The source is padded with trailing singleton axes to the output's rank, which is
the declaration-time form of the `reshape` loop `runop!` ran: torch prepends
singleton dims when the repeat spec is longer than the rank, and a prepend in
torch's order is an append in the reversed one.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("repeat.default")})
    a = operand(ec, op.ins[1])
    out = dest(ec)
    od = size(out)
    id = ntuple(k -> k <= ndims(a) ? size(a, k) : 1, length(od))
    M.dispatch!(ec.g, tilecopy!, (out, od, a, id), prod(od); name = op.id)
    return out
end

"""
`aten::sum.dim_IntList`, as one dispatch over the output.

`od` carries the reduction: the input's shape with a `1` on each reduced axis.
The output resource may have those axes dropped (`keepdim = false`), which
changes its shape and not its bytes, so the kernel indexes both linearly.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("sum.dim_IntList")})
    a = operand(ec, op.ins[1])
    out = dest(ec)
    id = size(a)
    dims = Tuple(jdim(d, length(id)) for d in ints(op.attrs["arg1"]))
    od = ntuple(k -> k in dims ? 1 : id[k], length(id))
    prod(od) == length(out) ||
        error("DNNKernels: `sum.dim_IntList` (op $(op.id)) reduces $(id) over " *
              "$(dims) to $(prod(od)) elements, and its output buffer holds " *
              "$(length(out)).")
    # `foldpremap` folds a map step into the reduction; `identity` is the
    # unfolded case, so there is one kernel rather than two.
    f = something(premap(op), identity)
    M.dispatch!(ec.g, sumdims!, (out, od, a, id, f), prod(od); name = op.id)
    return out
end

"""
`aten::arange.start_step(start, end, step)`, as one dispatch.

The three scalars may be **fractional** — DINOv3's rotary embedding asks for
`arange(0.5, 32, 1)`, the patch centres — so they are read with `numattr` and
not `intattr`.

The length is aten's own `ceil((end - start) / step)` and not a Julia range's.
Those disagree whenever the step is not 1: `arange(0, 5, 2)` is `[0, 2, 4]` to
torch, three elements, where `0:2:5` is also three but `0:2:4` and `0:2:5` are
not the same range to reason about. `test_arange.jl` pins both, and the length is
checked against the declared buffer rather than allocated to fit — a declared
graph has already placed those bytes.

`end` may be a host value the graph computed rather than an attribute, which is
why `op.ins` is consulted first.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("arange.start_step")})
    # NOT the positional accessor: that assumes every position is either an
    # attribute or an `ins` entry, and `start`/`step` here may be neither --
    # torch defaults them. Same three reads `runop!` made.
    start = numattr(ec.dims, something(get(op.attrs, "arg0", nothing), 0))
    stop  = length(op.ins) >= 1 ? operand(ec, op.ins[1]) :
                                  numattr(ec.dims, op.attrs["arg1"])
    step  = numattr(ec.dims, something(get(op.attrs, "arg2", nothing), 1))
    out = dest(ec)
    n = max(0, ceil(Int, (Float64(stop) - Float64(start)) / Float64(step)))
    n == length(out) || error(
        "DNNKernels: `arange.start_step` (op $(op.id)) is $start:$step:$stop, " *
        "which is $n elements, and its output buffer holds $(length(out)).")
    T = eltype(out)
    M.dispatch!(ec.g, arange!, (out, n, T(start), T(step)), n; name = op.id)
    return out
end

# ── batch norm, in training mode ─────────────────────────────────────────────

"""
`aten::_native_batch_norm_legit.no_stats`, as two passes.

Two dispatches, and a dispatch is a pass: the second reads what the first writes,
and a pass is the unit that may run concurrently, so the dependency between them
is a dependency between passes. Nothing here orders it — `bnstats!` writes `mean`
and `invstd`, `bnapply!` reads them, `argument_usage` reads that off both bodies
and `Barriers` derives the wait. That is the whole of what used to be an implicit
ordering inside a sequence of broadcasts.

All three of torch's results are declared: the normalised output, the mean and
the inverse standard deviation. The last two are what the backward pass reads,
and a graph that never reads them still has them placed, which costs `2C`
floats and keeps the op's shape honest.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("_native_batch_norm_legit.no_stats")})
    x = operand(ec, op.ins[1])
    gamma = operand(ec, op.ins[2])
    beta = operand(ec, op.ins[3])
    out, mean, invstd = dests(ec, 3)
    eps = Float32(op.attrs["arg5"])
    id = size(x)
    # torch's channel dim is 1, which in the reversed shape is `ndims - 1`.
    c = length(id) - 1
    C = id[c]
    cstride = prod(ntuple(k -> id[k], c - 1); init = 1)
    nouter = prod(ntuple(k -> id[c + k], length(id) - c); init = 1)
    M.dispatch!(ec.g, bnstats!, (mean, invstd, x, cstride, C, nouter, eps), C;
                name = "$(op.id).stats")
    M.dispatch!(ec.g, bnapply!,
                (out, x, mean, invstd, gamma, beta, length(out), cstride, C),
                length(out); name = op.id)
    return (out, mean, invstd)
end

# ── convolution ──────────────────────────────────────────────────────────────

"""
`aten::convolution`, forward and dense, as the implicit GEMM plus whatever
split-K needs.

**Everything the run used to decide is decided here, from shapes.** `convtiles`
and `convsplit` read `Cout`, `NPQ`, `CRS` and the shader-core count, all of which
the graph states or the device reports, so the tiling and the split factor are
`Val` parameters of one dispatch rather than a choice made per launch.

Split-K is the interesting half, and it is what `Workspace` was for. Each split
accumulates into the destination atomically, so:

  * the destination must START at the bias rather than being overwritten, which
    is a pass of its own (`ew!` broadcasting the bias along the channel axis, or
    `fill_kernel!` when there is none);
  * a partial sum cannot be clamped, so a fused activation has to wait until the
    splits are summed, and an fp16 output cannot be the accumulator at all
    because Vulkan 1.3 has no fp16 atomic add. Either case accumulates into an
    fp32 SCRATCH and converts once, in a third pass.

That scratch is a declared transient. `Workspace` bump-allocated it from an arena
that reset per op, so its bytes could only ever be reused by the same op; here
its liveness is what its uses say and the placer aliases it against the whole
graph.

**`splitk > 1` makes this convolution non-reproducible**, because the order the
atomics land in is not fixed and float addition is not associative. Measured on
`segment`'s `convolution_21` (fp32, 3x3, 512 -> 768, NPQ = 64, `splitk = 4`): two
runs on identical inputs differ by 3.05e-5 on 9% of elements, carrying to ~5e-7
at the graph's outputs. Not a bug, it is what buys the 4-8x, but it is the floor
for anything measured on a graph containing one.
"""
function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("convolution.default")})
    x = operand(ec, op.ins[1])
    w = operand(ec, op.ins[2])
    bias = length(op.ins) >= 3 ? operand(ec, op.ins[3]) : nothing
    out = dest(ec)
    stride = reverse(ints(op.attrs["arg3"]))
    pad = reverse(ints(op.attrs["arg4"]))
    dil = reverse(ints(op.attrs["arg5"]))
    groups = Int(op.attrs["arg8"])
    act = Symbol(get(op.attrs, "act", "none"))

    # What is declared so far is the dense forward 2-D case. The others are not
    # refusals on principle, they are unported: each has its own kernel in
    # `kernels/extern/` and its own reason to exist, and a graph that needs one
    # should say so here rather than run as something else. `runop!`'s note
    # applies unchanged — a `ConvTranspose2d` taken as an ordinary convolution
    # is a wrong picture with nothing in the numbers to point at it.
    get(op.attrs, "arg6", false) == true && error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) is TRANSPOSED (arg6), and only " *
        "the forward convolution is declared. `convolutiontranspose!` is the " *
        "kernel; it needs an `emitop!` of its own.")
    length(stride) == 2 || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) is $(length(stride))-D, and only " *
        "the 2-D convolution is declared. 1-D lifts to it and 3-D has " *
        "`convolution3d!`; both need an `emitop!` of their own.")
    groups == 1 || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) has $(groups) groups, and only " *
        "the dense convolution is declared. `convolution_direct!` is the " *
        "grouped kernel; it needs an `emitop!` of its own.")

    KWk, KHk, Cin, Cout = size(w)
    Wid, Hei = size(x, 1), size(x, 2)
    OW, OH, _, N = size(out)
    CRS = Cin * KHk * KWk
    NPQ = N * OH * OW
    T = eltype(out)
    ACC = accum(eltype(x))
    cores = M.caps(M.backend(ec.dev)).cores
    BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ = convtiles(Cout, NPQ; cores)
    nbk = cld(Cout, BS_K)
    nbn = cld(NPQ, BS_NPQ)
    splitk = convsplit(nbk, nbn, cld(CRS, BS_CRS); cores)

    # An fp32 destination with no fused activation can take the atomics itself;
    # anything else needs the scratch. Declared, not allocated.
    direct = splitk == 1 || (T === Float32 && act === :none)
    acc = direct ? out : scratch(ec, Float32, size(out)...)
    # Fold the activation into the write-back only when there is a single split.
    kact = (splitk == 1 && act === :relu) ? :relu : :none

    if splitk > 1
        od = size(acc)
        if bias === nothing
            M.dispatch!(ec.g, M.fill_kernel!, (acc, zero(eltype(acc))),
                        length(acc); name = "$(op.id).prefill")
        else
            # The bias lies along the channel axis, which is the third of four in
            # the reversed layout, so it broadcasts with a zero stride everywhere
            # else. One pass, no materialised copy.
            bd = ntuple(k -> k == 3 ? length(bias) : 1, length(od))
            M.dispatch!(ec.g, ew!,
                        (acc, od, (bias,), (bcstrides(od, bd),), identity),
                        prod(od); name = "$(op.id).prefill")
        end
    end

    args = (acc, x, w, bias,
            Val(ACC), Val(splitk), Val(kact),
            Val(BS_K), Val(BS_CRS), Val(BS_NPQ), Val(TS_K), Val(TS_NPQ),
            Val(KWk), Val(KHk),
            Val(stride[1]), Val(stride[2]), Val(pad[1]), Val(pad[2]),
            Val(dil[1]), Val(dil[2]),
            Cin, Cout, Wid, Hei, OW, OH, NPQ, CRS, nbn)
    # `bias` is passed as `nothing` when there is none, rather than omitted: the
    # kernel branches on `bias === nothing` at compile, and `nothing` is a
    # zero-size argument the compiled kernel has no parameter for
    # (`Mantle.NotPassed`), so there is no slot and nothing to declare.
    M.dispatch!(ec.g, conv2d_igemm_ki!, args, (nbk * WG, nbn * splitk);
                group = (WG, 1), name = op.id)

    # The third pass: convert the fp32 scratch down, applying the activation
    # once now that the splits have been summed.
    if acc !== out
        od = size(out)
        f = act === :relu ? (v -> max(v, zero(v))) : identity
        M.dispatch!(ec.g, ew!, (out, od, (acc,), (bcstrides(od, od),), f),
                    prod(od); name = "$(op.id).reduce")
    elseif splitk > 1 && act === :relu
        # In place: `out` is the destination AND the operand, so the walk reports
        # it read+write and the pass is ordered against the splits that wrote it.
        od = size(out)
        M.dispatch!(ec.g, ew!,
                    (out, od, (out,), (bcstrides(od, od),), v -> max(v, zero(v))),
                    prod(od); name = "$(op.id).act")
    end
    return out
end
