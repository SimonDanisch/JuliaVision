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

The uses are the thing capture could not state. The destination is written, the
operands are only read, so two ops that read the same weight no longer serialise
against each other.
"""
function elementwise!(ec::EmitCtx, op::Op, f, ins...)
    out = dest(ec)
    od = size(out)
    M.compute!(ec.g, op.id) do p
        ops, sts = operandtuples(p, od, ins)
        M.dispatch!(p, ew!, (M.use(p, out; write = true), od, ops, sts, f),
                    length(out))
    end
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

The second reason given was that `use(p, x; read = true)` needs each operand to
be a top-level argument. That was simply wrong; `use` is called here, at emit
time, where a walk over the operands does it.
"""
operandtuples(p, od::Dims, ::Tuple{}) = ((), ())
function operandtuples(p, od::Dims, ins::Tuple)
    x = first(ins)
    isresource(x) || error(
        "DNNKernels: elementwise operand of type $(typeof(x)) has no bytes to " *
        "index. A host scalar belongs in the function, as `Base.Fix2(f, x)`, " *
        "rather than in the operand list; `binary!` is where that is decided.")
    ops, sts = operandtuples(p, od, Base.tail(ins))
    return ((M.use(p, x; read = true), ops...),
            (bcstrides(od, size(x)), sts...))
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
    a, b = operand(ec, op.ins[1]), operand(ec, op.ins[2])
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
    elementwise!(ec, op, identity, operand(ec, op.ins[1]))

function emitop!(ec::EmitCtx, op::Op, ::Val{Symbol("leaky_relu.default")})
    x = operand(ec, op.ins[1])
    s = eltype(dest(ec))(something(get(op.attrs, "arg1", nothing), 0.01))
    # The slope is captured, so it travels in the closure and not as an operand.
    elementwise!(ec, op, v -> v >= zero(v) ? v : s * v, x)
end

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
    M.compute!(ec.g, op.id) do p
        M.dispatch!(p, tilecopy!,
                    (M.use(p, out; write = true), od,
                     M.use(p, a; read = true), id),
                    prod(od))
    end
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
    M.compute!(ec.g, op.id) do p
        M.dispatch!(p, sumdims!,
                    (M.use(p, out; write = true), od,
                     M.use(p, a; read = true), id, f),
                    prod(od))
    end
    return out
end

# ── batch norm, in training mode ─────────────────────────────────────────────

"""
`aten::_native_batch_norm_legit.no_stats`, as two passes.

Two `compute!`s and not two dispatches in one, because the second reads what the
first writes: a pass is the unit that may run concurrently, so a dependency
between dispatches is a dependency between passes. The uses state it and
`Barriers` derives the wait, which is the whole of what used to be an implicit
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
    M.compute!(ec.g, "$(op.id).stats") do p
        M.dispatch!(p, bnstats!,
                    (M.use(p, mean; write = true), M.use(p, invstd; write = true),
                     M.use(p, x; read = true), cstride, C, nouter, eps),
                    C)
    end
    M.compute!(ec.g, op.id) do p
        M.dispatch!(p, bnapply!,
                    (M.use(p, out; write = true), M.use(p, x; read = true),
                     M.use(p, mean; read = true), M.use(p, invstd; read = true),
                     M.use(p, gamma; read = true), M.use(p, beta; read = true),
                     length(out), cstride, C),
                    length(out))
    end
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
        M.compute!(ec.g, "$(op.id).prefill") do p
            od = size(acc)
            if bias === nothing
                M.dispatch!(p, M.fill_kernel!,
                            (M.use(p, acc; write = true), zero(eltype(acc))),
                            length(acc))
            else
                # The bias lies along the channel axis, which is the third of
                # four in the reversed layout, so it broadcasts with a zero
                # stride everywhere else. One pass, no materialised copy.
                bd = ntuple(k -> k == 3 ? length(bias) : 1, length(od))
                M.dispatch!(p, ew!,
                            (M.use(p, acc; write = true), od,
                             (M.use(p, bias; read = true),), (bcstrides(od, bd),),
                             identity),
                            prod(od))
            end
        end
    end

    M.compute!(ec.g, op.id) do p
        args = (M.use(p, acc; write = true), M.use(p, x; read = true),
                M.use(p, w; read = true),
                bias === nothing ? nothing : M.use(p, bias; read = true),
                Val(ACC), Val(splitk), Val(kact),
                Val(BS_K), Val(BS_CRS), Val(BS_NPQ), Val(TS_K), Val(TS_NPQ),
                Val(KWk), Val(KHk),
                Val(stride[1]), Val(stride[2]), Val(pad[1]), Val(pad[2]),
                Val(dil[1]), Val(dil[2]),
                Cin, Cout, Wid, Hei, OW, OH, NPQ, CRS, nbn)
        M.dispatch!(p, conv2d_igemm_ki!, args, (nbk * WG, nbn * splitk);
                    group = (WG, 1))
    end

    # The third pass: convert the fp32 scratch down, applying the activation
    # once now that the splits have been summed.
    if acc !== out
        M.compute!(ec.g, "$(op.id).reduce") do p
            od = size(out)
            f = act === :relu ? (v -> max(v, zero(v))) : identity
            M.dispatch!(p, ew!,
                        (M.use(p, out; write = true), od,
                         (M.use(p, acc; read = true),), (bcstrides(od, od),), f),
                        prod(od))
        end
    elseif splitk > 1 && act === :relu
        M.compute!(ec.g, "$(op.id).act") do p
            od = size(out)
            M.dispatch!(p, ew!,
                        (M.use(p, out; write = true), od,
                         (M.use(p, out; read = true),), (bcstrides(od, od),),
                         v -> max(v, zero(v))),
                        prod(od))
        end
    end
    return out
end
