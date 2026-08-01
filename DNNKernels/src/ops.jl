"""
ATen op methods.

Each is one `runop!` method keyed on the op name. Elementwise work is
broadcasting - `Lava.LavaArray <: AbstractGPUArray`, so that is already a fused
device kernel - and reductions come from `AcceleratedKernels`. Only convolution,
pooling and resampling reach a `launch!`.

`arg1` on a binary op means the second operand was a scalar and torch folded it
into the schema; `alpha` scales the second operand (`add.Tensor(a, b, alpha=k)`
is `a + k*b`).
"""

"""
Operands of a binary op.

torch folds a scalar operand into the schema, so it arrives in `attrs` under
its positional slot rather than in `ins`. Either side can be the scalar -
`1 - sigmoid(x)` exports as `sub.Tensor(ins=[sigmoid], arg0=1)` - so both
positions fall back to attrs and `ins` is consumed in order.
"""
function operand(ctx::Ctx, op::Op, pos::Int)
    key = "arg$(pos-1)"
    haskey(op.attrs, key) && return scalar(op.attrs[key])
    # the pos'th operand is a tensor: count how many earlier positions were scalars
    idx = pos - count(p -> haskey(op.attrs, "arg$(p-1)"), 1:(pos - 1))
    idx <= length(op.ins) ||
        error("$(op.aten) ($(op.id)) has no operand at position $pos")
    value(ctx, op.ins[idx])
end

lhs(ctx::Ctx, op::Op) = operand(ctx, op, 1)
rhs(ctx::Ctx, op::Op, i::Int=2) = operand(ctx, op, i)
alpha(op::Op) = get(op.attrs, "alpha", 1)

"""Float -> integer the way torch does it: saturate on infinities, NaN to zero."""
@inline function safetrunc(::Type{T}, v) where {T<:Integer}
    isnan(v) && return zero(T)
    v >= typemax(T) && return typemax(T)
    v <= typemin(T) && return typemin(T)
    return trunc(T, v)
end

"""Non-finite scalars are serialised as strings; see export_graphs.const."""
scalar(v) = v
scalar(v::AbstractString) = v == "-inf" ? -Inf32 : v == "inf" ? Inf32 :
                            v == "nan" ? NaN32 : v

"""
    intlist(ctx, v) -> Vector{Int}

An attr list whose elements may be constants or `"\$buffer"` references to host
scalars (see export_graphs: mixed lists keep their order in the attr).
"""
intlist(ctx::Ctx, v) = Int[el isa AbstractString && startswith(el, "\$") ?
                           Int(value(ctx, el[2:end])) : Int(el) for el in v]

# ---------------------------------------------------------------- elementwise

"""
    erf(x)

Abramowitz & Stegun 7.1.26, max error ~1.5e-7 — comfortably inside fp32.

`erf.default` has been in the elementwise table all along but the function it
names was never defined: neither MatAnyone nor BasicVSR++ contains an `erf`, so
the op would have thrown `UndefVarError` the first time one did. Wan's `gelu` is
that first time. Written as a rational approximation rather than pulled from
SpecialFunctions because it has to compile into a GPU kernel.
"""
@inline function erf(x)
    T = typeof(float(x))
    a = abs(x)
    t = one(T) / (one(T) + T(0.3275911) * a)
    y = one(T) - (((((T(1.061405429) * t - T(1.453152027)) * t) + T(1.421413741)) * t -
                   T(0.284496736)) * t + T(0.254829592)) * t * exp(-a * a)
    return x < zero(x) ? -y : y
end

for (name, f) in (("relu.default", :relu_), ("sigmoid.default", :sigmoid_),
                  ("tanh.default", :tanh), ("log.default", :log),
                  ("exp.default", :exp), ("sqrt.default", :sqrt),
                  ("rsqrt.default", :(x -> inv(sqrt(x)))), ("neg.default", :(-)),
                  ("abs.default", :abs), ("reciprocal.default", :inv),
                  ("bitwise_not.default", :(!)), ("logical_not.default", :(!)),
                  ("sin.default", :sin), ("cos.default", :cos),
                  ("erf.default", :erf), ("floor.default", :floor))
    # Writes into the op's planned slab slot instead of returning a fresh array;
    # `opdest` falls back to allocating when the buffer was not planned.
    @eval function runop!(ctx::Ctx, op::Op, ::Val{Symbol($name)})
        emit(ctx, Base.broadcasted($f, lhs(ctx, op)))
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("add.Tensor")})
    a, b, k = lhs(ctx, op), rhs(ctx, op), alpha(op)
    sum_ = k == 1 ? Base.broadcasted(+, a, b) :
                    Base.broadcasted(+, a, Base.broadcasted(*, k, b))
    # `act` comes from `foldrelu`; the relu op it replaced is gone and its buffer
    # aliases this one. Folded into the same broadcast, so it is free.
    if Symbol(get(op.attrs, "act", "none")) === :relu
        T = dtypeof(ctx, op.out)
        return emit(ctx, Base.broadcasted(max, sum_, zero(T)))
    end
    emit(ctx, sum_)
end
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("sub.Tensor")})
    a, b, k = lhs(ctx, op), rhs(ctx, op), alpha(op)
    emit(ctx, k == 1 ? Base.broadcasted(-, a, b) :
                       Base.broadcasted(-, a, Base.broadcasted(*, k, b)))
end
runop!(ctx::Ctx, op::Op, ::Val{Symbol("mul.Tensor")}) =
    emit(ctx, Base.broadcasted(*, lhs(ctx, op), rhs(ctx, op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("div.Tensor")}) =
    emit(ctx, Base.broadcasted(/, lhs(ctx, op), rhs(ctx, op)))
"""
Exponentiation by squaring.

`x^n` with a *runtime* integer `n` converts to a float exponent and lowers to
the device `pow()`, which is NaN for a negative base - while Julia's CPU `^`
special-cases integral exponents and returns the right answer. A literal `x^2`
also folds to `x*x` and hides the difference. `key_proj` squares a conv output
that is mostly negative (big_modules.py KeyProjection), so on GPU the whole
shrinkage term came back NaN.

Written without recursion: SPIR-V rejects an entry point whose call graph has a
cycle, so `inv(intpow(x, -n))` fails to validate rather than to run.
"""
@inline function intpow(x::T, n::Integer) where {T}
    k = n < 0 ? -n : n
    r = one(T)
    b = x
    while k > 0
        (k & 1) == 1 && (r *= b)
        b *= b
        k >>= 1
    end
    n < 0 ? inv(r) : r
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("pow.Tensor_Scalar")})
    a = lhs(ctx, op)
    e = rhs(ctx, op)
    (e isa Integer || (e isa Real && isinteger(e))) && return intpow.(a, Int(e))
    a .^ e
end
runop!(ctx::Ctx, op::Op, ::Val{Symbol("ge.Tensor")}) = lhs(ctx, op) .>= rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("ge.Scalar")}) = lhs(ctx, op) .>= rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("eq.Scalar")}) = lhs(ctx, op) .== rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("bitwise_and.Tensor")}) = lhs(ctx, op) .& rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("logical_and.default")}) = lhs(ctx, op) .& rhs(ctx, op)
# Converted to the declared dtype inside the broadcast, because `ifelse` does not
# promote: `ifelse(::Bool, ::Float16, ::Float32)` infers `Union{Float16,Float32}`,
# and a broadcast over an abstract eltype reaches Lava as a
# `BrokenBroadcast{AbstractFloat}` that fails to compile — "method lookup failure"
# hundreds of frames from the op that caused it. The two branches always had the
# same dtype until autocast put a cast boundary between them.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("where.self")})
    c = lhs(ctx, op)
    a, b = value(ctx, op.ins[2]), value(ctx, op.ins[3])
    # A zero *value*, not the type: a closure capturing `T` has a `Type{Float32}`
    # field, which is not isbits, and a kernel cannot take a non-bitstype argument.
    z = zero(dtypeof(ctx, op.out))
    emit(ctx, Base.broadcasted((p, x, y) -> ifelse(p, oftype(z, x), oftype(z, y)), c, a, b))
end
runop!(ctx::Ctx, op::Op, ::Val{Symbol("clamp.default")}) =
    (lo = get(op.attrs, "arg1", nothing); hi = get(op.attrs, "arg2", nothing);
     clamp.(lhs(ctx, op), lo === nothing ? -Inf32 : Float32(lo),
            hi === nothing ? Inf32 : Float32(hi)))
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("clone.default")})
    a = lhs(ctx, op)
    d = alloc(ctx, op.out, size(a)...)
    d .= a
    d
end

# autocast's dtype boundaries. Writing into the op's planned slot rather than
# `copy(...)`: `copy` calls `similar`, so all 118 of these a step allocated a
# fresh device buffer outside the static plan — and then `coerce` converted the
# result in a *second* pass. `opdest` is typed by the graph's declared dtype, so
# the conversion happens inside this broadcast and `coerce` becomes a no-op.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_to_copy.default")})
    a = lhs(ctx, op)
    d = opdest(ctx, a)
    # torch's float -> integer cast truncates toward zero; Julia's `convert`
    # throws `InexactError` on anything with a fractional part. The Wan VAE hits
    # this casting attention index arithmetic, where 0.25 is a legitimate input
    # that torch turns into 0.
    if eltype(d) <: Integer && !(eltype(a) <: Integer)
        # Saturating, not just truncating: torch turns +/-Inf into the integer
        # extremes and NaN into 0, while Julia's `trunc` throws `InexactError`.
        # T5's attention mask carries -Inf into exactly this cast.
        d .= safetrunc.(eltype(d), a)
    else
        d .= a
    end
    d
end

# aten::copy is functional: it returns src broadcast to dst's shape and must
# NOT write into dst. dst is usually a view, so an in-place `.=` would scribble
# through it into the parent buffer and silently corrupt an earlier value. The
# op's *own* planned slot is a different buffer, so writing there is both safe
# and free.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("copy.default")})
    dst = lhs(ctx, op)
    src = value(ctx, op.ins[2])
    out = dest(ctx, eltype(dst), size(dst)...)
    out .= src
    out
end

# --------------------------------------------------------------- constructors

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("full.default")})
    sz = evalshape(shapeof(ctx, op.out), ctx.dims)
    fill!(alloc(ctx, op.out, sz...), convert(dtypeof(ctx, op.out), scalar(op.attrs["arg1"])))
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("full_like.default")})
    a = lhs(ctx, op)
    T = dtypeof(ctx, op.out)
    fill!(alloc(ctx, T, size(a)...), convert(T, scalar(op.attrs["arg1"])))
end

"""
`empty.memory_format` allocates without initialising.

Zero-filled rather than handed back as whatever the slab last held. torch leaves
the contents undefined, so zeroing is a legal implementation of it and the only
one under which a graph that (wrongly) reads the result fails the same way twice
instead of intermittently. SAM 2's decoder uses it for a `(1, 0, 256)` tensor
concatenated onto the sparse embeddings — the branch where there are no boxes —
so here there is nothing to fill anyway.
"""
runop!(ctx::Ctx, op::Op, ::Val{Symbol("empty.memory_format")}) =
    fill!(alloc(ctx, op.out, evalshape(shapeof(ctx, op.out), ctx.dims)...),
          zero(dtypeof(ctx, op.out)))

runop!(ctx::Ctx, op::Op, ::Val{Symbol("scalar_tensor.default")}) =
    fill!(alloc(ctx, op.out), convert(dtypeof(ctx, op.out), scalar(op.attrs["arg0"])))

@inline arange_body(I, start, step) = start + (I[1] - 1) * step

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("arange.start_step")})
    start = something(get(op.attrs, "arg0", nothing), 0)
    stop = length(op.ins) >= 1 ? value(ctx, op.ins[1]) : op.attrs["arg1"]
    step = something(get(op.attrs, "arg2", nothing), 1)
    T = dtypeof(ctx, op.out)
    n = length(T(start):T(step):T(stop - step))
    launch!(arange_body, alloc(ctx, op.out, n), T(start), T(step))
end

shapeof(ctx::Ctx, id) = ctx.graph.buffers[id].shape
dtypeof(ctx::Ctx, id) = ctx.graph.buffers[id].dtype

# ------------------------------------------------------------------ reductions

torchdims(op::Op, n) = [jdim(d, n) for d in ints(op.attrs["arg1"])]
keepdim(op::Op) = get(op.attrs, "arg2", false) === true

function reduced(a, d, keep)
    keep && return a
    dropdims(a; dims=Tuple(d))
end

runop!(ctx::Ctx, op::Op, ::Val{Symbol("sum.dim_IntList")}) =
    (a = lhs(ctx, op); d = torchdims(op, ndims(a)); reduced(sum(a; dims=Tuple(d)), d, keepdim(op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("mean.dim")}) =
    (a = lhs(ctx, op); d = torchdims(op, ndims(a)); reduced(sum(a; dims=Tuple(d)) ./ prod(size(a, i) for i in d), d, keepdim(op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("prod.dim_int")}) =
    (a = lhs(ctx, op); d = [jdim(Int(op.attrs["arg1"]), ndims(a))];
     reduced(prod(a; dims=Tuple(d)), d, keepdim(op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("any.dim")}) =
    (a = lhs(ctx, op); d = [jdim(Int(op.attrs["arg1"]), ndims(a))];
     reduced(any(a; dims=Tuple(d)), d, keepdim(op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("all.dim")}) =
    (a = lhs(ctx, op); d = [jdim(Int(op.attrs["arg1"]), ndims(a))];
     reduced(all(a; dims=Tuple(d)), d, keepdim(op)))

# aten::max.dim returns (values, indices). Base's argmax hands back
# CartesianIndex objects, which is host-shaped; scanning the axis explicitly
# keeps both results on the execution backend.
@inline function maxdim_body(I, a, ::Val{D}, ::Val{WANTIDX}) where {D,WANTIDX}
    @inbounds begin
        J = ntuple(k -> k == D ? 1 : I[k], Val(length(I)))
        best = a[CartesianIndex(J)]
        bi = 1
        for i in 2:size(a, D)
            v = a[CartesianIndex(ntuple(k -> k == D ? i : I[k], Val(length(I))))]
            if v > best
                best = v
                bi = i
            end
        end
        WANTIDX ? oftype(best, bi - 1) : best
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("max.dim")})
    a = lhs(ctx, op)
    d = jdim(Int(op.attrs["arg1"]), ndims(a))
    sz = ntuple(k -> k == d ? 1 : size(a, k), ndims(a))
    vals = launch!(maxdim_body, alloc(ctx, eltype(a), sz...), a, Val(d), Val(false))
    inds = launch!(maxdim_body, alloc(ctx, eltype(a), sz...), a, Val(d), Val(true))
    keepdim(op) ? (vals, inds) : (dropdims(vals; dims=d), dropdims(inds; dims=d))
end

"""
One workgroup per softmax slice: max, sum and normalise in a single dispatch.

The three-pass version (`maximum(a; dims)`, `exp.(a .- m)`, `./ sum(e; dims)`)
is four kernels and three allocations, and its two partial reductions go through
`AcceleratedKernels` — which is built for reducing *large* extents, while
attention here softmaxes 16 or 32 elements at a time. Three of these cost
3.9 ms a step that way. Fused, the whole reduction lives in shared memory.

`SOFTMAX_WG` threads cooperate per slice; the reduced axis can be any length.
"""
const SOFTMAX_WG = 64

@kernel function softmax_kernel!(out, @Const(a), ::Val{WG}, pre, n) where {WG}
    lt, = @index(Local, NTuple)
    grp, = @index(Group, NTuple)
    @uniform T = eltype(out)
    sh = @localmem Float32 (WG,)
    # Values that have to survive a `@synchronize` need `@private` storage —
    # a plain local is not guaranteed to on the CPU backend.
    keep = @private Float32 (2,)

    # `base` — the first element of slice `grp-1` in the (pre, n, post) view of
    # `a`, with `pre` elements between consecutive entries along the reduced
    # axis — is recomputed in every phase rather than kept in a local. A plain
    # local does not survive a `@synchronize` on the CPU backend, which splits
    # the kernel into one loop per barrier-free region, so the binding is simply
    # not in scope in the next one (`UndefVarError: base not defined`). Three
    # integer ops is cheaper than a `@private` slot.
    @inbounds begin
        base = ((grp - 1) % pre) + pre * n * ((grp - 1) ÷ pre)
        acc = -Inf32
        i = lt
        while i <= n
            acc = max(acc, Float32(a[base + pre * (i - 1) + 1]))
            i += WG
        end
        sh[lt] = acc
    end
    @synchronize
    @inbounds if lt == 1
        m = sh[1]
        for k in 2:WG
            m = max(m, sh[k])
        end
        sh[1] = m
    end
    @synchronize
    @inbounds keep[1] = sh[1]
    @synchronize                       # every thread has read `m` before sh[1] is reused

    @inbounds begin
        base = ((grp - 1) % pre) + pre * n * ((grp - 1) ÷ pre)
        s = 0f0
        i = lt
        while i <= n
            s += exp(Float32(a[base + pre * (i - 1) + 1]) - keep[1])
            i += WG
        end
        sh[lt] = s
    end
    @synchronize
    @inbounds if lt == 1
        s = sh[1]
        for k in 2:WG
            s += sh[k]
        end
        sh[1] = s
    end
    @synchronize

    @inbounds begin
        base = ((grp - 1) % pre) + pre * n * ((grp - 1) ÷ pre)
        denom = sh[1]
        i = lt
        while i <= n
            j = base + pre * (i - 1) + 1
            out[j] = T(exp(Float32(a[j]) - keep[1]) / denom)
            i += WG
        end
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_softmax.default")})
    a = lhs(ctx, op)
    d = jdim(Int(op.attrs["arg1"]), ndims(a))
    # The kernel indexes `a` linearly over the (pre, n, post) view, so a wrapper
    # whose linear order is not a dense array's has to be collapsed first.
    a isa Union{SubArray,PermutedDimsArray,Base.ReshapedArray} &&
        (a = materialize(ctx.rec, ctx.backend, a))
    pre = prod(ntuple(k -> size(a, k), d - 1); init=1)
    n = size(a, d)
    post = length(a) ÷ (pre * n)
    out = dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, size(a)...)
    softmax_kernel!(ctx.backend, SOFTMAX_WG)(out, a, Val(SOFTMAX_WG), pre, n;
                                             ndrange = SOFTMAX_WG * pre * post)
    out
end

# aten::native_layer_norm returns (out, mean, rstd); normalized over the
# trailing torch dims, i.e. the leading Julia dims
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("native_layer_norm.default")})
    a = lhs(ctx, op)
    nshape = ints(op.attrs["arg1"])
    d = Tuple(1:length(nshape))
    eps = Float32(op.attrs["arg4"])
    n = prod(size(a, i) for i in d)
    γ = length(op.ins) >= 2 ? value(ctx, op.ins[2]) : nothing
    β = length(op.ins) >= 3 ? value(ctx, op.ins[3]) : nothing
    # One kernel instead of six passes, where the layout allows it: `a` dense
    # with the normalised axis fastest, which is what the reversed layout gives
    # (torch normalises over trailing dims, we see them leading). 0.341 -> see
    # `kernels/layernorm.jl` for the measurement and why two reduction passes
    # rather than one.
    # `LavaArray` rather than `AbstractArray`: dense and contiguous by
    # construction, which is what the kernel indexes on, and it keeps a lazy
    # `Broadcasted` or a permuted view out of a path that cannot take them.
    if LN_FUSED[] && a isa Lava.LavaArray && d == Tuple(1:length(d)) && length(a) % n == 0
        out = tupledest(ctx, 0, tupledtype(ctx, 0, eltype(a)), size(a)...)
        groups = length(a) ÷ n
        μ = tupledest(ctx, 1, Float32, groups)
        r = tupledest(ctx, 2, Float32, groups)
        layernorm!(out, μ, r, a, γ, β, n, eps; backend = ctx.backend)
        return (out, μ, r)
    end
    μ = sum(a; dims=d) ./ n
    # The centred tensor goes to workspace scratch, not to a fresh allocation.
    # Written the obvious way — `sum(abs2, a .- μ; dims=d)` — the dot syntax
    # materialises a full-size temporary per layer norm, and on SAM 2's image
    # encoder those were **1 131 MB of the 1 649 MB the graph asks the pool for
    # on every call**: the single largest source, ahead of everything else by 4x.
    # The workspace is reset per op, so all 96 layer norms share one buffer
    # instead of taking 96 distinct ones — the same reuse the convolution and
    # attention kernels already rely on.
    #
    # Not rewritten as E[x²] - μ², which needs no temporary at all: that form
    # cancels catastrophically when the mean dominates the variance, and mask
    # parity with PyTorch to five decimals is not worth trading for an
    # allocation.
    v = if ctx.ws === nothing
        sum(abs2, a .- μ; dims=d) ./ n         # verification path, no workspace
    else
        t = scratch!(ctx.ws, ctx.backend, eltype(a), size(a)...)
        t .= Base.broadcasted(-, a, μ)
        sum(abs2, t; dims=d) ./ n
    end
    r = 1 ./ sqrt.(v .+ eps)
    # One broadcast, one write, into the slot the planner reserved.
    #
    # Written as four statements (`y = (a .- μ) .* r`, then `y = y .* γ`, then
    # `y = y .+ β`) each one materialises: four passes over a tensor that is
    # 216 MiB in SAM 2's first stage, and four allocations outside the plan.
    # Kept lazy with `broadcasted` it is a single fused kernel, which is what
    # writing the whole thing as one expression would have given — but the shape
    # of the expression depends on whether the graph supplies γ and β.
    y = Base.broadcasted(*, Base.broadcasted(-, a, μ), r)
    length(op.ins) >= 2 && (y = Base.broadcasted(*, y, value(ctx, op.ins[2])))
    length(op.ins) >= 3 && (y = Base.broadcasted(+, y, value(ctx, op.ins[3])))
    out = tupledest(ctx, 0, tupledtype(ctx, 0, eltype(a)), size(a)...)
    out .= y
    (out, μ, r)
end

# ------------------------------------------------------------------- structural

# Left allocating on purpose. Writing the parts into a planned destination
# removes one allocation but costs one kernel launch *per part* instead of the
# single fused `cat`; measured, that is a net loss (43.9 -> 39.5 steps/s) because
# this workload is launch-bound, not allocation-bound.
"""
Concatenation, written into the op's planned slot.

`Base.cat` allocates its own result — so this op escaped the static plan
entirely — and reaches it through `cat_t`/`__cat`, whose per-part `copyto!` on a
GPU array goes the long way round. Writing each part into a view of the
destination is the same number of dispatches with none of that: 21 `cat`s a step
cost 5.2 ms before and a fifth of that after.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("cat.default")})
    # A strided `SubArray` source (e.g. `select` on a middle axis of a 5-D
    # tensor) has no pointer for the device copy to start from, so the
    # broadcast-assign below fails with "conversion to pointer not defined".
    # Materialising just those keeps the dense inputs copy-free.
    parts = [(v = value(ctx, i); v isa SubArray ? materialize(ctx.rec, ctx.backend, v) : v)
             for i in op.ins]
    n = ndims(parts[1])
    d = jdim(Int(get(op.attrs, "arg1", 0)), n)
    total = sum(size(p, d) for p in parts)
    dims = ntuple(k -> k == d ? total : size(parts[1], k), n)
    out = dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, dims...)
    off = 0
    for p in parts
        len = size(p, d)
        len == 0 && continue
        view(out, ntuple(k -> k == d ? ((off + 1):(off + len)) : Colon(), n)...) .= p
        off += len
    end
    out
end

"""Element of an outer repetition: the source tiles, so index it modulo its own size."""
@inline repeatouter(I, a, sz::NTuple{N,Int}) where {N} =
    @inbounds a[ntuple(k -> mod1(I[k], sz[k]), Val(N))...]

# One gather into the planned slot. `repeat(a; inner=all-ones, outer=reps)` is
# two allocations and two passes: Julia runs the inner phase first, which with
# every `inner` at 1 copies the array to produce exactly the array it was given,
# and then allocates the outer result. Both land outside the plan — 88 MB and
# 126 MB per call on SAM 2's encoder — for an op that reads each source element
# and writes it `prod(reps)` times.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("repeat.default")})
    a = lhs(ctx, op)
    reps = reverse(ints(op.attrs["arg1"]))
    # torch prepends singleton dims when the repeat spec is longer than the rank
    while ndims(a) < length(reps)
        a = reshape(a, size(a)..., 1)
    end
    sz = size(a)
    r = ntuple(k -> k <= length(reps) ? Int(reps[k]) : 1, length(sz))
    out = dest(ctx, eltype(a), map(*, sz, r)...)
    launch!(repeatouter, out, a, sz)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("constant_pad_nd.default")})
    a = value(ctx, op.ins[1])
    pad = intlist(ctx, op.attrs["arg1"])  # (lo, hi) per torch dim from the last
    v = convert(eltype(a), get(op.attrs, "arg2", 0))
    dims = collect(size(a))
    los = zeros(Int, ndims(a))
    for k in 1:(length(pad) ÷ 2)
        lo, hi = pad[2k - 1], pad[2k]     # torch dim -k, i.e. Julia dim k
        dims[k] += lo + hi
        los[k] = lo
    end
    # Into the planned slot, not a fresh buffer. Wan's VAE decoder pads before
    # each of its 116 3-D convolutions, and at 256x256x9 those temporaries are
    # hundreds of MB apiece — allocating them outside the plan is what took the
    # decode from a 1.2 GB slab to a 14.8 GB peak, i.e. from comfortable to OOM.
    out = fill!(dest(ctx, eltype(a), Tuple(dims)...), v)
    idx = ntuple(k -> (los[k] + 1):(los[k] + size(a, k)), ndims(a))
    out[idx...] = a
    out
end

# `copy` here allocated a fresh device buffer on every call, outside the static
# plan — with the pool near its limit that was 64 MiB of churn per step and an
# OOM-reclaim (a full device flush plus a GC) to go with it. The planned slot
# holds exactly this buffer; seeding it from the source is the same one pass the
# copy was.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("slice_scatter.default")})
    a = lhs(ctx, op)
    dst = dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, size(a)...)
    dst .= a
    src = value(ctx, op.ins[2])
    n = ndims(dst)
    d = jdim(Int(get(op.attrs, "arg2", 0)), n)
    lo = Int(get(op.attrs, "arg3", 0))
    hi = get(op.attrs, "arg4", nothing)
    hi = hi === nothing ? size(dst, d) : min(Int(hi), size(dst, d))
    idx = ntuple(i -> i == d ? ((lo + 1):hi) : Colon(), n)
    dst[idx...] = src
    dst
end

# --------------------------------------------------------------------- extern

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("convolution.default")})
    x = lhs(ctx, op)
    w = value(ctx, op.ins[2])
    bias = length(op.ins) >= 3 ? value(ctx, op.ins[3]) : nothing
    stride = reverse(ints(op.attrs["arg3"]))
    pad = reverse(ints(op.attrs["arg4"]))
    dil = reverse(ints(op.attrs["arg5"]))
    groups = Int(op.attrs["arg8"])
    # `aten::convolution` is also the TRANSPOSED convolution: arg6 says so and
    # arg7 carries its output padding. Reading neither means a `ConvTranspose2d`
    # would run here as an ordinary convolution — a wrong picture, no error, and
    # nothing in the numbers to point at it. SAM 2's mask decoder upsamples with
    # exactly that layer, so this refuses instead of guessing until the
    # transposed path exists (same reasoning as the 3-D convolution that silently
    # took the 2-D branch during the Wan port).
    # `act` is set by `foldrelu`, which deletes the relu op and aliases its
    # buffer onto this convolution's output. Every path below has to honour it —
    # dropping it silently would be a wrong answer, not a slow one.
    act = Symbol(get(op.attrs, "act", "none"))
    outpad = reverse(ints(get(op.attrs, "arg7", Int[])))
    if get(op.attrs, "arg6", false) == true
        # `aten::convolution` is also the TRANSPOSED convolution — arg6 says so.
        # Reading neither this nor arg7 is how a `ConvTranspose2d` silently runs
        # as an ordinary convolution: a wrong picture, no error, nothing in the
        # numbers to point at. SAM 2's mask decoder upsamples with two of them.
        length(stride) == 2 ||
            error("transposed convolution is implemented for 2-D only (op $(op.id))")
        ox = convtransposesize(size(x, 1), size(w, 1), stride[1], pad[1], dil[1], outpad[1])
        oy = convtransposesize(size(x, 2), size(w, 2), stride[2], pad[2], dil[2], outpad[2])
        out = alloc(ctx, eltype(x), ox, oy, size(w, 3), size(x, 4))
        convolutiontranspose!(out, x, w, bias, stride, pad, dil, outpad, groups; ws = ctx.ws)
        act === :relu && (out .= max.(out, zero(eltype(out))))
        return out
    end
    ox = convsize(size(x, 1), size(w, 1), stride[1], pad[1], dil[1])
    if length(stride) == 1                       # aten::convolution covers 1-D too
        out = alloc(ctx, eltype(x), ox, size(w, 3), size(x, 3))
        convolution1d!(out, x, w, bias, stride, pad, dil, groups)
        # No epilogue to fold into here; correctness first, and the 1-D
        # convolutions are the 9 channel-attention layers, not a hot path.
        act === :relu && (out .= max.(out, zero(eltype(out))))
    elseif length(stride) == 3                   # and 3-D, for the Wan VAE
        oy = convsize(size(x, 2), size(w, 2), stride[2], pad[2], dil[2])
        oz = convsize(size(x, 3), size(w, 3), stride[3], pad[3], dil[3])
        out = alloc(ctx, eltype(x), ox, oy, oz, size(w, 5), size(x, 5))
        convolution3d!(out, x, w, bias, stride, pad, dil, groups; act)
    else
        oy = convsize(size(x, 2), size(w, 2), stride[2], pad[2], dil[2])
        out = alloc(ctx, eltype(x), ox, oy, size(w, 4), size(x, 4))
        convolution!(out, x, w, bias, stride, pad, dil, groups; ws=ctx.ws, act)
    end
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_native_batch_norm_legit_no_training.default")})
    x = lhs(ctx, op)
    γ, β = value(ctx, op.ins[2]), value(ctx, op.ins[3])
    μ, v = value(ctx, op.ins[4]), value(ctx, op.ins[5])
    eps = Float32(op.attrs["arg6"])
    c = ndims(x) - 1                      # torch channel dim 1 -> Julia dim n-1
    rs = ntuple(i -> i == c ? length(γ) : 1, ndims(x))
    s = reshape(γ ./ sqrt.(v .+ eps), rs)
    b = reshape(β, rs) .- reshape(μ, rs) .* s
    (x .* s .+ b, similar(x, 0), similar(x, 0))
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_adaptive_avg_pool2d.default")})
    x = lhs(ctx, op)
    oy = Int(value(ctx, op.ins[2]))       # torch (H, W) -> Julia (y, x)
    ox = Int(value(ctx, op.ins[3]))
    out = alloc(ctx, eltype(x), ox, oy, size(x, 3), size(x, 4))
    adaptive_avg_pool2d!(out, x)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("upsample_bilinear2d.vec")})
    x = lhs(ctx, op)
    target = evalshape(shapeof(ctx, op.out), ctx.dims)
    out = alloc(ctx, eltype(x), target...)
    # arg2 is align_corners; the graphs use both conventions
    upsample_bilinear2d!(out, x; align_corners = Bool(something(get(op.attrs, "arg2", nothing), false)))
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("upsample_nearest2d.vec")})
    x = lhs(ctx, op)
    target = evalshape(shapeof(ctx, op.out), ctx.dims)
    out = alloc(ctx, eltype(x), target...)
    upsample_nearest2d!(out, x)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("cumsum.default")})
    a = lhs(ctx, op)
    out = alloc(ctx, op.out, size(a)...)
    cumsum_dim!(out, a, jdim(Int(op.attrs["arg1"]), ndims(a)))
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("max_pool2d_with_indices.default")})
    x = lhs(ctx, op)
    k = reverse(ints(op.attrs["arg1"]))
    s = haskey(op.attrs, "arg2") ? reverse(ints(op.attrs["arg2"])) : k
    p = haskey(op.attrs, "arg3") ? reverse(ints(op.attrs["arg3"])) : [0, 0]
    ox = (size(x, 1) + 2p[1] - k[1]) ÷ s[1] + 1
    oy = (size(x, 2) + 2p[2] - k[2]) ÷ s[2] + 1
    # `tupledest`, not `alloc`: this op's result is a tuple, so the planner
    # reserved it under `"<id>.0"` and asking under the bare id finds nothing.
    # Six of these missed their reservations for 32.9 MB per encode — allocated
    # fresh while the slab held space for them the whole time.
    out = tupledest(ctx, 0, eltype(x), ox, oy, size(x, 3), size(x, 4))
    maxpool2d!(out, x, k, s, p)
    (out, similar(out, Int64, 0))
end

# addmm(bias, a, b) = bias + a*b ; bmm is batched over the leading torch dim.
# Reversed layout swaps the operands: torch (m,k)x(k,n) is Julia (n,k)x(k,m).
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("addmm.default")})
    bias = lhs(ctx, op)
    a, b = value(ctx, op.ins[2]), value(ctx, op.ins[3])
    out = alloc(ctx, op.out, size(b, 1), size(a, 2))
    matmul!(out, b, a, bias; ws=ctx.ws)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("mm.default")})
    a, b = lhs(ctx, op), value(ctx, op.ins[2])
    out = alloc(ctx, op.out, size(b, 1), size(a, 2))
    matmul!(out, b, a; ws=ctx.ws)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_scaled_dot_product_efficient_attention.default")})
    q, k, v = value(ctx, op.ins[1]), value(ctx, op.ins[2]), value(ctx, op.ins[3])
    bias = length(op.ins) >= 4 ? value(ctx, op.ins[4]) : nothing
    s = get(op.attrs, "scale", nothing)
    scale = s === nothing ? inv(sqrt(size(q, 1))) : Float64(s)
    out = sdpa(q, k, v, bias, scale; backend=ctx.backend, ws=ctx.ws,
               out=tupledest(ctx, 0, tupledtype(ctx, 0, accum(eltype(q))),
                             size(v, 1), size(q, 2), size(q, 3), size(q, 4)))
    # (output, logsumexp, philox_seed, philox_offset); only the first is read
    (out, similar(out, 0), similar(out, 0), similar(out, 0))
end

# PyTorch picks the backend, and under autocast it picks flash for the unmasked
# attentions and mem-efficient for the masked ones, so both ATen ops appear in
# the same graph. Flash takes no attn_bias and carries an explicit scale.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_scaled_dot_product_flash_attention.default")})
    q, k, v = value(ctx, op.ins[1]), value(ctx, op.ins[2]), value(ctx, op.ins[3])
    s = get(op.attrs, "scale", nothing)
    scale = s === nothing ? inv(sqrt(size(q, 1))) : Float64(s)
    out = sdpa(q, k, v, nothing, scale; backend=ctx.backend, ws=ctx.ws,
               out=tupledest(ctx, 0, tupledtype(ctx, 0, accum(eltype(q))),
                             size(v, 1), size(q, 2), size(q, 3), size(q, 4)))
    e = similar(out, 0)
    # (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
    #  philox_seed, philox_offset, debug_attn_mask)
    (out, e, e, e, e, e, e, e, e)
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("bmm.default")})
    a, b = lhs(ctx, op), value(ctx, op.ins[2])
    # a: torch (B,m,k) -> Julia (k,m,B); b: torch (B,k,n) -> Julia (n,k,B)
    out = alloc(ctx, op.out, size(b, 1), size(a, 2), size(a, 3))
    batchedmatmul!(out, b, a)
    out
end

# ------------------------------------------------------------ BasicVSR++ ops

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("leaky_relu.default")})
    x = lhs(ctx, op)
    s = eltype(x)(something(get(op.attrs, "arg1", nothing), 0.01))
    emit(ctx, Base.broadcasted(v -> v >= zero(v) ? v : s * v, x))
end

"""
`flip` reverses whole axes, so it is a permutation of indices rather than
arithmetic — `reverse` on the resolved Julia dimensions, materialised because
the result feeds ops that want real storage.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("flip.default")})
    x = lhs(ctx, op)
    dims = ints(op.attrs["arg1"])
    jd = Tuple(jdim(d, ndims(x)) for d in (dims isa Integer ? [dims] : dims))
    out = alloc(ctx, eltype(x), size(x)...)
    flip!(out, x, jd)
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("avg_pool2d.default")})
    x = lhs(ctx, op)
    k = ints(op.attrs["arg1"])
    kk = k isa Integer ? (k, k) : Tuple(k)
    s = ints(get(op.attrs, "arg2", collect(kk)))
    ss = isempty(s) ? kk : (s isa Integer ? (s, s) : Tuple(s))
    p = ints(get(op.attrs, "arg3", [0, 0]))
    pp = p isa Integer ? (p, p) : Tuple(p)
    # torch orders these (H, W); Julia's first axis is W
    kw, kh = kk[end], kk[1]
    sw, sh = ss[end], ss[1]
    pw, ph = pp[end], pp[1]
    ow = (size(x, 1) + 2pw - kw) ÷ sw + 1
    oh = (size(x, 2) + 2ph - kh) ÷ sh + 1
    out = alloc(ctx, eltype(x), ow, oh, size(x, 3), size(x, 4))
    avg_pool2d!(out, x, kw, kh, sw, sh, pw, ph)
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("grid_sampler_2d.default")})
    x = value(ctx, op.ins[1])
    grid = value(ctx, op.ins[2])
    align = Bool(something(get(op.attrs, "arg4", nothing), true))
    # arg3 is torch's padding_mode enum: 0 = zeros, 1 = border, 2 = reflection.
    # BasicVSR++ uses both 0 and 1; treating border as zeros leaves a dark rim
    # that the next warp amplifies.
    pad = Int(something(get(op.attrs, "arg3", nothing), 0)) == 1 ? :border : :zeros
    # grid is (2, W, H, N) after reversal — coordinates first
    out = alloc(ctx, eltype(x), size(grid, 2), size(grid, 3), size(x, 3), size(x, 4))
    grid_sample2d!(out, x, grid; align_corners = align, padding = pad)
end

"""
torchvision's op, not an ATen one — it survives `run_decompositions` intact
because it is a registered custom operator, which is exactly what we want: one
graph node instead of a scatter of index arithmetic to re-fuse.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("torchvision.deform_conv2d.default")})
    x      = value(ctx, op.ins[1])
    w      = value(ctx, op.ins[2])
    offset = value(ctx, op.ins[3])
    mask   = length(op.ins) >= 4 ? value(ctx, op.ins[4]) : nothing
    bias   = length(op.ins) >= 5 ? value(ctx, op.ins[5]) : nothing
    a(k, d) = Int(something(get(op.attrs, k, nothing), d))
    sw, sh = a("arg5", 1), a("arg6", 1)          # torch: stride_h, stride_w
    pw, ph = a("arg7", 0), a("arg8", 0)
    dw, dh = a("arg9", 1), a("arg10", 1)
    groups = a("arg11", 1)
    dgroups = a("arg12", 1)
    ow = (size(x, 1) + 2pw - (size(w, 1) - 1) * dw - 1) ÷ sw + 1
    oh = (size(x, 2) + 2ph - (size(w, 2) - 1) * dh - 1) ÷ sh + 1
    out = alloc(ctx, eltype(x), ow, oh, size(w, 4), size(x, 4))
    deform_conv2d!(out, x, offset, mask, w, bias;
                   stride = (sw, sh), padding = (pw, ph), dilation = (dw, dh),
                   groups = groups, deform_groups = dgroups)
end


"""
`linalg_vector_norm` is the reduction inside RMS norm — 90 of the Wan VAE's
nodes. `arg1` is the order (2 everywhere here), `arg2` the dims, `arg3` keepdim.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("linalg_vector_norm.default")})
    x = lhs(ctx, op)
    ord = Float64(something(get(op.attrs, "arg1", nothing), 2))
    dims = get(op.attrs, "arg2", nothing)
    keep = Bool(something(get(op.attrs, "arg3", nothing), false))
    jd = dims === nothing ? collect(1:ndims(x)) :
         [jdim(d, ndims(x)) for d in ints(dims)]
    sq = ord == 2 ? abs2.(x) : abs.(x) .^ ord
    acc = sum(sq; dims = Tuple(jd))
    r = ord == 2 ? sqrt.(acc) : acc .^ (1 / ord)
    reduced(r, jd, keep)
end

"`_assert_tensor_metadata` is an export-time check with no runtime meaning."
runop!(ctx::Ctx, op::Op, ::Val{Symbol("_assert_tensor_metadata.default")}) =
    lhs(ctx, op)

runop!(ctx::Ctx, op::Op, ::Val{Symbol("mul.Scalar")}) =
    emit(ctx, Base.broadcasted(*, lhs(ctx, op), scalar(op.attrs["arg1"])))

"""
`index.Tensor` is torch's advanced indexing: `arg1` is one entry per dimension,
`nothing` meaning "take the whole axis" and a name meaning "gather with this
index tensor". The Wan VAE uses it for attention's row/column gather, where the
leading axes are `nothing` and the trailing two carry broadcast index tensors.

Torch indexes the un-reversed shape, so entry `k` of `arg1` addresses Julia
dimension `ndims - k + 1`; the index values are 0-based and become 1-based here.
Only the form the graphs actually use is supported — all-`nothing` prefixes then
index tensors — and anything else errors rather than quietly gathering the wrong
axis.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("index.Tensor")})
    x = lhs(ctx, op)
    spec = op.attrs["arg1"]
    n = ndims(x)
    idx = Vector{Any}(undef, n)
    fill!(idx, Colon())
    for (k, e) in enumerate(spec)
        e === nothing && continue
        jd = n - k + 1
        1 <= jd <= n || error("index.Tensor: dim $k out of range for $(n)-d input")
        iv = value(ctx, String(e)[2:end])          # attrs store "$name"
        idx[jd] = vec(Int.(collect(iv))) .+ 1      # torch indices are 0-based
    end
    materialize(ctx, view(x, idx...))
end

"""
`gelu` with torch's default (exact) formulation: `x/2 * (1 + erf(x/sqrt(2)))`.
`arg1 = "tanh"` selects the approximation, which differs by ~1e-3 and is a
different function, not a faster one — so it is dispatched, not assumed.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("gelu.default")})
    x = lhs(ctx, op)
    if String(get(op.attrs, "arg1", "none")) == "tanh"
        c = eltype(x)(0.7978845608028654)     # sqrt(2/pi)
        emit(ctx, Base.broadcasted(v -> 0.5f0 * v * (1 + tanh(c * (v + eltype(x)(0.044715) * v^3))), x))
    else
        emit(ctx, Base.broadcasted(v -> 0.5f0 * v * (1 + erf(v / sqrt(eltype(x)(2)))), x))
    end
end

"""
`view_as_complex` / `view_as_real` are the rotary embedding's pair-packing.

Torch stores the real/imaginary pair in the *last* axis, so after the reversal
this layout uses it becomes the *first* — which is exactly the layout
`reinterpret(reshape, ...)` converts without copying: a `(2, dims...)` real array
and a `(dims...)` complex one share memory.

The pair axis has to be first for that to hold. It always is here, because it is
torch's last, but a graph that permuted it would silently reinterpret the wrong
elements, so it is checked.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("view_as_complex.default")})
    x = lhs(ctx, op)
    size(x, 1) == 2 ||
        error("view_as_complex expects the real/imaginary pair on the first (reversed) axis, got size $(size(x))")
    materialize(reinterpret(reshape, Complex{eltype(x)}, x))
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("view_as_real.default")})
    z = lhs(ctx, op)
    eltype(z) <: Complex ||
        error("view_as_real expects a complex input, got $(eltype(z))")
    materialize(reinterpret(reshape, real(eltype(z)), z))
end

"""
`pow.Scalar` is `base ^ tensor` — the scalar is the *base*, not the exponent
(that is `pow.Tensor_Scalar`). Wan's sinusoidal time embedding is
`10000 ^ (-arange/half)`, which is this one.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("pow.Scalar")})
    # Position 1 is the scalar and position 2 the tensor — `operand` already
    # resolves that from the attrs, which is exactly why it exists. Reaching for
    # `attrs["arg0"]` *and* `lhs` asks for position 1 twice: the base comes back
    # correctly and the "tensor" is the base again, so the result is 0-d and then
    # broadcasts silently against everything downstream.
    b = operand(ctx, op, 1)
    emit(ctx, Base.broadcasted(v -> b^v, operand(ctx, op, 2)))
end

runop!(ctx::Ctx, op::Op, ::Val{Symbol("gt.Scalar")}) =
    emit(ctx, Base.broadcasted(>, lhs(ctx, op), scalar(op.attrs["arg1"])))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("lt.Scalar")}) =
    emit(ctx, Base.broadcasted(<, lhs(ctx, op), scalar(op.attrs["arg1"])))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("minimum.default")}) =
    emit(ctx, Base.broadcasted(min, lhs(ctx, op), rhs(ctx, op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("maximum.default")}) =
    emit(ctx, Base.broadcasted(max, lhs(ctx, op), rhs(ctx, op)))

"""
    embedding(weight, indices)

Token embedding: gather rows of `weight` by `indices`.

Reversed layout puts the embedding dimension first — `weight` is
`(dim, vocab)` and the result is `(dim, indices...)` — so this is a column
gather, and the indices are torch's 0-based ones.
"""
@kernel function embedding_kernel!(out, @Const(w), @Const(idx), n::Int32)
    k = @index(Global, Linear)
    @inbounds if k <= n
        d = size(out, 1)
        col = (Int32(k) - Int32(1)) ÷ Int32(d)          # which token
        row = (Int32(k) - Int32(1)) % Int32(d) + Int32(1)
        out[k] = w[row, Int32(idx[col + Int32(1)]) + Int32(1)]
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("embedding.default")})
    w = value(ctx, op.ins[1])
    idx = value(ctx, op.ins[2])
    d = size(w, 1)
    out = alloc(ctx, eltype(w), d, size(idx)...)
    backend = KernelAbstractions.get_backend(out)
    n = length(out)
    embedding_kernel!(backend)(out, w, reshape(idx, length(idx)), Int32(n); ndrange = n)
    KernelAbstractions.synchronize(backend)
    out
end
