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
    haskey(op.attrs, key) && return numattr(ctx, op.attrs[key])
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

"""
    pycomplex(s) -> ComplexF32 | nothing

Parse a Python complex literal — `"1j"`, `"-2j"`, `"(1+2j)"`, `"(-1.5-0.5j)"`.

`repr()` of a Python `complex` is not a number in JSON, so the exporter writes it
as a string alongside `"inf"`/`"nan"`. Kokoro's iSTFT multiplies a phase by `1j`
before `exp`, and without this the literal reaches a kernel as a `String`: the
failure is a GPU compilation error mentioning `LavaRefValue{String}` inside a
broadcast, which names neither the op nor the attribute.

The split between real and imaginary parts is the last `+`/`-` that is not the
leading sign and not an exponent marker — `1e-5j` has a `-` that belongs to the
exponent.
"""
function pycomplex(s::AbstractString)
    t = strip(String(s), ['(', ')'])
    (endswith(t, "j") && length(t) > 1) || return nothing
    t = t[1:prevind(t, lastindex(t))]
    i = findlast(k -> (t[k] == '+' || t[k] == '-') && k > firstindex(t) &&
                      !(t[prevind(t, k)] in ('e', 'E')), collect(eachindex(t)))
    part(str, dflt) = isempty(str) || str == "+" ? dflt :
                      str == "-" ? -dflt : parse(Float64, str)
    i === nothing && return ComplexF32(0, part(t, 1.0))
    return ComplexF32(parse(Float64, t[firstindex(t):prevind(t, i)]),
                      part(t[i:end], 1.0))
end

"""
    numattr(ctx, v)

A scalar attribute, with symbolic shape expressions resolved.

`scalar` handles the string encodings that are *values* — `"inf"`, `"nan"`,
`"1j"` — and hands anything else back unchanged. Once a graph carries symbolic
shapes, an attribute can also be a shape *expression*: `clamp`'s upper bound and
`full`'s fill value both turn into functions of the sequence length. Those must
be evaluated against the `dims` of this call, or they reach arithmetic as a
`String` and fail with `MethodError: no method matching Float32(::String)` —
which names neither the op nor the symbol.
"""
function numattr(ctx::Ctx, v)
    s = scalar(v)
    s isa AbstractString ? evalexpr(String(s), ctx.dims) : s
end

"""Non-finite and complex scalars are serialised as strings; see export_graphs.const."""
scalar(v) = v
function scalar(v::AbstractString)
    v == "-inf" && return -Inf32
    v == "inf" && return Inf32
    v == "nan" && return NaN32
    c = pycomplex(v)
    return c === nothing ? v : c
end

"""
    intlist(ctx, v) -> Vector{Int}

An attr list whose elements may be constants or `"\$buffer"` references to host
scalars (see export_graphs: mixed lists keep their order in the attr).
"""
intlist(ctx::Ctx, v) = Int[intattr(ctx, el) for el in v]

"""
    intattr(ctx, v) -> Int

One attribute element, whatever form it arrived in:

  * an integer, as most are;
  * `"\$name"`, a reference to a host scalar the graph computed;
  * **a symbolic expression** — `"t"`, `"2*f + 1"` — evaluated against the `dims`
    passed to `call`.

The third is what makes a graph reusable across sequence lengths. `evalshape`
already did it for buffer *shapes*; an attribute that carries a length (an
`arange` bound, a `view`'s target extent) needs the same treatment, and without
it the symbol reaches arithmetic as a `String` and fails with
`MethodError: no method matching -(::String, ::Int64)` — which names neither the
op nor the symbol.
"""
intattr(ctx::Ctx, v::Integer) = Int(v)
intattr(ctx::Ctx, v::Bool) = Int(v)
intattr(ctx::Ctx, v::AbstractString) =
    startswith(v, "\$") ? Int(value(ctx, v[2:end])) : evalexpr(String(v), ctx.dims)
intattr(ctx::Ctx, v) = Int(v)

# ---------------------------------------------------------------- elementwise

"""
    erf(x)

Abramowitz & Stegun 7.1.26, max error ~1.5e-7 — comfortably inside fp32.

`erf.default` has been in the elementwise table all along but the function it
names was never defined: neither MatAnyone nor BasicVSR++ contains an `erf`, so
the op would have thrown `UndefVarError` the first time one did. Wan's `gelu` is
that first time. Written as a rational approximation rather than pulled from
SpecialFunctions because it has to compile into a GPU kernel.

**Evaluated in `accum(T)`, and that is not a nicety.** "Max error ~1.5e-7"
describes the *formula*, not this function: A&S 7.1.26 ends in `1 - poly*exp`,
which for `|x| > 2` subtracts two quantities that agree to four digits, and fp16
has three. Run entirely in half it returns 0 at `x = -4` where the answer is
-1.27e-4, and is 8.5% wrong at `x = -3`. Whisper's encoder is 34 `gelu`s over
that range: with the half evaluation its gelu carried rel rms 8.94e-4 against
PyTorch's, and 2.48e-6 with this one — 360x, from the width alone, same
coefficients. The narrow return keeps the op's dtype the graph's, so only the
single store rounds.
"""
@inline function erf(x)
    S = typeof(float(x))
    T = accum(S)
    a = abs(T(x))
    t = one(T) / (one(T) + T(0.3275911) * a)
    y = one(T) - (((((T(1.061405429) * t - T(1.453152027)) * t) + T(1.421413741)) * t -
                   T(0.284496736)) * t + T(0.254829592)) * t * exp(-a * a)
    return S(x < zero(x) ? -y : y)
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

"""
`pow(a, e)` with a scalar exponent.

The exponent is known **here**, on the host, so the kernel does not carry a loop
for it: the exponents models actually use get their expression directly and
`intpow` stays as the general fallback. Every one of Kokoro's 48 is `2`.

**This is not a speed fix, and the measurement that suggested it was one is worth
recording.** Per-op serialised timing put `pow.Tensor_Scalar` at 157 ms of
Kokoro's vocoder — above its 70 batch-norms — which reads as a runtime
square-and-multiply loop being expensive. Timed directly on the two shapes it
runs at, interleaved and both orders:

    (256, 3400)     intpow(x, 2)  0.084 ms    x*x  0.083 ms    1.0x
    (128, 20401)    intpow(x, 2)  0.120 ms    x*x  0.112 ms    1.1x

The loop folds. The 3.3 ms per call the attribution implied was the instrument —
a sync around every op measures itself (`lavadnn-perf-attribution`), and this is
what that looks like when it lands on a cheap op called many times. The direct
forms are kept because they are free and clearer, not because they are faster.

`^` is deliberately not used even for them — see `intpow` for why `x^2` on a
mostly-negative tensor came back NaN.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("pow.Tensor_Scalar")})
    a = lhs(ctx, op)
    e = rhs(ctx, op)
    if e isa Integer || (e isa Real && isinteger(e))
        n = Int(e)
        n == 1 && return emit(ctx, Base.broadcasted(identity, a))
        n == 2 && return emit(ctx, Base.broadcasted(x -> x * x, a))
        n == 3 && return emit(ctx, Base.broadcasted(x -> x * x * x, a))
        n == -1 && return emit(ctx, Base.broadcasted(inv, a))
        return intpow.(a, n)
    end
    a .^ e
end
runop!(ctx::Ctx, op::Op, ::Val{Symbol("ge.Tensor")}) = lhs(ctx, op) .>= rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("ge.Scalar")}) = lhs(ctx, op) .>= rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("eq.Scalar")}) = lhs(ctx, op) .== rhs(ctx, op)
# `le` is `ge`'s mirror and arrived with Whisper's DECODER: it builds the causal
# mask over the KV cache, `(1,1,1,449)` for a 448-slot cache. It was the only op
# of that graph's 162 we did not already have.
runop!(ctx::Ctx, op::Op, ::Val{Symbol("le.Tensor")}) = lhs(ctx, op) .<= rhs(ctx, op)
runop!(ctx::Ctx, op::Op, ::Val{Symbol("le.Scalar")}) = lhs(ctx, op) .<= rhs(ctx, op)
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
"""
`clamp` with either bound optional — and either bound possibly **symbolic**: the
iSTFT clamps an index against the frame count, so `arg2` becomes a function of
the sequence length once the graph is length-generic.
"""
runop!(ctx::Ctx, op::Op, ::Val{Symbol("clamp.default")}) =
    (lo = get(op.attrs, "arg1", nothing); hi = get(op.attrs, "arg2", nothing);
     clamp.(lhs(ctx, op), lo === nothing ? -Inf32 : Float32(numattr(ctx, lo)),
            hi === nothing ? Inf32 : Float32(numattr(ctx, hi))))
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
    # The VALUE can be symbolic too, not just the shape: a graph that materialises
    # its own sequence length writes `full((1,), t)`. `scalar` hands back the
    # string unchanged when it is not inf/nan/complex, and `convert(Float32, ...)`
    # then fails with `MethodError: no method matching Float32(::String)`.
    fill!(alloc(ctx, op.out, sz...),
          convert(dtypeof(ctx, op.out), numattr(ctx, op.attrs["arg1"])))
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("full_like.default")})
    a = lhs(ctx, op)
    T = dtypeof(ctx, op.out)
    fill!(alloc(ctx, T, size(a)...), convert(T, numattr(ctx, op.attrs["arg1"])))
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
    fill!(alloc(ctx, op.out), convert(dtypeof(ctx, op.out), numattr(ctx, op.attrs["arg0"])))

@inline arange_body(I, start, step) = start + (I[1] - 1) * step

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("arange.start_step")})
    start = intattr(ctx, something(get(op.attrs, "arg0", nothing), 0))
    stop = length(op.ins) >= 1 ? value(ctx, op.ins[1]) : intattr(ctx, op.attrs["arg1"])
    step = intattr(ctx, something(get(op.attrs, "arg2", nothing), 1))
    T = dtypeof(ctx, op.out)
    n = length(T(start):T(step):T(stop - step))
    launch!(ctx, arange_body, alloc(ctx, op.out, n), T(start), T(step))
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
    vals = launch!(ctx, maxdim_body, alloc(ctx, eltype(a), sz...), a, Val(d), Val(false))
    inds = launch!(ctx, maxdim_body, alloc(ctx, eltype(a), sz...), a, Val(d), Val(true))
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
    #
    # The two forms do not produce identical bits — so "does the mask still
    # match" is a question about the whole model, not about the kernel. They do
    # now agree on the ACCUMULATOR, which is not a tidiness point: summing in
    # `Float16` saturates at 65504, and the squares of 768 features at magnitude
    # 30 already exceed that. The instance norm below hit exactly this and went
    # silently to zero (see `_native_batch_norm_legit.no_stats`); this path
    # reduces over far fewer elements, so it is a narrower window, not a closed
    # one.
    # This was a switch (`LN_FUSED`) so the two could be compared end to end in
    # one session; the fused form won and the switch is gone (review finding 3).
    if a isa Lava.LavaArray && d == Tuple(1:length(d)) && length(a) % n == 0
        out = tupledest(ctx, 0, tupledtype(ctx, 0, eltype(a)), size(a)...)
        groups = length(a) ÷ n
        μ = tupledest(ctx, 1, Float32, groups)
        r = tupledest(ctx, 2, Float32, groups)
        layernorm!(ctx, out, μ, r, a, γ, β, n, eps)
        return (out, μ, r)
    end
    A = accum(eltype(a))
    μ = sum(a; dims=d, init=zero(A)) ./ n
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
        sum(abs2, a .- μ; dims=d, init=zero(A)) ./ n   # verification path, no workspace
    else
        t = scratch!(ctx.ws, ctx.backend, eltype(a), size(a)...)
        t .= Base.broadcasted(-, a, μ)
        sum(abs2, t; dims=d, init=zero(A)) ./ n
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
    launch!(ctx, repeatouter, out, a, sz)
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

# ── the iSTFT pair ───────────────────────────────────────────────────────────
#
# `_fft_r2c` / `_fft_c2r` are Kokoro's inverse STFT, one of each per utterance,
# at length 20 batched over ~11.7k windows. They map onto the FFT ported from
# VkFFT (`Lava.rfft`, `Lava.fft`) — 20 is 4x5, which the mixed-radix plan covers.
#
# torch's `normalization` enum on both: 0 none, 1 by sqrt(n), 2 by n. It is read
# rather than assumed, because getting it wrong scales the audio by 20 and still
# sounds like speech.

"""Torch's FFT normalization enum applied to `x` for a transform of length `n`."""
fftnorm(x, mode::Integer, n::Integer) =
    mode == 0 ? x : mode == 1 ? x ./ sqrt(Float32(n)) : x ./ Float32(n)

"""
`_fft_r2c(self, dim, normalization, onesided)` — the forward real FFT.

Only a single transform axis, and it must be the contiguous one: `Lava.rfft`
transforms along dimension 1 and batches over the rest, and a graph asking for
any other axis would need a transpose that nothing has yet required.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_fft_r2c.default")})
    a = lhs(ctx, op)
    dims = ints(op.attrs["arg1"])
    length(dims) == 1 || error("_fft_r2c: one transform axis only (op $(op.id))")
    d = jdim(dims[1], ndims(a))
    d == 1 || error("_fft_r2c: transform axis must be contiguous, got Julia dim $d")
    Bool(get(op.attrs, "arg3", true)) ||
        error("_fft_r2c: two-sided output is not implemented (op $(op.id))")
    fftnorm(Lava.rfft(a), Int(get(op.attrs, "arg2", 0)), size(a, 1))
end

"""
`_fft_c2r(self, dim, normalization, last_dim_size)` — the inverse real FFT.

Lava has no `irfft`, so it is built from the pieces it does have: extend the
half spectrum to the full Hermitian one, inverse complex FFT, take the real
part. `X[n - k + 2] = conj(X[k])` is the extension, and it is a gather with a
reversed index rather than a `reverse` — a reversed *view* of a device array is
not something every path here handles.

`arg3` is the output length, and it is not redundant: 11 bins come from either
20 or 21 samples, and only the caller knows which.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_fft_c2r.default")})
    a = lhs(ctx, op)
    dims = ints(op.attrs["arg1"])
    length(dims) == 1 || error("_fft_c2r: one transform axis only (op $(op.id))")
    d = jdim(dims[1], ndims(a))
    d == 1 || error("_fft_c2r: transform axis must be contiguous, got Julia dim $d")
    n = Int(op.attrs["arg3"])
    nb = size(a, 1)
    full = alloc(ctx, ComplexF32, n, Base.tail(size(a))...)
    copyto!(selectdim(full, 1, 1:nb), a)
    if n > nb
        # Bin j of the full spectrum mirrors bin n - j + 2 of the half one, so
        # for n = 20, nb = 11 the sources are 10, 9, ... 2 — a REVERSED
        # CONTIGUOUS RANGE, written as one. An index vector would say the same
        # thing and could not reach a kernel: a `SubArray` holding a host
        # `Vector{Int}` fails to compile with "passing non-bitstype argument",
        # whereas a `StepRange` is isbits and rides along.
        rest = ntuple(_ -> Colon(), ndims(a) - 1)
        copyto!(selectdim(full, 1, (nb + 1):n),
                conj.(view(a, (n - nb + 1):-1:2, rest...)))
    end
    # `fftany!`, not `fft!`: the iSTFT length is 20 = 4x5 and `fft!` is
    # power-of-two only. `rfft` reaches for the same dispatcher and says why —
    # Whisper's 400-point mel is 200 complex, DeepFilterNet3's 960 is 480.
    y = real.(Lava.fftany!(similar(full), full; inverse = true))
    fftnorm(y, Int(get(op.attrs, "arg2", 0)), n)
end

# ── Kokoro's six ─────────────────────────────────────────────────────────────
#
# `gather` for the text half; the other five for the iSTFTNet vocoder's
# harmonic-plus-noise source and its overlap-add.

"""
`gather` along one axis: `out[I] = a[I with I[d] = index[I]]`, torch's
`arg1` naming the axis.

Only the form Kokoro produces is supported — every axis other than `d` is a
singleton in both operands, which makes the whole thing `a[index]` — and anything
else errors rather than silently gathering the wrong axis. The general N-d gather
wants a kernel; this one is a lookup, and writing the kernel before a graph needs
it would be inventing a requirement.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("gather.default")})
    a = lhs(ctx, op)
    idx = value(ctx, op.ins[2])
    n = ndims(a)
    d = jdim(Int(op.attrs["arg1"]), n)
    all(k -> k == d || size(a, k) == 1, 1:n) && all(k -> k == d || size(idx, k) == 1, 1:ndims(idx)) ||
        error("gather: only singleton batch axes are supported (op $(op.id), " *
              "a $(size(a)), index $(size(idx)), dim $d)")
    flat = vec(Int.(collect(idx))) .+ 1        # torch indices are 0-based
    out = alloc(ctx, op.out, size(idx)...)
    copyto!(out, reshape(collect(vec(a))[flat], size(idx)))
    out
end

"""
`remainder` is **`mod`, not `rem`**: torch's result takes the sign of the
*divisor*, so `remainder(-0.25, 1) == 0.75`. Kokoro's sine generator relies on
exactly that — it wraps an accumulated phase into `[0, 1)`, and `rem` would hand
back negative phases for half the samples and a different waveform.
"""
runop!(ctx::Ctx, op::Op, ::Val{Symbol("remainder.Scalar")}) =
    emit(ctx, Base.broadcasted(mod, lhs(ctx, op), numattr(ctx, op.attrs["arg1"])))

"""
`angle` of a complex array — `atan(imag, real)`, elementwise.

Written as a two-argument `atan` rather than `angle` because the branch at the
negative real axis has to be the quadrant-correct one; `atan(y/x)` is not.
"""
runop!(ctx::Ctx, op::Op, ::Val{Symbol("angle.default")}) =
    emit(ctx, Base.broadcasted(z -> atan(imag(z), real(z)), lhs(ctx, op)))

"""
    hostnoise(ctx, op, f, dims)

`rand`/`randn` generated on the host and uploaded.

**Lava has no device RNG**, so this is the honest implementation rather than the
fast one: Kokoro's `SineGen` wants `randn_like` over `(1, 58800, 9)` — about
2 MB per call — plus a 9-element `rand` for the initial phase. A counter-based
(Philox-style) device generator is the follow-up; until then this is correct,
and correctness is what the vocoder's noise floor needs first.

The values come from `ctx.noise` rather than from `Random` directly, so a parity
run can substitute [`ZeroNoise`](@ref) and compare the deterministic path against
a reference that has had the same thing done to it.
"""
function hostnoise(ctx::Ctx, op::Op, f, dims)
    T = dtypeof(ctx, op.out)
    out = alloc(ctx, op.out, dims...)
    copyto!(out, draw(ctx.noise, f, T, dims))
    out
end

runop!(ctx::Ctx, op::Op, ::Val{Symbol("rand.default")}) =
    hostnoise(ctx, op, rand, Tuple(evalshape(shapeof(ctx, op.out), ctx.dims)))

runop!(ctx::Ctx, op::Op, ::Val{Symbol("randn_like.default")}) =
    hostnoise(ctx, op, randn, size(lhs(ctx, op)))

"""
`unfold(a, dim, size, step)` — the sliding window, and in torch a *view*.

`out[..., w, k] = a[..., (w-1) * step + k]`. Kokoro uses it three times for the
iSTFT's overlap-add, at `size = 20, step = 5` over 58820 samples, so the windows
overlap four ways and the result is 4x the input in elements.

Materialised here rather than viewed: the stride pattern is expressible, but a
`step < size` view aliases itself, and every consumer downstream would have to
be safe against that. The copy is one pass.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("unfold.default")})
    a = lhs(ctx, op)
    n = ndims(a)
    d = jdim(Int(op.attrs["arg1"]), n)
    sz = Int(op.attrs["arg2"])
    st = Int(op.attrs["arg3"])
    nw = (size(a, d) - sz) ÷ st + 1
    # torch appends the window axis LAST, which in the reversed layout is FIRST;
    # the windowed axis itself keeps its position and shrinks to the window count.
    out = alloc(ctx, op.out, sz, ntuple(k -> k == d ? nw : size(a, k), n)...)
    for k in 1:sz
        # offset k within every window: elements k, k+st, k+2st, ...
        copyto!(selectdim(out, 1, k), selectdim(a, d, k:st:(k + st * (nw - 1))))
    end
    out
end

"""
`aten::lstm` — a whole recurrent layer as ONE op.

Kept undecomposed on purpose (`kernels/extern/lstm.jl` explains the arithmetic
and `tools/export_kokoro.py:decomptable` does the export side): letting
`run_decompositions()` unroll it turns 1 node into 532 per 12 timesteps, and the
count grows with the sequence.

The reversed layout puts the sequence in the middle: torch `(N, T, D)` with
`batch_first` is Julia `(D, T, N)`, and the `(2, N, H)` initial state is
`(H, N, 2)`. Only the shapes Kokoro produces are accepted — one layer, batch 1,
`batch_first`, inference — and anything else errors rather than quietly running a
different recurrence. Multi-layer is a loop over this; batching is a wider GEMM
and a batch axis in the kernel; neither has a caller yet.

Returns only `(output,)`. `aten::lstm` also returns the final `h` and `c`, which
`getitem` would pick out — no graph here reads them, and returning a placeholder
that looks like state would be worse than not returning it.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("lstm.input")})
    x = lhs(ctx, op)                                  # (D, T, N)
    hx = [value(ctx, String(e)[2:end]) for e in op.attrs["arg1"]]
    ps = [value(ctx, String(e)[2:end]) for e in op.attrs["arg2"]]
    hasbias = Bool(something(get(op.attrs, "arg3", nothing), true))
    nlayers = Int(something(get(op.attrs, "arg4", nothing), 1))
    bidir = Bool(something(get(op.attrs, "arg7", nothing), false))
    batchfirst = Bool(something(get(op.attrs, "arg8", nothing), false))
    hasbias || error("lstm: has_biases = false is not implemented (op $(op.id))")
    nlayers == 1 || error("lstm: num_layers = $nlayers is not implemented (op $(op.id))")
    batchfirst || error("lstm: batch_first = false is not implemented (op $(op.id))")
    size(x, 3) == 1 || error("lstm: batch $(size(x, 3)) is not implemented (op $(op.id))")

    D, T = size(x, 1), size(x, 2)
    H = size(ps[2], 1)                                # w_hh is (H, 4H)
    ndir = bidir ? 2 : 1
    length(ps) == 4 * ndir || error("lstm: $(length(ps)) parameters for " *
                                    "$(ndir) direction(s) (op $(op.id))")
    out = alloc(ctx, op.out, ndir * H, T, 1)
    x2 = reshape(x, D, T)
    h0, c0 = hx[1], hx[2]
    for d in 0:(ndir - 1)
        w_ih, w_hh, b_ih, b_hh = ps[4d + 1], ps[4d + 2], ps[4d + 3], ps[4d + 4]
        lstm!(ctx, reshape(out, ndir * H, T), x2, w_ih, w_hh, b_ih, b_hh,
              view(h0, :, 1, d + 1), view(c0, :, 1, d + 1), T, H, d * H, d == 1)
    end
    (out,)
end

"""
    scatteradd_kernel!(dst, src, idx, n)

`dst[idx[j]] += src[j]`, atomically, for every `j`.

**The atomic is the whole point, not a precaution.** `idx` repeats — that is what
distinguishes a scatter-add from a scatter — and on Kokoro's iSTFT overlap-add
235220 entries land in 58820 slots. A plain `dst[idx[j]] += src[j]` from parallel
lanes loses every contribution but the last one per slot, which is exactly the
failure the host implementation was written to avoid: three of every four windows
vanished, the envelope had 1451 zeros, and dividing by it gave NaN audio.

fp32 only, because `VK_EXT_shader_atomic_float` gives a hardware `OpAtomicFAdd`
for it; `index_put` keeps its host path for every other dtype.

**This makes the op non-deterministic in the last ULP, and that is inherent.**
Atomics complete in whatever order the scheduler gives them and floating-point
addition is not associative, so two runs of the same input differ by ~1e-6
relative. PyTorch's `index_put_(accumulate=True)` on CUDA behaves identically —
it is one of the ops `torch.use_deterministic_algorithms` refuses — and
`KokoroRunner`'s suite asserts a tolerance here rather than equality.
"""
@kernel function scatteradd_kernel!(dst, @Const(src), @Const(idx), n::Int32)
    j = @index(Global, Linear)
    @inbounds if j <= n
        Atomix.@atomic dst[Int(idx[j])] += src[j]
    end
end

"""
`index_put` is the scatter half of `index.Tensor`, and it is how a KV cache is
written: `cache[:, :, position] = k` for a `position` that is a tensor, not a
literal, so `slice_scatter` cannot express it.

`arg1` is one entry per torch dimension exactly as in `index.Tensor` — `nothing`
for a whole axis, `"\$name"` for an index tensor — and the *last* input is the
values. `arg3` is `accumulate`: torch's `index_put_(..., accumulate=True)` is
`+=` and not `=`, a different op, so it is read rather than assumed.

Consecutive indices become a range. That is not cosmetic: a `UnitRange` view is
strided and assignable on the device, while a view built from an index *vector*
would need a scatter kernel. A cache write is always one slot, so the range path
is the one that runs; the vector path exists so a future multi-slot write fails
loudly at the assignment instead of silently taking a wrong branch.

**This copies the whole tensor to write one slot.** Whisper's decoder does that
eight times per token (~18 MB) and then `cat`s the results back into the stacked
cache (~18 MB again). It is the price of a functional graph, ~12% on a 2.07 ms
token; folding select/index_put/cat into an in-place write on the input buffer
needs the graph to express output-aliases-input, which it does not today.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("index_put.default")})
    a = lhs(ctx, op)
    src = value(ctx, op.ins[end])
    dst = dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, size(a)...)
    dst .= a
    n = ndims(a)
    idx = Vector{Any}(undef, n)
    fill!(idx, Colon())
    nz = Int[]
    for (k, e) in enumerate(op.attrs["arg1"])
        e === nothing && continue
        jd = n - k + 1
        1 <= jd <= n || error("index_put: dim $k out of range for $(n)-d input")
        push!(nz, jd)
        iv = vec(Int.(collect(value(ctx, String(e)[2:end])))) .+ 1  # 0-based in torch
        idx[jd] = iv == collect(first(iv):last(iv)) ? (first(iv):last(iv)) : iv
    end
    if Bool(something(get(op.attrs, "arg3", nothing), false))
        # ── accumulate = true: a SCATTER-ADD, and duplicates are the point.
        #
        # `view(dst, idx) .+= src` is NOT this. When `idx` repeats an index —
        # which it does here, 235220 entries landing in 58820 slots — a Julia
        # broadcast assignment keeps one contribution per slot and silently drops
        # the rest, while torch sums them. That is the iSTFT's overlap-add
        # envelope: three of every four windows vanished, the envelope had 1451
        # zeros, and the division by it produced NaN and Inf in the audio.
        #
        # **On the device, through `OpAtomicFAdd`.** This used to be a host loop,
        # on the grounds that every use was a constant subgraph folded at load —
        # "a runtime use would want a kernel with atomics". Kokoro's iSTFT is that
        # runtime use: two calls per utterance, and an OPDOUBLE ablation put them
        # at **+204 ms of an 845 ms vocoder**, because the host path downloads the
        # whole tensor, loops on one core and uploads it again, synchronising the
        # queue twice around it.
        #
        # `VK_EXT_shader_atomic_float` gives a hardware `OpAtomicFAdd` for fp32,
        # which is exactly the primitive the duplicates need. Anything else keeps
        # the host path — correctness first, and an fp16 atomic add is not a
        # device feature we ask for.
        length(nz) == 1 || error("index_put: accumulate with $(length(nz)) index " *
                                 "tensors is not implemented (op $(op.id))")
        d = nz[1]
        all(k -> k == d || size(a, k) == 1, 1:n) ||
            error("index_put: accumulate needs singleton batch axes (op $(op.id))")
        ivals = idx[d]
        if eltype(dst) === Float32 && dst isa Lava.LavaArray
            iv32 = toback(ctx.backend, Int32.(collect(ivals)))
            m = length(src)
            scatteradd_kernel!(ctx.backend)(vec(dst), vec(src), iv32, Int32(m);
                                            ndrange = m)
            return dst
        end
        hv = vec(Array(dst))
        hs = vec(Array(src))
        @inbounds for (j, i) in enumerate(ivals)
            hv[i] += hs[j]
        end
        copyto!(dst, reshape(hv, size(dst)))
        return dst
    end
    v = view(dst, idx...)
    v .= reshape(src, size(v))
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
        if length(stride) == 1
            # 1-D is 2-D with a singleton second spatial axis. Kokoro's iSTFTNet
            # upsamples with `ConvTranspose1d`, and rather than a second kernel
            # the operands are lifted: `(L, Cin, N) -> (L, 1, Cin, N)` and
            # `(k, Cout, Cin) -> (k, 1, Cout, Cin)`, with stride/pad/dilation/
            # output-padding all 1 on the added axis. The 2-D path is the one
            # that has been measured and tested, so the lift is preferable to a
            # parallel implementation that could drift from it.
            ox1 = convtransposesize(size(x, 1), size(w, 1), stride[1], pad[1],
                                    dil[1], outpad[1])
            x2 = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
            w2 = reshape(w, size(w, 1), 1, size(w, 2), size(w, 3))
            # `size(w, 2)` is C_out PER GROUP — the transposed weight is
            # `(k, C_out/groups, C_in)`. Depthwise gives 1 there and 512 groups.
            out2 = alloc(ctx, eltype(x), ox1, 1, groups * size(w, 2), size(x, 3))
            convolutiontranspose!(ctx, out2, x2, w2, bias, [stride[1], 1],
                                  [pad[1], 0], [dil[1], 1], [outpad[1], 0], groups)
            out1 = reshape(out2, ox1, groups * size(w, 2), size(x, 3))
            act === :relu && (out1 .= max.(out1, zero(eltype(out1))))
            return out1
        end
        length(stride) == 2 ||
            error("transposed convolution is implemented for 1-D and 2-D only " *
                  "(op $(op.id), $(length(stride))-D)")
        ox = convtransposesize(size(x, 1), size(w, 1), stride[1], pad[1], dil[1], outpad[1])
        oy = convtransposesize(size(x, 2), size(w, 2), stride[2], pad[2], dil[2], outpad[2])
        out = alloc(ctx, eltype(x), ox, oy, groups * size(w, 3), size(x, 4))
        convolutiontranspose!(ctx, out, x, w, bias, stride, pad, dil, outpad, groups)
        act === :relu && (out .= max.(out, zero(eltype(out))))
        return out
    end
    ox = convsize(size(x, 1), size(w, 1), stride[1], pad[1], dil[1])
    if length(stride) == 1                       # aten::convolution covers 1-D too
        # **1-D is 2-D with a singleton second spatial axis, and lifting it is
        # what puts it on the tensor cores.** `convolution1d!` is a direct scalar
        # kernel — one thread per output element, no reuse, no cooperative
        # matrices — and for a vocoder that is the whole cost: Kokoro's 90 1-D
        # convolutions are 235.7 GFLOP at **0.52 TFLOP/s**, against 42.4 for an
        # fp16 GEMM on this card.
        #
        # The 2-D path already has everything needed. It materialises im2col and
        # pads BOTH axes — the reduction axis by `padtile(CRS)` and the pixel
        # count by `padtile(NPQ)`, whose rows the im2col kernel simply zero-fills
        # — so a sequence length that lands on no tiling is not a refusal there.
        # Measured on the dominant shape, padding M from 20401 to 20416 is worth
        # **13.5x** even paying for the copy.
        #
        # The transposed 1-D case has lifted this way since the Kokoro port; this
        # is the forward one, and the same argument applies: reuse the path that
        # is measured and tested rather than tune a second implementation.
        #
        # `conv_coopmat_plan` decides. When it declines — fp32 operands, a `Cout`
        # off the tile, an im2col too large to be worth materialising — the
        # direct kernel is still the right answer and still runs.
        # ONE allocation for both branches, in the layout the op declares; the
        # 2-D path takes a `reshape` of it. Allocating a separate 4-D buffer
        # would spend a slab slot on every convolution that then declines, which
        # is 29 of Kokoro's 90.
        Cin1, Cout1 = size(w, 2), size(w, 3)
        out = alloc(ctx, eltype(x), ox, Cout1, size(x, 3))
        x2 = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
        w2 = reshape(w, size(w, 1), 1, Cin1, Cout1)
        out2 = reshape(out, ox, 1, Cout1, size(x, 3))
        if groups == 1 && ctx.ws !== nothing &&
           conv_coopmat_plan(ctx.dev, out2, x2, w2) isa ConvCoopMatPlan
            convolution!(ctx, out2, x2, w2, bias, [stride[1], 1], [pad[1], 0],
                         [dil[1], 1], 1; act)
        else
            convolution1d!(ctx, out, x, w, bias, stride, pad, dil, groups)
            # The direct kernel has no epilogue to fold into.
            act === :relu && (out .= max.(out, zero(eltype(out))))
        end
    elseif length(stride) == 3                   # and 3-D, for the Wan VAE
        oy = convsize(size(x, 2), size(w, 2), stride[2], pad[2], dil[2])
        oz = convsize(size(x, 3), size(w, 3), stride[3], pad[3], dil[3])
        out = alloc(ctx, eltype(x), ox, oy, oz, size(w, 5), size(x, 5))
        convolution3d!(out, x, w, bias, stride, pad, dil, groups; act)
    else
        oy = convsize(size(x, 2), size(w, 2), stride[2], pad[2], dil[2])
        out = alloc(ctx, eltype(x), ox, oy, size(w, 4), size(x, 4))
        convolution!(ctx, out, x, w, bias, stride, pad, dil, groups; act)
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

"""
`InstanceNorm2d`, which is what an `nn.InstanceNorm2d` decomposes to: a batch
norm that computes its statistics from the input because there are none stored.

The `_no_training` overload above is the folded case — running mean and variance
come in as weights, so the whole thing collapses to one affine map. This one
cannot fold: the statistics are a function of the input, so they are reduced per
channel on every call. That is the difference between the two overloads and the
reason this needed its own method rather than a rename.

**Biased variance, deliberately.** Training-mode batch norm normalises with
`1/N`, and only the running-statistics update (which `no_stats` does not have)
uses the `1/(N-1)` correction. Using the unbiased estimator here is a silent
`N/(N-1)` error in the output — invisible at 128x128 where it is 6e-5, and not
invisible in a 4x4 feature map.

Returns `(output, save_mean, save_invstd)` as the ATen signature does. The
graph's inference path never reads the second and third, but they are cheap and
already computed, so they are returned rather than stubbed.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_native_batch_norm_legit.no_stats")})
    x = lhs(ctx, op)
    γ, β = value(ctx, op.ins[2]), value(ctx, op.ins[3])
    eps = Float32(op.attrs["arg5"])
    c = ndims(x) - 1                      # torch channel dim 1 -> Julia dim n-1
    rd = ntuple(i -> i < c ? i : i + 1, ndims(x) - 1)   # every dim but the channel
    rs = ntuple(i -> i == c ? length(γ) : 1, ndims(x))

    # **The reduction accumulates in fp32 even for an fp16 input, and it has to.**
    #
    # `sum` over a `Float16` array accumulates in `Float16`, whose largest finite
    # value is 65504. This norm reduces over the *whole sequence*: on Kokoro's
    # vocoder that is 18961 elements per channel at a magnitude around 27, so the
    # sum of squares reaches ~1.4e7 and saturates to `Inf`. Then `invstd` is
    # `1/sqrt(Inf) = 0` and **every output is silently zero** — no NaN, no error,
    # and only for inputs long enough to overflow, so a short utterance passes
    # and a long one comes out as silence or, once a later op divides by it, as
    # NaN several ops downstream.
    #
    # `accum` is the same widening the convolution and matmul paths use. It is
    # applied through `init` rather than `sum(a -> A(a), x)` because a closure
    # over a *type* is not a bitstype and cannot enter a GPU kernel — that form
    # fails to compile with "Argument 5 to your kernel function ... is not a
    # bitstype", which names the argument and not the type it closed over.
    #
    # Costs nothing: `d` stays a lazy broadcast so nothing extra is materialised,
    # and the output dtype is unchanged — `eps` and `γ` are fp32, so the result
    # was already promoted.
    A = accum(eltype(x))
    n = length(x) ÷ size(x, c)
    μ = sum(x; dims = rd, init = zero(A)) ./ n
    d = x .- μ                            # `A`, because `μ` is
    v = sum(abs2, d; dims = rd, init = zero(A)) ./ n   # biased; see above
    invstd = 1 ./ sqrt.(v .+ eps)

    s = invstd .* reshape(γ, rs)
    (d .* s .+ reshape(β, rs), vec(μ), vec(invstd))
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_adaptive_avg_pool2d.default")})
    x = lhs(ctx, op)
    oy = Int(value(ctx, op.ins[2]))       # torch (H, W) -> Julia (y, x)
    ox = Int(value(ctx, op.ins[3]))
    out = alloc(ctx, eltype(x), ox, oy, size(x, 3), size(x, 4))
    adaptive_avg_pool2d!(ctx, out, x)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("upsample_bilinear2d.vec")})
    x = lhs(ctx, op)
    target = evalshape(shapeof(ctx, op.out), ctx.dims)
    out = alloc(ctx, eltype(x), target...)
    # arg2 is align_corners; the graphs use both conventions
    upsample_bilinear2d!(ctx, out, x;
                         align_corners = Bool(something(get(op.attrs, "arg2", nothing), false)))
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("upsample_nearest2d.vec")})
    x = lhs(ctx, op)
    target = evalshape(shapeof(ctx, op.out), ctx.dims)
    out = alloc(ctx, eltype(x), target...)
    upsample_nearest2d!(ctx, out, x)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("cumsum.default")})
    a = lhs(ctx, op)
    out = alloc(ctx, op.out, size(a)...)
    cumsum_dim!(ctx, out, a, jdim(Int(op.attrs["arg1"]), ndims(a)))
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
    maxpool2d!(ctx, out, x, k, s, p)
    (out, similar(out, Int64, 0))
end

# addmm(bias, a, b) = bias + a*b ; bmm is batched over the leading torch dim.
# Reversed layout swaps the operands: torch (m,k)x(k,n) is Julia (n,k)x(k,m).
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("addmm.default")})
    bias = lhs(ctx, op)
    a, b = value(ctx, op.ins[2]), value(ctx, op.ins[3])
    out = alloc(ctx, op.out, size(b, 1), size(a, 2))
    # `act` is set by `foldgelu`, which deleted the activation op and aliased its
    # buffer onto this one. It is applied inside the GEMM's store, so the fused
    # form reads and writes the result once instead of three times.
    matmul!(ctx, out, b, a, bias; epi=actfn(Symbol(get(op.attrs, "act", "none"))))
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("mm.default")})
    a, b = lhs(ctx, op), value(ctx, op.ins[2])
    out = alloc(ctx, op.out, size(b, 1), size(a, 2))
    matmul!(ctx, out, b, a)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_scaled_dot_product_efficient_attention.default")})
    q, k, v = value(ctx, op.ins[1]), value(ctx, op.ins[2]), value(ctx, op.ins[3])
    bias = length(op.ins) >= 4 ? value(ctx, op.ins[4]) : nothing
    s = get(op.attrs, "scale", nothing)
    scale = s === nothing ? inv(sqrt(size(q, 1))) : Float64(s)
    out = sdpa(ctx, q, k, v, bias, scale;
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
    out = sdpa(ctx, q, k, v, nothing, scale;
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
    batchedmatmul!(ctx, out, b, a)
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
    emit(ctx, Base.broadcasted(*, lhs(ctx, op), numattr(ctx, op.attrs["arg1"])))

"""
`index.Tensor` is torch's advanced indexing: `arg1` is one entry per dimension,
`nothing` meaning "take the whole axis" and a name meaning "gather with this
index tensor". The Wan VAE uses it for attention's row/column gather, where the
leading axes are `nothing` and the trailing two carry broadcast index tensors.

Torch indexes the un-reversed shape, so entry `k` of `arg1` addresses Julia
dimension `ndims - k + 1`; the index values are 0-based and become 1-based here.

**Several index tensors are PAIRED, not a Cartesian product.** This is the part
that is easy to get silently wrong, because Julia spells the other thing the same
way: `x[i, j]` with two vectors selects `length(i) * length(j)` elements, while
torch broadcasts `i` against `j` and selects `length(i)` of them, one per
position. So `mask[idx0, idx1]` with `idx0` of shape `(1,1,1,1)` and `idx1` of
`(1,1,1,30)` is 30 elements, and `view(x, idx0, idx1)` is 30 *rows* — a wrong
answer, or, as Kokoro's ALBERT found, a `BoundsError` from a 30-element index
landing on a size-1 axis.

One index tensor is the same function either way, which is why the single-index
form was right for a year and nothing noticed.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("index.Tensor")})
    x = lhs(ctx, op)
    spec = op.attrs["arg1"]
    n = ndims(x)
    dims, devs = Int[], Any[]
    for (k, e) in enumerate(spec)
        e === nothing && continue
        jd = n - k + 1
        1 <= jd <= n || error("index.Tensor: dim $k out of range for $(n)-d input")
        push!(dims, jd)
        push!(devs, value(ctx, String(e)[2:end]))    # attrs store "\$name"
    end
    isempty(dims) && return materialize(ctx, x)
    # Ascending Julia dimension. `spec` is in torch order, so reversing each
    # entry's axis walks the Julia dims *backwards* and `dims` comes out
    # descending — which `indexpaired` does not care about (it sums) and
    # `indexseparable` very much does: it pairs the j-th indexed dimension with
    # the j-th broadcast axis, and unsorted that pairing is transposed.
    p = sortperm(dims)
    dims, devs = dims[p], devs[p]
    if length(dims) == 1 || indexseparable(dims, devs)
        idx = Vector{Any}(undef, n)
        fill!(idx, Colon())
        # The index stays ON THE DEVICE. `Int.(collect(iv)) .+ 1` brings it home
        # and the resulting `SubArray{...,Tuple{Vector{Int64},...}}` cannot reach
        # a kernel at all — it fails to compile with "passing non-bitstype
        # argument", naming the broadcast machinery rather than the index. The
        # `.+ 1` (torch is 0-based) is a broadcast, so it runs on the device and
        # gives back a device vector.
        for (i, d) in enumerate(dims)
            idx[d] = vec(devs[i]) .+ 1
        end
        return materialize(ctx, view(x, idx...))
    end
    indexpaired(ctx, x, dims, [Int.(collect(d)) .+ 1 for d in devs])
end

"""
    indexseparable(dims, arrs) -> Bool

Whether these index tensors form an **outer product** over the axes they index —
the one multi-index case Julia's `view` already computes, and therefore the one
that can stay on the device.

Torch pairs index tensors by broadcasting them against each other, and Julia's
`x[i, j]` crosses them. Those are different functions *in general*, and the same
function when each index tensor varies along its own broadcast axis and is
size-1 in the others: then the broadcast enumerates every combination, which is
what the cross does.

SAM 2's position-embedding interpolation is exactly that and is why this exists.
It indexes a `(7, 7, 144, 1)` table with a `(256,)` and a `(1, 256)` — one row
selector and one column selector — for a `(256, 256, 144, 1)` result, sixteen
times. Requiring every axis to be indexed sent it to [`indexpaired`](@ref), which
refused; before the pairing fix it took `view` and was *correct by accident*,
because for this shape the two agree.

Three conditions, all necessary:

  * the broadcast has exactly one axis per indexed dimension — otherwise the
    result has a rank the sliced axes cannot be appended to. Kokoro's ALBERT
    mask broadcasts a `(1,1,1,1)` against a `(t,1,1,1)` over a 2-d input: four
    broadcast axes, two indexed dims, and the answer is genuinely paired;
  * index `i` is the only one that varies along broadcast axis `i`;
  * the indexed dims are **contiguous**. Torch leaves the gathered axes where
    they were only when the advanced indices are adjacent, and moves them to the
    front otherwise; `view` always leaves them in place, so the two agree only in
    the adjacent case.

`size(a, k)` past `ndims(a)` is 1, so a lower-rank index array needs no padding.
"""
function indexseparable(dims::Vector{Int}, arrs)
    dims == collect(dims[1]:dims[end]) || return false
    nb = maximum(ndims, arrs)
    nb == length(dims) || return false
    bs = ntuple(k -> maximum(a -> size(a, k), arrs), nb)
    for (i, a) in enumerate(arrs), k in 1:nb
        size(a, k) == (k == i ? bs[k] : 1) || return false
    end
    true
end

"""
    indexpaired(ctx, x, dims, arrs) -> array

Torch's advanced indexing with more than one index tensor: broadcast the index
arrays against each other, then take one element of `x` per position.

Every axis of `x` must be indexed. The mixed case — some axes indexed, some
sliced — is where torch also has to decide *where* to put the gathered axis; the
separable half of it is handled before this is reached (see
[`indexseparable`](@ref)), and what is left errors rather than picking one of the
two conventions and being right half the time.

The claim this docstring used to make — that no graph here produces the mixed
case — was false, and cost SAM 2: its encoder has sixteen of them, and requiring
every axis to be indexed made `Model` throw while folding them as constants.

The gather runs on the **host**. Every use of this form so far is attention-mask
or position metadata — tens to thousands of elements, produced by
`arange`/`unsqueeze` — so a round trip is cheaper than the alternative, and the
alternative is not free: a `view` with a host `Vector{Int}` index cannot reach a
kernel at all (`passing non-bitstype argument ... Vector{Int64}`), so it would
need the index uploaded and a gather kernel written for it.

The single-index path above stays on the device, which is where the large
gathers are — the Wan VAE's attention row/column selects.
"""
function indexpaired(ctx::Ctx, x, dims::Vector{Int}, arrs::Vector{<:Array})
    n = ndims(x)
    length(dims) == n ||
        error("index.Tensor: $(length(dims)) index tensors for a $(n)-d input; " *
              "mixing indexed and sliced axes is not implemented")
    # The broadcast determines the result shape — no need to compute it first,
    # and `Broadcast.broadcast_shapes` is not in every Julia this runs on.
    strides = cumprod([1; collect(size(x))[1:end - 1]])
    lin = reduce((u, v) -> u .+ v,
                 ((a .- 1) .* strides[d] for (d, a) in zip(dims, arrs))) .+ 1
    flat = collect(vec(x))[vec(lin)]
    out = alloc(ctx, eltype(x), size(lin)...)
    copyto!(out, reshape(flat, size(lin)))
    out
end

"""
    actfn(name) -> function

The activation a fused epilogue applies, by name.

**These have to be the same expressions the standalone ops use**, character for
character, or folding one into a GEMM changes the model's output. `geluexact`
below is `runop!(::Val{Symbol("gelu.default")})`'s branch with the operand type
made implicit.

Both evaluate in `accum(T)` and round once, which is what PyTorch does for a
half tensor (`opmath_type<scalar_t>` is `float`). The half evaluation this
replaces lost accuracy twice over — inside [`erf`](@ref), and again at the
`1 + erf` that follows it, where an argument below -2 makes the sum cancel to a
couple of significant bits. Measured on Whisper's `gelu_2` against PyTorch's own
fp16: rel rms 8.94e-4 narrow, 2.48e-6 wide.
"""
@inline function geluexact(v)
    T = accum(typeof(float(v)))
    x = T(v)
    oftype(v, T(0.5) * x * (one(T) + erf(x / sqrt(T(2)))))
end
"""
`gelu` with `approximate="tanh"`: `x/2 * (1 + tanh(sqrt(2/pi) (x + 0.044715 x³)))`.

Wide for the same two reasons `geluexact` is, plus a third of its own: `x³` in
half overflows above |x| = 40, which is inside the range Whisper's residual
stream reaches.
"""
@inline function gelutanh(v)
    T = accum(typeof(float(v)))
    x = T(v)
    c = T(0.7978845608028654)             # sqrt(2/pi)
    oftype(v, T(0.5) * x * (one(T) + tanh(c * (x + T(0.044715) * x^3))))
end
@inline relu_epi(v) = max(v, zero(v))
@inline actfn(name::Symbol) = name === :gelu ? geluexact :
                              name === :relu ? relu_epi : identity

"""
`gelu` with torch's default (exact) formulation. `arg1 = "tanh"` selects the
approximation, which differs by ~1e-3 and is a different function, not a faster
one — so it is dispatched, not assumed.

The two branches *call* [`geluexact`](@ref) / [`gelutanh`](@ref) rather than
restating them. When they were written out here as well, the epilogue and the
standalone op were two copies of one expression that had to agree character for
character for a fold to be a no-op, and the comment saying so was the only thing
keeping them in step.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("gelu.default")})
    x = lhs(ctx, op)
    f = String(get(op.attrs, "arg1", "none")) == "tanh" ? gelutanh : geluexact
    emit(ctx, Base.broadcasted(f, x))
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
    emit(ctx, Base.broadcasted(>, lhs(ctx, op), numattr(ctx, op.attrs["arg1"])))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("lt.Scalar")}) =
    emit(ctx, Base.broadcasted(<, lhs(ctx, op), numattr(ctx, op.attrs["arg1"])))
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
