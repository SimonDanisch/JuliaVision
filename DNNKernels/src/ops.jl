"""
ATen op methods.

Each is one `runop!` method keyed on the op name. Elementwise work is
broadcasting - `Mantle.LavaArray <: AbstractGPUArray`, so that is already a fused
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
# The positional attribute names, interned. `"arg$(pos-1)"` built a fresh
# `String` on EVERY operand access — 463 KB per Kokoro utterance, measured, and
# the `count` below built one more per earlier position. The set is tiny, fixed,
# and known at load.
const ARGKEY = ntuple(i -> "arg$(i-1)", 12)
@inline argkey(pos::Int) = @inbounds pos <= length(ARGKEY) ? ARGKEY[pos] : "arg$(pos-1)"

function operand(ctx::Ctx, op::Op, pos::Int)
    key = argkey(pos)
    haskey(op.attrs, key) && return numattr(ctx, op.attrs[key])
    # the pos'th operand is a tensor: count how many earlier positions were scalars
    idx = pos - count(p -> haskey(op.attrs, argkey(p)), 1:(pos - 1))
    idx <= length(op.ins) ||
        error("$(op.aten) ($(op.id)) has no operand at position $pos")
    value(ctx, op.ins[idx])
end

lhs(ctx::Ctx, op::Op) = operand(ctx, op, 1)
rhs(ctx::Ctx, op::Op, i::Int=2) = operand(ctx, op, i)
alpha(op::Op) = get(op.attrs, "alpha", 1)

"""
Float -> integer the way torch does it: saturate on infinities, NaN to zero.

`unsafe_trunc` and not `trunc` for the last step, and the three guards above are
what make it safe: NaN, over and under are all handled, so what reaches it is in
range by construction. `trunc(T, v)` carries an `InexactError` branch for the
cases that cannot happen here, and a throw inside a GPU kernel is not free — the
exception has to be ALLOCATED, which on a device means a hostcall. AMDGPU.jl
reports "Global hostcalls detected" and leaves a host thread servicing `malloc`
for the rest of the session; Lava emits the path as dead code and its
`replace_unreachable!` pass warns about lowering it. Neither is wanted for a
branch the guards have already ruled out.
"""
@inline function safetrunc(::Type{T}, v) where {T<:Integer}
    isnan(v) && return zero(T)
    v >= typemax(T) && return typemax(T)
    v <= typemin(T) && return typemin(T)
    return unsafe_trunc(T, v)
end

"""
    fastfloor(v) -> Int

`floor(Int, v)` for a value whose range the caller has already bounded.

Same reason as [`safetrunc`](@ref)'s last line: `floor(Int, v)` is a `trunc` with
an `InexactError` branch, and that branch allocates its exception inside the
kernel. A caller that clamps its own index — which is what a resampling kernel
does, because the edge case is a pixel and not an error — pays for a throw it
has made unreachable.
"""
@inline fastfloor(v::AbstractFloat) = unsafe_trunc(Int, floor(v))

"""
[`safetrunc`](@ref) with its target type as a TYPE PARAMETER rather than a
broadcast argument.

`safetrunc.(eltype(d), a)` puts a `Type` in the broadcast, and a `Type` is not a
value every backend can carry: `Base.broadcastable` wraps it in a `Ref`, and on
AMDGPU that arrives in the kernel as a `ROCRefType{Int64}` — the `Type{}` gone —
so `safetrunc` is called with an `Int64` where it wants a type, finds no method,
and dispatches DYNAMICALLY inside the kernel. That allocates, which on a GPU is
a hostcall: AMDGPU.jl reports "Global hostcalls detected" and keeps a host
thread servicing `malloc` for the rest of the session.

A zero-size singleton carries the type with nothing to marshal, so the kernel is
fully typed on every backend. Lava keeps the `Type{}` and gets the static call,
so a closure over the type works there and not elsewhere.
"""
struct SafeTrunc{T} end
@inline (::SafeTrunc{T})(v) where {T<:Integer} = safetrunc(T, v)

"""
`convert` to a fixed type, as a singleton callable.

`v -> T(v)` would be a closure with a `Type{Float32}` field, which is not isbits,
and a kernel cannot take a non-bitstype argument -- the same trap `where.self`
avoids by capturing `zero(T)` instead of `T`. As a parametric singleton it has no
field at all, so it costs no slot (`Mantle.NotPassed`).
"""
struct ToType{T} end
@inline (::ToType{T})(v) where {T} = convert(T, v)

"""
torch's cast to `Bool`, which is `v != 0` and NOT Julia's `convert`.

`convert(Bool, 2)` throws `InexactError` and `trunc(Bool, 0.5)` is `false`;
`.to(torch.bool)` answers `true` to both, because it is defined as a comparison
against zero for every source type. So a `Bool` destination is neither of the
other two rules and has its own.

`convert` inside a KERNEL is worse than merely wrong: the throw path makes Lava
emit `gpu_gc_pool_alloc` and warn that lowering its `unreachable` leaves an
undef POINTER for the caller to dereference. That
warning is what found this, on kokorotext's BERT half, which casts an `Int64`
mask to `Bool`. Its mask is all ones, so the path was never taken and no
number was ever wrong, which is the whole reason it needed finding.
"""
struct ToBool end
@inline (::ToBool)(v) = !iszero(v)

"""
    castfn(::Type{T}, ::Type{S}) -> callable

What `_to_copy` applies for an `S` source and a `T` destination, as a singleton
a kernel can take.

Three rules, in one place: this file's `runop!` and `emit.jl`'s emit both need
them, and two copies of a rule drift. Every one is torch's definition and not
Julia's:

  * a `Bool` destination is a comparison against zero;
  * a float truncated to an integer saturates, rather than throwing;
  * everything else is `convert`, which for a float narrowing rounds once.

Written as four methods rather than a chain of `if`s so a new rule is a method.
The `Bool` case is spelled for both source families because that is what keeps
it strictly more specific than the float-to-integer one instead of ambiguous
with it.
"""
castfn(::Type{Bool}, ::Type{S}) where {S<:AbstractFloat} = ToBool()
castfn(::Type{Bool}, ::Type{S}) where {S<:Integer} = ToBool()
castfn(::Type{T}, ::Type{S}) where {T<:Integer,S<:AbstractFloat} = SafeTrunc{T}()
castfn(::Type{T}, ::Type{S}) where {T,S} = ToType{T}()

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
function numattr(dims::NamedTuple, v)
    s = scalar(v)
    s isa AbstractString ? evalexpr(String(s), dims) : s
end
# Both contexts carry `dims`, and an attribute that is a shape expression means
# the same thing to either: the interpreted path asked with a `Ctx` and `emitop!`
# asks with an `EmitCtx`, so the shared half takes the `dims` alone.
numattr(ctx::Ctx, v) = numattr(ctx.dims, v)

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
intlist(ctx::Ctx, v) = intlist(ctx.dims, ctx.values, v)
intlist(dims::NamedTuple, scalars, v) = Int[intattr(dims, scalars, el) for el in v]

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

On `(dims, scalars)` and not on a context, because BOTH paths need the same
answer: the emit reads the same attributes and a second copy of this rule is a
second set of forms it might not cover. `scalars` is where a `"\$name"`
reference resolves — `ctx.values` for a run, `emitctx.res` for a declaration —
and a graph whose attributes are all literal never touches it.
"""
intattr(dims::NamedTuple, scalars, v::Bool) = Int(v)
intattr(dims::NamedTuple, scalars, v::Integer) = Int(v)
intattr(dims::NamedTuple, scalars, v::AbstractString) =
    startswith(v, "\$") ? Int(scalars[v[2:end]]) : evalexpr(String(v), dims)
intattr(dims::NamedTuple, scalars, v) = Int(v)

intattr(ctx::Ctx, v) = intattr(ctx.dims, ctx.values, v)

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

"""`rsqrt` as a named function, not `x -> inv(sqrt(x))`.

An anonymous function gets a fresh type per definition site, so the one in this
table and the one a fusion pass would build are different types naming the same
arithmetic — two kernels, two frozen entries, for one operation. Named once, they
are the same singleton everywhere."""
rsqrt_(x) = inv(sqrt(x))

"""
Every ATen op that is exactly one function of one operand, and which function.

One table, read twice: `runop!` generates a method per entry below, and
[`fusedfunc`](@ref) reads the same entry to put that function into a `FusedOp`.
They were separate lists for about an hour, which is how long it took to notice
that a fusion emitting `x -> inv(sqrt(x))` would not be the `rsqrt.default`
anybody had tested.
"""
const UNARY_FUSED = Dict{String,Any}(
    "relu.default" => relu_, "sigmoid.default" => sigmoid_,
    "tanh.default" => tanh, "log.default" => log,
    "exp.default" => exp, "sqrt.default" => sqrt,
    "rsqrt.default" => rsqrt_, "neg.default" => -,
    "abs.default" => abs, "reciprocal.default" => inv,
    "bitwise_not.default" => !, "logical_not.default" => !,
    "sin.default" => sin, "cos.default" => cos,
    "erf.default" => erf, "floor.default" => floor)

for (name, f) in UNARY_FUSED
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
    # `act` comes from `foldrelu`: the relu op it folded in is dropped and its
    # buffer aliases this one. Same broadcast, so it is free.
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

**Not a speed fix.** Per-op serialised timing puts `pow.Tensor_Scalar` at 157 ms
of Kokoro's vocoder — above its 70 batch-norms — which reads as a runtime
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
# `le` is `ge`'s mirror. Whisper's decoder builds its causal mask over the KV
# cache with it, `(1,1,1,449)` for a 448-slot cache.
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

Through `emit`, like every other elementwise op, and NOT `clamp.(...)`. The
dotted call allocates through `similar`, so it lands outside the static plan —
and because the bounds are `Float32`, it also *promotes*: `clamp.(::Float16,
::Float32, ::Float32)` has `eltype` `Float32`, so an fp16 op wrote an unplanned
fp32 buffer and needed a second pass to get back. Whisper has 32 of these on
`(1500, 1280)` fp16, and `opdouble` priced them at **4.22 ms of a 124.66 ms
encode** — against 0.29 ms for `mul.Tensor`, which moves exactly the same bytes
over exactly the same shape. Isolated, the two forms are 17.2x apart:

    clamp.(a, f32, f32)        1186.6 us    10 GB/s   <- was shipped
    d .= clamp.(a, f32, f32)      68.9 us   111 GB/s
    d .= a * 0.125f0              60.1 us   128 GB/s  <- the reference

This does NOT put the clamp back on the fusion path, and must not: `fuse.jl`
lists `clamp.default` among the three atens that were added to `FUSABLE` and
taken back out, because fusing them saved dispatches that were not the cost and
moved SAM 2's mask IoU from 1.00000 to 0.98750 — a fused chain keeps
intermediates in fp32 registers where the reference rounds to fp16 at every step.
`emit` materialises whenever the id is not in `ctx.lazy`, and clamp's never is,
so the store still rounds to the declared dtype exactly as the reference does.

An op that allocates its own output instead of taking the planned one is worth
grepping for: `_to_copy.default` carries the same note.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("clamp.default")})
    lo = get(op.attrs, "arg1", nothing)
    hi = get(op.attrs, "arg2", nothing)
    x = lhs(ctx, op)
    # Bounds in the operand's OWN type when that type is an integer. Float32
    # bounds promote the result to `Float32`, and storing that into an integer
    # destination is a `convert` with an `InexactError` path — which is a THROW
    # inside the kernel, and a throw allocates its exception. On a GPU that
    # allocation is a hostcall (AMDGPU.jl: "Global hostcalls detected", plus a
    # host thread servicing `malloc` afterwards); on the other backends it is
    # dead code the compiler still has to emit. torch's `clamp` on an integer
    # tensor stays integral too, so this is also the closer answer.
    #
    # Only when the bounds ARE integral: `clamp(::Int, 0.25, …)` is a legitimate
    # call the Wan VAE makes, and rounding the bound there would change it.
    T = eltype(x)
    l, h = clampbounds(T, ctx.dims, lo, hi)
    emit(ctx, Base.broadcasted(clamp, x, l, h))
end
"""The pair of bounds `clamp` should be broadcast with: see `clamp.default`."""
function clampbounds(::Type{T}, dims::NamedTuple, lo, hi) where {T}
    l = lo === nothing ? -Inf32 : Float32(numattr(dims, lo))
    h = hi === nothing ? Inf32 : Float32(numattr(dims, hi))
    T <: Integer || return (l, h)
    (isinteger(l) || isinf(l)) && (isinteger(h) || isinf(h)) || return (l, h)
    return (l == -Inf32 ? typemin(T) : T(l), h == Inf32 ? typemax(T) : T(h))
end

"""
    operand(ctx, id, a) -> dense array

One operand of a fused op, forced into the slot reserved for `id`.

`d .= a` does both jobs at once: it evaluates a **lazy** producer (an elementwise
op like the `mul.Tensor` that scales q stays unevaluated — `value` returns a
broadcast the consumer must force) and collapses a **permuted view**.

**The slot must be keyed by `id`.** Calling `materialize(ctx, ...)` once per
operand instead returns the same shared scratch every time, so q, k and v all
aliased one buffer and the last write won — the check inside `runop!` reported
`maxabs_q == maxabs_k == maxabs_v` to the last digit while `sdpa` itself was
computing them correctly to 5.4e-5. A fused op with N operands needs N distinct
destinations, and `dest(ctx, id, ...)` is where the planner already reserved one.
"""
@inline function operand(ctx::Ctx, id::AbstractString, a)
    d = dest(ctx, id, eltype(a), size(a)...)
    d .= a
    d
end

"""
`fused.sdpa` — the attention [`fuseattention`](@ref) collapses a
`bmm -> softmax -> [clone] -> bmm` into.

Its operands are `(q, k, v)` as `(D, S, H, B)`, which is what `sdpa` wants and
what the exporter's own 4-D buffers already are. The scale is usually **not**
applied here: the `mul` that scaled q is left in the graph, so the default is
`scale = 1`. A fusion that had to reach past that `mul` to find an fp16 operand
carries the product it folded as the op's `scale`, and then this reads it — the
declared form takes the same attribute, and the two paths disagreeing about it
is a `rel 1.8` difference in the attention output that nothing else reports.

The declared output keeps the second `bmm`'s shape, `(H, S, D)` in torch order
and `(D, S, H)` in Julia's — the same elements as `sdpa`'s `(D, S, H, B)` with
`B == 1`, so the result is reshaped rather than copied.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.maskedattention")})
    q,k,v,mask = map(id -> value(ctx,id),op.ins)
    scale = op.attrs["scale"]
    if get(op.attrs,"staged",true) && maskedprefill_applicable(ctx,q,k,v,mask)
        out = dest(ctx,eltype(q),size(q)...)
        maskedprefill!(ctx,out,q,k,v,mask,scale)
        return reshape(out,size(out,1),size(out,2),size(out,3))
    end
    plan = flashcm_plan(ctx.dev,q,k,v,nothing;clamp=true)
    epad = plan isa FlashCMPlan ? flashepad(ctx.dev,plan.EP) : 0
    rpad = plan isa FlashCMPlan ? flashrpad(ctx.dev,plan.BR) : 0
    if ctx.dev.coopmatsubgroup == 32 && size(q,1) == 128
        short = size(q,2) <= 16
        nk = size(k,2)
        # A long query wants `(8, 8)` padding, not `(16, 0)`, and a key extent
        # this kernel has to sweep in full wants a bigger tile with it. Measured
        # per shape at lq = 4096 (`tools/bench_masked_flash.jl`, recorded), ms:
        #
        #     lk     (16,32,8,16,0)   this pick
        #     512         3.17        2.86  (16,32,8,8,8)
        #     1024        6.21        5.43  (16,32,8,8,8)
        #     2048       12.46       10.54  (16,32,8,8,8)
        #     4096       47.51       38.01  (32,64,8,8,8)
        #
        # `(16, 0)` was tuned at lk = 512 and then applied to every long query,
        # which is how the 4096-slot chunks of a long prompt ended up paying 25%.
        big = !short && nk >= 4096
        tuned = flashcm_plan(ctx.dev,q,k,v,nothing;clamp=true,
                            BR=big ? 32 : 16,BC=big ? 64 : 32,
                            NW=short && nk<512 ? 4 : 8,
                            rego=short && nk<512)
        tepad, trpad = short ? (8,0) : (8,8)
        if tuned isa FlashCMPlan &&
           flashcmshared(tuned.EP,tuned.BR,tuned.BC,tepad,trpad) <= ctx.dev.sharedbudget
            plan = tuned
            epad, rpad = tepad, trpad
            if short && nk >= 512
                p = plan
                plan = FlashCMPlan(p.BR,p.BC,p.NW,p.NT,p.E,p.EP,p.clamp,
                    p.rego,p.held,p.rescale,p.onepass,p.lazyrescale,8,false)
            end
        end
    end
    if plan isa FlashCMPlan
        out = dest(ctx,eltype(q),size(q)...)
        sdpaflashcm!(ctx,out,plan,q,k,v,scale;mask,epad,rpad)
    else
        out = sdpa(ctx,q,k,v,mask,scale)
    end
    reshape(out,size(out,1),size(out,2),size(out,3))
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.sdpa")})
    q, k, v = value(ctx, op.ins[1]), value(ctx, op.ins[2]), value(ctx, op.ins[3])
    # `sdpa` reads these as dense operands; q/k/v here are usually views over the
    # QKV projection, and `contiguous` is what every other op does with those.
    # Each operand into ITS OWN slot — see `operand`.
    qc = operand(ctx, op.ins[1], q)
    kc = operand(ctx, op.ins[2], k)
    vc = operand(ctx, op.ins[3], v)
    # Return `sdpa`'s own result and let `execute!`'s `coerce` put it in the
    # declared dtype — `sdpa` accumulates in `accum(eltype(q))` = Float32 while
    # this buffer is Float16.
    #
    # Do NOT `alloc(ctx, op.out, ...)` and copy into it. `execute!` sets
    # `ctx.outid[] = op.out` before the call, so `sdpa`'s internal `dest` hands
    # back THE SAME planned slot that `alloc` does — the copy was a buffer
    # assigned to itself through a differently-typed view, and the model came out
    # a constant. Two different guesses at the destination both produced the
    # bit-identical wrong answer, which is what said the fault was the slot and
    # not the dtype.
    scale = Float64(get(op.attrs, "scale", 1.0))
    o = sdpa(ctx, qc, kc, vc, nothing, scale)
    FUSEATTENTIONCHECK[] && checksdpa(qc, kc, vc, o, scale)
    reshape(o, size(o, 1), size(o, 2), size(o, 3))
end

"""
    FUSEATTENTIONCHECK[] = true

Recompute one head of the attention on the CPU **from inside `runop!`** and
compare it with what `sdpa` returned.

It has to happen here. Reading the operands after `execute!` returns tells you
nothing: transients are reused, so `mul` and `select` no longer hold what this
op saw, and a ratio computed from them reports a missing scale that is not
missing. Inside the op is the only place the operands are certainly the right
bytes.
"""
const FUSEATTENTIONCHECK = Ref(false)

function checksdpa(q, k, v, o, scale = 1.0)
    Q = Float32.(Array(q)[:, :, 1, 1]); K = Float32.(Array(k)[:, :, 1, 1])
    V = Float32.(Array(v)[:, :, 1, 1])
    S = (K' * Q) .* Float32(scale)
    S .-= maximum(S; dims = 1)
    P = exp.(S); P ./= sum(P; dims = 1)
    R = V * P
    G = Float32.(Array(o)[:, :, 1, 1])
    @info "fused.sdpa check" maxabs_q=maximum(abs.(Q)) maxabs_k=maximum(abs.(K)) maxabs_v=maximum(abs.(V)) rel=maximum(abs.(G .- R)) / max(1f-6, maximum(abs.(R)))
    nothing
end

# The identity, as an op. `foldcacheupdate` turns a `cat` that reassembled a
# tensor its own inputs were already views of into this, and `aten::alias` means
# exactly that too — the output shares storage with the input.
# Rotary position embedding, as one op, written by `fuserope`.
#
# `rotate_half` is a slice, a negate and a concatenate, and the embedding is then
# two multiplies and an add: six dispatches per projection, twice a layer, for an
# elementwise function of one tensor. The concatenated half is never needed as a
# tensor — it is `-x[j + H/2]` for the low half and `x[j - H/2]` for the high one,
# which is an index, not a buffer.
@kernel cpu=false function rope_kernel!(out, @Const(x), @Const(cs), @Const(sn),
                                        H::Int32, half::Int32, HT::Int32, n::Int64)
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            l = Int32(i) - Int32(1)
            j = l % H                       # 0-based position within the head dim
            # `cos`/`sin` are per TOKEN as well as per component: they are
            # `(H, T)` in Julia order against `x`'s `(H, T, heads, batch)`, so the
            # index is the position within the first two axes, not within one
            # head. Indexing by `j` alone is right at decode, where `T == 1`, and
            # silently wrong for every prompt longer than one token.
            c = l % HT
            o = j < half ? (l + half) : (l - half)
            r = Float32(x[o + Int32(1)])
            j < half && (r = -r)
            out[i] = eltype(out)(muladd(Float32(x[i]), Float32(cs[c + Int32(1)]),
                                        r * Float32(sn[c + Int32(1)])))
        end
    end
end

# The INTERLEAVED rotary, as one op — see `fusepairrope`.
#
# One thread per PAIR, so both of a pair's elements are read and written by the
# same thread and the store is a contiguous 2-element write. `P` and `H` are
# `Val`s because the only arithmetic here besides the rotation is two integer
# divisions, and by a runtime divisor those cost more than the rotation does.
@kernel cpu=false function pairrope_kernel!(out, @Const(x), @Const(cs), @Const(sn),
                                            ::Val{P}, ::Val{H}, n::Int32) where {P,H}
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            i0 = Int32(i) - Int32(1)
            j = i0 % Int32(P)                 # position within the half head
            # `cos`/`sin` are per TOKEN as well as per component and do not
            # carry the head axis, so the head is divided out rather than
            # masked off.
            t = i0 ÷ Int32(P * H)
            c = j + Int32(P) * t
            o = Int32(2) * i0
            a = Float32(x[o + Int32(1)])
            b = Float32(x[o + Int32(2)])
            cv = Float32(cs[c + Int32(1)])
            sv = Float32(sn[c + Int32(1)])
            out[o + Int32(1)] = eltype(out)(a * cv - b * sv)
            out[o + Int32(2)] = eltype(out)(a * sv + b * cv)
        end
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.pairrope")})
    x = lhs(ctx, op)
    cv = value(ctx, op.ins[2])
    sv = value(ctx, op.ins[3])
    P = Int(op.attrs["P"])
    ob = ctx.graph.buffers[ctx.outid[]]
    out = dest(ctx, ob.dtype, evalshape(ob.shape, ctx.dims)...)
    xd = x isa GPUArrays.AbstractGPUArray ? x : materialize(ctx.rec, ctx.backend, x)
    npair = length(xd) ÷ 2
    H = size(xd, 2)
    pairrope_kernel!(ctx.backend, 256)(out, xd, vec(cv), vec(sv), Val(P), Val(H),
                                       Int32(npair); ndrange = npair)
    out
end

# SwiGLU as one op.
#
# The export spells `silu(gate) * up` as a widening cast, a `sigmoid`, a
# multiply, a narrowing cast and a second multiply — five dispatches over a
# 26624-wide tensor, per layer, and two of them exist only to move it between
# fp16 and fp32. `sigmoid` is evaluated in fp32 here exactly as the traced graph
# does, so this changes the dispatch count and not the arithmetic.
@kernel cpu=false function swiglu_kernel!(out, @Const(gate), @Const(up), n::Int64)
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            x = Float32(gate[i])
            out[i] = eltype(out)(Float32(eltype(out)(x / (1f0 + exp(-x)))) * Float32(up[i]))
        end
    end
end

@kernel cpu=false function swiglu_strided_kernel!(out, @Const(gate), @Const(up), gb, ub, gs, us)
    i = @index(Global, Cartesian)
    @inbounds begin
        gi = gb; ui = ub
        for d in 1:length(gs)
            gi += Int32(i[d]-1)*gs[d]
            ui += Int32(i[d]-1)*us[d]
        end
        x = Float32(gate[gi])
        out[i] = eltype(out)(Float32(eltype(out)(x / (1f0 + exp(-x)))) * Float32(up[ui]))
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.swiglu")})
    g = lhs(ctx, op); u = value(ctx, op.ins[2])
    ob = ctx.graph.buffers[ctx.outid[]]
    out = dest(ctx, ob.dtype, evalshape(ob.shape, ctx.dims)...)
    gr, ur = stridedroot(g), stridedroot(u)
    if size(g) == size(u) == size(out) &&
       gr !== nothing && ur !== nothing &&
       max(length(gr[1]),length(ur[1])) <= typemax(Int32)
        swiglu_strided_kernel!(ctx.backend, 256)(out,
            reshape(gr[1],length(gr[1])),reshape(ur[1],length(ur[1])),
            Int32(gr[2]+1),Int32(ur[2]+1),Int32.(strides(g)),Int32.(strides(u));
            ndrange=size(out))
        return out
    end
    # `AbstractGPUArray`, the question being asked: is this already a dense device
    # array, or does `materialize` have to make one? It named `Mantle.LavaArray`
    # until 2026-09-14, so on every other backend a dense operand still paid for a
    # full copy of itself. Nine sites across the four `fused.*` ops below have the
    # same line for the same reason.
    gd = g isa GPUArrays.AbstractGPUArray ? g : materialize(ctx.rec, ctx.backend, g)
    ud = u isa GPUArrays.AbstractGPUArray ? u : materialize(ctx.rec, ctx.backend, u)
    swiglu_kernel!(ctx.backend, 256)(out, gd, ud, Int64(length(gd)); ndrange = length(gd))
    out
end

# RoPE that stores straight into the KV cache.
#
# The rotated key is written to a temporary and then copied into the cache slot
# by an `index_put` whose only job is that copy: 128 extra dispatches and 3.1 ms
# of a 166 ms decode step, for a store the rope kernel could have done itself.
# Cartesian for the same reason `indexput_kernel!` is — the destination is a
# strided view into the cache, and indexing one linearly costs more than the
# launch shape saves.
@kernel cpu=false function rope_store_kernel!(dst, @Const(x), @Const(cs), @Const(sn),
                                              @Const(iv), H::Int32, half::Int32,
                                              HT::Int32, ::Val{N}, ::Val{SZ},
                                              n::Int64) where {N,SZ}
    lin = @index(Global, Linear)
    if lin <= n
    @inbounds begin
        I = CartesianIndices(SZ)[lin]
        # `x` is `(H, T, heads, batch)`; the cache slice is `(H, maxlen, heads, batch)`
        # and `iv[t]` is where token `t` belongs in it.
        j = Int32(I[1]) - Int32(1)
        l = j + H * (Int32(I[2]) - Int32(1))         # position within (H, T)
        o = j < half ? (l + half) : (l - half)
        xo = CartesianIndex(ntuple(k -> k == 1 ? (Int(o % H) + 1) :
                                        (k == 2 ? (Int(o ÷ H) + 1) : I[k]), Val(N)))
        r = Float32(x[xo])
        j < half && (r = -r)
        v = muladd(Float32(x[I]), Float32(cs[l + Int32(1)]), r * Float32(sn[l + Int32(1)]))
        t = Int(iv[I[2]]) + 1
        dst[CartesianIndex(ntuple(k -> k == 2 ? t : I[k], Val(N)))] = eltype(dst)(v)
    end
    end
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.ropecache")})
    x = lhs(ctx, op)
    cs = value(ctx, op.ins[2]); sn = value(ctx, op.ins[3])
    dst = value(ctx, op.ins[4]); iv = value(ctx, op.ins[5])
    xd = x isa GPUArrays.AbstractGPUArray ? x : materialize(ctx.rec, ctx.backend, x)
    cv = cs isa GPUArrays.AbstractGPUArray ? cs : materialize(ctx.rec, ctx.backend, cs)
    sv = sn isa GPUArrays.AbstractGPUArray ? sn : materialize(ctx.rec, ctx.backend, sn)
    H = size(xd, 1)
    rope_store_kernel!(ctx.backend, 256)(dst, xd, vec(cv), vec(sv), vec(iv),
                                         Int32(H), Int32(H ÷ 2), Int32(H * size(xd, 2)),
                                         Val(ndims(xd)), Val(size(xd)), Int64(length(xd));
                                         ndrange = length(xd))
    dst
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.rope")})
    x = lhs(ctx, op)
    cs = value(ctx, op.ins[2]); sn = value(ctx, op.ins[3])
    # Through the RECYCLER, not `contiguous`. The rotated operand is a permuted
    # view, so it has to be laid out densely first — and doing that with a fresh
    # allocation is a pool allocation per layer per projection, which costs more
    # than the five dispatches this op replaces.
    xd = x isa GPUArrays.AbstractGPUArray ? x : materialize(ctx.rec, ctx.backend, x)
    cv = cs isa GPUArrays.AbstractGPUArray ? cs : materialize(ctx.rec, ctx.backend, cs)
    sv = sn isa GPUArrays.AbstractGPUArray ? sn : materialize(ctx.rec, ctx.backend, sn)
    H = size(xd, 1)
    ob = ctx.graph.buffers[ctx.outid[]]
    out = dest(ctx, ob.dtype, evalshape(ob.shape, ctx.dims)...)
    HT = H * size(xd, 2)
    rope_kernel!(ctx.backend, 256)(out, xd, vec(cv), vec(sv),
                                   Int32(H), Int32(H ÷ 2), Int32(HT), Int64(length(xd));
                                   ndrange = length(xd))
    out
end

# The whole grouped RMS norm, written by `fusegroupedrms`.
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.groupedrms")})
    x = lhs(ctx, op)
    γ = value(ctx, op.ins[2])
    C = Int(op.attrs["C"]); NG = Int(op.attrs["ng"]); ε = Float32(op.attrs["eps"])
    ob = ctx.graph.buffers[ctx.outid[]]
    out = dest(ctx, ob.dtype, evalshape(ob.shape, ctx.dims)...)
    xd = x isa GPUArrays.AbstractGPUArray ? x : materialize(ctx.rec, ctx.backend, x)
    n = length(xd) ÷ C
    groupedrms_kernel!(ctx.backend, LN_WG)(
        out, xd, γ, Int32(C), Int32(NG), ε,
        Val(Bool(get(op.attrs, "midround", false))); ndrange = n * LN_WG)
    out
end

function runop!(ctx::Ctx, op::Op, ::Val{Symbol("alias.default")})
    value(ctx, op.ins[1])
end

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
    # `castfn` is the rule, and it is the same object the emit dispatches, so
    # the two routes cannot cast differently. Each of its three cases is torch's
    # definition rather than Julia's -- see its docstring.
    d .= castfn(eltype(d), eltype(a)).(a)
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

"""
`aten::arange.start_step(start, end, step)`, whose three scalars may be
**fractional** — DINOv3's rotary embedding asks for `arange(0.5, 32, 1)`, the
patch centres, and `intattr` threw `InexactError: Int64(0.5)` on it.

The length is aten's own `ceil((end - start) / step)` and not a Julia range's.
Those disagree whenever the step is not 1: `arange(0, 5, 2)` is `[0, 2, 4]`, three
elements, while `length(0:2:5-2)` is two. The old formula was
`length(start:step:stop-step)`, which is right for a unit step and silently one
short otherwise — so this fixes a latent miscount as well as the fractional case,
and nothing in the tree had a non-unit integer step to expose it.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("arange.start_step")})
    start = numattr(ctx, something(get(op.attrs, "arg0", nothing), 0))
    stop = length(op.ins) >= 1 ? value(ctx, op.ins[1]) : numattr(ctx, op.attrs["arg1"])
    step = numattr(ctx, something(get(op.attrs, "arg2", nothing), 1))
    T = dtypeof(ctx, op.out)
    n = max(0, ceil(Int, (Float64(stop) - Float64(start)) / Float64(step)))
    launch!(ctx, arange_body, alloc(ctx, op.out, n), T(start), T(step))
end

shapeof(ctx::Ctx, id) = ctx.graph.buffers[id].shape
dtypeof(ctx::Ctx, id) = ctx.graph.buffers[id].dtype

# ------------------------------------------------------------------ reductions

torchdims(op::Op, n) = [jdim(d, n) for d in ints(op.attrs["arg1"])]
# Through `atenarg` because `keepdim=True` is a keyword in almost every PyTorch
# source that writes it, and a reduction that silently keeps or drops the wrong
# axis is the same trap `gelu`'s `approximate` was — see `atenarg`'s docstring.
# Every graph in the tree happens to carry it positionally; that is a property of
# what has been exported so far, not of the format.
keepdim(op::Op) = atenarg(op, 2, "keepdim", false) === true

function reduced(a, d, keep)
    keep && return a
    dropdims(a; dims=Tuple(d))
end

"""
The map step `foldpremap` folded into this reduction, or `nothing`.

`sum(f, a; dims)` is `mapreduce(f, +, a; dims)`, so the whole of premapping is
passing `f` one level down — no new kernel, and the elementwise pass that would
have materialised `f.(a)` disappears.

Behind a barrier, and for the same reason `addmm_epi!` is: the `FusedOp` comes
out of a `Dict{String,Any}`, so without one the *per-element* call inside the
reduction would be a dynamic dispatch. One dispatch per op, none per element.
"""
premap(op::Op) = get(op.attrs, "premap", nothing)
@inline premapsum(a, f, d) = sum(f, a; dims = d)
@inline premapprod(a, f, d) = prod(f, a; dims = d)

runop!(ctx::Ctx, op::Op, ::Val{Symbol("sum.dim_IntList")}) =
    (a = lhs(ctx, op); d = torchdims(op, ndims(a)); f = premap(op);
     reduced(f === nothing ? sum(a; dims=Tuple(d)) : premapsum(a, f, Tuple(d)), d, keepdim(op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("mean.dim")}) =
    (a = lhs(ctx, op); d = torchdims(op, ndims(a)); f = premap(op);
     reduced((f === nothing ? sum(a; dims=Tuple(d)) : premapsum(a, f, Tuple(d))) ./ prod(size(a, i) for i in d), d, keepdim(op)))
runop!(ctx::Ctx, op::Op, ::Val{Symbol("prod.dim_int")}) =
    (a = lhs(ctx, op); d = [jdim(Int(op.attrs["arg1"]), ndims(a))]; f = premap(op);
     reduced(f === nothing ? prod(a; dims=Tuple(d)) : premapprod(a, f, Tuple(d)), d, keepdim(op)))
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

"""Largest `n` `topk_body` will scan before it refuses. Chosen so the redundant
work stays under a sort's constant factor, not from a measurement — the only
caller today has `n = 8`."""
const TOPK_MAX_N = 256

# aten::topk returns (values, indices). One thread per OUTPUT element, which
# means the thread producing rank `r` re-runs the selection `r+1` times — O(k*n)
# per slice with no sort, no shared memory and no cross-thread agreement needed.
#
# That trade is chosen for the shape this exists for. Hunyuan3D's MoE gate takes
# the top 2 of 8 experts, so `k*n` is 16 comparisons and a sorting network would
# be slower than the redundancy it removes. It is the wrong kernel for a large
# `n`, and the guard in `runop!` says so rather than degrading quietly.
#
# The ordering is `(value descending, index ascending)`: round `r` picks the
# largest key strictly below the previous round's, so equal values resolve to the
# lower index — which is what torch returns. Written as a bounded scan rather
# than a rank count so every arm produces a real result; there is no
# "unreachable" branch handing back a plausible number.
@inline function topk_body(I, a, ::Val{D}, ::Val{WANTIDX}) where {D,WANTIDX}
    @inbounds begin
        N = size(a, D)
        at(i) = a[CartesianIndex(ntuple(k -> k == D ? i : I[k], Val(length(I))))]
        prev_v = at(1)
        prev_i = 0                       # 0 = "no upper bound yet", round 1
        for _ in 1:I[D]
            best_v = at(1)
            best_i = 0
            for i in 1:N
                v = at(i)
                below = prev_i == 0 || v < prev_v || (v == prev_v && i > prev_i)
                better = best_i == 0 || v > best_v || (v == best_v && i < best_i)
                if below && better
                    best_v = v
                    best_i = i
                end
            end
            prev_v = best_v
            prev_i = best_i
        end
        WANTIDX ? oftype(prev_v, prev_i - 1) : prev_v
    end
end

"""
`aten::topk(self, k, dim, largest, sorted)` -> `(values, indices)`.

`largest=false` is not implemented: nothing in the ported models asks for it and
a smallest-k selection is the opposite comparison throughout `topk_body`, not a
flag this can pass through. It errors rather than returning the largest ones
under a different name.

Indices come back in the VALUE dtype, as `max.dim` above already does. Torch
declares them `int64`; carrying an Int64 index array onto the device to hold
values in `0:7` would cost more than it states, and every consumer here
(`scatter`, `gather`, `index`) converts with `Int(...)` on read. The exactness
limit is real and worth naming: a Float16 index is only exact to 2048, so this
would be wrong for a large `n` — the same guard that rejects a large `n` for
being slow also keeps it inside that range.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("topk.default")})
    a = lhs(ctx, op)
    k = Int(op.attrs["arg1"])
    d = jdim(Int(get(op.attrs, "arg2", -1)), ndims(a))
    Bool(something(get(op.attrs, "arg3", nothing), true)) ||
        error("topk: largest=false is not implemented (op $(op.id))")
    n = size(a, d)
    k <= n || error("topk: k=$k exceeds dim $d of size $n (op $(op.id))")
    n <= TOPK_MAX_N || error(
        "topk: dim $d has $n elements and this kernel is O(k*n) per output — " *
        "above $TOPK_MAX_N it needs a sorting network, not this (op $(op.id))")
    sz = ntuple(i -> i == d ? k : size(a, i), ndims(a))
    vals = launch!(ctx, topk_body, tupledest(ctx, 0, tupledtype(ctx, 0, eltype(a)), sz...),
                   a, Val(d), Val(false))
    inds = launch!(ctx, topk_body, tupledest(ctx, 1, eltype(a), sz...),
                   a, Val(d), Val(true))
    (vals, inds)
end

"""
    scatter_kernel!(dst, idx, src, Val(D), Val(N))

`dst[..., idx[I], ...] = src[I]` along Julia dim `D`, one thread per element of
`src`.

No atomic, unlike [`scatteradd_kernel!`](@ref), and the difference is the op
rather than an oversight: `aten::scatter` without a `reduce` is *undefined* when
two entries of `idx` collide in the same slice — torch documents the result as
non-deterministic — so there is nothing to accumulate and nothing to order.
"""
# `dst[.., iv[l], ..] = src[.., l, ..]` along dim `D`, with the index staying ON
# THE DEVICE.
#
# Distinct from `scatter_kernel!` only in how `iv` is addressed: `scatter` carries
# an index the same shape as `src`, `index_put` carries one vector over the
# indexed axis. The point of it is what it does NOT do — see `index_put.default`.
# `dst[.., iv[l], ..] = src[.., l, ..]` along dim `D`, with the index staying ON
# THE DEVICE.
#
# FLAT launch, CARTESIAN indexing, and both halves of that were measured.
#
# A multidimensional `ndrange` is partitioned into multidimensional workgroups,
# and for a `(128, 1, 8, 1)` write that is **15.44 us per launch against 2.74**
# for a flat one — 64 of those a step is 1.5 ms of pure launch shape.
#
# But indexing `dst` linearly to go with the flat launch is worse still: 4.88
# tok/s against 6.05. `dst` is a `view` into the KV cache, and a strided view
# recomputes its Cartesian position on every linear store. So: flat range,
# converted back to a `CartesianIndex` in the kernel, where it costs three
# divisions instead of a store's worth of index arithmetic.
#
# Distinct from `scatter_kernel!` only in how `iv` is addressed: `scatter` carries
# an index the same shape as `src`, `index_put` carries one vector over the
# indexed axis. The point of it is what it does NOT do — see `index_put.default`.
@kernel cpu=false function indexput_kernel!(dst, @Const(src), @Const(iv), ::Val{D},
                                            ::Val{N}, ::Val{SZ}, n::Int64) where {D,N,SZ}
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            I = CartesianIndices(SZ)[i]
            j = Int(iv[I[D]]) + 1
            dst[CartesianIndex(ntuple(k -> k == D ? j : I[k], Val(N)))] = src[I]
        end
    end
end

@kernel function scatter_kernel!(dst, @Const(idx), @Const(src), ::Val{D}, ::Val{N}) where {D,N}
    I = @index(Global, Cartesian)
    @inbounds begin
        j = Int(idx[I]) + 1                        # torch indices are 0-based
        dst[CartesianIndex(ntuple(k -> k == D ? j : I[k], Val(N)))] = src[I]
    end
end

"""
`aten::scatter.src(self, dim, index, src)` — `self` with the entries `index`
names along `dim` replaced by `src`.

Out of place: the result is a copy of `self` with the writes applied, so `self`
survives for whatever else reads it. The copy is a full pass over `self` and the
scatter touches `length(index)` of it; for the MoE gate that is 8 columns
written from 2, which is the cheap direction.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("scatter.src")})
    a = lhs(ctx, op)
    d = jdim(Int(op.attrs["arg1"]), ndims(a))
    idx = operand(ctx, op, 3)
    src = operand(ctx, op, 4)
    size(idx) == size(src) || error(
        "scatter: index $(size(idx)) and src $(size(src)) must have the same " *
        "shape (op $(op.id))")
    ndims(idx) == ndims(a) || error(
        "scatter: index is $(ndims(idx))-d and self is $(ndims(a))-d (op $(op.id))")
    dst = dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, size(a)...)
    dst .= a
    scatter_kernel!(ctx.backend)(dst, idx, src, Val(d), Val(ndims(a));
                                 ndrange = size(idx))
    dst
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
    # `AbstractGPUArray` rather than `AbstractArray`: dense and contiguous by
    # construction, which is what the kernel indexes on, and it keeps a lazy
    # `Broadcasted` or a permuted view out of a path that cannot take them. Every
    # wrapper the fallback exists for — `SubArray`, `PermutedDimsArray`,
    # `ReshapedArray` — is outside it, and so is a host `Array`, which matters
    # because the kernel is `cpu=false`.
    #
    # It named `Mantle.LavaArray` until 2026-09-14, which is the same predicate
    # with one backend's spelling, and it is what the fallback cost: on AMDGPU a
    # `ROCArray` failed the test, so SAM 2.1's 96 layer norms replayed as 854
    # graph passes against Lava's 96 — 8.9 launches apiece for a form whose whole
    # point is to be one.
    #
    # The two forms do not produce identical bits — so "does the mask still
    # match" is a question about the whole model, not about the kernel. They do
    # now agree on the ACCUMULATOR, which is not a tidiness point: summing in
    # `Float16` saturates at 65504, and the squares of 768 features at magnitude
    # 30 already exceed that. The instance norm below hit exactly this and went
    # silently to zero (see `_native_batch_norm_legit.no_stats`); this path
    # reduces over far fewer elements, so it is a narrower window, not a closed
    # one.
    # The fused form wins, so there is no switch to compare the two.
    if a isa GPUArrays.AbstractGPUArray && d == Tuple(1:length(d)) && length(a) % n == 0
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
    t = scratch!(ctx.ws, ctx.backend, eltype(a), size(a)...)
    t .= Base.broadcasted(-, a, μ)
    v = sum(abs2, t; dims=d, init=zero(A)) ./ n
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

"""
`aten::_fused_rms_norm` returns `(out, rstd)` — `x * rsqrt(mean(x^2) + eps) * γ`,
over the trailing torch dims, i.e. the leading Julia ones.

`nn.RMSNorm` survives `run_decompositions()` whole rather than breaking into
`pow`/`mean`/`rsqrt`/`mul`, so this is one op and not four.
There is no `β`: RMS norm has a scale and no shift, and no mean to subtract.

`eps` is optional in torch's schema, and when the module was built without one
the reference falls back to `finfo(dtype).eps` — not to zero. Defaulting to zero
would be a silently different function on a tensor whose mean square underflows.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("_fused_rms_norm.default")})
    a = lhs(ctx, op)
    nshape = ints(op.attrs["arg1"])
    d = Tuple(1:length(nshape))
    n = prod(size(a, i) for i in d)
    γ = length(op.ins) >= 2 ? value(ctx, op.ins[2]) : nothing
    epsattr = get(op.attrs, "arg3", nothing)
    ε = epsattr === nothing ? Float32(eps(float(eltype(a)))) : Float32(epsattr)

    # Same layout contract as `native_layer_norm` above: `AbstractGPUArray` rather
    # than `AbstractArray`, because the kernel indexes linearly over a dense
    # buffer and a permuted view's linear order is not that. Hunyuan3D's q/k norms
    # feed exactly such a view — `q.reshape(B,N,h,d).transpose(1,2)` — so the
    # fallback below is the path this model takes today, not a corner.
    if a isa GPUArrays.AbstractGPUArray && length(a) % n == 0
        out = tupledest(ctx, 0, tupledtype(ctx, 0, eltype(a)), size(a)...)
        r = tupledest(ctx, 1, Float32, length(a) ÷ n)
        rmsnorm!(ctx, out, r, a, γ, n, ε)
        return (out, r)
    end
    A = accum(eltype(a))
    # `sqaccum`, not `abs2` — and the difference is the whole point of this line.
    # `sum(abs2, a; init = zero(Float32))` squares each element at the ELEMENT's
    # precision and only then widens, so a Float16 above 256 gives `Inf` before
    # the accumulator is ever reached; `rsqrt(Inf)` is 0 and the group comes back
    # all zeros with nothing raised. The layer norm above is not exposed to this
    # because it reduces `a .- μ` and `μ` is Float32, so its subtraction has
    # already widened. RMS norm has no mean to subtract. Pinned by
    # `test_hunyuan3d_ops.jl`, which caught it here.
    r = 1 ./ sqrt.(sum(sqaccum, a; dims=d, init=zero(A)) ./ n .+ ε)
    y = Base.broadcasted(*, a, r)
    γ === nothing || (y = Base.broadcasted(*, y, γ))
    out = tupledest(ctx, 0, tupledtype(ctx, 0, eltype(a)), size(a)...)
    out .= y
    (out, r)
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
# every `inner` at 1 copies the array to produce exactly the array it was
# handed, and then allocates the outer result. Both land outside the plan — 88 MB and
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
# VkFFT (`Mantle.rfft`, `Mantle.fft`) — 20 is 4x5, which the mixed-radix plan covers.
#
# torch's `normalization` enum on both: 0 none, 1 by sqrt(n), 2 by n. It is read
# rather than assumed, because getting it wrong scales the audio by 20 and still
# sounds like speech.

"""Torch's FFT normalization enum applied to `x` for a transform of length `n`."""
fftnorm(x, mode::Integer, n::Integer) =
    mode == 0 ? x : mode == 1 ? x ./ sqrt(Float32(n)) : x ./ Float32(n)

"""
`_fft_r2c(self, dim, normalization, onesided)` — the forward real FFT.

Only a single transform axis, and it must be the contiguous one: `Mantle.rfft`
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
    fftnorm(Mantle.rfft(a), Int(get(op.attrs, "arg2", 0)), size(a, 1))
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
    y = real.(Mantle.fftany!(similar(full), full; inverse = true))
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

This is the immediate executor's host implementation. The declared executor
uses a counter-derived device kernel instead, because recording these uploaded
values would replay the same noise forever. Kokoro's `SineGen` wants
`randn_like` over `(1, 58800, 9)` — about 2 MB per call — plus a 9-element
`rand` for the initial phase.

The values come from `ctx.noise` rather than from `Random` directly, so a parity
run can substitute [`ZeroNoise`](@ref) and compare the deterministic path against
a reference that has had the same thing done to it.
"""
function hostnoise(ctx::Ctx, op::Op, f, dims)
    if Mantle.recordingpass() !== nothing
        ctx.noise isa ZeroNoise || throw(ArgumentError(
            "recording a host noise draw would freeze randomness; use record=false or ZeroNoise()"))
        out = alloc(ctx, op.out, dims...)
        fill!(out, zero(eltype(out)))
        return out
    end
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
`(H, N, 2)`. Which configurations are accepted is `lstmconfig`, beside the
kernel, because it is the kernel's contract and the declared route reads the
same one.

Returns only `(output,)`. `aten::lstm` also returns the final `h` and `c`, which
`getitem` would pick out — no graph here reads them, and returning a placeholder
that looks like state would be worse than not returning it.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("lstm.input")})
    x = lhs(ctx, op)                                  # (D, T, N)
    hx = [value(ctx, String(e)[2:end]) for e in op.attrs["arg1"]]
    ps = [value(ctx, String(e)[2:end]) for e in op.attrs["arg2"]]
    D, T, H, ndir = lstmconfig(op, x, ps)
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

`dst[idx[j] + 1] += src[j]`, atomically, for every `j`; `idx` uses torch's
zero-based convention.

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
        Atomix.@atomic dst[Int(idx[j]) + 1] += src[j]
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
    # `inplace` is set by `foldcacheupdate`, which has checked that `a` is a view
    # of a graph input whose pre-write contents nothing else reads. Writing
    # through it skips a full copy of the tensor per call — 1 GiB per token on a
    # 64-layer KV cache — and makes the value the caller handed in the value it
    # gets back.
    inplace = Bool(something(get(op.attrs, "inplace", nothing), false))
    dst = inplace ? a : dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, size(a)...)
    inplace || (dst .= a)
    n = ndims(a)
    accum = Bool(something(get(op.attrs, "arg3", nothing), false))
    # Hold the index operands as they came — ON THE DEVICE, and not `collect`ed
    # here for every dim to decide a contiguity test only the `view` path at the
    # bottom consumes. On
    # Kokoro that cost **17.5 MB of host allocation per utterance** (measured,
    # `--track-allocation`): the iSTFT overlap-add scatters 235220 entries, so
    # the index alone is 1.9 MB of Int64, downloaded here and again as
    # `Int32.(collect(ivals))` below. Each download drags a queue sync with it,
    # which is the part that shows up as time.
    raw = Vector{Any}(undef, n)
    fill!(raw, nothing)
    nz = Int[]
    for (k, e) in enumerate(op.attrs["arg1"])
        e === nothing && continue
        jd = n - k + 1
        1 <= jd <= n || error("index_put: dim $k out of range for $(n)-d input")
        push!(nz, jd)
        raw[jd] = value(ctx, String(e)[2:end])
    end
    idx = Vector{Any}(undef, n)
    fill!(idx, Colon())
    # Only the paths that genuinely need host indices pay for them.
    hostidx(jd) = (iv = vec(Int.(collect(raw[jd]))) .+ 1;  # 0-based in torch
                   iv == collect(first(iv):last(iv)) ? (first(iv):last(iv)) : iv)
    if accum
        # ── accumulate = true: a SCATTER-ADD, and duplicates are the point.
        #
        # `view(dst, idx) .+= src` is NOT this. When `idx` repeats an index —
        # which it does here, 235220 entries landing in 58820 slots — a Julia
        # broadcast assignment keeps one contribution per slot and silently drops
        # the rest, while torch sums them. That is the iSTFT's overlap-add
        # envelope: three of every four windows vanished, the envelope had 1451
        # zeros, and the division by it produced NaN and Inf in the audio.
        #
        # **On the device, through `OpAtomicFAdd`.** A host loop would do for a
        # use that is always a constant subgraph folded at load. Kokoro's iSTFT is that
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
        if eltype(dst) === Float32
            # The kernel consumes torch's zero-based index directly. It is a
            # backend-independent KernelAbstractions kernel; whether the target
            # lowers Atomix's fp32 add to its native atomic is the backend's job.
            iv = vec(raw[d])
            m = length(src)
            scatteradd_kernel!(ctx.backend)(vec(dst), vec(src), iv, Int32(m);
                                            ndrange = m)
            return dst
        end
        # Host fallback (non-fp32: there is no fp16 atomic add to ask for) still
        # needs the indices on the host, and pays for them here rather than for
        # every caller.
        ivals = hostidx(d)
        hv = vec(Array(dst))
        hs = vec(Array(src))
        @inbounds for (j, i) in enumerate(ivals)
            hv[i] += hs[j]
        end
        copyto!(dst, reshape(hv, size(dst)))
        return dst
    end
    # ── ONE index tensor: scatter on the device, and never look at the values.
    #
    # `hostidx` below downloads the index to build a Julia `view`, and a download
    # is a `flush!` plus `vkWaitSemaphores` — it drains the queue. A KV cache
    # write does this twice a layer, so K2 Horizon 32B's 64 layers stalled the
    # pipeline **128 times per token**: 1232 of the 1458 profile samples inside
    # `execute!`, i.e. 85% of a decode step, spent waiting for a GPU that had
    # nothing left to run. The index is `cache_position`, a graph input, and
    # nothing about this op needs its value on the host.
    #
    # Strictly better than the range path it replaces, which was cheap on the
    # device (a strided assign) and only ever ran because the download had
    # already happened.
    if length(nz) == 1 && ndims(src) == n
        d = nz[1]
        indexput_kernel!(ctx.backend, 256)(dst, src, vec(raw[d]), Val(d), Val(n),
                                           Val(size(src)), Int64(length(src));
                                           ndrange = length(src))
        return dst
    end
    # More than one index tensor is advanced indexing; still host-side, and rare
    # enough that it has never shown up in a profile.
    for jd in nz
        idx[jd] = hostidx(jd)
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
        if groups == 1 &&
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
    #
    # `epilogue` is the general form of the same thing, set by `foldepilogue`: any
    # unary elementwise expression, as a `FusedOp`. Through a barrier because it
    # comes out of a `Dict{String,Any}`, and `matmul!` has to specialise on the
    # concrete callable or the GEMM's store would dispatch per element.
    epi = get(op.attrs, "epilogue", nothing)
    epi === nothing ?
        matmul!(ctx, out, b, a, bias; epi = actfn(Symbol(get(op.attrs, "act", "none")))) :
        addmm_epi!(ctx, out, b, a, bias, epi)
    out
end

"""The barrier for a `FusedOp` epilogue: one dynamic dispatch per op, none per
element."""
@inline addmm_epi!(ctx::Ctx, out, b, a, bias, epi) = matmul!(ctx, out, b, a, bias; epi)

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
        # `unsafe_view`, not `view`. `checkbounds` against a DEVICE index array
        # has to read its values on the host: a full sync per call on the plain
        # path, and simply WRONG on a recorded one, where the `.+ 1` above is a
        # launch that has not run yet — the check then reads an unwritten buffer
        # and throws `BoundsError: ... at index [1:1280, [1]]` for an index that
        # is plainly in range. The dims are checked above and the values are the
        # traced model's own.
        J = Base.to_indices(x, Tuple(idx))
        return materialize(ctx, Base.unsafe_view(x, J...))
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

The mixed case DOES occur: SAM 2's encoder has sixteen, so requiring every axis
to be indexed makes `Model` throw while folding them as constants.

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
`gelu` with torch's default (exact) formulation. `approximate = "tanh"` selects
the approximation, which differs by ~1e-3 and is a different function, not a
faster one — so it is dispatched, not assumed.

Read through [`atenarg`](@ref): the attribute arrives as `arg1` or as
`approximate` depending on how the model's source wrote the call, and reading
only `arg1` — which is what this did — meant no graph in the tree ever took the
tanh branch.

The two branches *call* [`geluexact`](@ref) / [`gelutanh`](@ref) rather than
restating them. When they were written out here as well, the epilogue and the
standalone op were two copies of one expression that had to agree character for
character for a fold to be a no-op, and the comment saying so was the only thing
keeping them in step.
"""
function runop!(ctx::Ctx, op::Op, ::Val{Symbol("gelu.default")})
    x = lhs(ctx, op)
    f = String(atenarg(op, 1, "approximate", "none")) == "tanh" ? gelutanh : geluexact
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
    out
end
