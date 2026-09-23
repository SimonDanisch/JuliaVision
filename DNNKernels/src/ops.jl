"""
What an ATen op is READ as, and the kernels several of them share.

This file used to hold a `runop!` method per op name as well — a second
implementation of everything `emitop!` declares, kept because it was the only
thing that ran without a plan. Nothing runs without a plan now, and the arms are
gone. What was never duplicated stayed:

  * the ATTRIBUTE readers. torch folds a scalar operand into the schema, so it
    arrives in `attrs` under its positional slot rather than in `ins` — either
    side can be the scalar, since `1 - sigmoid(x)` exports as
    `sub.Tensor(ins = [sigmoid], arg0 = 1)`. `alpha` scales the second operand,
    so `add.Tensor(a, b, alpha = k)` is `a + k*b`. `ARGKEY`, `intattr`,
    `numattr`, `keepdim` and `torchdims` are what read all of that, and `emit.jl`
    calls them through its own `operand`.

  * the KERNELS with no elementwise spelling — rope, scatter, embedding,
    softmax, swiglu — which the declared path dispatches directly.
"""
# The positional attribute names, interned. `"arg$(pos-1)"` built a fresh
# `String` on EVERY operand access — 463 KB per Kokoro utterance, measured, and
# the `count` below built one more per earlier position. The set is tiny, fixed,
# and known at load.
const ARGKEY = ntuple(i -> "arg$(i-1)", 12)
@inline argkey(pos::Int) = @inbounds pos <= length(ARGKEY) ? ARGKEY[pos] : "arg$(pos-1)"
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

"""Non-finite and complex scalars are serialised as strings; see export_graphs.const."""
scalar(v) = v
function scalar(v::AbstractString)
    v == "-inf" && return -Inf32
    v == "inf" && return Inf32
    v == "nan" && return NaN32
    c = pycomplex(v)
    return c === nothing ? v : c
end
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
    InFloat{T}(f)

`f`, applied to its operand converted to `T` first.

Every float-valued entry in [`UNARY_FUSED`](@ref) — `sqrt`, `log`, `exp`, `tanh`,
`sin`, `cos`, `inv`, `rsqrt_` — promotes an INTEGER argument through `float(x)`,
and `float(::Int64)` is `Float64`. A GPU kernel handed that computes in double
precision: on a device that has no `double` it does not compile at all (Kokoro's
`rsqrt` of an `Int64` token count is an `InvalidIRError`, "unsupported use of
double value", from `sqrt` at `math.jl:1544`), and on one that does it silently
pays for a precision the destination cannot hold. Torch computes such an op in
the RESULT dtype, which is what converting first does.

A named callable and not `x -> f(T(x))`, for the reason [`rsqrt_`](@ref) is
named: an anonymous function gets a fresh type per definition site, so the same
arithmetic would compile and freeze as many kernels as there are places that
wrote it. `InFloat{Float32}(sqrt)` is one type everywhere.
"""
# `T` FIRST, so that `InFloat{Float32}` names the target type and not the function:
# with the parameters the other way round, `InFloat{Float32}(rsqrt_)` binds `Float32`
# to the function slot and builds `InFloat{typeof(rsqrt_),typeof(rsqrt_)}`, whose call
# is `rsqrt_(rsqrt_(x))` — a `MethodError` that on a GPU appears as a throw the
# compiler cannot emit rather than as an error anybody can read.
struct InFloat{T,F} <: Function
    f::F
end
InFloat{T}(f::F) where {T,F} = InFloat{T,F}(f)
@inline (g::InFloat{T,F})(x) where {T,F} = g.f(T(x))

"""
    infloat(f, Tout, Tin) -> f or InFloat

[`InFloat`](@ref) when a float-valued `f` would otherwise promote an integer operand
to `Float64`, and `f` itself when it would not. `Bool` is excluded: `!` is in the same
table, and `!(Float32(x))` is not a function.
"""
infloat(f, ::Type{Tout}, ::Type{Tin}) where {Tout,Tin} =
    (Tout <: AbstractFloat && Tin <: Integer && Tin !== Bool) ? InFloat{Tout}(f) : f

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
"""The pair of bounds `clamp` should be broadcast with: see `clamp.default`."""
function clampbounds(::Type{T}, dims::NamedTuple, lo, hi) where {T}
    l = lo === nothing ? -Inf32 : Float32(numattr(dims, lo))
    h = hi === nothing ? Inf32 : Float32(numattr(dims, hi))
    T <: Integer || return (l, h)
    (isinteger(l) || isinf(l)) && (isinteger(h) || isinf(h)) || return (l, h)
    return (l == -Inf32 ? typemin(T) : T(l), h == Inf32 ? typemax(T) : T(h))
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
function rope_kernel!(out, x, cs, sn,
                                        H::Int32, half::Int32, HT::Int32, n::Int64)
    i = KI.get_global_id().x
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
    return nothing
end

# The INTERLEAVED rotary, as one op — see `fusepairrope`.
#
# One thread per PAIR, so both of a pair's elements are read and written by the
# same thread and the store is a contiguous 2-element write. `P` and `H` are
# `Val`s because the only arithmetic here besides the rotation is two integer
# divisions, and by a runtime divisor those cost more than the rotation does.
function pairrope_kernel!(out, x, cs, sn,
                                            ::Val{P}, ::Val{H}, n::Int32) where {P,H}
    i = KI.get_global_id().x
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
    return nothing
end

# SwiGLU as one op.
#
# The export spells `silu(gate) * up` as a widening cast, a `sigmoid`, a
# multiply, a narrowing cast and a second multiply — five dispatches over a
# 26624-wide tensor, per layer, and two of them exist only to move it between
# fp16 and fp32. `sigmoid` is evaluated in fp32 here exactly as the traced graph
# does, so this changes the dispatch count and not the arithmetic.
function swiglu_kernel!(out, gate, up, n::Int64)
    i = KI.get_global_id().x
    if i <= n
        @inbounds begin
            x = Float32(gate[i])
            out[i] = eltype(out)(Float32(eltype(out)(x / (1f0 + exp(-x)))) * Float32(up[i]))
        end
    end
    return nothing
end

function swiglu_strided_kernel!(out, gate, up, gb, ub, gs, us)
    # Flat launch, decomposed here — see `scatter_kernel!`.
    lin = KI.get_global_id().x
    lin <= length(out) || return nothing
    i = CartesianIndices(size(out))[lin]
    @inbounds begin
        gi = gb; ui = ub
        for d in 1:length(gs)
            gi += Int32(i[d]-1)*gs[d]
            ui += Int32(i[d]-1)*us[d]
        end
        x = Float32(gate[gi])
        out[i] = eltype(out)(Float32(eltype(out)(x / (1f0 + exp(-x)))) * Float32(up[ui]))
    end
    return nothing
end

# The same SwiGLU where the innermost axis is a RUN on all three operands, `V`
# elements to a thread.
#
# Identical arithmetic to the kernel above and the same `exp` in the same place;
# what differs is how many bytes a thread moves. One fp16 element per thread is
# six bytes against a Cartesian index computed per element, and on Qwen-Image
# 2.1's `12288 x 4118` that measured 26.6 GB/s with the `exp` TAKEN OUT — so the
# sigmoid was never the ceiling, and no group shape fixes it either (12.1 ms at
# the best of six against 14.8 at the default). At `V = 8` the same op runs at
# 104.7 GB/s: 12.06 ms to 2.90. `V = 16` gives it back (3.64), which is why
# `swiglurun` stops at eight.
function swiglu_run_kernel!(out, gate, up,
                                              gb, ub, ob, gs, us, os,
                                              ::Val{V}, nd) where {V}
    # Flat launch, decomposed here — see `scatter_kernel!`. `nd` is the run
    # shape: axis one is `size(out, 1) ÷ V`, so it is not `size(out)`.
    lin = KI.get_global_id().x
    lin <= prod(nd) || return nothing
    i = CartesianIndices(nd)[lin]
    @inbounds begin
        # Axis one is the run, so it advances `V` at a time; every other axis is
        # walked exactly as the scalar kernel walks it.
        gi = gb + (Int32(i[1]) - Int32(1)) * Int32(V)
        ui = ub + (Int32(i[1]) - Int32(1)) * Int32(V)
        oi = ob + (Int32(i[1]) - Int32(1)) * Int32(V)
        for d in 2:length(gs)
            gi += Int32(i[d]-1)*gs[d]
            ui += Int32(i[d]-1)*us[d]
            oi += Int32(i[d]-1)*os[d]
        end
        for v in Int32(0):Int32(V - 1)
            x = Float32(gate[gi + v])
            out[oi + v] = eltype(out)(Float32(eltype(out)(x / (1f0 + exp(-x)))) *
                                      Float32(up[ui + v]))
        end
    end
    return nothing
end

"""
Off sends every SwiGLU back to [`swiglu_strided_kernel!`](@ref).

A declared switch rather than an edit, because the two kernels can only be
compared on ONE machine at ONE temperature: this M5 drifts about 10% over six
back-to-back denoiser steps, which is the size of the difference being measured.
Flip it, rebuild the plan in the same session, and the two numbers are
comparable.
"""
const SWIGLU_RUN = Ref(true)

"""
    swiglurun(dims, gs, us, os) -> V or nothing

How many elements of `dims` one thread of [`swiglu_run_kernel!`](@ref) may take.

All three operands have to be a RUN on the innermost axis — that is what makes
`+ v` the right address for all of them — and the extent has to divide `V`,
because the kernel has no tail. Eight where it fits, then four, then two;
measured on Qwen-Image 2.1, sixteen is past the turn.
"""
function swiglurun(dims, gs, us, os)
    SWIGLU_RUN[] || return nothing
    isempty(dims) && return nothing
    (gs[1] == 1 && us[1] == 1 && os[1] == 1) || return nothing
    for V in (8, 4, 2)
        dims[1] % V == 0 && return V
    end
    return nothing
end

# RoPE that stores straight into the KV cache.
#
# The rotated key is written to a temporary and then copied into the cache slot
# by an `index_put` whose only job is that copy: 128 extra dispatches and 3.1 ms
# of a 166 ms decode step, for a store the rope kernel could have done itself.
# Cartesian for the same reason `indexput_kernel!` is — the destination is a
# strided view into the cache, and indexing one linearly costs more than the
# launch shape saves.
function rope_store_kernel!(dst, x, cs, sn,
                                              iv, H::Int32, half::Int32,
                                              HT::Int32, ::Val{N}, ::Val{SZ},
                                              n::Int64) where {N,SZ}
    lin = KI.get_global_id().x
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
    return nothing
end

@inline arange_body(I, start, step) = start + (I[1] - 1) * step

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
function indexput_kernel!(dst, src, iv, ::Val{D},
                                            ::Val{N}, ::Val{SZ}, n::Int64) where {D,N,SZ}
    i = KI.get_global_id().x
    if i <= n
        @inbounds begin
            I = CartesianIndices(SZ)[i]
            j = Int(iv[I[D]]) + 1
            dst[CartesianIndex(ntuple(k -> k == D ? j : I[k], Val(N)))] = src[I]
        end
    end
    return nothing
end

function scatter_kernel!(dst, idx, src, ::Val{D}, ::Val{N}) where {D,N}
    # `@index(Global, Cartesian)` was KernelAbstractions decomposing an
    # N-dimensional ndrange; `KernelInterface` gives three linear axes. Launched
    # flat and decomposed here, which is what `bmm_nsplit_reduce!` already does.
    lin = KI.get_global_id().x
    lin <= length(idx) || return nothing
    I = CartesianIndices(size(idx))[lin]
    @inbounds begin
        j = Int(idx[I]) + 1                        # torch indices are 0-based
        dst[CartesianIndex(ntuple(k -> k == D ? j : I[k], Val(N)))] = src[I]
    end
    return nothing
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

function softmax_kernel!(out, a, ::Val{WG}, pre, n) where {WG}
    lt, = Tuple(KI.get_local_id())
    grp, = Tuple(KI.get_group_id())
    T = eltype(out)
    sh = KI.localmemory(Float32, Val((WG,)), Val(1))
    # Values that have to survive a `@synchronize` need `@private` storage —
    # a plain local is not guaranteed to on the CPU backend.
    keep = StaticArrays.MArray{Tuple{2}, Float32}(undef)

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
    KI.barrier()
    @inbounds if lt == 1
        m = sh[1]
        for k in 2:WG
            m = max(m, sh[k])
        end
        sh[1] = m
    end
    KI.barrier()
    @inbounds keep[1] = sh[1]
    KI.barrier()                       # every thread has read `m` before sh[1] is reused

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
    KI.barrier()
    @inbounds if lt == 1
        s = sh[1]
        for k in 2:WG
            s += sh[k]
        end
        sh[1] = s
    end
    KI.barrier()

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
    return nothing
end

"""Element of an outer repetition: the source tiles, so index it modulo its own size."""
@inline repeatouter(I, a, sz::NTuple{N,Int}) where {N} =
    @inbounds a[ntuple(k -> mod1(I[k], sz[k]), Val(N))...]

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
function scatteradd_kernel!(dst, src, idx, n::Int32)
    j = KI.get_global_id().x
    @inbounds if j <= n
        Atomix.@atomic dst[Int(idx[j]) + 1] += src[j]
    end
    return nothing
end

"""The barrier for a `FusedOp` epilogue: one dynamic dispatch per op, none per
element."""
@inline addmm_epi!(ctx::Ctx, out, b, a, bias, epi) = matmul!(ctx, out, b, a, bias; epi)

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

# The same table read the other way, for a backend library that has its own node for
# one of these. Declared HERE, next to the functions, because which activation
# `geluexact` is is a property of the function and not of any device — see
# `Mantle.activationkind`. A library's node is its own arithmetic: Apple's `erf` is a
# real one where [`erf`](@ref) above is Abramowitz-Stegun 7.1.26, so a backend that
# takes this route is measured against the reference dump rather than against the
# kernel it replaces.
M.activationkind(::typeof(geluexact)) = :gelu
M.activationkind(::typeof(gelutanh)) = :gelu_tanh
M.activationkind(::typeof(relu_epi)) = :relu

"""
    embedding(weight, indices)

Token embedding: gather rows of `weight` by `indices`.

Reversed layout puts the embedding dimension first — `weight` is
`(dim, vocab)` and the result is `(dim, indices...)` — so this is a column
gather, and the indices are torch's 0-based ones.
"""
function embedding_kernel!(out, w, idx, n::Int32)
    k = KI.get_global_id().x
    @inbounds if k <= n
        d = size(out, 1)
        col = (Int32(k) - Int32(1)) ÷ Int32(d)          # which token
        row = (Int32(k) - Int32(1)) % Int32(d) + Int32(1)
        out[k] = w[row, Int32(idx[col + Int32(1)]) + Int32(1)]
    end
    return nothing
end

# ── shape and attribute readers that outlived the interpreted runner ─────────
#
# These three were in `execute.jl` and are the only things in it that the
# DECLARED path calls. `emit.jl` and four host rewrites use them; the rest of
# that file was `execute!` and the machinery under it, and is gone.

"""
    jdim(d, n) -> Int

Torch dimension index (0-based, possibly negative) to Julia dimension for an
`n`-dimensional array whose shape is reversed.

**Refuses a dim outside torch's own range**, which is `-n:n-1`. The arithmetic
alone answers 0 for `d == n` and `n + 1` for `d == -(n + 1)`, an axis that does
not exist, returned as confidently as one that does, and it goes straight into
`colstrides` indexing from there. Five call sites out of thirty-five had grown
their own version of this check, each in its own words: that is the same fact in
six places, and the five that had it were the ops whose author happened to think
of it rather than the ops that needed it.

torch raises `IndexError` here, so refusing is also what the graph's producer
does.
"""
function jdim(d::Integer, n::Integer)
    j = d >= 0 ? n - Int(d) : -Int(d)
    1 <= j <= n || error(
        "DNNKernels: torch dim $(d) is out of range for a $(n)-d tensor, whose " *
        "dims are $(-n):$(n - 1). Reversed that would be Julia axis $(j), which " *
        "is not an axis.")
    return j
end

ints(x::Vector{Int}) = x

ints(x) = x isa AbstractVector ? Int.(x) : Int(x)

"""
    staticints(x) -> Vector{Int} | Nothing

`ints(x)`, but `nothing` when any element is not a number.

For a HOST-SIDE graph rewrite, which sees an attribute before any `dims` exist and so
cannot resolve a symbolic one. An exported attribute is `Any[...]` and an entry may be
a `"\$sym"` reference — a pad amount computed from a sequence length is — and there
`ints` throws `Int(::String)`. A rewrite that cannot read the number has to decline
the op, not fail the load, so the answer is `nothing` rather than an error.
"""
staticints(x::Vector{Int}) = x

function staticints(x)
    x isa AbstractVector || return x isa Number ? [Int(x)] : nothing
    all(v -> v isa Number, x) || return nothing
    Int[Int(v) for v in x]
end