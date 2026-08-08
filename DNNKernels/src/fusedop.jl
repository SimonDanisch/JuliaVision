"""
One elementwise expression as a value: several functions and how their operands
wire together, callable as a single function.

    fo = FusedOp((+, *), ((In(1), In(2)), (Tmp(1), In(3))))
    fo(a, b, c) == (a + b) * c

## Why this exists

Fusion here is currently a *runtime* trick: an elementwise op whose result is
read once returns its `Base.Broadcasted` unmaterialised, and the consumer's
broadcast nests it. That works, and it costs three things.

The graph never learns a fusion happened, so `lifetimes` walks fusion chains with
depth guards to find when an operand is really last read, `planslab` special-cases
values that have no storage, and discovery has to see through the same laziness.
Every one of those is reconstructing, later and approximately, a fact the fusion
knew exactly.

The type of a nested `Broadcasted` encodes the whole tree, so each distinct
expression is a distinct type, a distinct specialisation and a distinct kernel to
compile — and because the nesting only happens while *executing*, that set cannot
be enumerated ahead of time. It has to be discovered by running, which is why the
frozen kernel cache is populated by a workload rather than derived.

And it is Base's broadcasting, which means Base's dispatch: the `BroadcastStyle`
methods that route it here are type piracy over `SubArray` and `PermutedDimsArray`,
and they invalidate Base's own precompiled broadcast code on load.

A `FusedOp` is the same expression as a **plain callable**. It is not a broadcast,
it has no style, and it names no Base method. What it is for is being handed to a
kernel — any kernel — as `f`.

## Three places it is meant to be applied

That is why it is a bare callable rather than anything cleverer:

  * a standalone elementwise kernel, replacing `dst .= f.(a, b)`
  * a matmul's write-out epilogue, where `coopmat_setcomp` already applies a
    function per component (it was built for the GEMM's gelu epilogue)
  * a reduction's map step — `mapreduce(fo, +, x)` instead of materialising
    `f.(x)` and reducing it

An expression that can only be broadcast can do the first. This does all three,
and all three are implemented: `fuseops` for the standalone kernel,
`foldepilogue` for the GEMM's write-out, `foldpremap` for the reduction's map
step. Measured across SAM 2 and MatAnyone, they fire 69, 11 and 4 times.

The map step is the small one, and for a structural reason rather than a
tuning one: 111 of the 136 reductions in these graphs read a value that
something else reads too, usually a residual stream's `add.Tensor`, which has to
be materialised whatever happens downstream.

## Shape

Operands are named rather than positional-by-convention, because a fused
expression is a DAG and not a chain: `(a + b) * (c - d)` has no reading as
`f3(f2(f1(...)))`. `In(i)` is the i-th argument the caller passes, `Tmp(i)` is
what the i-th function produced, and `Konst(v)` is a literal the graph carried in
an attribute. Functions are applied in order, so `Tmp(i)` is only legal after
function `i` — which the constructor checks, because the alternative is a
`MethodError` inside a GPU kernel.
"""

"""The i-th argument passed to the `FusedOp`."""
struct In{I} end
In(i::Integer) = In{Int(i)}()

"""What the i-th function produced. Only legal after that function has run."""
struct Tmp{I} end
Tmp(i::Integer) = Tmp{Int(i)}()

"""A literal operand — `alpha` on an `add.Tensor`, the exponent on a `pow`.

Holds the value rather than encoding it in the type: two graphs differing only in
a scale factor would otherwise compile to two kernels, and the value is read once
per element from a register either way."""
struct Konst{T}
    v::T
end

const Operand = Union{In,Tmp,Konst}

"""
A dtype conversion, as a concrete callable.

`_to_copy` is the most frequent op in an exported graph — 603 of SAM 2's 1353
before the cast-folding passes — so this is the common case, not a corner.

The obvious spelling is to put the type itself in `funcs`: `FusedOp((*, Float16,
+), …)`. That types the slot as `DataType`, which is **not** a concrete callable,
so calling it is a dynamic dispatch — per element, inside a kernel. Inference
gives up and returns `Any`, which is how this was caught: `@inferred` on the
mixed-dtype chain.

As a type parameter it is a singleton, and the call inlines to one convert.
"""
struct Cast{T} end
Cast(::Type{T}) where {T} = Cast{T}()
@inline (::Cast{T})(x) where {T} = T(x)

Base.show(io::IO, ::Cast{T}) where {T} = print(io, T)

"""
`f`, then rounded to `T` — what materialising the result into a `T` array does.

**Fusion has to round where the unfused chain stored, or it is not the same
arithmetic.** An exported graph carries scalars as plain numbers, and a Float64
one promotes: `x + 1.0e-6` on a `Float32` tensor evaluates in Float64. Unfused
that is immediately stored into a Float32 destination and rounded; fused, the
Float64 would carry into the next function and the one after.

It shows up exactly where floating point is least forgiving. On
`(2 - sqrt(x + 1e-6)) * 3` at `x = 4` the subtraction cancels to ~1e-7, and the
two orders of rounding differ in the fourth significant digit: -7.4999995e-7
against -7.1525574e-7.

Free where it changes nothing — `Float32(::Float32)` is the identity and compiles
away — so every step gets one rather than trying to work out which steps promote.

**It does not make a fused chain bit-equal to a materialised one, and cannot.**
A `mul` followed by an `add` contracts to an FMA in the SPIR-V backend, straight
through the conversion between them: on `sigmoid -> mul -> add` in Float16 the
device result matches `fma(x2, t1, x3)` on 16384 of 16384 elements and the
per-step-rounded result on none of them, one ulp (2^-10) apart. That is one
rounding where materialising does two, so it is the *more* accurate answer, and
it is not reachable from Julia. The runtime fusion in `fuse.jl` has always
contracted the same way, which is why the two agree exactly.

What `Rounded` is for is the part that *is* controllable: a Float64 scalar
attribute promoting the rest of the chain, which is a change of working
precision rather than a fused multiply-add.
"""
struct Rounded{T,F}
    f::F
end
Rounded(::Type{T}, f::F) where {T,F} = Rounded{T,F}(f)
@inline (r::Rounded{T})(args::Vararg{Any,N}) where {T,N} = T(r.f(args...))

Base.show(io::IO, r::Rounded{T}) where {T} = print(io, r.f)

"""
    FusedOp(funcs, args)

`funcs[k]` is applied to `args[k]`, in order, and the last result is returned.
"""
struct FusedOp{Fs<:Tuple,As<:Tuple}
    funcs::Fs
    args::As

    function FusedOp(funcs::Fs, args::As) where {Fs<:Tuple,As<:Tuple}
        length(funcs) == length(args) || throw(ArgumentError(
            "FusedOp: $(length(funcs)) functions but $(length(args)) argument lists"))
        isempty(funcs) && throw(ArgumentError("FusedOp: needs at least one function"))
        # A `Tmp(j)` referring to a function that has not run yet reads an
        # undefined slot. Caught here rather than in a kernel, where it is a
        # MethodError several frames inside a launch.
        for (k, as) in enumerate(args), a in as
            a isa Tmp && (tmpindex(a) < k || throw(ArgumentError(
                "FusedOp: function $k reads Tmp($(tmpindex(a))), which runs at or after it")))
        end
        new{Fs,As}(funcs, args)
    end
end

tmpindex(::Tmp{I}) where {I} = I
inindex(::In{I}) where {I} = I

"""How many arguments this expression expects — the largest `In` it names."""
nargs(fo::FusedOp) = maximum((a isa In ? inindex(a) : 0 for as in fo.args for a in as); init = 0)

"""How many functions deep it is; the number of kernels it replaces."""
Base.length(fo::FusedOp) = length(fo.funcs)

"""
Unrolled to straight-line code: `t1 = f1(x1, x2); t2 = f2(t1, x3); …; tn`.

**Generated, and it has to be.** The natural version threads the temporaries
through a tuple that grows by one per function and recurses. That is perfectly
type stable — inference returns the right concrete type at every depth — but it
stops being *inlined* past three functions, and the growing tuples then land on
the heap:

    depth 2    0 bytes       depth 5   112
    depth 3    0             depth 6   176
    depth 4   48             depth 8   304

Per element, inside a kernel, that is not a slow path, it is a broken one.
Unrolling gives every temporary its own SSA value, so there is no tuple to
allocate at any depth and the expression becomes the straight-line arithmetic it
always was.

Type stability was never the problem here, which is worth saying because it is
the thing one reaches for first: `@code_warntype` is clean on the recursive
version at every depth. `@allocated` is what shows it.
"""
@generated function (fo::FusedOp{Fs,As})(ins::Vararg{Any,N}) where {Fs,As,N}
    n = length(Fs.parameters)
    body = Expr(:block, Expr(:meta, :inline))
    for k in 1:n
        ops = As.parameters[k].parameters
        args = map(enumerate(ops)) do (j, A)
            if A <: In
                :(ins[$(A.parameters[1])])
            elseif A <: Tmp
                Symbol(:t, A.parameters[1])
            else                     # Konst — read the value from the field
                :(fo.args[$k][$j].v)
            end
        end
        push!(body.args, :($(Symbol(:t, k)) = fo.funcs[$k]($(args...))))
    end
    push!(body.args, Symbol(:t, n))
    body
end

function Base.show(io::IO, fo::FusedOp)
    print(io, "FusedOp(")
    for (k, (f, as)) in enumerate(zip(fo.funcs, fo.args))
        k > 1 && print(io, "; ")
        print(io, "t", k, " = ", f, "(")
        for (j, a) in enumerate(as)
            j > 1 && print(io, ", ")
            print(io, a isa In ? "x$(inindex(a))" : a isa Tmp ? "t$(tmpindex(a))" : repr(a.v))
        end
        print(io, ")")
    end
    print(io, ")")
end
