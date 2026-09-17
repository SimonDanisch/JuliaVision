"""
Elementwise, as one declared dispatch — and the kernel a fused GROUP compiles to.

`runop!` wrote these as `emit(ctx, Base.broadcasted(f, xs...))`, which returned
the `Broadcasted` itself when `fuse.jl` said the value had exactly one consumer,
so the fused kernel was generated at the CONSUMER's launch by GPUArrays'
broadcast machinery — pairwise, per materialisation point, at run time. That is
why SAM 2.1's encoder shows ONE `fused.elementwise` op and 1,014 broadcast
dispatches.

The fusion DECISION was always static (`fuseops`, at model build). What was at
run time was the codegen. Declared, there is nothing to defer: the group is known
when the graph is built, so it emits one dispatch over one kernel.

**Macro-free.** `KI.get_global_id()` rather than `@index`, so the kernel is a
plain function that `KI.kernel_function` compiles for whichever backend asks —
`hipfunction` on ROCm, Lava's own compile on Vulkan. Two things about the
intrinsics that are easy to get wrong and cost a compile each: `get_global_id()`
returns `@NamedTuple{x::Int, y::Int, z::Int}` and is **1-based**, and there is no
implicit ndrange bounds check the way `@kernel` had, so every kernel guards its
own tail.

ONE kernel over a tuple of operands, not one per arity.

It was `ew1!`/`ew2!`/`ew3!`, and the reason given was that `Mantle.resolve` is
applied per element of a dispatch's argument tuple, so a tuple-of-resources
argument arrived unresolved. That was true, and the gap was in core rather than
a reason for three kernels. Three walks had to agree that a tuple argument is the
argument list grouped:

  * `resolve` and `storage` take a `::Tuple` method, so the operands reach the
    kernel as device arrays instead of `Buffer` handles.
  * `devicepointeroffsets` does not count a tuple as a level of nesting
    (`Mantle.nestinglevels`). Counting it stops one short of the addresses
    inside, so a recorded plan notes only its output and a `resize!` of an
    operand leaves it reading freed storage with nothing to see anywhere.
  * `find_tlas_in_args` walks into one, so an accel passed in a tuple still
    enables ray query.

`holdleaves!` already did the right thing, and the packer itself needed nothing:
its generic branch inlines any isbits aggregate and hands core the aggregate's
type. So the arity limit was the API bending around a gap, which is the wrong
direction.

The second reason given was that `use(p, x; read = true)` needs each operand to
be a top-level argument. That was simply wrong — `use` is called at emit time,
where a loop over the operands does it.
"""

"""
    bcindex(lin, od, st) -> Int

The 1-based index into an operand whose effective strides are `st`, for the
`lin`-th element (0-based) of an output shaped `od`.

`st[k] == 0` is how a broadcast axis is expressed: an operand of extent 1 on axis
`k` contributes nothing to the offset, so a `(C, 1, 1)` bias meets a `(W, H, C)`
activation with no materialised copy and no separate kernel.

Divisions, one per axis. The alternative is a 3-D ndrange, which only works up to
three axes and SAM 2's tensors are four; `toLE_tiled_*!` in `attention.jl` shows
the shape a specialisation would take if this ever measures.
"""
@inline function bcindex(lin::Int, od::NTuple{N,Int}, st::NTuple{N,Int}) where {N}
    off = 0
    rem = lin
    @inbounds for k in 1:N
        off += (rem % od[k]) * st[k]
        rem = rem ÷ od[k]
    end
    return off + 1
end

@inline function ewguard(od::NTuple{N,Int}) where {N}
    i = KI.get_global_id().x
    return i, i <= prod(od)
end

"""
Each operand's element for output element `lin`, in order.

A recursive tuple walk rather than a loop or an `ntuple`: the operands have
different types, so only the recursion is type-stable without a `Val` for the
count — and it unrolls, so the kernel has no tuple indexing at run time. Same
shape as `operandtuples` uses on the host side.
"""
# The two signatures differ ONLY in the operand tuples, so `Tuple{}` is strictly
# more specific and the recursion terminates. The base case first had an untyped
# `od`, which makes the two AMBIGUOUS rather than ordered: more specific in the
# operands, less specific in `od`. An ambiguity inside a kernel is a
# `jl_f_throw_methoderror`, which the compiler reports as "method lookup
# failure" four frames from anything that mentions dispatch.
@inline gather(::Tuple{}, ::Tuple{}, lin::Int, od::NTuple{N,Int}) where {N} = ()
@inline gather(ops::Tuple, sts::Tuple, lin::Int, od::NTuple{N,Int}) where {N} =
    (@inbounds(first(ops)[bcindex(lin, od, first(sts))]),
     gather(Base.tail(ops), Base.tail(sts), lin, od)...)

"""
    ew!(out, od, ops, sts, f)

`out .= f.(ops...)` over the output shape `od`, with `sts[k]` the effective
strides of `ops[k]` — zero on every axis it is broadcast along.

Any number of operands. `clone` is this with one operand and `f = identity`; a
folded bias-and-activation group is three.
"""
function ew!(out, od::NTuple{N,Int}, ops::Tuple, sts::Tuple, f) where {N}
    i, ok = ewguard(od)
    ok || return
    @inbounds out[i] = f(gather(ops, sts, i - 1, od)...)
    return
end

"""
    colstrides(d) -> NTuple

Column-major strides of a shape: `(1, d[1], d[1]*d[2], …)`.

`Val(N)`-specialised because this one runs INSIDE kernels, unlike `bcstrides`
below, which runs at emit time on shapes the graph states.
"""
@inline colstrides(d::NTuple{N,Int}) where {N} = ntuple(k -> begin
        s = 1
        for j in 1:(k - 1)
            s *= d[j]
        end
        s
    end, Val(N))

"""
    bcstrides(od, id) -> NTuple

An operand's effective strides against an output shape: its own column-major
strides, with ZERO on every axis it is broadcast along.

Computed at emit time, from shapes the graph already states, so the kernel does
no shape arithmetic and the dispatch carries no shape objects.
"""
function bcstrides(od::NTuple{N,Int}, id::NTuple{M,Int}) where {N,M}
    st = ntuple(N) do k
        k > M && return 0
        id[k] == 1 && od[k] != 1 && return 0
        s = 1
        for j in 1:(k - 1)
            s *= id[j]
        end
        s
    end
    return st
end
