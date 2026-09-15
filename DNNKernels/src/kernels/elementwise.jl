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

Arity-specialised rather than variadic over a tuple of operands, and the reason
is `Mantle.resolve`: it is applied per element of a dispatch's argument tuple, so
a tuple-of-resources argument would arrive as a tuple of unresolved handles. One
method per operand count keeps every resource a top-level argument, which is also
what makes `use(p, x; read = true)` able to name each one.
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

"""One operand: `out .= f.(a)`. `clone` is this with `f = identity`."""
function ew1!(out, od::NTuple{N,Int}, a, sa::NTuple{N,Int}, f) where {N}
    i, ok = ewguard(od)
    ok || return
    @inbounds out[i] = f(a[bcindex(i - 1, od, sa)])
    return
end

"""Two operands: `out .= f.(a, b)`."""
function ew2!(out, od::NTuple{N,Int}, a, sa::NTuple{N,Int}, b, sb::NTuple{N,Int}, f) where {N}
    i, ok = ewguard(od)
    ok || return
    @inbounds out[i] = f(a[bcindex(i - 1, od, sa)], b[bcindex(i - 1, od, sb)])
    return
end

"""Three operands, which is what a folded bias-and-activation group needs."""
function ew3!(out, od::NTuple{N,Int}, a, sa::NTuple{N,Int}, b, sb::NTuple{N,Int},
              c, sc::NTuple{N,Int}, f) where {N}
    i, ok = ewguard(od)
    ok || return
    @inbounds out[i] = f(a[bcindex(i - 1, od, sa)], b[bcindex(i - 1, od, sb)],
                         c[bcindex(i - 1, od, sc)])
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
