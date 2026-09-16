"""
The ops that move elements rather than compute them, as declared dispatches.

`repeat` and a reduction are the two shapes of index arithmetic a graph needs
that broadcasting cannot express: one reads its source modulo the source extent,
the other reads a whole axis per output element. Both are one kernel over the
OUTPUT, so the destination is the planned buffer and there is no intermediate.

What this replaces is not a kernel but an allocation. `runop!`'s `repeat` was a
`launch!` into `dest`, which was already right; its `sum` was `Base.sum(a;
dims)`, which GPUArrays answers by allocating the result and, for a non-trivial
`dims`, a mapreduce workspace as well. Neither is in any plan.
"""

"""
    tilecopy!(out, od, a, id)

`out = repeat(a, reps)`: element `i` of the output reads its source coordinate
modulo the source extent on every axis.

Outer repetition only, which is what `aten::repeat` means. `runop!` wrote the
same arithmetic as `mod1(I[k], sz[k])` over a Cartesian index; the 0-based form
here is the same thing with the index decomposed by hand, because a macro-free
kernel gets a linear id.
"""
function tilecopy!(out, od::NTuple{N,Int}, a, id::NTuple{N,Int}) where {N}
    i = KI.get_global_id().x
    i <= prod(od) || return
    ist = colstrides(id)
    off = 0
    r = i - 1
    @inbounds for k in 1:N
        off += ((r % od[k]) % id[k]) * ist[k]
        r = r ÷ od[k]
    end
    @inbounds out[i] = a[off + 1]
    return
end

"""
    sumdims!(out, od, a, id, f)

`out = sum(f, a; dims)`, one thread per OUTPUT element.

`od` is the input's shape with a `1` on every reduced axis, so the reduced
extents are exactly the axes the output holds as one, and the kernel needs no
separate list of dims. The output buffer's own shape may have them dropped
instead; both orders are column-major over the same elements, so the linear
index is the same either way.

**The accumulator is `accum(eltype(a))` and that is a contract, not a
preference.** `foldincasts` removes a widening cast in front of a reduction on
the grounds that "its accumulator was already the wide type"; a reduction that
accumulated in the operand's type would make that fold change the answer. A
`Float16` sum over a long axis also saturates at 65504, which is silent.

`f` is the map step `foldpremap` folded in, `identity` when there is none: the
same argument-not-wrapper decision `ew!` makes, so a premapped sum is still one
dispatch and one pass.
"""
function sumdims!(out, od::NTuple{N,Int}, a, id::NTuple{N,Int}, f, scale = nothing) where {N}
    i = KI.get_global_id().x
    i <= prod(od) || return
    ist = colstrides(id)
    # The output coordinate, placed on the axes that survive. A reduced axis has
    # `od[k] == 1`, so it contributes nothing here and everything below.
    base = 0
    r = i - 1
    @inbounds for k in 1:N
        base += (r % od[k]) * ist[k]
        r = r ÷ od[k]
    end
    rd = ntuple(k -> od[k] == 1 ? id[k] : 1, Val(N))
    acc = zero(accum(eltype(a)))
    @inbounds for j in 0:(prod(rd) - 1)
        off = base
        q = j
        for k in 1:N
            off += (q % rd[k]) * ist[k]
            q = q ÷ rd[k]
        end
        acc += f(a[off + 1])
    end
    # `scale` is `mean.dim`: the reduction divided by how many elements it summed.
    # In the SAME kernel, and folded into the accumulator's type rather than a
    # second elementwise pass, because the count is a host scalar the emit
    # already knows — a pass to multiply by a constant is a whole extra
    # round-trip of the result through memory. `nothing` is `sum`, and the
    # branch is on a type, so neither form pays for the other.
    @inbounds out[i] = scale === nothing ? acc : acc * scale
    return
end

"""
    arange!(out, n, start, step)

`out[i] = start + (i - 1) * step`, for `n` elements.

The scalars arrive already converted to the output's element type, because the
length is not derivable from them here: aten's is `ceil((end - start) / step)`,
which disagrees with a Julia range's whenever the step is not 1, and the host is
where that was decided. See the `arange.start_step` emit.
"""
function arange!(out, n::Int, start, step)
    i = KI.get_global_id().x
    i <= n || return
    @inbounds out[i] = start + (i - 1) * step
    return
end

# ── advanced indexing ────────────────────────────────────────────────────────
#
# `index.Tensor` was two host-side shapes: an outer product, which Julia's
# `view` already computes and which could stay on the device, and torch's
# PAIRED broadcast, which had no Julia spelling and so gathered on the HOST --
# `collect(vec(x))[vec(lin)]`, a full round trip. That is exactly what a
# recorded plan forbids, so the paired form could never be replayed.
#
# Declared, both are a gather, and the index arrays are ordinary read operands.
# Nothing comes to the host and nothing needs the `.+ 1` pass the interpreted
# path ran to make torch's 0-based values into Julia ones: a 0-based value IS
# the coordinate, and the one `+ 1` is on the final linear index.

"""
The source coordinate for output axis `k` at output coordinate `c` (0-based).

`c` itself where the axis is sliced, and `idxs[j][c + 1]` where it is the `j`-th
indexed axis. A recursive walk rather than a lookup because the index arrays
have different types; it unrolls, so the kernel has no tuple indexing at run
time. Both signatures are typed identically so `Tuple{}` is strictly more
specific — untyped trailing arguments make the two AMBIGUOUS, which inside a
kernel is a `jl_f_throw_methoderror` reported as "method lookup failure".
"""
@inline srccoord(::Tuple{}, ::Tuple{}, k::Int, c::Int) = c
@inline function srccoord(idxs::Tuple, dims::Tuple, k::Int, c::Int)
    first(dims) == k && return Int(@inbounds first(idxs)[c + 1])
    return srccoord(Base.tail(idxs), Base.tail(dims), k, c)
end

"""
    indexgather!(out, od, x, xd, idxs, dims)

`out = x[..., idxs[1], ..., idxs[2], ...]` where the index tensors form an OUTER
PRODUCT over the axes they index — the separable case, and the only one Julia's
`view` computes the same way.

`out` has `x`'s rank with each indexed axis replaced by that index's length, so
one linear pass over `out` decomposes into output coordinates and each one maps
through `srccoord`.
"""
function indexgather!(out, od::NTuple{N,Int}, x, xd::NTuple{N,Int},
                      idxs::Tuple, dims::Tuple) where {N}
    i = KI.get_global_id().x
    i <= prod(od) || return
    xst = colstrides(xd)
    off = 0
    r = i - 1
    @inbounds for k in 1:N
        c = r % od[k]
        r = r ÷ od[k]
        off += srccoord(idxs, dims, k, c) * xst[k]
    end
    @inbounds out[i] = x[off + 1]
    return
end

"""The contribution of each paired index tensor to the source offset."""
@inline pairedoffset(::Tuple{}, ::Tuple{}, xst::Tuple, lin::Int,
                     od::NTuple{N,Int}, j::Int) where {N} = 0
@inline pairedoffset(idxs::Tuple, sts::Tuple, xst::Tuple, lin::Int,
                     od::NTuple{N,Int}, j::Int) where {N} =
    Int(@inbounds first(idxs)[bcindex(lin, od, first(sts))]) * @inbounds(xst[j]) +
    pairedoffset(Base.tail(idxs), Base.tail(sts), xst, lin, od, j + 1)

"""
    indexpaired!(out, od, x, xd, idxs, sts)

Torch's advanced indexing with more than one index tensor and no outer-product
structure: the index arrays are broadcast against EACH OTHER, and each output
position takes one element of `x`.

Every axis of `x` is indexed here, so the source coordinate is complete — the
mixed case is separable and handled by [`indexgather!`](@ref). `sts` is each
index tensor's effective strides against `od`, computed at emit time, so the
broadcast costs a `bcindex` and no materialised copy.
"""
function indexpaired!(out, od::NTuple{N,Int}, x, xd::Tuple, idxs::Tuple,
                      sts::Tuple) where {N}
    i = KI.get_global_id().x
    i <= prod(od) || return
    off = pairedoffset(idxs, sts, colstrides(xd), i - 1, od, 1)
    @inbounds out[i] = x[off + 1]
    return
end

"""
    catcopy!(out, od, part, pd, d, off)

One input of a `cat` into its slice of the output: `out[..., off+1:off+pd[d], ...] = part`.

A kernel and not a `slice`, because a `cat` along anything but the last axis is
not contiguous in the output — the slice a `Mantle.slice` names is a linear
range, and this one is strided. One dispatch per input, each writing a disjoint
region, which is why the graph may run them concurrently: the walk reports `out`
written by each and `Barriers` finds no hazard between them.
"""
function catcopy!(out, od::NTuple{N,Int}, part, pd::NTuple{N,Int},
                  ::Val{D}, off::Int) where {N,D}
    i = KI.get_global_id().x
    i <= prod(pd) || return
    ost = colstrides(od)
    o = 0
    r = i - 1
    @inbounds for k in 1:N
        c = r % pd[k]
        r = r ÷ pd[k]
        o += (k == D ? c + off : c) * ost[k]
    end
    @inbounds out[o + 1] = part[i]
    return
end

"""
    stridedcopy!(out, od, a, ast, off)

`out[i] = a[off + Σ_k c_k * ast[k]]` where `c` is `i`'s coordinate in `od`.

One kernel for every view whose read moves elements, because each of them is the
same thing — the parent read at a strided coordinate — and only the strides
differ:

  * a PERMUTE reorders the parent's strides;
  * an EXPAND gives a repeated axis stride 0, which is what `bcstrides` already
    computes;
  * a SLICE adds an offset and multiplies one axis's stride by the step;
  * a SELECT adds an offset and drops the axis.

Four ops, four ways of filling `ast` and `off` on the host, one kernel and one
pass. `contiguous` did the permute case with `permutedims!` and the rest stayed
lazy Julia wrappers; declared, a wrapper is not a thing a kernel can be handed,
so the copy is a pass the placer can alias like any other.
"""
function stridedcopy!(out, od::NTuple{N,Int}, a, ast::NTuple{N,Int},
                      off::Int) where {N}
    i = KI.get_global_id().x
    i <= prod(od) || return
    o = off
    r = i - 1
    @inbounds for k in 1:N
        o += (r % od[k]) * ast[k]
        r = r ÷ od[k]
    end
    @inbounds out[i] = a[o + 1]
    return
end
