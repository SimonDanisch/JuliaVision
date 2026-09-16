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
function sumdims!(out, od::NTuple{N,Int}, a, id::NTuple{N,Int}, f) where {N}
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
    @inbounds out[i] = acc
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
