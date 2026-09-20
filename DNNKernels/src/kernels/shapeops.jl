"""
The ops that move elements rather than compute them, as declared dispatches.

`repeat` and a reduction are the two shapes of index arithmetic a graph needs
that broadcasting cannot express: one reads its source modulo the source extent,
the other reads a whole axis per output element. Both are one kernel over the
OUTPUT, so the destination is the planned buffer and there is no intermediate.

What these replace is not a kernel but an allocation: `Base.sum(a; dims)` is
answered by GPUArrays allocating the result and, for a non-trivial `dims`, a
mapreduce workspace as well. Neither is in any plan.
"""

"""
    rowbias!(out, bias, n, Val(rows))

Add a dense matrix's row bias in place.  GEMM output is column-major here, so
the row index is simply the linear index modulo `rows`; keeping this as a
dedicated kernel avoids the rank-wide coordinate decomposition in the generic
broadcast kernel for what is one modulo and two contiguous memory accesses.

`rows` is a `Val` because it is a graph shape.  GPU compilers can therefore
strength-reduce the modulo, including the common power-of-two widths, without
making this kernel backend-specific.
"""
function rowbias!(out, bias, n::Int, ::Val{ROWS}) where {ROWS}
    i = KI.get_global_id().x
    i <= n || return
    @inbounds out[i] = out[i] + bias[(i - 1) % ROWS + 1]
    return
end

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
    folddims!(out, od, a, id, f, combine, init, post = identity)

`out = post(reduce(combine, map(f, a); dims))`, one thread per OUTPUT element.

`od` is the input's shape with a `1` on every reduced axis, so the reduced
extents are exactly the axes the output holds as one, and the kernel needs no
separate list of dims. The output buffer's own shape may have them dropped
instead; both orders are column-major over the same elements, so the linear
index is the same either way.

**`init` carries the accumulator's type and that is a contract, not a
preference.** `foldincasts` removes a widening cast in front of a reduction on
the grounds that "its accumulator was already the wide type"; a reduction that
accumulated in the operand's type would make that fold change the answer. A
`Float16` sum over a long axis also saturates at 65504, which is silent. The
emit states it once, in `foldinit`.

`f` is the map step `foldpremap` folded in, `identity` when there is none: the
same argument-not-wrapper decision `ew!` makes, so a premapped reduction is
still one dispatch and one pass.

`combine` is why this is ONE kernel and not four. `sum`, `mean`, `prod` and
`any` differ in the operator and its identity and in nothing else — same index
arithmetic, same loop, same store — so writing `+` into it leaves
`prod.dim_int` and `any.dim` with no declared form at all.

`post` is the accumulator's last step, applied once per output element inside
this kernel rather than as a pass over the result. A `scale` value covers
`mean`'s division by the count and not a norm's `sqrt`, which wants the same
place, so the function is the one mechanism and `mean` passes the multiply. `identity` is every reduction that has no such step,
and it costs nothing: it is a type, so the store specialises on it.
"""
function folddims!(out, od::NTuple{N,Int}, a, id::NTuple{N,Int}, f, combine, init,
                   post = identity) where {N}
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
    acc = init
    @inbounds for j in 0:(prod(rd) - 1)
        off = base
        q = j
        for k in 1:N
            off += (q % rd[k]) * ist[k]
            q = q ÷ rd[k]
        end
        acc = combine(acc, f(a[off + 1]))
    end
    # In the SAME kernel, and in the accumulator's type rather than a second
    # elementwise pass: `mean`'s count and a norm's order are host scalars the
    # emit already knows, and a pass to apply one is a whole extra round-trip of
    # the result through memory.
    @inbounds out[i] = post(acc)
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
    gatherdim!(out, od, a, ad, idx, d)

`aten::gather` along one axis: `out[c] = a[c with c[d] = idx[c]]`.

The index has the OUTPUT's shape and supplies the coordinate on axis `d` only;
every other axis takes its own coordinate straight through. That is what
separates this from [`indexgather!`](@ref), where an index array is one
coordinate list for a whole axis and the indexed axes form an outer product:
there the index varies along its own axis, here it varies along all of them.

`idx` holds torch's 0-based values, so it goes into the offset unshifted and the
one `+ 1` is on the final linear index, as everywhere else here.

The interpreted path only ever did the case where every axis but `d` is a
singleton, which makes the whole thing `a[idx]` and needs no kernel. It
`collect`ed both operands to the host and indexed there, so it could not be
recorded. The general form is the same index arithmetic as the rest of this file
and no harder to write than the restriction was to state.
"""
function gatherdim!(out, od::NTuple{N,Int}, a, ad::NTuple{N,Int}, idx,
                    d::Int) where {N}
    i = KI.get_global_id().x
    i <= prod(od) || return
    ast = colstrides(ad)
    off = 0
    r = i - 1
    @inbounds for k in 1:N
        c = r % od[k]
        r = r ÷ od[k]
        off += (k == d ? Int(idx[i]) : c) * ast[k]
    end
    @inbounds out[i] = a[off + 1]
    return
end

"""
    blockcopy!(out, od, part, pd, off)

`part` into the box of `out` that starts at `off`: `out[off .+ c] = part[c]` for
every coordinate `c` of `part`.

A kernel and not a `slice`, because a box is not a linear range unless it spans
every axis but the last — the window a `Mantle.slice` names IS a linear range,
and this one is strided.

`off` is one offset PER AXIS, which is what makes this one kernel for three ops:

  * `cat` offsets the concatenated axis and nothing else;
  * `slice_scatter` offsets the scattered axis and nothing else;
  * `constant_pad_nd` offsets every axis at once, by its low pad.

It was `Val{D}, off::Int` for the first of those, with `k == D ? c + off : c`
inside the loop — the tuple form has no branch at all, so the general kernel is
also the cheaper one.

One dispatch per part, each writing a disjoint region, which is why a graph may
run several concurrently: the walk reports `out` written by each and `Barriers`
finds no hazard between them.
"""
function blockcopy!(out, od::NTuple{N,Int}, part, pd::NTuple{N,Int},
                    off::NTuple{N,Int}) where {N}
    i = KI.get_global_id().x
    i <= prod(pd) || return
    ost = colstrides(od)
    o = 0
    r = i - 1
    @inbounds for k in 1:N
        o += ((r % pd[k]) + off[k]) * ost[k]
        r = r ÷ pd[k]
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

"""
    stridedcopy32!(out, exts, a, ast, off, n)

[`stridedcopy!`](@ref) with the linear-to-Cartesian decomposition in 32 bits.

**The division chain is the cost of this kernel, not the traffic.** At rank 6 the
form above is six emulated 64-bit divisions and six modulos per element;
`Mantle.cart32` over `FastDiv32` extents is a high multiply and a shift each.
That is the same measurement Lava's broadcast records for the same mistake —
~15 us per division per 2.36 M elements, linear in the rank — and SAM 2's
encoder copies 940 MiB through this kernel at rank 4 and 6.

`exts` is `Mantle.broadcastextents(od)` and `n` the element count, because the
extents no longer arrive in a form the guard can multiply. Only for a copy whose
largest index fits `Int32`, which is what [`stridedcopydispatch!`](@ref) checks.
"""
function stridedcopy32!(out, exts, a, ast::NTuple{N,Int32}, off::Int32,
                        n::Int32) where {N}
    i = KI.get_global_id().x
    i <= n || return
    c = M.cart32(UInt32(i - 1), exts)
    o = off
    @inbounds for k in 1:N
        o += Int32(c[k] - 1) * ast[k]
    end
    @inbounds out[i] = a[o + Int32(1)]
    return
end

"""
    transposecast_f32_f16!(out, src, M, N)

Transpose each dense `N × M` plane into an `M × N` plane while narrowing
Float32 to Float16.  The padded shared tile makes both global-memory directions
coalesced; the generic strided elementwise path necessarily leaves one side of
this transpose strided.
"""
@kernel cpu=false function transposecast_f32_f16!(out, @Const(src),
                                                   M::Int32, N::Int32)
    tile = @localmem Float32 (33, 32)
    tx, ty = @index(Local, NTuple)
    gx, gy, gz = @index(Group, NTuple)
    n0 = Int32(gx - 1) * Int32(32)
    m0 = Int32(gy - 1) * Int32(32)
    base = Int32(gz - 1) * M * N
    @inbounds begin
        for j in Int32(0):Int32(7)
            n = n0 + Int32(tx)
            m = m0 + Int32(ty) + Int32(4) * j
            tile[tx, ty + 4j] = n <= N && m <= M ?
                src[base + (m - Int32(1)) * N + n] : 0.0f0
        end
        @synchronize
        for j in Int32(0):Int32(7)
            m = m0 + Int32(tx)
            n = n0 + Int32(ty) + Int32(4) * j
            m <= M && n <= N &&
                (out[base + (n - Int32(1)) * M + m] =
                    Float16(tile[ty + 4j, tx]))
        end
    end
end

# Literal shared-memory element types are required by the GPU compiler, so the
# two same-type transpose kernels are generated rather than parameterised by a
# run-time type value.
for T in (Float16, Float32)
    @eval @kernel cpu=false function $(Symbol("transposecopy_", nameof(T), "!"))(
            out, @Const(src), M::Int32, N::Int32)
        tile = @localmem $(nameof(T)) (33, 32)
        tx, ty = @index(Local, NTuple)
        gx, gy, gz = @index(Group, NTuple)
        n0 = Int32(gx - 1) * Int32(32)
        m0 = Int32(gy - 1) * Int32(32)
        base = Int32(gz - 1) * M * N
        @inbounds begin
            for j in Int32(0):Int32(7)
                n = n0 + Int32(tx)
                m = m0 + Int32(ty) + Int32(4) * j
                tile[tx, ty + 4j] = n <= N && m <= M ?
                    src[base + (m - Int32(1)) * N + n] : zero($(nameof(T)))
            end
            @synchronize
            for j in Int32(0):Int32(7)
                m = m0 + Int32(tx)
                n = n0 + Int32(ty) + Int32(4) * j
                m <= M && n <= N &&
                    (out[base + (n - Int32(1)) * M + m] = tile[ty + 4j, tx])
            end
        end
    end
end

transposecopykernel(::Type{Float16}) = transposecopy_Float16!
transposecopykernel(::Type{Float32}) = transposecopy_Float32!

"""
    transposeadd_f16_f32_f32!(out, a, b, M, N)

Transpose matching dense planes while adding an fp16 and an fp32 source into
an fp32 destination.  Positional embedding addition has precisely this shape:
the generic elementwise kernel otherwise performs a full coordinate division
chain and leaves both reads strided.  One fp32 shared tile holds the sum, so
both reads and the destination write are coalesced.
"""
@kernel cpu=false function transposeadd_f16_f32_f32!(out, @Const(a), @Const(b),
                                                      M::Int32, N::Int32)
    tile = @localmem Float32 (33, 32)
    tx, ty = @index(Local, NTuple)
    gx, gy, gz = @index(Group, NTuple)
    n0 = Int32(gx - 1) * Int32(32)
    m0 = Int32(gy - 1) * Int32(32)
    base = Int32(gz - 1) * M * N
    @inbounds begin
        for j in Int32(0):Int32(7)
            n = n0 + Int32(tx)
            m = m0 + Int32(ty) + Int32(4) * j
            tile[tx, ty + 4j] = n <= N && m <= M ?
                Float32(a[base + (m - Int32(1)) * N + n]) +
                Float32(b[base + (m - Int32(1)) * N + n]) : 0.0f0
        end
        @synchronize
        for j in Int32(0):Int32(7)
            m = m0 + Int32(tx)
            n = n0 + Int32(ty) + Int32(4) * j
            m <= M && n <= N &&
                (out[base + (n - Int32(1)) * M + m] = tile[ty + 4j, tx])
        end
    end
end

"""
Transpose one fp16 plane while adding an already destination-ordered fp16 plane.
The transposed source is staged; the dense operand and destination are touched
coalesced after the tile turns. This is the residual-add layout between SAM 2's
window stages.
"""
@kernel cpu=false function transposeadd_f16_dense_f16!(out, @Const(a), @Const(b),
                                                        M::Int32, N::Int32)
    tile = @localmem Float16 (33, 32)
    tx, ty = @index(Local, NTuple)
    gx, gy, gz = @index(Group, NTuple)
    n0 = Int32(gx - 1) * Int32(32)
    m0 = Int32(gy - 1) * Int32(32)
    base = Int32(gz - 1) * M * N
    @inbounds begin
        for j in Int32(0):Int32(7)
            n = n0 + Int32(tx)
            m = m0 + Int32(ty) + Int32(4) * j
            tile[tx, ty + 4j] = n <= N && m <= M ?
                a[base + (m - Int32(1)) * N + n] : zero(Float16)
        end
        @synchronize
        for j in Int32(0):Int32(7)
            m = m0 + Int32(tx)
            n = n0 + Int32(ty) + Int32(4) * j
            if m <= M && n <= N
                i = base + (n - Int32(1)) * M + m
                out[i] = tile[ty + 4j, tx] + b[i]
            end
        end
    end
end
