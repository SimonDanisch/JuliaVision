"""
One generic kernel, so op bodies are plain functions.

Everything elementwise or reducing is already provided for any
`AbstractGPUArray` - and `Mantle.LavaArray <: GPUArraysCore.AbstractGPUArray` -
by broadcasting, `fill!`, and `AcceleratedKernels`' backend-agnostic
`sum`/`prod`/`maximum`/`any`/`all`/`sort`. None of that is reimplemented here.

What is left is the genuinely index-shaped work: convolution, pooling,
resampling, the memory read. Those launch through `launch!`, which supplies the
`@index` boilerplate once so a kernel body is an ordinary `@inline` function of
`(I, args...)` returning the value at `I`. Arguments are passed explicitly
rather than captured, which keeps the bodies concatenable when emit_kernels
fuses a chain.
"""

"""
    materialize(v)
    materialize(rec, backend, v)

Detach a view into a dense array on its own backend.

The second form takes the destination from a `Recycler`, so the copy lands at the
same address on every step. Views are materialised on the hot path — every `cat`
operand that is a slice, the alpha and the mask `step!` carries into the next
frame — and a fresh `similar` each time is exactly the drift that stops a
recorded command sequence from being replayable.
"""
function materialize(v::AbstractArray)
    d = similar(v)
    d .= v
    d
end

function materialize(rec, backend, v::AbstractArray)
    rec === nothing && return materialize(v)
    d = recycle!(rec, backend, eltype(v), size(v))
    d .= v
    d
end

"""
    materialize(ctx, v)

Into the slot the planner reserved for the op currently running.

For an op *output* this is what the two forms above should have been: the
transient is already in the plan, so taking it from the `Recycler` instead
allocates memory that is reserved and then never used. `index.Tensor` did
exactly that, and its 16 gathers were **604 MB** of SAM 2's encoder — the
single largest thing left outside the plan once the fused values and the
`contiguous` copies were in it. Writing them into the reservation costs no
slab at all, because it was already sized for them.

The `(rec, backend, v)` form stays for the callers that have no op id to place
under: a view chain resolved inside `makeview`, and the two slices `step!`
carries between frames.
"""
function materialize(ctx, v::AbstractArray)
    d = dest(ctx, eltype(v), size(v)...)
    d .= v
    d
end

"""
    bidx(a, I)

Index `a` at the output index `I`, collapsing its extent-1 axes. PyTorch aligns
trailing dimensions when broadcasting and these arrays carry the reversed shape,
so trailing becomes leading and this only has to handle the size-1 axes.
"""
@inline function bidx(a, I::CartesianIndex)
    sz = size(a)
    CartesianIndex(ntuple(d -> @inbounds(sz[d] == 1 ? 1 : I[d]), Val(length(sz))))
end

@kernel function ndmap!(f::F, out, args::Vararg{Any,N}) where {F,N}
    I = @index(Global, NTuple)
    @inbounds out[I...] = f(I, args...)
end

"""
    launch!(f, out, args...; backend=get_backend(out))

Evaluate `f(I, args...)` at every index `I` of `out`. N-D dispatch by
construction, so no kernel body divides a runtime value to recover its
coordinates.

Flattening this to a 1-D `ndrange` with the coordinates recovered from a
`CartesianIndices` was tried and reverted: it was **wrong** (the model's matte
went from 2.8e-4 to 0.27 against PyTorch) *and* slower, 31.7 -> 34.6 ms. It
looked promising because the same flattening is worth 6x in Lava's broadcast,
but that win came from removing the per-element div/mod that
`getindex(::Broadcasted, ::Integer)` performs on an N-D tree — not from the
launch geometry, which on its own was only 54 -> 103 GB/s of the 54 -> 332
total. These kernels already take an N-D index and do no such division, so
flattening buys nothing and costs a `CartesianIndices` lookup.
"""
# Flat variant of `ndmap!`, and what `launch!` takes for any multi-dimensional
# linearly-indexable destination.
#
# The decomposition is `Mantle.cart32` over `Mantle.FastDiv32` extents, not
# `CartesianIndices`, and that is the whole reason this variant is worth
# re-measuring: it used to cost N-1 real integer divisions, which is exactly what
# the flat launch was switched off for. A magic-number multiply is ~5 cycles where
# the divide was ~25.
@kernel function ndmap_flat!(f::F, out, sz, n, args::Vararg{Any,N}) where {F,N}
    lin = @index(Global, Linear)
    if lin <= n
        @inbounds out[lin] = f(Mantle.cart32(UInt32(lin) - UInt32(1), sz), args...)
    end
end

# ── Why the flat launch, and what would overturn it ──────────────────────────
#
# This was a switch (`LAUNCH_FLAT`), on by default. It was off first, for a good
# reason that stopped being true; the switch is gone and the winner is inlined
# (`kernel-library-review.md` finding 3, tier two).
#
# The rule is: flatten when the index arithmetic you need is cheaper than the
# fragmentation you are paying for. An N-D `ndrange` gets an N-D workgroup, so
# consecutive lanes stop walking consecutive memory; flattening fixes that but
# `ndmap!` hands `f` a full Cartesian index, so it has to rebuild one — N-1
# divisions where the N-D launch needed none.
#
# Measured that way it lost, 31.69 ms against 28.83. Then `Mantle.FastDiv32` made a
# decomposition a magic-number multiply instead of a division (~5 cycles against
# ~25) and the same A/B, interleaved in one session on SAM 2's encoder, reverses:
#
#     flat = false    p50 264.79 ms
#     flat = true     p50 253.41 ms      11.38 ms, 4.3%
#
# Encoder outputs are bit-identical between the two, and correctness was never
# the question — all 8 graphs verified on this path when it was written, and the
# end-to-end matte was 2.754e-4 against 2.768e-4.
#
# The balance depends on the cost of a division, and that has moved once. If it
# moves again this is the paragraph to re-measure against; the A/B is the two
# branches of `launch!` below, not a global.

"""
    launchgroup(sz) -> Dims

`Mantle.launchgroup`, re-exported so the launch sites here read the same as the
ones in Lava. One definition: the rule about which axis a workgroup fills first
is a property of the backend, and two copies of it would drift.

Note in particular the warning it carries about `kernel(backend, wg)` versus the
`workgroupsize` launch keyword — every launch in this package uses the keyword.
"""
# 256 threads measured best and is the default. A literal rather than a global
# because it was read in exactly one place — here — and a caller holding a
# context goes through `launchgroup(ctx.dev)` instead, which is the same number
# clamped to what the device will actually launch.
@inline launchgroup(sz::Dims, target::Int = 256) = Mantle.launchgroup(sz, target)
@inline launchgroup(ctx::Ctx, sz::Dims) = Mantle.launchgroup(sz, launchgroup(ctx.dev))

"""
    flat4(a) -> (ndrange, W, H, C)

The launch geometry for a rank-4 kernel that indexes through [`coords4`](@ref):
a **1-D** ndrange over `length(a)`, and the three leading extents as `Int32`.

Both come off the same array in one call, so the ndrange and the extents the
kernel divides by cannot drift apart — passing `size(a)` to one and a stale
`W`/`H`/`C` to the other silently reads the wrong elements, and nothing checks it.
"""
@inline flat4(a) = ((length(a),), Int32(size(a, 1)), Int32(size(a, 2)), Int32(size(a, 3)))

"""
    coords4(q::Int32, W, H, C) -> (i, j, c, n)

A 1-based rank-4 coordinate recovered from a **linear** thread index, in Int32.

`@index(Global, NTuple)` is the obvious way to write a rank-4 kernel and costs
**3.5x** on this backend. `KA.expand` recovers the block coordinate by indexing a
`CartesianIndices` with a flattened index — `2(N-1)` integer divisions by
*runtime* extents, in Int64. `Lava.directdispatch` removes that for rank <= 3 by
reading the dispatch builtins directly, but a rank-4 block grid with a non-unit
axis past the third is flattened by `pad_to_3d` and cannot use it.

Doing the same recovery ourselves, in Int32, over a 1-D ndrange gets it back.
A 4-D copy of 1920x1152x4, measured interleaved:

    rank-4 @index(Global, NTuple)   1.036 ms    68 GB/s
    rank-1 @index(Global, Linear)   0.293      241   3.53x
    1-D ndrange + coords4           0.310      228   3.34x

within 6% of a bare linear copy. This is why `grid_sample2d_kernel!` looked like
a slow gather: a plain copy over its ndrange cost 88% of it, and the sampling was
never the expense. See [[ka-ntuple-index-costs-3.6x]] in the project notes.

**Indexing stays multi-dimensional** — `out[i, j, c, n]`, not `out[q]` — so this
is correct for a strided view as well; only `length(a) == W*H*C*N` is assumed,
which `flat4` guarantees by construction.
"""
@inline function coords4(q::Int32, W::Int32, H::Int32, C::Int32)
    i = q % W;  r = q ÷ W
    j = r % H;  s = r ÷ H
    c = s % C;  n = s ÷ C
    (Int(i) + 1, Int(j) + 1, Int(c) + 1, Int(n) + 1)
end

"""
    launch!(ctx, f, out, args...)

The form every op body and kernel entry point uses: the backend and the launch
probe both come off the context, so a launch site needs no argument of its own to
be measurable. See [`Diagnostics`](@ref).
"""
@inline launch!(ctx::Ctx, f::F, out, args...) where {F} =
    launch!(f, out, args...; backend = ctx.backend, probe = ctx.diag.launches)

function launch!(f::F, out, args...; backend=KernelAbstractions.get_backend(out),
                 probe=nothing) where {F}
    p = probe
    if p !== nothing && ndims(out) <= 1
        sz = size(out); wg = Mantle.staticgroup(sz)
        grp = ntuple(i -> cld(sz[i], wg[i]), length(sz))
        c, _ = get(p, (sz, wg), (0, grp))
        p[(sz, wg)] = (c + 1, grp)
    end
    if ndims(out) > 1 && IndexStyle(out) === IndexLinear()
        n = length(out)
        ndmap_flat!(backend)(f, out, map(Mantle.FastDiv32, size(out)), n, args...; ndrange=n)
    else
        sz = size(out)
        # Workgroup in the kernel's TYPE, via `staticgroup` so it never trips
        # `Mantle.WORKGROUP_FALLBACK`. That keeps the index arithmetic
        # compile-time constant, which is worth ~2x on these kernels. The cost is
        # a separate SPIR-V module per (body, workgroup shape); measured below.
        ndmap!(backend, Mantle.staticgroup(sz))(f, out, args...; ndrange=sz)
    end
    out
end

"""
    launched!(f, out, args...)

`launch!` followed by a synchronize. Only for the verification path; the
recorded command stream never synchronizes between passes.
"""
function launched!(f::F, out, args...) where {F}
    backend = KernelAbstractions.get_backend(out)
    launch!(f, out, args...; backend)
    KernelAbstractions.synchronize(backend)
    out
end

# Scalar bodies. Shared with the fused kernels, which are literally these
# concatenated with SSA renaming.
@inline relu_(x) = max(x, zero(x))
@inline sigmoid_(x) = one(x) / (one(x) + exp(-x))
@inline hardsigmoid_(x) = clamp(x / 6 + oftype(x, 0.5), zero(x), one(x))
