"""
One generic kernel, so op bodies are plain functions.

Everything elementwise or reducing is already provided for any
`AbstractGPUArray` - and `Lava.LavaArray <: GPUArraysCore.AbstractGPUArray` -
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
# The decomposition is `Lava.cart32` over `Lava.FastDiv32` extents, not
# `CartesianIndices`, and that is the whole reason this variant is worth
# re-measuring: it used to cost N-1 real integer divisions, which is exactly what
# the flat launch was switched off for. A magic-number multiply is ~5 cycles where
# the divide was ~25.
@kernel function ndmap_flat!(f::F, out, sz, n, args::Vararg{Any,N}) where {F,N}
    lin = @index(Global, Linear)
    if lin <= n
        @inbounds out[lin] = f(Lava.cart32(UInt32(lin) - UInt32(1), sz), args...)
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
# Measured that way it lost, 31.69 ms against 28.83. Then `Lava.FastDiv32` made a
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
    LAUNCH_GROUP[] :: Int

Threads per workgroup `launch!` asks for. 256 measured best; see `launchgroup`.
"""
const LAUNCH_GROUP = Ref(256)

"""
    launchgroup(sz) -> Dims

`Lava.launchgroup`, re-exported so the launch sites here read the same as the
ones in Lava. One definition: the rule about which axis a workgroup fills first
is a property of the backend, and two copies of it would drift.

Note in particular the warning it carries about `kernel(backend, wg)` versus the
`workgroupsize` launch keyword — every launch in this package uses the keyword.
"""
@inline launchgroup(sz::Dims, target::Int = LAUNCH_GROUP[]) = Lava.launchgroup(sz, target)

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
        sz = size(out); wg = Lava.staticgroup(sz)
        grp = ntuple(i -> cld(sz[i], wg[i]), length(sz))
        c, _ = get(p, (sz, wg), (0, grp))
        p[(sz, wg)] = (c + 1, grp)
    end
    if ndims(out) > 1 && IndexStyle(out) === IndexLinear()
        n = length(out)
        ndmap_flat!(backend)(f, out, map(Lava.FastDiv32, size(out)), n, args...; ndrange=n)
    else
        sz = size(out)
        # Workgroup in the kernel's TYPE, via `staticgroup` so it never trips
        # `Lava.WORKGROUP_FALLBACK`. That keeps the index arithmetic
        # compile-time constant, which is worth ~2x on these kernels. The cost is
        # a separate SPIR-V module per (body, workgroup shape); measured below.
        ndmap!(backend, Lava.staticgroup(sz))(f, out, args...; ndrange=sz)
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
