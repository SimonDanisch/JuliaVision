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
# Flat variant of `ndmap!`; see `LAUNCH_FLAT`.
@kernel function ndmap_flat!(f::F, out, cis, n, args::Vararg{Any,N}) where {F,N}
    lin = @index(Global, Linear)
    if lin <= n
        @inbounds out[lin] = f(Tuple(cis[lin]), args...)
    end
end

"""
    LAUNCH_FLAT[] :: Bool

Launch `launch!` kernels over a flat range instead of `size(out)`.

**Off, because for these kernels it is 10% slower** — 31.69 ms against 28.83,
interleaved A/B in one session. It is *correct* (all 8 graphs verify, and the
end-to-end matte is 2.754e-4 against 2.768e-4), so this is a performance
decision, not a correctness one.

Worth understanding, because the same flattening is worth ~3 ms elsewhere — the
convolution epilogue, im2col and both GEMM epilogues all gained from it. The
difference is what the decomposition costs:

  * Those kernels needed only one or two divisions, hand-written for their
    access pattern (`conv_epilogue_kernel!` recovers pixel/channel/image with
    two), and their fast `ndrange` dimension was shorter than a warp — a
    15-wide `OW` means a warp spans 15 contiguous outputs and then jumps.
  * `ndmap!` hands `f` a full Cartesian index, so flattening costs a complete
    `CartesianIndices` lookup — N-1 divisions — and `f` then re-derives its own
    offsets from it. That exceeds the coalescing it buys.

So the rule is not "always flatten": flatten when the index arithmetic you need
is cheaper than the fragmentation you are paying for.
"""
const LAUNCH_FLAT = Ref(false)

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
    LAUNCH_PROBE[] :: Union{Nothing,Dict}

Set to a dict to record every `launch!` as `(ndrange, workgroup) => (count, groups)`.
Off by default and free when off.

For finding launches that do not fill the device. A grid of 64 workgroups on a
48-SM card leaves most of it idle however good the kernel is, and that is
invisible in a per-op timing table — it shows up only as one op being
inexplicably slow. `Lava.with_dispatch_timing` says *which dispatch*; this says
*which launch site and what shape*.
"""
const LAUNCH_PROBE = Ref{Any}(nothing)

function launch!(f::F, out, args...; backend=KernelAbstractions.get_backend(out)) where {F}
    p = LAUNCH_PROBE[]
    if p !== nothing && !(LAUNCH_FLAT[] && ndims(out) > 1)
        sz = size(out); wg = Lava.staticgroup(sz)
        grp = ntuple(i -> cld(sz[i], wg[i]), length(sz))
        c, _ = get(p, (sz, wg), (0, grp))
        p[(sz, wg)] = (c + 1, grp)
    end
    if LAUNCH_FLAT[] && ndims(out) > 1 && IndexStyle(out) === IndexLinear()
        n = length(out)
        ndmap_flat!(backend)(f, out, CartesianIndices(out), n, args...; ndrange=n)
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
