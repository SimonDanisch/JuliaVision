"""
Scratch for kernels that need working storage the graph does not name.

`plan.jl` packs every buffer the *graph* declares into one slab, but an
implementation is free to want more — the tensor-core convolution needs an
im2col matrix, an fp32 GEMM destination and, when the reduction is split, a
plane per split. None of those appear in the exported graph, so none of them are
in the plan, and allocating them per call is not a small cost: it drove Lava's
pool into repeated `reclaim on OOM retry`, which idles the GPU, and made the
tensor-core path *slower* than the scalar one it replaces (2.05 ms against
0.34 ms on the 120x64 layer).

A bump pointer reset at the top of each op is enough. Ops run one after another
and nothing here outlives the op that asked for it, so successive ops reuse the
same bytes; the buffer settles at the high-water mark after a few steps and
never grows again.
"""

mutable struct Workspace
    buf::Any            # device Vector{UInt8}, or nothing until first use
    used::Int
    backend::Any
    retired::Vector{Any}
end

Workspace(backend) = Workspace(nothing, 0, backend, Any[])

"""
Start a fresh op: everything handed out before is free again.

Also drops the buffers a *previous* op retired. Retention is only needed within
the op that grew the workspace — a convolution takes `col`, then `C`, then the
split planes, and a grow on the second call would strand the first. By the time
the next op starts, every dispatch that could reference the old buffer has been
recorded, and `pin_leaves!` has pinned it into the batch, so the batch holds it
alive until the GPU is done with it.

Never clearing this was expensive: growth is geometric (1.5x), so going from
1 MB to ~1 GB is ~18 reallocations and the retained set sums to roughly *twice*
the final size. On SAM 2's encoder that was gigabytes of dead buffers held for
the lifetime of the model.
"""
function reset!(ws::Workspace)
    isempty(ws.retired) || empty!(ws.retired)
    ws.used = 0
    ws
end
reset!(::Nothing) = nothing

"""
    scratch!(ws, T, dims...) -> array

An uninitialised `T` array of shape `dims` inside the workspace. Alignment is
the slab's 256 bytes so `GPUArrays.derive` lands on an exact element offset.
"""
function scratch!(ws::Workspace, backend, ::Type{T}, dims::Integer...) where {T}
    off = cld(ws.used, 256) * 256
    need = off + prod(dims) * sizeof(T)
    if ws.buf === nothing || need > length(ws.buf)
        # Keep the old buffer alive rather than dropping it. Arrays already
        # handed out this op point into it and their dispatches are already
        # recorded — a convolution takes `col`, then `C`, then the split planes,
        # so a grow on the second or third call would strand the first. Letting
        # the finalizer run under them is a use-after-free that surfaces much
        # later as `sync_access!: buffer is not ALIVE`. Syncing instead of
        # retaining does *not* fix it: the stranded array is still in use by a
        # dispatch this op has yet to record. Growth is geometric and stops after
        # warm-up, so the retained set stays tiny.
        ws.buf === nothing || push!(ws.retired, ws.buf)
        ws.buf = KernelAbstractions.allocate(backend, UInt8, max(need + need ÷ 2, 1 << 20))
    end
    ws.used = need
    # `slabview`, not `GPUArrays.derive` directly: the CPU backend's buffer is an
    # ordinary `Vector{UInt8}` and has no `derive` method (see execute.jl).
    slabview(T, ws.buf, dims, off)
end

# Without a workspace (the CPU verification path, and any direct kernel call)
# fall back to a plain allocation. Same results, just not reused.
scratch!(::Nothing, backend, ::Type{T}, dims::Integer...) where {T} =
    KernelAbstractions.allocate(backend, T, dims...)

"""
    scratch!(ctx, T, dims...) -> array

The form the kernel entry points use. The workspace and the backend are both on
the context, so this is one argument where `scratch!(ws, backend, T, …)` was two
— the same trade the entry points themselves make.
"""
@inline scratch!(ctx::Ctx, ::Type{T}, dims::Integer...) where {T} =
    scratch!(ctx.ws, ctx.backend, T, dims...)
