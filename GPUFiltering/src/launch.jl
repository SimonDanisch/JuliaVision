# Where an operation's kernels go.
#
# Every operation here is written once, as kernel launches through `launch!`.
# Its first argument says where they go: a device runs them now, and a Mantle
# graph declares them as nodes of the graph, ordered by what they read and write,
# to run on every `run!` of its plan. The operation itself does not know which: it
# launches, and never synchronizes or reads a result back, so the same code is
# correct in both. Mantle's recording redesign makes the graph the location of
# graph arrays, after which `launch!` becomes `Mantle.dispatch!` and the explicit
# first argument is no longer needed; see `docs/recording-plan.md` in Mantle.
#
# Every operation also has a method without the device, which launches on the
# device of its output.
#
# A kernel is a plain function over KernelInterface's intrinsics. A launch is
# whole work-groups, so the last group along each axis overhangs the range: a
# kernel takes its index from `workitem` and returns when it is outside.

"""
    launch!(device, kernel, args::Tuple, ndrange; name = nothing)

Launch the plain-function `kernel` over `ndrange` with `args`: now, on a device
(any KernelInterface backend: a Mantle `Device`, lavapipe included), or as a node
of a Mantle graph.

A graph orders its nodes by the array objects they name, so name each array by
one object in every operation of a graph: two views of one buffer, such as two
calls of `Mantle.storage` or `reshape`, are two resources to it, and nothing
orders a write through one before a read through the other.
"""
function launch!(device::KA.Backend, kernel, args::Tuple, ndrange; name = nothing)
    k = KI.kernel_function(device, kernel, KI.argument_types(device, args))
    k(args...; ndrange, workgroupsize = something(group(ndrange), ()))
    return nothing
end

function launch!(g::Mantle.Graph, kernel, args::Tuple, ndrange; name = nothing)
    Mantle.dispatch!(g, kernel, args, ndrange; group = group(ndrange), name)
    return nothing
end

# An image-shaped range launches 16x16 workgroups: Mantle's default fills the
# first axis first, which for a 2-D range is one row of 64 (see `Mantle.dispatch!`).
group(::Integer) = nothing
group(::NTuple{1, Integer}) = nothing
group(::NTuple{2, Integer}) = (16, 16)
group(::NTuple{3, Integer}) = (16, 16, 1)

"""
    workitem(range) -> (index, inside)

This work-item's 1-based index in a launch over `range`, and whether it is
inside it. `range` is the launch's `ndrange`: an extent gives an `Int`, a tuple of
extents a `CartesianIndex` of as many axes.
"""
@inline function workitem(n::Integer)
    i = Int(KI.get_global_id().x)
    return i, i <= n
end
@inline function workitem(dims::NTuple{N, Integer}) where {N}
    id = KI.get_global_id()
    g = (id.x, id.y, id.z)
    I = CartesianIndex(ntuple(d -> Int(g[d]), Val(N)))
    return I, all(ntuple(d -> I[d] <= dims[d], Val(N)))
end

"""
    device(a) -> backend

Where an operation launches when it is not told: the device of `a`.
"""
device(a::AbstractArray) = KA.get_backend(a)
