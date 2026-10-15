# Declaring GPUFiltering's operations into a Mantle graph: the kernels become
# nodes of the graph, ordered by what they read and write, and run on every
# `run!` of its plan.
module GPUFilteringMantleExt

import GPUFiltering, Mantle

# An image-shaped range launches 16x16 workgroups: Mantle's default fills the
# first axis first, which for a 2-D range is one row of 64 (see `Mantle.dispatch!`).
group(::Integer) = nothing
group(::NTuple{1, Integer}) = nothing
group(::NTuple{2, Integer}) = (16, 16)
group(::NTuple{3, Integer}) = (16, 16, 1)

function GPUFiltering.launch!(g::Mantle.Graph, kernel, args::Tuple, ndrange; name = nothing)
    Mantle.dispatch!(g, kernel, args, ndrange; group = group(ndrange), name)
    return nothing
end

end
