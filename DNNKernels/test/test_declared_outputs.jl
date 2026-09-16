# What a graph's OUTPUTS are, and what a declaration may not leak.
#
# Two faults, both found by running SAM 2 end to end and both silent:
#
#   * `viewfor` built every materialised view as a `Transient.Buffer`. SAM 2's
#     decoder returns two of them (`slice_2`, `slice_3`) that nothing inside the
#     graph reads, so both had a degenerate live range and the placer -- doing
#     exactly its job -- overlapped them. The mask's first three elements came
#     back holding the IoU scores. Shapes right, dtypes right, 196,605 of 196,608
#     elements right.
#
#   * `destor` was `something(maybedest(emitctx, i), scratch(emitctx, T, dims...))`, and
#     `something` evaluates both arguments: the scratch was declared even when
#     the export had a destination for that result. A transient no pass then
#     touches is one `Liveness` refuses by name, so this is loud -- but only on a
#     graph where those results ARE declared, which is `keepall` below.
#
# Structural, on the real exported decoder, and no numbers: what is being
# asserted is that a declaration cannot express the two shapes above, not that
# some particular tensor came out right. The numerical gate is
# `SAM2Runner`'s `verifygraph`, which is what found both of these.

using Test
import DNNKernels
import Mantle

const DKO = DNNKernels
const MO = Mantle

include(joinpath(@__DIR__, "fixtures.jl"))

"""
The resource that owns `r`'s bytes: a view's parent, transitively.

`Mantle.storage` would answer the device array, which is the same object for two
aliased transients only after `Place` has run. This asks the declaration.
"""
ownerof(r) = r
ownerof(r::MO.ResourceView) = ownerof(r.parent)

"""
Zeroed DEVICE buffers of each weight's declared shape.

Nothing here runs, but the emit still has to see real operands: `mmplan` decides
from `Core.Typeof` and `size` of the weight, so a placeholder of the wrong rank
or on the host declines to a path that then refuses for being non-dense. The
shape and dtype come from the graph, which is where they are declared.

Every `:weight` buffer, not `weightkeys(g)`: that function answers what the ops
READ, following views, and `declare!` needs one per weight buffer the export
carries — SAM 2's decoder holds `model.maskmem_tpos_enc`, which no op in it
reads.
"""
function stubweights(dev, g)
    w = Dict{String,Any}()
    for (_, b) in g.buffers
        (b.kind === :weight && !isempty(b.key)) || continue
        haskey(w, b.key) && continue
        w[b.key] = MO.Buffer(dev, zeros(b.dtype, DKO.evalshape(b.shape, (res = 1024,))))
    end
    return w
end

function declaredoutputs(dev, label)
@testset "a graph output owns its bytes — $label" begin
    g = Fixtures.sam2("sam2_decoder")
    mantlegraph, emitctx = DKO.emitgraph(dev, g, stubweights(dev, g), (res = 1024,))

    @test !isempty(g.outputs)
    for id in g.outputs
        # Resolved at all: an output view is materialised by nobody else, since
        # the caller's read is not an op. This was a `KeyError` in `planfor`.
        @test haskey(emitctx.res, id)
        # And OWNED. A transient's bytes are the placer's to hand on at its last
        # use, and an output's last use is after the plan has run.
        @test !(ownerof(emitctx.res[id]) isa MO.TransientBuffer)
    end
end

@testset "nothing declares a transient it does not use — $label" begin
    g = Fixtures.sam2("sam2_decoder")
    # `keepall` is the mode where every result of a multi-output op has a
    # declared destination, so `destor` takes `maybedest` on all of them -- and
    # an eager scratch alongside it then belongs to no pass. Stated as the
    # transient-registration check rather than as `Plan`, so the assertion names
    # the fault instead of reporting an allocator message about it.
    mantlegraph, _ = DKO.emitgraph(dev, g, stubweights(dev, g), (res = 1024,); keepall = true)
    used = Set{Any}()
    for p in mantlegraph.passes, (rid, _) in MO.usages(p)
        push!(used, rid)
    end
    orphans = [size(t) for (rid, t) in mantlegraph.transient_by_id if !(rid in used)]
    @test isempty(orphans)
    # …and every transient the graph holds is one some pass declared, which is
    # the same statement from the other side: a scratch that reached no
    # `dispatch!` never gets an id at all.
    @test length(mantlegraph.transients) == length(mantlegraph.transient_by_id)
end
end

let bes = MO.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the declared-output checks"
    for be in bes
        declaredoutputs(MO.Device(be), string(nameof(typeof(be))))
    end
end
