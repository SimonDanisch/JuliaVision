"""
Static buffer planning.

`torch.export` hands us a lifetime for every transient: `Buffer.live` is the
`[first_op, last_op]` range over which the value is needed. Two buffers whose
lifetimes do not overlap can occupy the same memory, so the whole graph's
scratch space can be laid out *once*, at load time, into a single slab — and
then nothing allocates at run time at all.

That is the point. It is not a cheaper allocator: a pool still makes one
allocation call per op, and this makes none. Measured on this model, the
lifetimes are worth a lot:

    graph                 sum of transients   peak concurrent   reuse
    encode_image                    53.1 MB           4.2 MB    12.6x
    segment                         81.4 MB          25.5 MB     3.2x
    readout_query                    9.8 MB           0.5 MB    18.0x

The graphs run one after another inside a step, so a single slab sized to the
largest peak — 25.5 MB — serves all eight, against 179 MB of separate
allocations.

The second reason to want this is that it pins device addresses: with a fixed
slab the pointer for every op is the same on every step, which is the
precondition for capturing the launch sequence once and replaying it.
"""

"""Round up to a 256-byte boundary — the alignment every GPU backend is happy with."""
# The SLAB ALLOCATOR was here, deleted 2026-09-15: `Slab`, `place(::Slab, …)`,
# `planslab`, `checkslab` and `lifetimes`.
#
# It laid every intermediate of a graph out in one `KA.allocate`d byte buffer,
# aliasing by live range, because Mantle could not: passes were discovered by
# CAPTURE, so placement could not run before the ops had run. Declared, Mantle
# owns the arena — `Transient.Buffer` plus `Liveness`/`Place`/`Aliasing` — and a
# second placer is a second answer to one question.
#
# `lifetimes` goes with it, and its finding does not: **the export's `b.live` is
# not safe to place against.** A buffer stays reachable long after torch
# considers it dead, so reusing its bytes at the annotated point corrupts the
# result, and the ranges had to be re-derived by walking the ops and following
# view chains to their root. Mantle derives liveness from declared USE, which is
# the same walk done once by the party that knows — so `b.live` must stay a
# cross-check and never an input.
#
# What stays in this file is the view vocabulary emit needs: `escaping`,
# `rootbuffer`, `SHAPEONLY_VIEWS` and `materialisedview` — the last of which
# names exactly the 51 buffers (290.2 MB on SAM 2's encoder) whose read forces a
# copy, which declared become a transient and a pass rather than a run-time
# `contiguous`.

alignup(n::Integer, a::Integer = 256) = ((n + a - 1) ÷ a) * a


# `place` for the plan this file produces — see its docstring in execute.jl.


"""
Buffer ids that outlive the graph: its declared outputs, plus anything they are
views of. A step chains eight graphs through one slab, so a value that escapes
into the next graph must not sit in memory the next graph will overwrite.
"""
function escaping(graph::Graph)
    esc = Set{String}()
    function mark(id)
        id in esc && return
        push!(esc, id)
        b = get(graph.buffers, id, nothing)
        b === nothing && return
        isempty(b.of) || mark(b.of)
    end
    foreach(mark, graph.outputs)
    esc
end


"""
    rootbuffer(graph, id) -> id

Follow a view chain to the buffer that owns the storage. `lifetimes` keys its
ranges by this, since a view has no lifetime of its own.
"""
function rootbuffer(graph::Graph, id::AbstractString, depth::Int = 0)
    depth > 64 && return String(id)
    b = get(graph.buffers, id, nothing)
    (b === nothing || isempty(b.of)) ? String(id) : rootbuffer(graph, b.of, depth + 1)
end

"""View ops that only reinterpret the shape — the same list `makeview` uses."""
# `alias.default` is the identity: torch's `alias` is a view with the parent's
# own shape AND strides, so there is nothing to reinterpret and nothing to move.
# It belongs here rather than in `viewstrides` for that reason, and
# `materialisedview` reads it correctly too -- an alias of a permute is
# materialised exactly as a reshape of one is.
const SHAPEONLY_VIEWS = ("view.default", "_unsafe_view.default", "unsqueeze.default",
                         "squeeze.dims", "squeeze.dim", "alias.default")

"""
    materialisedview(graph, b) -> Bool

Whether reading view `b` will force a copy.

`makeview` reshapes a shape-only view of its parent, and Julia cannot reshape a
`PermutedDimsArray` without collapsing it first — so a shape-only view *of a
permute* is materialised by `contiguous`, and every other view stays lazy. That
is decidable from the graph, which is what makes these plannable: on SAM 2's
encoder the predicate names exactly the 51 buffers, totalling 290.2 MB, that the
allocation trace shows `contiguous` allocating at run time.
"""
function materialisedview(graph::Graph, b::Buffer)
    b.kind === :view || return false
    b.viewop in SHAPEONLY_VIEWS || return false
    p = get(graph.buffers, b.of, nothing)
    p === nothing && return false
    p.viewop == "permute.default"
end


