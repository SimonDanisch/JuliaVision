"""
Fuse matmuls that share an activation into one.

A transformer layer multiplies the same normalised hidden state by three separate
projections (Q, K, V) and, after the second norm, by two more (gate, up). Each is
its own `mm`, and at decode each is a matrix-VECTOR product, where throughput is
set by how much of the device the launch can fill rather than by the arithmetic.
K and V are the small ones — `1024 x 5120`, 5.24 MB — and they run at 77 GB/s on
a device that reaches 227. Stacked with Q into a single `10240 x 5120` they run
at the same rate as Q does.

The stacking is exact, not an approximation: `[A; B] * x` is `[A*x; B*x]`, and
because the int8 scales are per OUTPUT CHANNEL they are unchanged by which matrix
a channel is considered to belong to. The results are bit-identical.

Each original output becomes a `slice.Tensor` view of the fused one, so every
consumer reads what it read before, from a row range rather than its own buffer.
"""

"""
    FUSEQKV[] = true

Turn the stacking off, to A/B it against the unfused graph on one model. The
claim it rests on — that `[A; B] * x` and `[A*x; B*x]` agree bit for bit, int8
scales included, because those scales are per output channel — is the kind that
should be checkable rather than argued.

Worth 6.07 tok/s against 5.60 on K2 Horizon 32B. Leaving Q, K and V as three
matmuls does NOT let them overlap here, which was the other thing worth checking:
they are independent, and unfusing them is still slower.
"""
const FUSEQKV = Ref(true)

"""
    fuseqkv(graphs, weights) -> (graphs, weights, n)

Replace each group of `mm`s sharing one activation with a single `mm` over the
row-stacked weight. `n` counts the groups fused.
"""
function fuseqkv(graphs::AbstractDict, weights::AbstractDict)
    FUSEQKV[] || return (Dict{String,Graph}(graphs), Dict{String,Any}(weights), 0)
    w = Dict{String,Any}(weights)
    out = Dict{String,Graph}()
    total = 0
    for (name, g) in graphs
        g2, n = fuseqkv(g, w)
        out[name] = g2
        total += n
    end
    (out, w, total)
end

function fuseqkv(g::Graph, weights::Dict{String,Any})
    buffers = Dict{String,Buffer}(g.buffers)
    # Group `mm`s by the activation they read. Order within a group follows the
    # graph, so the stacking order is deterministic.
    groups = Dict{String,Vector{Op}}()
    for op in g.ops
        op.aten == "mm.default" || continue
        length(op.ins) == 2 || continue
        push!(get!(groups, viewroot(g, op.ins[1]), Op[]), op)
    end

    # How many ops read each buffer: an output that something other than the
    # graph consumes is still fine (it becomes a view), but the WEIGHT must not
    # be read anywhere else, since it is about to stop existing on its own.
    wreads = Dict{String,Int}()
    for op in g.ops, id in op.ins
        b = get(g.buffers, id, nothing)
        b !== nothing && b.kind === :weight && !isempty(b.key) &&
            (wreads[b.key] = get(wreads, b.key, 0) + 1)
    end

    ops = copy(g.ops)
    order = copy(g.order)
    drop = Set{String}()
    n = 0
    for (act, mms) in groups
        length(mms) >= 2 || continue
        wb = [get(g.buffers, o.ins[2], nothing) for o in mms]
        all(b -> b !== nothing && b.kind === :weight && !isempty(b.key) &&
                 length(b.shape) == 2, wb) || continue
        # Same reduction extent, and each weight read exactly here.
        K = wb[1].shape[1]
        all(b -> b.shape[1] == K, wb) || continue
        all(b -> get(wreads, b.key, 0) == 1, wb) || continue
        dt = wb[1].dtype
        all(b -> b.dtype === dt, wb) || continue
        obs = [g.buffers[o.out] for o in mms]
        all(b -> length(b.shape) == 2 && b.shape[1] == obs[1].shape[1], obs) || continue
        all(b -> b.dtype === obs[1].dtype, obs) || continue

        widths = [Int(b.shape[2]) for b in wb]
        total = sum(widths)
        fid = "fuseqkv_" * mms[1].id
        fkey = fid * "|w"
        weights[fkey] = RowCat([weights[b.key] for b in wb])
        buffers[fid * "_w"] = Buffer(fid * "_w", :weight, Any[K, total], dt, fkey,
                                     (0, 0), "", "", Dict{String,Any}())
        buffers[fid] = Buffer(fid, :transient, Any[obs[1].shape[1], total], obs[1].dtype,
                              "", (0, 0), "", "", Dict{String,Any}())
        # The fused product takes the first group member's place, so it lands
        # before every consumer of any of the outputs.
        ops[findfirst(==(mms[1]), ops)] =
            Op(fid, "mm.default", [mms[1].ins[1], fid * "_w"], fid, Dict{String,Any}())
        off = 0
        for (i, o) in enumerate(mms)
            buffers[o.out] = Buffer(o.out, :view, obs[i].shape, obs[i].dtype, "", (0, 0),
                                    fid, "slice.Tensor",
                                    Dict{String,Any}("arg1" => 1, "arg2" => off))
            off += widths[i]
            i == 1 || push!(drop, o.id)
        end
        push!(order, fid * "_w")
        push!(order, fid)
        n += 1
    end
    n == 0 && return (g, 0)
    keep = [o for o in ops if !(o.id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, keep, g.fusion), n)
end
