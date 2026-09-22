# Where a declared model's frame actually goes, one op at a time.
#
# Mantle's profiler does not populate per-pass GPU time on every backend, and the
# obvious substitutes both mislead: a low DISPATCH overhead says the GPU is busy and
# nothing about whether it is busy at peak, and timing graph PREFIXES is not
# monotone because `dropdead` gives each truncation a different set of live ops.
#
# So each op is emitted on its own, with its first input promoted to a graph input,
# and replayed. That over-counts slightly (an op alone pays a submission the frame
# amortises) and it does not see fusion between neighbours, but it finds the ops that
# are orders of magnitude off, which is what a first pass is for.
#
# It found RIFE's: seven transposed convolutions at 0.03 TFLOP/s, 438 ms of a 581 ms
# frame, while the forward convolutions beside them ran on Apple's library at 2.5-9.7.
using Mantle, KernelAbstractions, DNNKernels
const DK = DNNKernels
const M = Mantle

"""`op` alone as a graph, its first input promoted to a graph input."""
function soloop(g::DK.Graph, op::DK.Op)
    bufs = Dict{String,DK.Buffer}()
    for id in [op.ins; op.out]
        b = g.buffers[id]
        bufs[id] = id == op.ins[1] ?
            DK.Buffer(b.id, :input, b.shape, b.dtype, b.key, b.live, b.of, b.viewop,
                      b.attrs) : b
    end
    DK.Graph(g.name, g.symbols, [op.ins[1]], [op.out], bufs, [op.ins; op.out], [op],
             Vector{String}[])
end

"""Milliseconds for `op` alone, and how many of its dispatches were library calls."""
function timeop(dev, g, weights, dims, op; R = 4)
    gr = soloop(g, op)
    mg, ec = DK.emitgraph(dev, gr, DK.residentweights(dev, gr, weights), dims)
    nlib = sum(count(d -> d.ndrange === nothing, p.dispatches) for p in mg.passes)
    p = M.Plan(mg); M.record!(p)
    # After `record!`: a graph input has no storage until the plan is materialised.
    xs = M.storage(ec.res[op.ins[1]])
    if xs isa AbstractArray && eltype(xs) <: AbstractFloat
        copyto!(xs, convert(Array{eltype(xs)},
                reshape(Float32.((1:length(xs)) .% 37) ./ 37 .- 0.5, size(xs))))
    end
    M.run!(p); M.waitidle(dev)
    t = minimum([@elapsed (M.run!(p); M.waitidle(dev)) for _ in 1:R])
    M.free!(p); DK.freeowned!(ec)
    (t * 1000, nlib, length(mg.passes))
end

"""Every op of `g`, slowest first. Ops that cannot stand alone are reported as such."""
function census(dev, g, weights, dims; top = 15)
    rows = Any[]
    for op in g.ops
        isempty(op.ins) && continue
        try
            ms, nlib, np = timeop(dev, g, weights, dims, op)
            push!(rows, (op.id, op.aten, ms, nlib, np))
        catch e
            push!(rows, (op.id, op.aten, NaN, -1, 0))
        end
    end
    ok = [r for r in rows if !isnan(r[3])]
    tot = sum(r[3] for r in ok; init = 0.0)
    sort!(ok; by = r -> r[3], rev = true)
    println("summed ", round(tot, digits=1), " ms over ", length(ok), " ops (",
            count(r -> isnan(r[3]), rows), " could not stand alone)")
    println(rpad("op", 22), rpad("aten", 30), lpad("ms", 8), lpad("%", 7), lpad("lib", 5))
    for r in first(ok, top)
        println(rpad(r[1], 22), rpad(r[2], 30), lpad(round(r[3], digits=2), 8),
                lpad(round(100r[3]/tot, digits=1), 7), lpad(r[4], 5))
    end
    bykind = Dict{String,Float64}()
    for r in ok; bykind[r[2]] = get(bykind, r[2], 0.0) + r[3]; end
    println("\nby aten:")
    for (k, v) in first(sort(collect(bykind); by = last, rev = true), 10)
        println("  ", rpad(k, 32), lpad(round(v, digits=1), 8), " ms  ",
                lpad(round(100v/tot, digits=1), 5), "%")
    end
    ok
end
