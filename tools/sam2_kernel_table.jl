"""
One table: every kernel SAM 2.1's encoder runs, ours against PyTorch's.

    julia --project=. tools/sam2_kernel_table.jl

The point is to stop optimising by guess. Each row is an aten op with the device
time both runtimes spend in it, the arithmetic it performs, and therefore the
rate each achieves — so "which kernel is behind, and by how much" is a lookup
rather than an experiment.

Needs `gen/graphs/sam2-large/pytorch_kernels.json`, written by
`uv run tools/sam2_pytorch_kernels.py`. Our side comes from
`DNNKernels.OPTIMES`, which serialises around every op; see its docstring for why
that is the only measurement that attributes device time to a source-level op,
and why the total therefore exceeds a free-running encode.

FLOPs are counted from the graph, not measured: `2*M*N*K` for the matmuls,
`2*prod(out)*prod(weight spatial × Cin)` for convolution, `4*prod(q)*Lk` for
attention (QK^T and PV). Ops that move data rather than compute get no count and
show a rate of `-`; for those the honest comparison is bytes, and the ratio
column still says whether we are behind.
"""

using Printf
using Lava, DNNKernels, SAM2Runner, KernelAbstractions
# Through DNNKernels rather than as a direct dependency, the way its own
# test suite does: this reads one JSON file and does not warrant a dep.
const JSON3 = DNNKernels.JSON3
using DNNKernels: encode, toback, readsafetensors, evalshape

const KA = KernelAbstractions
# A tool, and it wants a file the runner has no accessor for, so it takes the
# directory as an argument instead of reaching into the package. Defaults to the
# artifact via the runner's own model loader path, which is the only sanctioned
# way in; override on the command line to point at a local export.
const TORCH = get(ENV, "SAM2_KERNEL_TABLE_TORCH_JSON") do
    length(ARGS) >= 1 ? ARGS[1] :
        error("""
              pass the path to pytorch_kernels.json:
                  julia --project=. tools/sam2_kernel_table.jl <path>
              It is developer material and does not ship in the `sam2-large`
              artifact, so there is nothing to resolve it from.""")
end

"""Arithmetic per aten op, summed over the graph, from the declared shapes."""
function graphflops(g, dims)
    shp(id) = (b = get(g.buffers, id, nothing); b === nothing ? nothing : evalshape(b.shape, dims))
    out = Dict{String,Float64}()
    for o in g.ops
        f = 0.0
        so = shp(o.out)
        if o.aten == "convolution.default" && so !== nothing
            w = shp(o.ins[2])
            w !== nothing && (f = 2 * prod(so) * prod(w[1:(end - 1)]))
        elseif o.aten in ("addmm.default", "mm.default")
            a = o.aten == "addmm.default" ? shp(o.ins[2]) : shp(o.ins[1])
            (so !== nothing && a !== nothing) && (f = 2 * prod(so) * a[1])
        elseif o.aten == "bmm.default"
            a = shp(o.ins[1])
            (so !== nothing && a !== nothing) && (f = 2 * prod(so) * a[1])
        elseif occursin("scaled_dot_product", o.aten)
            q = shp(o.ins[1]); k = shp(o.ins[2])
            (q !== nothing && k !== nothing) && (f = 4 * prod(q) * k[2])
        end
        f > 0 && (out[o.aten] = get(out, o.aten, 0.0) + f)
    end
    out
end

"""`{aten => (calls, ms)}` for one encode, device time, serialised per op."""
function ourtimes(model, img; iters = 5)
    encode(model, img); KA.synchronize(model.model.backend)   # warm
    t = Dict{String,Tuple{Int,Float64}}()
    DNNKernels.OPTIMES[] = t
    for _ in 1:iters
        encode(model, img)
    end
    KA.synchronize(model.model.backend)
    DNNKernels.OPTIMES[] = nothing
    Dict(k => (v[1] ÷ iters, v[2] / iters) for (k, v) in t)
end

function main()
    isfile(TORCH) || error("no $TORCH — run `uv run tools/sam2_pytorch_kernels.py` first")
    torch = JSON3.read(read(TORCH, String))
    tk = Dict{String,Tuple{Int,Float64}}(String(k) => (Int(v[1]), Float64(v[2]))
                                         for (k, v) in pairs(torch.encode_kernels_ms))

    model = SAM2Runner.sam2model()
    # `sam2refs()`, not a path built on `assetdir()`. The old form read
    # refs.safetensors out of the MODEL artifact, which does not contain it —
    # they live in `sam2-large-refs`, deliberately, so a caller who only wants to
    # segment a picture never fetches 1.2 GB of fixtures.
    refs = SAM2Runner.sam2refs()
    img = toback(model.model.backend, refs["sam2_encoder/in0"])

    ours = ourtimes(model, img)
    fl = graphflops(model.model.graphs["sam2_encoder"], (res = model.res,))

    clk = strip(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits`, String))
    ourtot = sum(x -> x[2], values(ours); init = 0.0)
    thtot = sum(x -> x[2], values(tk); init = 0.0)
    @printf("SAM 2.1 encode, res %d, @ %s MHz (torch measured @ %s MHz)\n",
            model.res, clk, torch.sm_clock_mhz)
    @printf("ours %.1f ms serialised, torch %.1f ms device\n\n", ourtot, thtot)
    @printf("%-38s %6s %9s %9s %7s %9s %8s %8s\n",
            "aten op", "calls", "ours ms", "torch ms", "ratio", "GFLOP", "ourTF/s", "thTF/s")

    keys_ = union(keys(ours), keys(tk))
    rows = [(k, get(ours, k, (0, 0.0)), get(tk, k, (0, 0.0)), get(fl, k, 0.0)) for k in keys_]
    sort!(rows; by = r -> -(r[2][2]))
    for (k, o, t, f) in rows
        (o[2] < 0.05 && t[2] < 0.05) && continue
        rate(ms) = (f > 0 && ms > 0) ? @sprintf("%8.1f", f / 1e12 / (ms / 1e3)) : "       -"
        ratio = t[2] > 0 ? @sprintf("%6.2fx", o[2] / t[2]) : "      -"
        @printf("%-38s %6d %9.2f %9.2f %7s %9.1f %8s %8s\n",
                k, o[1], o[2], t[2], ratio, f / 1e9, rate(o[2]), rate(t[2]))
    end
    @printf("\nTotal graph arithmetic: %.1f GFLOP\n", sum(values(fl); init = 0.0) / 1e9)
end

# `@__FILE__ && main()` would parse as `@__FILE__(&& main())` — a macro
# slurps the rest of the line, so this needs to be a block.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
