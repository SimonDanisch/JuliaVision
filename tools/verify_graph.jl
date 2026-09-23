# SUPERSEDED, and it does not load. It runs the interpreted path over a
# `planslab` slab, and `planslab`/`Workspace`/`scratchfor`/`recordplan` went with
# the declared path. The slab is not droppable here -- see the note below on why
# the graph needs it -- so the replacement is not a smaller edit but the DECLARED
# path, where `Mantle.Place` does the placement `planslab` did:
# `DNNKernels.verifygraph` for the verification (`tools/two_route_parity.jl` was
# the other answer here and went with the interpreted runner it compared against),
# `tools/bench_all.jl` for the timing. It also reads a local `gen/` export tree
# rather than the artifact, so it cannot run on a machine that has not run the
# exporter. Kept for the measurements in its header.
"""
Any exported graph against the reference its own export wrote.

    julia --project=. tools/verify_graph.jl [dir ...] [--cpu]

With no arguments it runs every `gen/graphs/trellis2*` export, which is the
conditioner, the sparse-structure flow model, its depth slices and the conv3d
decoder. A directory may be named bare (`wandit-full-fp32`) or by path.

Generic over the graph rather than over one model. The inputs come from
`realinputs`, which drops the `c_`-prefixed constants `torch.export` lifted out
of the module and which are satisfied from the weight table, and each is looked
up in the reference by the same id — `export_graphs.py` files them under the
names `torch.export` took from the forward's parameters, so the ids and the keys
are the same thing. A reference missing one of those keys is an export whose
`ARGNAMES` disagrees with the model's own signature, and it says so rather than
guessing. The reference file is `reference.safetensors` or `refs.safetensors`;
both spellings exist in `gen/graphs` and neither is worth migrating.

Both "% of range" figures are printed. They are not interchangeable and the
distinction mattered: the gelu bug (see `atenarg`) showed up as 0.0125% mean and
0.053% max at three blocks, and only the mean matched what the substitution
predicts.

`Mantle.trim_gpu_pool!()` between models, which is what makes a sweep of several
1B-parameter graphs fit in one session at all. A finished graph's pool blocks are
held by the batch that recorded them until something flushes, so without it the
second model allocates on top of the first's high-water mark and a 20 GB card
runs out.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics, Lava
using Mantle: LavaBackend   # Mantle owns it; Lava does not re-export it
using DNNKernels: loadgraph, execute!, readsafetensors, toback, realinputs,
                  ascomplex, planslab, fusableset, Workspace, Ctx, value
const KA = KernelAbstractions

const GRAPHS = normpath(joinpath(@__DIR__, "..", "gen", "graphs"))

"""The one graph JSON in an export directory — `op_histogram.json`,
`preprocess.json` and `node_stats.json` sit beside it and are not graphs."""
function graphjson(dir)
    side = ("op_histogram.json", "preprocess.json", "node_stats.json")
    c = filter(f -> endswith(f, ".json") && !(f in side), readdir(dir))
    length(c) == 1 || error("expected one graph json in $dir, found $c")
    joinpath(dir, only(c))
end

"""Either spelling of the reference file, whichever the exporter wrote."""
function referencefile(dir)
    for f in ("reference.safetensors", "refs.safetensors")
        isfile(joinpath(dir, f)) && return joinpath(dir, f)
    end
    error("no reference in $dir")
end

function verify(dir, backend)
    g = loadgraph(graphjson(dir))
    # Everything in the file, not `todevice`'s live subset. `execute!` demands
    # every *declared* weight whether or not an op reads it, and the pipeline gets
    # away with pruning only because `prepare` rebuilds the graph without the dead
    # ones first — unprepared, DINOv3's `mask_token` is declared, unread, and
    # fatal. Uploading the lot is the right policy for a one-shot check anyway;
    # the pruning is a memory optimisation that belongs to a resident pipeline.
    #
    # `ascomplex` is kept, because it is not an optimisation: safetensors has no
    # complex dtype, so a complex constant arrives as its real view with a leading
    # pair axis, and Wan's rotary table fed as a real array survives 40 ops before
    # surfacing as a `cat` bounds error.
    bykey = Dict{String,Any}()
    for (id, b) in g.buffers
        b.kind === :weight && (bykey[b.key] = b)
        b.kind === :external && startswith(id, "c_") && (bykey[id[3:end]] = b)
    end
    w = Dict{String,Any}(k => toback(backend, ascomplex(v, get(bykey, k, nothing)))
                         for (k, v) in readsafetensors(joinpath(dir, "weights.safetensors")))
    ref = readsafetensors(referencefile(dir))

    ids = realinputs(g)
    absent = [i for i in ids if !haskey(ref, i)]
    isempty(absent) ||
        error("$(basename(dir)): reference has no $(absent); it holds " *
              "$(sort(collect(keys(ref)))). Re-export — ARGNAMES does not match " *
              "the model's forward.")
    ins = Dict{String,Any}(i => toback(backend, ref[i]) for i in ids)
    # The lifted constants are declared inputs and `execute!` insists on every
    # one, whether or not an op reads it. Same rule `wan.jl`'s `rungraph` uses.
    for id in g.inputs
        haskey(ins, id) && continue
        haskey(w, id[3:end]) ||
            error("$(basename(dir)): input $id is neither in the reference nor a " *
                  "lifted constant in the weight table")
        ins[id] = w[id[3:end]]
    end

    plan = planslab(g, (;))
    slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
    out = execute!(g, ins, w; dims = (;), backend, slab, plan,
                   ws = Workspace(backend), lazy = fusableset(g))
    got = Array(value(Ctx(out, g, (;), backend), only(g.outputs)))

    want = ref["out"]
    length(got) == length(want) ||
        error("$(length(got)) elements vs reference $(length(want))")

    lo, hi = extrema(want)
    span = hi - lo
    d = abs.(vec(got) .- vec(want))
    c = cor(Float64.(vec(got)), Float64.(vec(want)))
    @printf("%-24s %5d ops  max %.6f%%  mean %.6f%%  corr %.9f\n",
            basename(dir), length(g.ops), 100maximum(d) / span, 100mean(d) / span, c)
    (; dir = basename(dir), max = 100maximum(d) / span, mean = 100mean(d) / span, corr = c)
end

function main(args)
    backend = "--cpu" in args ? KA.CPU() : LavaBackend()
    dirs = filter(a -> !startswith(a, "--"), args)
    if isempty(dirs)
        dirs = sort(filter(d -> startswith(basename(d), "trellis2"),
                           joinpath.(GRAPHS, readdir(GRAPHS))))
        filter!(isdir, dirs)
    else
        dirs = [isdir(d) ? d : joinpath(GRAPHS, d) for d in dirs]
    end
    println("reference range is the denominator for both percentages\n")
    for d in dirs
        verify(d, backend)
        backend isa LavaBackend && Mantle.trim_gpu_pool!()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
