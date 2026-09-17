# Which exported graphs the declared path covers, per installed model artifact.
#
#     julia --project=. tools/declared_coverage.jl
#
# Static: no device, no weights, no run. `emitop!` dispatches on
# `Val(Symbol(aten))`, so its `Val` parameters ARE the ported set and a list
# beside them would go stale; a graph's requirement is the `aten` field of its
# ops after `dropdead`, since a dead op is not a requirement.
#
# Only artifacts already in the store are read, so nothing downloads.
#
# **"ALL DECLARED" is about op KINDS and not about shapes.** An op with an
# `emitop!` may still refuse the shape it is given: `convolution.default` covers
# 1-D and 2-D, dense and split-K, and refuses the 1-D TRANSPOSED form that 5 of
# kokorovoc's 90 convolutions are. So this sweep is a lower bound on what is
# missing, and `KokoroRunner`'s suite is what found that one. The same blind spot
# hid `fused.sdpa`, which a fusion pass creates and which no raw graph names.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":0")
using DNNKernels, Artifacts, Printf
const DK = DNNKernels

"""
Is this JSON an exported graph?

The artifacts hold others beside them — `op_histogram.json`, Kokoro's `vocab`
and `lexicon`, SAM 2's refs — and they are not failures to be skipped. This was
`try loadgraph catch; continue; end`, which also swallowed a graph that failed to
load for a real reason: a schema change would have read as "no graphs here" and
the sweep would have reported full coverage of nothing.
"""
isatengraph(path) = haskey(DK.JSON3.read(read(path, String)), :buffers)

ported = Set{String}()
for m in methods(DK.emitop!)
    p = Base.unwrap_unionall(m.sig).parameters
    length(p) >= 4 || continue
    T = p[4]
    T isa DataType && T <: Val || continue
    isempty(T.parameters) && continue
    T.parameters[1] isa Symbol || continue
    push!(ported, String(T.parameters[1]))
end
@printf("emitop! covers %d aten ops\n\n", length(ported))

runners = ["BasicVSRRunner", "DeepFilterRunner", "DemucsRunner",
           "DepthAnythingRunner", "Hunyuan3DRunner", "KokoroRunner",
           "MatAnyoneRunner", "NeuralLUTRunner", "ProPainterRunner",
           "RIFERunner", "SAM2Runner", "WhisperRunner"]
root = normpath(joinpath(@__DIR__, ".."))
ndone, ngaps = Ref(0), Ref(0)
for r in runners
    toml = joinpath(root, r, "Artifacts.toml")
    isfile(toml) || continue
    for (name, meta) in Artifacts.load_artifacts_toml(toml)
        h = get(meta isa Vector ? first(meta) : meta, "git-tree-sha1", nothing)
        h === nothing && continue
        dir = Artifacts.artifact_path(Base.SHA1(h))
        isdir(dir) || continue
        for sub in (dir, joinpath(dir, "graphs"))
            isdir(sub) || continue
            for f in sort(readdir(sub))
                endswith(f, ".json") || continue
                path = joinpath(sub, f)
                isatengraph(path) || continue
                g = DK.loadgraph(path)
                need = sort(unique(o.aten for o in first(DK.dropdead(g)).ops))
                isempty(need) && continue
                gaps = filter(a -> !(a in ported), need)
                isempty(gaps) ? (ndone[] += 1) : (ngaps[] += 1)
                @printf("%-26s %-24s %3d kinds  %s\n", name, splitext(f)[1],
                        length(need),
                        isempty(gaps) ? "ALL DECLARED" :
                        "missing: " * join(gaps, ", "))
            end
        end
    end
end
@printf("\n%d graphs fully declared, %d with gaps\n", ndone[], ngaps[])
nothing
