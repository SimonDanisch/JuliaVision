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
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":0")
using DNNKernels, Artifacts, Printf
const DK = DNNKernels

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
                endswith(f, ".json") && f != "op_histogram.json" || continue
                g = try
                    DK.loadgraph(joinpath(sub, f))
                catch
                    continue
                end
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
