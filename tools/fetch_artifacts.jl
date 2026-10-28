#=
Install developer-side artifacts and print where they are.

    julia tools/fetch_artifacts.jl sam2-src sam2-large-ckpt

One path per name, in order. `tools/artifacts.py` runs this on a miss, so Python
never downloads or verifies an artifact itself. Uses `tools/Artifacts.toml` by
default; `--toml <runner/Artifacts.toml>` resolves a runner's assets instead.
=#

using Pkg.Artifacts

toml = joinpath(@__DIR__, "Artifacts.toml")
if length(ARGS) >= 2 && ARGS[1] == "--toml"
    popfirst!(ARGS)
    toml = popfirst!(ARGS)
end

isempty(ARGS) && error("usage: julia tools/fetch_artifacts.jl <artifact name>...")
for name in ARGS
    println(ensure_artifact_installed(name, toml))
end
