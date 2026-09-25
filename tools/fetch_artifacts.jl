"""
Install developer-side artifacts and print where they are.

    julia tools/fetch_artifacts.jl sam2-src sam2-large-ckpt

One path per name, in order. `tools/artifacts.py` runs this on a miss, so Python
never downloads or verifies an artifact itself. Every artifact named here is
bound in `tools/Artifacts.toml`.
"""

using Pkg.Artifacts

const TOML = joinpath(@__DIR__, "Artifacts.toml")

isempty(ARGS) && error("usage: julia tools/fetch_artifacts.jl <artifact name>...")
for name in ARGS
    println(ensure_artifact_installed(name, TOML))
end
