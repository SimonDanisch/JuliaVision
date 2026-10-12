# Build an immutable local artifact and a transportable tarball. This does not
# publish a release or invent a download URL.
using Pkg.Artifacts
source = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "..", "gen", "graphs", "oidn")
package = normpath(joinpath(@__DIR__, "..", "OIDNRunner"))
hash = create_artifact() do dir
    for name in ("rt_alb_large", "rt_nrm_large", "rt_hdr_calb_cnrm_large"), ext in (".json", ".safetensors")
        cp(joinpath(source, name * ext), joinpath(dir, name * ext))
    end
    cp(joinpath(source, "provenance.json"), joinpath(dir, "provenance.json"))
    cp(joinpath(package, "LICENSE"), joinpath(dir, "LICENSE"))
    cp(joinpath(package, "NOTICE"), joinpath(dir, "NOTICE"))
end
bind_artifact!(joinpath(package, "Artifacts.toml"), "oidn", hash; force = true, lazy = true)
archive_artifact(hash, joinpath(source, "oidn-models.tar.gz"))
println("OIDN artifact: ", hash)
