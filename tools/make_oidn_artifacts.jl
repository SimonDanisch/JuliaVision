# Build an immutable artifact and tarball. Preserve published downloads for
# unchanged content; an optional second argument names an actual uploaded URL.
# This script does not create releases or upload assets.
using Pkg.Artifacts
using TOML
source = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "..", "gen", "graphs", "oidn")
package = normpath(joinpath(@__DIR__, "..", "OIDNRunner"))
artifact_file = joinpath(package, "Artifacts.toml")
previous = isfile(artifact_file) ? get(TOML.parsefile(artifact_file), "oidn", Dict()) : Dict()
hash = create_artifact() do dir
    for name in ("rt_alb_large", "rt_nrm_large", "rt_hdr_calb_cnrm_large"), ext in (".json", ".safetensors")
        cp(joinpath(source, name * ext), joinpath(dir, name * ext))
    end
    cp(joinpath(source, "provenance.json"), joinpath(dir, "provenance.json"))
    cp(joinpath(package, "LICENSE"), joinpath(dir, "LICENSE"))
    cp(joinpath(package, "NOTICE"), joinpath(dir, "NOTICE"))
end
archive_sha = archive_artifact(hash, joinpath(source, "oidn-models.tar.gz"))
downloads = if length(ARGS) >= 2
    [(ARGS[2], archive_sha)]
elseif get(previous,"git-tree-sha1","") == string(hash)
    [(d["url"],d["sha256"]) for d in get(previous,"download",[])]
else
    Tuple{String,String}[]
end
bind_artifact!(artifact_file, "oidn", hash; force = true, lazy = true, download_info = downloads)
println("OIDN artifact: ", hash)
println("archive sha256: ", archive_sha)
