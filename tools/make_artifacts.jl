"""
Pack the small-models exports into artifact tarballs and bind them.

    julia --project=. tools/make_artifacts.jl              # the ported three
    julia --project=. tools/make_artifacts.jl neurallut    # just one
    julia --project=. tools/make_artifacts.jl --tag assets-v2

For each model this creates the artifact from `gen/graphs/<name>`, archives it to
`gen/artifacts/<name>.tar.gz`, and writes the `[<name>]` entry into that model's
`Artifacts.toml` with `lazy = true` and the release URL.

**This is the step that publishes a re-export.** A runner reads its assets from
its artifact and from nowhere else, so exporting a graph changes nothing until it
is bound here — that is deliberate, and it is why re-running this is part of the
edit rather than an afterthought. The binding takes effect locally as soon as it
is written, because `create_artifact` has already put the tree in the store.

**Uploading is separate, and is how it reaches anyone else.** `bind_artifact!`
records a URL and a sha256; it does not put anything at that URL. A machine
without the tree in its store — anyone else's — resolves the binding, tries the
URL and fails until the tarball is attached to the release. The command is
printed at the end and is deliberately not run from here: it publishes.

**What ships and what does not.** The tarball carries the graph JSON, the
weights and the op histogram — what a caller needs to *run* the model. It leaves
out `reference*.safetensors`, which only `tools/verify_*.jl` reads and which the
exporter regenerates in one command. That is the same split SAM 2 makes, and it
matters most for RIFE, whose reference frames are ~80 MB against 22 MB of
weights: a user segmenting a clip should not download the test fixtures.

**`git-tree-sha1` is over the unpacked content, `sha256` over the tarball.** The
two are not interchangeable and `Pkg` checks both — the tree hash is what names
the artifact in the store, the sha256 is what validates the download. Both come
out of `create_artifact`/`archive_artifact` here rather than being computed by
hand, because a hand-computed tree hash that happens to be wrong fails at
`Pkg.instantiate` on someone else's machine and nowhere earlier.
"""

using Pkg.Artifacts, Printf, SHA

const ROOT = normpath(joinpath(@__DIR__, ".."))
const JV = joinpath(ROOT, "dev", "JuliaVision")

# artifact name => (package that binds it, the export directory under gen/graphs)
const MODELS = Dict(
    "neurallut"     => ("NeuralLUTRunner", "neurallut"),
    "rife"          => ("RIFERunner", "rife"),
    "depthanything" => ("DepthAnythingRunner", "depthanything"),
    # Not ported yet — listed so the instruction their `assetdir()` prints is one
    # that actually runs. Each errors here until its exporter has been run.
    "whisper"       => ("WhisperRunner", "whisper"),
    "basicvsrpp"    => ("BasicVSRRunner", "basicvsrpp-fp32"),
    "deepfilternet" => ("DeepFilterRunner", "deepfilternet"),
    "demucs"        => ("DemucsRunner", "demucs"),
    "kokoro"        => ("KokoroRunner", "kokoro"),
    "propainter"    => ("ProPainterRunner", "propainter"),
)

# Files a caller needs to run the model. Anything else in the export directory —
# above all the references — is developer material and stays out of the tarball.
const SHIPPED = ["op_histogram.json", "weights.safetensors"]   # + "<name>.json"

repo_url(tag, name) =
    "https://github.com/SimonDanisch/JuliaVision/releases/download/$tag/$name.tar.gz"

function pack(name::AbstractString, tag::AbstractString)
    pkg, dirname_ = MODELS[name]
    src = joinpath(ROOT, "gen", "graphs", dirname_)
    isdir(src) || error("no export at $src — run `uv run tools/export_$(name).py`")

    want = vcat("$name.json", SHIPPED)
    for f in want
        isfile(joinpath(src, f)) || error("$src is missing $f")
    end

    # `create_artifact` hands the callback a fresh directory and returns the
    # tree hash of whatever was put in it. Copying file by file rather than the
    # whole directory is what keeps the references out — and keeps the hash
    # stable when a new developer-only file appears beside them.
    hash = create_artifact() do dir
        for f in want
            cp(joinpath(src, f), joinpath(dir, f))
        end
    end

    outdir = joinpath(ROOT, "gen", "artifacts")
    mkpath(outdir)
    tarball = joinpath(outdir, "$name.tar.gz")
    isfile(tarball) && rm(tarball)
    sha = archive_artifact(hash, tarball)

    toml = joinpath(JV, pkg, "Artifacts.toml")
    bind_artifact!(toml, name, hash;
                   download_info = [(repo_url(tag, name), sha)],
                   lazy = true, force = true)

    bytes = filesize(tarball)
    @printf("%-14s %7.1f MiB  tree %s\n", name, bytes / 2^20, string(hash)[1:12])
    @printf("%-14s          sha256 %s\n", "", sha[1:12])
    @printf("%-14s          -> %s\n", "", relpath(toml, ROOT))
    return (; name, tarball, bytes, sha, hash)
end

args = copy(ARGS)
tag = "assets-v1"
i = findfirst(==("--tag"), args)
if i !== nothing
    tag = args[i + 1]
    deleteat!(args, i:i+1)
end
names = isempty(args) ? ["depthanything", "neurallut", "rife"] : args   # the ported ones
for n in names
    haskey(MODELS, n) || error("unknown model $n; known: $(join(sort(collect(keys(MODELS))), ", "))")
end

println("binding artifacts against release tag `$tag`\n")
made = [pack(n, tag) for n in names]

total = sum(m -> m.bytes, made)
@printf("\n%d tarballs, %.1f MiB total, in gen/artifacts/\n", length(made), total / 2^20)
println("""
Not yet uploaded. To publish them:

    gh release upload $tag \\
""" * join(["        gen/artifacts/$(m.name).tar.gz" for m in made], " \\\n") * """
 \\
        --repo SimonDanisch/JuliaVision --clobber

Until then the binding is local: `assetdir()` resolves to the tree just created,
because that is what `create_artifact` put in the store. Uploading is how it
reaches anyone else.""")
