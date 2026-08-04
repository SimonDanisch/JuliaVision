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
    # The decoder is a SECOND artifact for the same package, not a bigger first
    # one: the encoder alone is the useful component for alignment and
    # embeddings, and it is 1.33 GiB before the decoder's 909 MiB is added.
    "whisper-decoder" => ("WhisperRunner", "whisperdec"),
    "basicvsrpp"    => ("BasicVSRRunner", "basicvsrpp-fp32"),
    "deepfilternet" => ("DeepFilterRunner", "deepfilternet"),
    "demucs"        => ("DemucsRunner", "demucs"),
    "kokoro"        => ("KokoroRunner", "kokoro-dyn"),
    "propainter"    => ("ProPainterRunner", "propainter"),
)

# Files a caller needs to run the model. Anything else in the export directory —
# above all the references — is developer material and stays out of the tarball.
const SHIPPED = ["op_histogram.json", "weights.safetensors"]   # + "<name>.json"

# Models whose runnable set is not just `<name>.json`, and why.
#
#   * the Whisper decoder ships TWO graphs — the step and the cross-attention
#     K/V projection that feeds it — because they run at different rates (per
#     token vs per 30 s window) and fusing them would recompute 1500 positions
#     for every token.
#   * Kokoro ships two graphs because the model decides its own output length in
#     the middle (see `KokoroRunner`), plus three host-side tables: the phoneme
#     vocabulary the embedding indexes with, the 54 voice packs, and the G2P
#     lexicon. Without those the package loads and cannot be handed a sentence.
const GRAPHFILES = Dict(
    "whisper-decoder" => ["whisperdec.json", "whispercross.json"],
    "kokoro" => ["kokorotext.json", "kokorovoc.json",
                 "vocab.json", "voices.safetensors", "lexicon.json"],
)

# ── Upstream checkpoints, for models that are FETCHED but not yet TRACED.
#
# `MODELS` above packs `gen/graphs/<name>` — *our* export. These pack `gen/<name>`
# — the upstream `.pth`/`.th` the exporter reads. The two are different things and
# confusing them is what made this table necessary: a model whose checkpoints are
# sitting in `gen/` looks, from the outside, exactly like one that was never
# downloaded, and the port then starts by re-fetching several hundred MB by hand
# from HuggingFace and GitHub releases.
#
# Binding them makes that fetch content-addressed and reproducible on any machine,
# which is the whole point of doing it BEFORE the port rather than after. The
# runner's `assetdir()` resolves; its `ready()` stays false, because a checkpoint
# is not a graph. When the export lands, re-pack the same artifact NAME from
# `gen/graphs/<name>` via `MODELS` and the binding is replaced.
#
# artifact name => (package that binds it, the fetch directory under gen/)
const CHECKPOINTS = Dict(
    "propainter-ckpt"    => ("ProPainterRunner", "propainter"),
    "kokoro-ckpt"        => ("KokoroRunner", "kokoro"),
    "basicvsrpp-ckpt"    => ("BasicVSRRunner", "basicvsrpp"),
    "deepfilternet-ckpt" => ("DeepFilterRunner", "deepfilternet"),
    "demucs-ckpt"        => ("DemucsRunner", "demucs"),
)

"""
Pack the upstream checkpoints for `name`, whole directory.

Everything in `gen/<dir>` goes in, unlike [`pack`](@ref) — there is no
run-it/verify-it split to make yet, and a `.pth` this table names is by
definition a file the port needs. Anything genuinely developer-only in there
(`basicvsrpp`'s `refs.safetensors` and `node_stats.json`) is small next to the
checkpoints and not worth a second table to exclude.
"""
function packcheckpoints(name::AbstractString, tag::AbstractString)
    pkg, dirname_ = CHECKPOINTS[name]
    src = joinpath(ROOT, "gen", dirname_)
    isdir(src) || error("no checkpoints at $src — fetch them with `uv run tools/fetch.py $dirname_`")
    files = sort(filter(f -> isfile(joinpath(src, f)), readdir(src)))
    isempty(files) && error("$src is empty")

    hash = create_artifact() do dir
        for f in files
            cp(joinpath(src, f), joinpath(dir, f))
        end
    end

    return finishartifact(name, pkg, hash, tag, String[])
end

"""
    finishartifact(name, pkg, hash, tag, have) -> NamedTuple

Tarball, bind, report. The shared tail of every packer: `archive_artifact` for
the sha256 the download is validated against, `bind_artifact!` for the tree hash
that names it in the store, and a line saying which `Artifacts.toml` moved.

**The two hashes are not interchangeable and `Pkg` checks both** — the tree hash
names the artifact, the sha256 validates the tarball. Both come out of the
Artifacts API here rather than being computed by hand, because a hand-computed
tree hash that happens to be wrong fails at someone else's `Pkg.instantiate` and
nowhere earlier.

`have` is only for the report; an empty one prints no bracket.
"""
function finishartifact(name::AbstractString, pkg::AbstractString, hash,
                        tag::AbstractString, have = String[])
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
    @printf("%-14s %7.1f MiB  tree %s%s\n", name, bytes / 2^20, string(hash)[1:12],
            isempty(have) ? "" : "  [" * join(have, ", ") * "]")
    @printf("%-14s          sha256 %s\n", "", sha[1:12])
    @printf("%-14s          -> %s\n", "", relpath(toml, ROOT))
    return (; name, tarball, bytes, sha, hash)
end

# Reference artifacts: the PyTorch activations the layer-by-layer parity tests
# compare against. Separate from a model's own tarball for the reason `SHIPPED`
# states — someone matting a clip should not download a gigabyte of fixtures —
# and `sam2-large-refs` already set the shape beside `sam2-large`.
#
# This table did not exist, which is why `matanyone-refs` was bound in no
# `Artifacts.toml` anywhere and the MatAnyone parity gate silently ran ONE
# assertion instead of 61 while reporting green. `MatAnyoneRunner.refsdir()`'s
# error told you to run `make_artifacts.jl matanyone-refs`, which would have
# died on a `KeyError`.
#
# artifact name => (package that binds it, precisions to look for)
const REFS = Dict(
    "matanyone-refs" => ("MatAnyoneRunner", ["autocast", "fp32"]),
)

# ── Test fixtures: a named list of files out of an export directory.
#
# The same principle as `REFS` — what only the tests read does not belong in the
# artifact a caller downloads to run the model — but without MatAnyone's
# per-precision layout, which does not generalise.
#
# The reason these are an artifact at all rather than a path into `gen/` is the
# one `DNNKernels/src/assets.jl` gives at length: a test whose fixtures come from
# a local export tree passes for whoever ran the exporter and is *unreachable*
# for everyone else, so the parity gate — the one check that catches a kernel
# which is fast and subtly wrong — runs on exactly one machine.
#
# artifact name => (package that binds it, export dir under gen/graphs, files)
const FIXTURES = Dict(
    "kokoro-refs" => ("KokoroRunner", "kokoro-dyn",
                      ["refs_speak.safetensors", "refs_speak.json",
                       "g2p_reference.json"]),
)

"""
Pack the test fixtures for `name`: the named files, and nothing else.
"""
function packfixtures(name::AbstractString, tag::AbstractString)
    pkg, dirname_, files = FIXTURES[name]
    src = joinpath(ROOT, "gen", "graphs", dirname_)
    for f in files
        isfile(joinpath(src, f)) || error("$src is missing $f — regenerate it first")
    end
    hash = create_artifact() do dir
        for f in files
            cp(joinpath(src, f), joinpath(dir, f))
        end
    end
    return finishartifact(name, pkg, hash, tag, files)
end

"""
Pack the reference activations for `name`.

Layout is what the runner reads and nothing more: `refs-<p>.safetensors` and
`refs_manifest-<p>.json` at the root, `graphs/aten-<p>/` beside them. A precision
with no dumped references is skipped rather than fatal — one is enough to make
the gate real, and `matanyoneprecisions()` reports whichever are present.
"""
function packrefs(name::AbstractString, tag::AbstractString)
    pkg, precisions = REFS[name]
    gen = joinpath(ROOT, "gen")
    have = filter(p -> isfile(joinpath(gen, "refs-$p.safetensors")), precisions)
    isempty(have) && error("""
        no reference dumps under $gen for $name. Produce them with, per precision:
            uv run tools/dump_refs.py --precision <p> --max-size 128""")

    hash = create_artifact() do dir
        for p in have
            cp(joinpath(gen, "refs-$p.safetensors"), joinpath(dir, "refs-$p.safetensors"))
            mf = joinpath(gen, "refs_manifest-$p.json")
            isfile(mf) || error("$mf is missing; re-run dump_refs.py --precision $p")
            cp(mf, joinpath(dir, "refs_manifest-$p.json"))
            src = joinpath(gen, "graphs", "aten-$p")
            isdir(src) || error("$src is missing; re-run export_graphs.py --precision $p")
            mkpath(joinpath(dir, "graphs"))
            cp(src, joinpath(dir, "graphs", "aten-$p"))
        end
    end

    return finishartifact(name, pkg, hash, tag, have)
end

repo_url(tag, name) =
    "https://github.com/SimonDanisch/JuliaVision/releases/download/$tag/$name.tar.gz"

function pack(name::AbstractString, tag::AbstractString)
    pkg, dirname_ = MODELS[name]
    src = joinpath(ROOT, "gen", "graphs", dirname_)
    isdir(src) || error("no export at $src — run `uv run tools/export_$(name).py`")

    # `<name>.json` by default; see `GRAPHFILES` for the two that need more.
    want = vcat(get(GRAPHFILES, name, ["$name.json"]), SHIPPED)
    for f in want
        isfile(joinpath(src, f)) || error("$src is missing $f")
    end

    # Files that come from the UPSTREAM checkpoint directory rather than from our
    # export. The decoder needs two: `tokenizer.json` (50257 vocabulary entries
    # and 50000 ranked merges — the BPE table, without which the runner returns
    # token ids instead of text) and `generation_config.json` (the suppression
    # lists and special-token ids the decoding policy reads). Neither is produced
    # by `torch.export`, and shipping them here is what makes `whisper()` a
    # complete transcriber from a single download.
    extra = name == "whisper-decoder" ?
        [joinpath(ROOT, "gen", "whisper", f)
         for f in ("tokenizer.json", "generation_config.json")] : String[]
    for f in extra
        isfile(f) || error("missing $f — fetch the upstream checkpoint first")
    end

    # `create_artifact` hands the callback a fresh directory and returns the
    # tree hash of whatever was put in it. Copying file by file rather than the
    # whole directory is what keeps the references out — and keeps the hash
    # stable when a new developer-only file appears beside them.
    hash = create_artifact() do dir
        for f in want
            cp(joinpath(src, f), joinpath(dir, f))
        end
        for f in extra
            cp(f, joinpath(dir, basename(f)))
        end
    end

    return finishartifact(name, pkg, hash, tag, String[])
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
    haskey(MODELS, n) || haskey(REFS, n) || haskey(CHECKPOINTS, n) ||
        haskey(FIXTURES, n) || error(
        "unknown target $n; known: " *
        join(sort(vcat(collect(keys(MODELS)), collect(keys(REFS)),
                       collect(keys(CHECKPOINTS)), collect(keys(FIXTURES)))), ", "))
end

println("binding artifacts against release tag `$tag`\n")
made = [haskey(REFS, n) ? packrefs(n, tag) :
        haskey(FIXTURES, n) ? packfixtures(n, tag) :
        haskey(CHECKPOINTS, n) ? packcheckpoints(n, tag) : pack(n, tag) for n in names]

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
