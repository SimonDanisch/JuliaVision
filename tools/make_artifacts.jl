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
# Where the packages live, which depends on how `tools/` was reached. Symlinked
# into a parent repo (the VideoEdit layout) `ROOT` is that parent and the
# monorepo sits under `dev/JuliaVision`; run from inside a plain JuliaVision
# checkout, `ROOT` *is* the monorepo. Asking which one has the packages beats
# hardcoding either — the wrong guess fails inside `bind_artifact!` with a
# tempname error naming a path nobody wrote.
const JV = let nested = joinpath(ROOT, "dev", "JuliaVision")
    isdir(joinpath(nested, "DNNKernels")) ? nested :
    isdir(joinpath(ROOT, "DNNKernels"))   ? ROOT   :
    error("cannot find the JuliaVision packages from $ROOT")
end

# artifact name => (package that binds it, the export directory under gen/graphs)
const MODELS = Dict(
    # `neurallut` stays fp32 — measured, not assumed. Its autocast export is
    # **1.02x**, i.e. nothing, and `compare` flags it `trustworthy = false`. The
    # model is 39 ops and 0.46 ms of PyTorch; there is no tensor-core work in it
    # to win.
    "neurallut"     => ("NeuralLUTRunner", "neurallut"),
    # The fp16 exports, measured against the fp32 ones they replace, interleaved
    # on one clock plateau (2026-08-12):
    #
    #     rife           206.5 -> 168.1 ms   1.23x   rel rms 9.4e-06
    #     depthanything   46.7 ->  40.5      1.15x   rel rms 4.5e-04
    #
    # rife's largest per-channel deviation is 0.0039, which is exactly one 8-bit
    # level; depthanything's is 4.9e-03 on a 1.18..4.04 depth range.
    #
    # **Neither was shippable before `bmmpad`.** Depth Anything's autocast export
    # was 3.75x SLOWER than fp32 until two bugs in the batched-GEMM router were
    # fixed — see `planewise_worth` and `bmmpad` in DNNKernels' matmul.jl. Do not
    # read this pair as "fp16 always wins": it did not, for six days, and the
    # reason was ours.
    "rife"          => ("RIFERunner", "rife-fp16"),
    "depthanything" => ("DepthAnythingRunner", "depthanything-fp16"),
    # Not ported yet — listed so the instruction their `assetdir()` prints is one
    # that actually runs. Each errors here until its exporter has been run.
    # The **fp16** export, not the fp32 one beside it. Half the download (1.19 GiB
    # of weights against 2.37) and 6.3x the encode, for output that is
    # token-for-token identical: checked on three clips, including 19 s of real
    # speech where the two transcriptions match character for character. The two
    # two bugs that make fp16 unusable — an accumulator width in Lava's scalar
    # GEMM and `erf` evaluated in half — are both fixed.
    "whisper"       => ("WhisperRunner", "whisper-fp16"),
    # The fp32 encoder, shipped beside the fp16 one rather than instead of it.
    # `WhisperRunner.assetdir(:fp32)` resolves this; both are lazy, so a caller
    # downloads the one it asks for. It is not a slower default, it is the two
    # things half precision cannot be: the reference `verify_whisper.jl`'s
    # three-way gate measures against, and the fallback for a device without
    # cooperative matrices for fp16.
    "whisper-fp32"  => ("WhisperRunner", "whisper"),
    # The decoder is a SECOND artifact for the same package, not a bigger first
    # one: the encoder alone is the useful component for alignment and
    # embeddings, and it is 1.33 GiB before the decoder's 909 MiB is added.
    "whisper-decoder" => ("WhisperRunner", "whisperdec"),
    "basicvsrpp"    => ("BasicVSRRunner", "basicvsrpp-fp32"),
    "deepfilternet" => ("DeepFilterRunner", "deepfilternet"),
    "demucs"        => ("DemucsRunner", "demucs"),
    # `kokoro-fp16`, NOT `kokoro-dyn`. Both are length-generic (`symbols = ["f"]`),
    # so nothing is given up — but `kokoro-dyn` has zero fp16 buffers, and
    # `conv_coopmat_plan` gates on both operands being fp16 while this device has
    # no fp32 tensor-core path at all. Every one of the 90 convolutions therefore
    # declined the cooperative-matrix kernel, which an `opdouble` ablation prices
    # at +341 ms of a 477 ms utterance — about 72% of the model.
    #
    # Measured, same code, only the asset tree differing:
    #     kokoro-dyn (published)   min 380.9 ms
    #     kokoro-fp16              min 227.1 ms      1.68x
    #
    # The mixed-precision export (#71) landed but this mapping was never moved to
    # it, so the artifact kept shipping the fp32 graphs. Re-binding needs
    # `tools/make_artifacts.jl kokoro` AND an upload — the Artifacts.toml download
    # URL is content-addressed, so rewriting the hash without publishing leaves a
    # fresh clone unable to fetch it.
    "kokoro"        => ("KokoroRunner", "kokoro-fp16"),
    "propainter"    => ("ProPainterRunner", "propainter"),
    # Hunyuan3D-2.1 shape pipeline. Four artifacts rather than one tree: the
    # conditioner runs once per generation, the denoiser fifty times, the VAE
    # once and the geometry decoder some seven thousand times, so a caller has
    # no reason to fetch all four to use one. The denoiser is also 5.7 GiB on
    # its own — past what a GitHub release asset can hold, so publishing it
    # needs somewhere without that cap regardless of how it is bound here.
    "hunyuan3d"      => ("Hunyuan3DRunner", "hunyuan3d"),
    "hunyuan3d-cond" => ("Hunyuan3DRunner", "hunyuan3d-cond"),
    "hunyuan3d-vae"  => ("Hunyuan3DRunner", "hunyuan3d-vae"),
    "hunyuan3d-geo"  => ("Hunyuan3DRunner", "hunyuan3d-geo"),
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
#   * `whisper-fp32` is the first artifact whose NAME differs from its graph's.
#     Two artifacts carry the same encoder at two precisions, and `whispergraph`
#     opens `whisper.json` in whichever one it was given — so the fp32 tarball
#     must contain `whisper.json`, not `whisper-fp32.json`. Without this entry
#     `pack` looks for a file that does not exist, and had it not, it would have
#     shipped a graph under a name the runner never opens.
# Models whose weights ship as shards, so the graph artifact carries no weights.
const NOWEIGHTS = Set(["hunyuan3d"])

# Weight shards. A Julia artifact is one tree and a GitHub release asset caps at
# 2 GiB, so a 5.7 GiB tensor file cannot be published at all in one piece.
# `tools/shard_safetensors.jl` splits it byte-for-byte into pieces that fit, each
# bound separately; `Hunyuan3DRunner.hunyuan3dweights` merges them on load, which
# is why nothing downstream knows shards exist.
#
# artifact name => (package that binds it, dir under gen/shards, file)
const SHARDS = Dict(
    "hunyuan3d-dit-w1" => ("Hunyuan3DRunner", "hunyuan3d-dit", "weights-1of4.safetensors"),
    "hunyuan3d-dit-w2" => ("Hunyuan3DRunner", "hunyuan3d-dit", "weights-2of4.safetensors"),
    "hunyuan3d-dit-w3" => ("Hunyuan3DRunner", "hunyuan3d-dit", "weights-3of4.safetensors"),
    "hunyuan3d-dit-w4" => ("Hunyuan3DRunner", "hunyuan3d-dit", "weights-4of4.safetensors"),
    # Bonsai's official PTQ1 GGUF is 5.95 GB, also above GitHub's per-asset
    # limit. These are byte ranges rather than tensor shards: BonsaiRunner
    # concatenates them once into Scratch.jl's cache and verifies the original
    # upstream SHA-256 before the mmap-based GGUF reader sees it.
    "bonsai2-ptq1-p1" => ("BonsaiRunner", "bonsai2", "Ternary-Bonsai-2-27B-PTQ1_0-part-1of4"),
    "bonsai2-ptq1-p2" => ("BonsaiRunner", "bonsai2", "Ternary-Bonsai-2-27B-PTQ1_0-part-2of4"),
    "bonsai2-ptq1-p3" => ("BonsaiRunner", "bonsai2", "Ternary-Bonsai-2-27B-PTQ1_0-part-3of4"),
    "bonsai2-ptq1-p4" => ("BonsaiRunner", "bonsai2", "Ternary-Bonsai-2-27B-PTQ1_0-part-4of4"),
    # Qwen-Image 2.1's two compact checkpoints, 7.26 GB and 6.31 GB. Six and
    # five pieces rather than four each: `shard_safetensors.jl` splits on
    # tensor boundaries, so the pieces come out uneven, and four would have put
    # the largest past 2 GiB. As published they run 1.11 to 1.48 GiB, the last
    # of which is the largest asset in the release.
    #
    # Both merge back byte-exact — checked by reading the original and the
    # merged shards and comparing every tensor, 649 and 1762 of them, zero
    # mismatches. That check is what makes it safe for `QwenImageRunner` to
    # treat the merge as the checkpoint.
    ("qwenimage21-dit-w$i" => ("QwenImageRunner", "qwenimage21-dit",
                               "weights-$(i)of6.safetensors") for i in 1:6)...,
    ("qwenimage21-enc-w$i" => ("QwenImageRunner", "qwenimage21-enc",
                               "weights-$(i)of5.safetensors") for i in 1:5)...,
)

function packshard(name::AbstractString, tag::AbstractString)
    pkg, dir, file = SHARDS[name]
    src = joinpath(ROOT, "gen", "shards", dir, file)
    isfile(src) || error("no shard at $src — make them with " *
                         "`julia --project=. tools/shard_safetensors.jl <weights> <outdir> <n>`")
    hash = create_artifact() do d
        cp(src, joinpath(d, file))
        copylicense(name, d)
    end
    return finishartifact(name, pkg, hash, tag, String[])
end

# ── Licences that have to travel WITH the weights.
#
# Most of what is packed here is permissively licensed and says so once, in the
# package. Qwen-Image 2.1 is the first that does not: section 3 of the Qwen
# Research License Agreement permits redistribution only if every recipient is
# given a copy of the Agreement, modified files carry a notice saying they were
# changed, and a `Notice` file carries the attribution text quoted in 3(c).
#
# The obligation attaches to the MATERIALS, not to the repository, and each
# artifact is separately downloadable — someone who fetches one weight shard has
# received the Materials and has to receive the Agreement with them. So the
# licence goes into every tarball of the family rather than once into the small
# one, which costs 15 KiB against 14 GiB.
#
# artifact name => directory under tools/licenses holding LICENSE and NOTICE
const LICENSED = Dict{String,String}(
    n => "qwenimage21" for n in vcat(
        ["qwenimage21", "qwenimage21-vae"],
        ["qwenimage21-dit-w$i" for i in 1:6],
        ["qwenimage21-enc-w$i" for i in 1:5]))

function copylicense(name::AbstractString, dest::AbstractString)
    haskey(LICENSED, name) || return nothing
    src = joinpath(JV, "tools", "licenses", LICENSED[name])
    for f in ("LICENSE", "NOTICE")
        path = joinpath(src, f)
        isfile(path) || error("$name must ship $f and $path does not exist")
        cp(path, joinpath(dest, f))
    end
    return nothing
end

const GRAPHFILES = Dict(
    "whisper-decoder" => ["whisperdec.json", "whispercross.json"],
    "whisper-fp32" => ["whisper.json"],
    "kokoro" => ["kokorotext.json", "kokorovoc.json",
                 "vocab.json", "voices.safetensors", "lexicon.json"],
    # `export_hunyuan3d.py` names the graph after the PART, so none of the four
    # match their artifact name. The VAE also ships `scale_factor.json` (the
    # sampler works in a scaled latent space and cannot reconstruct without it)
    # and the conditioner ships `preprocess.json`.
    "hunyuan3d"      => ["hunyuan3d_dit.json"],
    "hunyuan3d-cond" => ["hunyuan3d_cond.json", "preprocess.json"],
    "hunyuan3d-vae"  => ["hunyuan3d_vae.json", "scale_factor.json"],
    "hunyuan3d-geo"  => ["hunyuan3d_geo.json"],
)

# ── Exports whose runnable set is an explicit list rather than `SHIPPED`.
#
# `pack` above assumes one graph called `<name>.json` beside a
# `weights.safetensors` and an `op_histogram.json`. Qwen-Image 2.1 has none of
# those names: three graphs exported at different times, one `*_constants`
# file per graph holding the tensors the trace lifted out, one histogram each,
# and a tokenizer directory that is not an export at all. Bending `pack` to
# fit would mean a second exclusion set beside `NOWEIGHTS`, so the list is
# given directly. Entries may name directories as well as files.
#
# The split is the same one `SHIPPED` makes: what a caller needs to RUN the
# model. `vae_reference.safetensors` (17 MB of PyTorch activations that only
# `tools/verify_qwenimage21.py` reads) stays out, and `processor/` ships the
# three files `QwenTokenizer` opens rather than the seven the repository has —
# `tokenizer.json` alone is 11 MB of BPE table this package never reads.
#
# The denoiser graph is exported at ONE prompt length. See `context_tokens` in
# `qwenimage21_export.json`: its text stream is part of a joint attention and
# the export passes `attention_mask=None`, so a shorter prompt cannot be padded
# into it. The graph here is the 22-token one `examples/generate.jl` uses.
# Making it length-generic means threading the mask back through
# `tools/export_qwenimage21.py` and is not something the packaging can fix.
#
# artifact name => (package that binds it, dir under gen/graphs, what to take)
const TREES = Dict(
    "qwenimage21" => ("QwenImageRunner", "qwenimage21",
        ["qwenimage21_transformer.json", "qwenimage21_text_encoder.json",
         "qwenimage21_export.json", "qwenimage21_text_encoder_export.json",
         "transformer_constants.safetensors", "text_encoder_constants.safetensors",
         "transformer_op_histogram.json", "text_encoder_op_histogram.json",
         "processor"]),
    # The VAE is its own artifact for the reason Hunyuan3D's four are: it is
    # 644 MB and runs once per image, against the denoiser's 6.8 GB twenty
    # times. Its graph travels with its weights so one `vaedir()` resolves
    # both.
    "qwenimage21-vae" => ("QwenImageRunner", "qwenimage21-vae",
        ["qwenimage21_vae_decoder.json", "vae.safetensors", "vae_op_histogram.json"]),
    # SAM 2.1: two graphs, so `pack`'s "one `<name>.json`" does not fit either.
    # It was bound by hand before this entry existed, which is how the decoder
    # came to ship as the fp32 export long after `--decoder-precision` had been
    # measured at 11.30 -> 3.93 ms in autocast: nothing rebuilt it from the
    # exporter, so nothing carried the flag. `refs.safetensors` stays out — it
    # is 1.2 GB and only the tests read it, which is what `sam2-large-refs` is
    # for.
    "sam2-large" => ("SAM2Runner", "sam2-large",
        ["sam2_encoder.json", "sam2_decoder.json",
         "op_histogram.json", "weights.safetensors"]),
)

"""
Pack `name` from the explicit file list in [`TREES`](@ref).

Unlike [`pack`](@ref) nothing is implied: every file and directory that ends up
in the tarball is named in the table, which is what lets an export with its own
naming scheme ship without teaching `pack` about it.
"""
function packtree(name::AbstractString, tag::AbstractString)
    pkg, dirname_, want = TREES[name]
    src = joinpath(ROOT, "gen", "graphs", dirname_)
    isdir(src) || error("no export at $src — run `uv run tools/export_$(name).py`")
    for f in want
        ispath(joinpath(src, f)) || error("$src is missing $f")
    end
    hash = create_artifact() do dir
        for f in want
            cp(joinpath(src, f), joinpath(dir, f))
        end
        copylicense(name, dir)
    end
    return finishartifact(name, pkg, hash, tag, want)
end

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
    # The per-node PyTorch activations `verify_sam2.jl` and `bench_sam2.jl`
    # compare against. They are dumped under the SAME precision policy the
    # graphs were exported with — `dump_sam2_refs.py` says why — so this
    # artifact and `sam2-large` have to be rebound together or the node-by-node
    # pass reports a dtype difference as a bug.
    "sam2-large-refs" => ("SAM2Runner", "sam2-large", ["refs.safetensors"]),
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
    # `NOWEIGHTS` drops `weights.safetensors`: a model whose weights are sharded
    # ships them as their own artifacts, and leaving them here too would both
    # double the download and push this tarball back over the release cap.
    shipped = name in NOWEIGHTS ? filter(!=("weights.safetensors"), SHIPPED) : SHIPPED
    want = vcat(get(GRAPHFILES, name, ["$name.json"]), shipped)
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
        haskey(FIXTURES, n) || haskey(SHARDS, n) || haskey(TREES, n) || error(
        "unknown target $n; known: " *
        join(sort(vcat(collect(keys(MODELS)), collect(keys(REFS)),
                       collect(keys(CHECKPOINTS)), collect(keys(FIXTURES)),
                       collect(keys(SHARDS)), collect(keys(TREES)))), ", "))
end

println("binding artifacts against release tag `$tag`\n")
made = [haskey(REFS, n) ? packrefs(n, tag) :
        haskey(FIXTURES, n) ? packfixtures(n, tag) :
        haskey(TREES, n) ? packtree(n, tag) :
        haskey(SHARDS, n) ? packshard(n, tag) :
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
