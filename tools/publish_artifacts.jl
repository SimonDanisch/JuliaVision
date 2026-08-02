"""
Package the model assets as Julia artifacts and bind them.

    julia --project=. tools/publish_artifacts.jl --dry-run     # tar + hash only
    julia --project=. tools/publish_artifacts.jl               # …and upload

The assets are generated, not committed: `tools/export_sam2.py`,
`tools/export_graphs.py` and `tools/convert_weights.py` produce them from
upstream checkpoints into `gen/`. That is fine for the machine that runs those
tools and useless for anybody else, so this turns the generated tree into
artifacts that Julia downloads on first use.

What it does, per artifact: copy the named files into a staging directory,
compute the `git-tree-sha1` Julia identifies it by, write a `.tar.gz`, upload it
to a GitHub release, and write the binding into the owning package's
`Artifacts.toml` with `lazy = true`.

`--dry-run` stops before uploading and prints the hashes, which is enough to
check that the split is right before spending a gigabyte of bandwidth.

The split is deliberate: reference activations are 1.2 GB and only the test
suite wants them, so they are their own artifact. Anybody segmenting a picture
downloads 943 MB, not 2.2 GB.
"""

using Pkg.Artifacts

const ROOT = normpath(joinpath(@__DIR__, ".."))
const GEN = joinpath(ROOT, "gen")
const PKGS = joinpath(ROOT, "dev", "JuliaVision")

"""GitHub release the tarballs are attached to. One tag for all of them, since
they are versioned together by the exporter that produced them."""
const REPO = "SimonDanisch/JuliaVision"
const TAG = "assets-v1"

"""
Each artifact: its name, the package whose `Artifacts.toml` binds it, and the
files it contains as `source in gen/` => `name inside the artifact`.

Listed explicitly rather than globbed. `gen/graphs/sam2-large` also holds
`op_histogram.json` and `pytorch_baseline.json`, which are measurements about
the model rather than part of it, and shipping them would be noise.
"""
const ARTIFACTS = [
    (name = "sam2-large", pkg = "SAM2Runner", files = [
        "graphs/sam2-large/sam2_encoder.json"  => "sam2_encoder.json",
        "graphs/sam2-large/sam2_decoder.json"  => "sam2_decoder.json",
        "graphs/sam2-large/weights.safetensors" => "weights.safetensors",
    ]),
    (name = "sam2-large-refs", pkg = "SAM2Runner", files = [
        "graphs/sam2-large/refs.safetensors"   => "refs.safetensors",
        "graphs/sam2-large/refs_manifest.json" => "refs_manifest.json",
    ]),
    (name = "matanyone", pkg = "MatAnyoneRunner", files = [
        "graphs/aten-autocast" => "graphs",          # a directory, copied whole
        "weights.safetensors"  => "weights.safetensors",
    ]),
]

"""Stage one artifact's files into `dir`, returning the bytes staged."""
function stage!(dir, files)
    total = 0
    for (src, dst) in files
        from = joinpath(GEN, src)
        ispath(from) || error("missing asset: $from — run the export tools first")
        to = joinpath(dir, dst)
        mkpath(dirname(to))
        cp(from, to)
        total += isdir(from) ?
            sum(filesize(joinpath(r, f)) for (r, _, fs) in walkdir(from) for f in fs; init = 0) :
            filesize(from)
    end
    total
end

function main(; dryrun = "--dry-run" in ARGS)
    isdir(GEN) || error("no $GEN — nothing to package")
    for a in ARTIFACTS
        toml = joinpath(PKGS, a.pkg, "Artifacts.toml")
        println("\n── $(a.name)  →  $(a.pkg)/Artifacts.toml")

        hash = create_artifact() do dir
            n = stage!(dir, a.files)
            println("   staged $(round(n / 1e6, digits = 1)) MB")
        end
        tarball = joinpath(tempdir(), "$(a.name).tar.gz")
        sha = archive_artifact(hash, tarball)
        println("   git-tree-sha1 $(bytes2hex(hash.bytes))")
        println("   sha256        $sha")
        println("   tarball       $tarball ($(round(filesize(tarball) / 1e6, digits = 1)) MB)")

        if dryrun
            println("   --dry-run: not uploading, not binding")
            continue
        end

        url = "https://github.com/$REPO/releases/download/$TAG/$(a.name).tar.gz"
        # Create the release once; every later artifact just uploads into it.
        notes = "Exported graphs, weights and reference activations, bound " *
                "lazily by SAM2Runner and MatAnyoneRunner."
        success(`gh release view $TAG --repo $REPO`) ||
            run(`gh release create $TAG --repo $REPO --title "Model assets" --notes $notes`)
        run(`gh release upload $TAG $tarball --repo $REPO --clobber`)

        bind_artifact!(toml, a.name, hash; download_info = [(url, sha)],
                       lazy = true, force = true)
        println("   bound in $toml")
    end
    dryrun && println("\nNothing uploaded. Drop --dry-run to publish.")
end

# `@__FILE__ && main()` would parse as `@__FILE__(&& main())` — a macro
# slurps the rest of the line, so this needs to be a block.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
