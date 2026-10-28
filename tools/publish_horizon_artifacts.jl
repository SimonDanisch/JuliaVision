#=
Publish HorizonRunner's fp16 export without keeping packaging duplicates.

    julia --project=. tools/publish_horizon_artifacts.jl <export> <tokenizer> <tiny> <workdir>

Uses gh's existing login and uploads to JuliaVision's assets-v1 release. Requires
GNU tar, python3 and gzip (or HORIZON_GZIP pointing to pigz). Each tensor is copied byte for
byte into a safetensors shard and verified against the source before upload.
Only two compressed archives are staged at once; verified uploads are removed.
The original export is never removed by this script. Re-running reuses immutable
release assets after checking GitHub's SHA-256 digest.
=#
using Pkg.Artifacts, SHA, JSON3, TOML
include(joinpath(@__DIR__, "shard_safetensors.jl"))

const REPO = "SimonDanisch/JuliaVision"
const TAG = "assets-v1"
const BINDINGS = joinpath(@__DIR__, "..", "HorizonRunner", "Artifacts.toml")
const LICENSEDIR = joinpath(@__DIR__, "licenses", "horizon32b")

function notices(dest)
    for f in readdir(LICENSEDIR)
        cp(joinpath(LICENSEDIR, f), joinpath(dest, f))
    end
end

function tensordigests(path)
    # hashlib uses the platform's accelerated SHA implementation; hashing this
    # 65 GiB export with scalar Julia SHA needlessly adds several minutes.
    script = joinpath(@__DIR__, "safetensors_hashes.py")
    JSON3.read(read(`python3 $script $path`, String), Dict{String,String})
end

function remoteasset(name)
    release = JSON3.read(read(`gh api repos/$REPO/releases/tags/$TAG`, String))
    # The release has more than one page of assets.
    endpoint = "repos/$REPO/releases/$(release.id)/assets?per_page=100"
    query = ".[] | select(.name == \"$name\")"
    found = strip(read(`gh api --paginate $endpoint --jq $query`, String))
    isempty(found) ? nothing : JSON3.read(found)
end

function horizongroups(entries)
    # The two 2.39 GiB embedding/head tensors compress below the release cap.
    # Keep them whole. All other shards fit even before compression. Unlike a
    # fixed shard count, this never leaves an oversized tail bucket.
    out, cur, bytes = Vector{typeof(entries)}(), similar(entries, 0), 0
    for entry in entries
        n = entry[2].data_offsets[2] - entry[2].data_offsets[1]
        if !isempty(cur) && bytes + n > 3 * 2^29
            push!(out, cur)
            cur, bytes = similar(entries, 0), 0
        end
        push!(cur, entry)
        bytes += n
    end
    isempty(cur) || push!(out, cur)
    out
end

function publish(name, hash, work)
    archive = "$name-$(hash).tar.gz"
    path = joinpath(work, archive)
    a = remoteasset(archive)
    # A checked-in binding already attests to both hashes. Reuse it when the
    # server still has those exact bytes, even if this machine uses gzip rather
    # than pigz (whose compressed streams differ for the same artifact tree).
    if a !== nothing && isfile(BINDINGS)
        pinned = get(TOML.parsefile(BINDINGS), name, nothing)
        if pinned !== nothing && pinned["git-tree-sha1"] == string(hash)
            for download in get(pinned, "download", [])
                if download["url"] == a.browser_download_url && a.digest == "sha256:$(download["sha256"])"
                    println("Reused verified remote $name"); flush(stdout)
                    return (; name, hash, digest=String(download["sha256"]), url=String(a.browser_download_url))
                end
            end
        end
    end
    compressor = get(ENV, "HORIZON_GZIP", "gzip")
    opts = occursin("pigz", basename(compressor)) ? ["-1", "-p", "8", "-n", "-c"] : ["-1", "-n", "-c"]
    run(pipeline(pipeline(`tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C $(artifact_path(hash)) -cf - .`, `$compressor $opts`); stdout=path))
    filesize(path) < 2^31 || error("$archive exceeds GitHub's 2 GiB limit; original retained")
    digest = bytes2hex(open(sha256, path))
    if a === nothing
        println("Uploading $archive ($(round(filesize(path)/2^30; digits=2)) GiB)"); flush(stdout)
        run(`gh release upload $TAG $path --repo $REPO`)
        a = remoteasset(archive)
    end
    a !== nothing && a.digest == "sha256:$digest" || error("remote digest mismatch for $archive")
    rm(path)
    println("Verified remote $name: $digest"); flush(stdout)
    (; name, hash, digest, url=String(a.browser_download_url))
end

function main(exportdir, tokenizer, tiny, work)
    mkpath(work)
    src = joinpath(exportdir, "weights.safetensors")
    info = shardinfo(src)
    gs = horizongroups(info.entries)
    snapshot = joinpath(work, "source-tensors.json")
    stamp = (; path=abspath(src), bytes=filesize(src), mtime=mtime(src))
    cached = isfile(snapshot) ? JSON3.read(read(snapshot, String)) : nothing
    expected = if cached !== nothing && cached.path == stamp.path &&
                  cached.bytes == stamp.bytes && cached.mtime == stamp.mtime
        Dict{String,String}(String(k) => String(v) for (k, v) in pairs(cached.tensors))
    else
        hashes = tensordigests(src)
        open(snapshot, "w") do io
            JSON3.write(io, merge(stamp, (; tensors=hashes)))
        end
        hashes
    end
    println("Source verified: $(length(expected)) tensors, $(length(gs)) shards"); flush(stdout)
    jobs = Task[]
    results = Any[]
    parts = Any[]
    function enqueue(name, hash)
        push!(jobs, @async publish(name, hash, work))
        if length(jobs) >= 2
            push!(results, fetch(popfirst!(jobs)))
        end
    end
    for (i, group) in enumerate(gs)
        name = "horizon32b-fp16-w$i"
        file = "weights-$i-of-$(length(gs)).safetensors"
        hash = create_artifact() do dir
            dst = joinpath(dir, file)
            writeshard(src, dst, group, info.base)
            got = tensordigests(dst)
            all(got[k] == expected[k] for k in keys(got)) || error("tensor mismatch in $file")
            notices(dir)
        end
        push!(parts, (; artifact=name, file))
        println("Verified local shard $i/$(length(gs)): $hash"); flush(stdout)
        enqueue(name, hash)
    end
    graphhash = create_artifact() do dir
        for f in readdir(exportdir)
            endswith(f, ".json") && cp(joinpath(exportdir, f), joinpath(dir, f))
        end
        tokdir = joinpath(dir, "tokenizer"); mkpath(tokdir)
        for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
                  "generation_config.json", "configuration_k2_horizon.py", "modeling_k2_horizon.py")
            cp(joinpath(tokenizer, f), joinpath(tokdir, f))
        end
        open(joinpath(dir, "weights.index.json"), "w") do io
            JSON3.write(io, (; format=1, shards=parts, tensor_sha256=expected))
        end
        notices(dir)
    end
    enqueue("horizon32b", graphhash)
    tinyhash = create_artifact() do dir
        for f in readdir(tiny)
            cp(joinpath(tiny, f), joinpath(dir, f))
        end
        notices(dir)
    end
    enqueue("horizon32b-tiny", tinyhash)
    append!(results, fetch.(jobs))
    # Commit the binding only after EVERY asset is uploaded and verified.
    table = BINDINGS * ".tmp"
    isfile(table) && rm(table)
    for r in results
        bind_artifact!(table, r.name, r.hash; lazy=true, download_info=[(r.url, r.digest)])
    end
    mv(table, BINDINGS; force=true)
    println("Published $(length(results)) verified artifacts; bound $BINDINGS")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 4 || error("expected <export> <tokenizer> <tiny> <workdir>")
    main(ARGS...)
end
