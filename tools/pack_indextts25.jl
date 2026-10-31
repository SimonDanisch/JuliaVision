"""
Build IndexTTS25Runner's lazy artifacts, including separate weight components.

    include("tools/pack_indextts25.jl")
    pack_indextts25("/path/to/gen/indextts25"; tag="indextts25-v1")

The result is installed locally and ready for release upload. Packing does not
publish: it writes the exact release-upload command after all files are ready.
"""

using Pkg.Artifacts, JSON3, SHA, Tar
include(joinpath(@__DIR__, "shard_safetensors.jl"))

function pack_indextts25(generated::AbstractString; tag::String = "indextts25-v1",
                         modeldir::AbstractString = expanduser("~/models/IndexTTS-2.5"),
                         germandir::AbstractString = expanduser("~/models/index-tts-2.5-german"),
                         upstream::AbstractString = expanduser("~/tools/index-tts"))
    pkg = normpath(joinpath(@__DIR__, "..", "IndexTTS25Runner"))
    toml = joinpath(pkg, "Artifacts.toml")
    graphs = joinpath(generated, "graphs")
    archives = joinpath(generated, "artifacts")
    mkpath(archives)
    parts = ["w2v", "campplus", "emo", "codec", "codec_fp16", "regulator", "dit", "dit_fp16", "bigvgan",
             "gpt_prefill", "gpt_de_prefill", "emotion_text"]
    manifests = Dict{String,Any}()
    names = String[]
    function notices(dest; german = false)
        cp(joinpath(modeldir, "LICENSE"), joinpath(dest, "LICENSE"))
        cp(joinpath(pkg, "NOTICE"), joinpath(dest, "NOTICE"))
        cp(joinpath(pkg, "APACHE_LICENSE"), joinpath(dest, "APACHE_LICENSE"))
        if german
            cp(joinpath(germandir, "LICENSE"), joinpath(dest, "GERMAN_LICENSE"))
            cp(joinpath(germandir, "NOTICE.md"), joinpath(dest, "GERMAN_NOTICE.md"))
        end
    end
    function finish(name, hash)
        tarball = joinpath(archives, "$name.tar.gz")
        pending = tarball * ".partial"
        # Parallel gzip at level 1 is standard gzip, so Pkg can install it on
        # every platform; compression does not affect the artifact's tree hash.
        if Sys.which("pigz") === nothing
            archive_artifact(hash, pending)
        else
            run(`tar -I 'pigz -1' -cf $pending -C $(artifact_path(hash)) .`)
        end
        filesize(pending) < 2^31 || error("release asset exceeds 2 GiB: $name")
        digest = bytes2hex(open(sha256, pending))
        mv(pending, tarball; force = true)
        url = "https://github.com/SimonDanisch/JuliaVision/releases/download/$tag/$name.tar.gz"
        bind_artifact!(toml, name, hash; lazy = true, force = true, download_info = [(url, digest)])
        push!(names, name)
        println(name, " ", hash, " ", round(filesize(tarball) / 2^20; digits = 1), " MiB")
        flush(stdout)
    end
    function completed(name)
        meta = artifact_meta(name, toml)
        meta === nothing && return false
        artifact_exists(Base.SHA1(meta["git-tree-sha1"])) || return false
        tarball = joinpath(archives, "$name.tar.gz")
        isfile(tarball) || return false
        digest = bytes2hex(open(sha256, tarball))
        return any(d -> d["sha256"] == digest, get(meta, "download", []))
    end
    for part in parts
        src = joinpath(graphs, part, "weights.safetensors")
        nshards = filesize(src) > 1800 * 2^20 ? 2 : 1
        components = ["indextts25-$part-weights" * (nshards == 1 ? "" : "-$i") for i in 1:nshards]
        if all(completed, components)
            manifests[part] = [(; artifact = name, file = nshards == 1 ? "weights.safetensors" : "weights-$(i)of2.safetensors")
                               for (i, name) in enumerate(components)]
            append!(names, components)
            println("Reusing verified ", part, " artifacts")
            flush(stdout)
            continue
        end
        files = if filesize(src) > 1800 * 2^20
            output = joinpath(generated, "shards", part)
            mkpath(output)
            shard(src, output, 2)
            [joinpath(output, "weights-$(i)of2.safetensors") for i in 1:2]
        else
            [src]
        end
        manifest = Any[]
        for (i, file) in enumerate(files)
            suffix = length(files) == 1 ? "" : "-$i"
            name = "indextts25-$part-weights$suffix"
            basename_ = basename(file)
            hash = create_artifact() do dest
                cp(file, joinpath(dest, basename_))
                notices(dest; german = startswith(part, "gpt_de"))
            end
            finish(name, hash)
            push!(manifest, (; artifact = name, file = basename_))
        end
        manifests[part] = manifest
    end
    hash = create_artifact() do dest
        for part in parts
            target = joinpath(dest, "graphs", part)
            mkpath(target)
            jsons = part == "emotion_text" ? ["emotion_prefill.json", "emotion_decode.json", "config.json",
                                             "vocab.json", "merges.txt", "added_tokens.json", "tokenizer.json", "chat_template.jinja"] :
                    ["$part.json"]
            for file in jsons
                cp(joinpath(graphs, part, file), joinpath(target, file))
            end
            write(joinpath(target, "weights.json"), JSON3.write(manifests[part]))
        end
        mkpath(joinpath(dest, "graphs", "aux"))
        cp(joinpath(graphs, "aux", "matrices.safetensors"), joinpath(dest, "graphs", "aux", "matrices.safetensors"))
        target = joinpath(dest, "graphs", "normalizer")
        mkpath(target)
        for lang in ("en", "zh"), kind in ("tagger", "verbalizer")
            file = "$(lang)_$kind.safetensors"
            cp(joinpath(graphs, "normalizer", file), joinpath(target, file))
        end
        cp(joinpath(modeldir, "multilingual_zh_ja_yue_char_del.tiktoken"), joinpath(dest, "tokenizer.tiktoken"))
        notices(dest; german = true)
    end
    finish("indextts25", hash)
    hash = create_artifact() do dest
        cp(joinpath(generated, "refs"), joinpath(dest, "refs"))
        mkpath(joinpath(dest, "examples"))
        clips = Set{String}()
        for file in readdir(joinpath(generated, "refs"); join = true)
            endswith(file, ".json") || continue
            meta = JSON3.read(read(file, String))
            for key in (:spk, :emo)
                haskey(meta, key) && push!(clips, String(meta[key]))
            end
        end
        for clip in clips
            cp(joinpath(upstream, "examples", clip), joinpath(dest, "examples", clip))
        end
        for part in ("normalizer", "emotion_text")
            mkpath(joinpath(dest, part))
            for file in readdir(joinpath(graphs, part))
                (startswith(file, "reference") || file == "battery.json") || continue
                cp(joinpath(graphs, part, file), joinpath(dest, part, file))
            end
        end
        notices(dest; german = true)
    end
    finish("indextts25-refs", hash)
    println("Ready to upload:")
    println("gh release upload $tag ", join((joinpath(archives, "$name.tar.gz") for name in names), " "), " --repo SimonDanisch/JuliaVision")
    return names
end
