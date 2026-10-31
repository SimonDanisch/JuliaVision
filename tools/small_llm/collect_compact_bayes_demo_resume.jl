using SHA,JSON3

function save_verified_demo_artifacts(directory,artifacts,allowed)
    Set(String.(keys(artifacts)))⊆Set(allowed)||error("Unexpected artifact")
    for (name,item) in pairs(artifacts)
        basename(name)==name||error("Unsafe name")
        !isempty(item.bytes)&&bytes2hex(sha256(item.bytes))==item.sha256||error("Empty/changed transport bytes")
        path=joinpath(directory,name);islink(path)&&error("Linked destination")
        isfile(path)&&read(path)!=item.bytes&&error("Preserve different immutable artifact")
    end
    for (name,item) in pairs(artifacts)
        path=joinpath(directory,name);isfile(path)&&continue
        temporary,io=mktemp(directory)
        try
            write(io,item.bytes);flush(io);close(io)
            bytes2hex(sha256(read(temporary)))==item.sha256||error("Written artifact mismatch")
            mv(temporary,path;force=false)
        finally
            isopen(io)&&close(io)
            ispath(temporary)&&rm(temporary)
        end
    end
    length(artifacts)
end

function collect_compact_bayes_demo_resume(file_session;directory=joinpath(@__DIR__,"compact_bayes_demo_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    names=vcat([String(j.id)*suffix for j in p.jobs for suffix in (".json","_attempt.json","_lost.json")],
        ["recovery_gate.json"])
    artifacts=file_session(names) do names
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/compact_bayes_demo_results"
        islink(root)&&error("Linked result root")
        found=Dict{String,NamedTuple{(:bytes,:sha256),Tuple{Vector{UInt8},String}}}()
        for name in names
            basename(name)==name||error("Unsafe artifact name");path=joinpath(root,name)
            islink(path)&&error("Linked artifact")
            if isfile(path)
                bytes=read(path);isempty(bytes)&&error("Empty remote artifact")
                found[name]=(;bytes,sha256=bytes2hex(Base.invokelatest(S.sha256,bytes)))
            end
        end
        found
    end
    save_verified_demo_artifacts(directory,artifacts,names)
end
