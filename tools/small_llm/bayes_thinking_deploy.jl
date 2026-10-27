isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")

function deploy_bayes_thinking(file_session;directory=joinpath(@__DIR__,"bayes_thinking_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    p.ready_for_generation&&length(p.jobs)==16&&p.decoding.thinking===true||error("Unfrozen cohort")
    updates=Dict{String,Vector{UInt8}}()
    for (file,hash) in pairs(p.source_sha256)
        file=String(file)
        basename(file)==file&&(endswith(file,".jl")||endswith(file,".py")||file in DOCUMENT_STUDY_JSON_SOURCES)||error("Public basename source only")
        file in ("bayes_thinking_admission.jl","score_bayes_thinking.jl","bayes_extraction_references.jl")&&error("No private grading code")
        bytes=read(joinpath(@__DIR__,file));bytes2hex(sha256(bytes))==hash||error("Source drift")
        updates[file]=bytes
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    actual=file_session(read(path)) do bytes
        S=Base.require(Main,:SHA);root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"bayes_thinking_results");path=joinpath(directory,"conditions.json")
        any(islink,(root,directory,path))&&error("Linked registration")
        if isfile(path);read(path)==bytes||error("Preserve conditions")
        else;mkpath(directory);open(io->write(io,bytes),path,"w")
        end
        for file in ("bayes_thinking_admission.jl","score_bayes_thinking.jl","bayes_extraction_references.jl")
            ispath(joinpath(root,file))&&error("Private grading material present")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==bytes2hex(sha256(read(path)))||error("Registration drift")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_bayes_thinking(file_session;directory=joinpath(@__DIR__,"bayes_thinking_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do ids
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/bayes_thinking_results"
        islink(root)&&error("Linked result root");found=Dict{String,Vector{UInt8}}()
        for id in ids
            basename(id)==id||error("Unsafe ID")
            for suffix in (".json","_attempt.json")
                path=joinpath(root,id*suffix)
                islink(path)&&error("Linked receipt")
                if isfile(path);found[id*suffix]=read(path);end
            end
            haskey(found,id*".json")&&!haskey(found,id*"_attempt.json")&&error("Missing attempt")
        end
        found
    end
    save_immutable_example_outputs(directory,files,ids)
end
