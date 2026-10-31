isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")

function deploy_compact_bayes(file_session;directory=joinpath(@__DIR__,"compact_bayes_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    p.ready_for_generation&&length(p.jobs)==8&&p.decoding.thinking===false||error("Frozen compact cohort required")
    updates=Dict{String,Vector{UInt8}}()
    for (f,h) in pairs(p.source_sha256)
        f=String(f);basename(f)==f&&(endswith(f,".jl")||endswith(f,".py")||f in DOCUMENT_STUDY_JSON_SOURCES)||error("Public source basename only")
        f in ("compact_bayes_admission.jl","compact_bayes_json.jl","score_compact_bayes.jl","bayes_extraction_references.jl")&&error("No private grading/solver data")
        bytes=read(joinpath(@__DIR__,f));bytes2hex(sha256(bytes))==h||error("Public source drift");updates[f]=bytes
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    actual=file_session(read(path)) do bytes
        S=Base.require(Main,:SHA);root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"compact_bayes_results");path=joinpath(directory,"conditions.json")
        any(islink,(root,directory,path))&&error("Linked registration")
        if isfile(path);read(path)==bytes||error("Preserve registration")
        else;mkpath(directory);open(io->write(io,bytes),path,"w")
        end
        for f in ("compact_bayes_admission.jl","compact_bayes_json.jl","score_compact_bayes.jl","bayes_extraction_references.jl")
            ispath(joinpath(root,f))&&error("Private compact grading material present")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==bytes2hex(sha256(read(path)))||error("Condition drift")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_compact_bayes(file_session;directory=joinpath(@__DIR__,"compact_bayes_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do ids
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/compact_bayes_results"
        islink(root)&&error("Linked result root");found=Dict{String,Vector{UInt8}}()
        for id in ids
            basename(id)==id||error("Unsafe ID")
            if isfile(joinpath(root,id*".json"))
                for suffix in (".json","_attempt.json")
                    path=joinpath(root,id*suffix);isfile(path)&&!islink(path)||error("Unpaired/linked receipt")
                    found[id*suffix]=read(path)
                end
            end
        end
        found
    end
    save_immutable_example_outputs(directory,files,ids)
end
