isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")
using JSON3,SHA
function deploy_bayes_extraction(file_session;directory=joinpath(@__DIR__,"bayes_extraction_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    p.ready_for_generation&&length(p.jobs)==16||error("Unfrozen Bayesian cohort")
    parentpath=joinpath(@__DIR__,"plot_interface_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    bytes2hex(sha256(read(parentpath)))==p.parent_runtime_conditions_sha256||error("Parent drift")
    Dict(String(k)=>String(v) for (k,v) in pairs(p.source_sha256))==Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256))||error("Model runtime changed")
    updates=Dict{String,Vector{UInt8}}()
    for (file,hash) in pairs(p.source_sha256)
        file=String(file);basename(file)==file&&(endswith(file,".jl")||endswith(file,".py")||file in DOCUMENT_STUDY_JSON_SOURCES)||error("Public source names only")
        startswith(file,"bayes_extraction")&&error("No study/private files on inference worker")
        bytes=read(joinpath(@__DIR__,file));bytes2hex(sha256(bytes))==hash||error("Source drift")
        updates[file]=bytes
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    actual=file_session(read(path)) do bytes
        S=Base.require(Main,:SHA);root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"bayes_extraction_results");path=joinpath(directory,"conditions.json")
        any(islink,(root,directory,path))&&error("Linked registration")
        if isfile(path);read(path)==bytes||error("Preserve conditions")
        else;mkpath(directory);open(io->write(io,bytes),path,"w")
        end
        for file in ("bayes_extraction_references.jl","score_bayes_extraction.jl","qualify_bayes_extraction.jl")
            ispath(joinpath(root,file))&&error("Private Bayesian material on inference worker")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==bytes2hex(sha256(read(path)))||error("Registration drift")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_bayes_extraction(file_session;directory=joinpath(@__DIR__,"bayes_extraction_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do ids
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/bayes_extraction_results"
        islink(root)&&error("Linked output directory");found=Dict{String,Vector{UInt8}}()
        for id in ids
            basename(id)==id||error("Unsafe ID")
            output=id*".json";attempt=id*"_attempt.json"
            if isfile(joinpath(root,output))
                isfile(joinpath(root,attempt))||error("Missing attempt")
                (islink(joinpath(root,output))||islink(joinpath(root,attempt)))&&error("Linked receipt")
                found[output]=read(joinpath(root,output));found[attempt]=read(joinpath(root,attempt))
            end
        end
        found
    end
    save_immutable_example_outputs(directory,files,ids)
end
