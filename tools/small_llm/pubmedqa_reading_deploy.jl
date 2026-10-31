isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")
using JSON3,SHA
function deploy_pubmedqa_reading(file_session;directory=joinpath(@__DIR__,"pubmedqa_reading_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    p.ready_for_generation&&length(p.jobs)==48||error("Unfrozen cohort")
    parentpath=joinpath(@__DIR__,"alias_transfer3_results/conditions.json")
    bytes2hex(sha256(read(parentpath)))==p.parent_runtime_conditions_sha256||error("Parent runtime drift")
    parent=JSON3.read(read(parentpath,String))
    expected_sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict("run_pubmedqa_reading.jl"=>bytes2hex(sha256(read(joinpath(@__DIR__,"run_pubmedqa_reading.jl"))))))
    Dict(String(k)=>String(v) for (k,v) in pairs(p.source_sha256))==expected_sources||error("Unapproved source closure")
    updates=Dict{String,Vector{UInt8}}()
    for (file,hash) in pairs(p.source_sha256)
        file=String(file);basename(file)==file&&(endswith(file,".jl")||endswith(file,".py")||file in DOCUMENT_STUDY_JSON_SOURCES)||error("Public runtime source names only")
        file in ("score_pubmedqa_reading.jl","pubmedqa_metrics.jl")&&error("Never deploy private scoring")
        data=read(joinpath(@__DIR__,file));bytes2hex(sha256(data))==hash||error("Runtime source drift")
        updates[file]=data
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    expected=bytes2hex(sha256(read(path)))
    actual=file_session(read(path)) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"pubmedqa_reading_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("Linked registration")
        if isfile(path)
            read(path)==data||error("Preserve conditions")
        else
            mkpath(directory);open(io->write(io,data),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==expected||error("Registration transfer drift")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_pubmedqa_reading(file_session;directory=joinpath(@__DIR__,"pubmedqa_reading_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do ids
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/pubmedqa_reading_results"
        islink(root)&&error("Linked remote outputs")
        found=Dict{String,Vector{UInt8}}()
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
