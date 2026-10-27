isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")
using JSON3,SHA
function deploy_plot_interface(file_session;directory=joinpath(@__DIR__,"plot_interface_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    p.ready_for_generation&&length(p.jobs)==16||error("Unfrozen plot cohort")
    parentpath=joinpath(@__DIR__,"pubmedqa_retrieval_results/conditions.json")
    bytes2hex(sha256(read(parentpath)))==p.parent_runtime_conditions_sha256||error("Parent runtime drift")
    parent=JSON3.read(read(parentpath,String))
    Dict(String(k)=>String(v) for (k,v) in pairs(p.source_sha256))==Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256))||error("Inference source closure changed")
    updates=Dict{String,Vector{UInt8}}()
    for (file,hash) in pairs(p.source_sha256)
        file=String(file);basename(file)==file&&(endswith(file,".jl")||endswith(file,".py")||file in DOCUMENT_STUDY_JSON_SOURCES)||error("Public source names only")
        startswith(file,"plot_interface")&&error("No plot references/frontends/scorers deployed")
        data=read(joinpath(@__DIR__,file));bytes2hex(sha256(data))==hash||error("Runtime source drift")
        updates[file]=data
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    actual=file_session(read(path)) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"plot_interface_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("Linked registration")
        if isfile(path)
            read(path)==data||error("Preserve conditions")
        else
            mkpath(directory);open(io->write(io,data),path,"w")
        end
        # Explicitly reject accidental reference deployment, not just task-file
        # allowlisting. This experiment's new private files must not be here.
        for file in ("plot_interface_references.jl","score_plot_interface.jl","qualify_plot_interface_references.jl")
            ispath(joinpath(root,file))&&error("Private plot material present on model worker")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==bytes2hex(sha256(read(path)))||error("Plot registration transfer drift")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_plot_interface(file_session;directory=joinpath(@__DIR__,"plot_interface_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do ids
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/plot_interface_results"
        islink(root)&&error("Linked output directory")
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
