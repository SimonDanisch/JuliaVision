isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")
using JSON3,SHA
function deploy_lint_revision(file_session;directory=joinpath(@__DIR__,"lint_revision_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    parentpath=joinpath(@__DIR__,"alias_transfer3_results/conditions.json")
    bytes2hex(sha256(read(parentpath)))==p.parent_conditions_sha256||error("Parent registration drift")
    parent=JSON3.read(read(parentpath,String))
    Dict(String(k)=>String(v) for (k,v) in pairs(p.source_sha256))==
        Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256))||error("Inference source closure changed")
    updates=Dict{String,Vector{UInt8}}()
    for (f,h) in pairs(p.source_sha256)
        name=String(f);basename(name)==name||error("Unsafe source path")
        bytes=read(joinpath(@__DIR__,name));bytes2hex(sha256(bytes))==h||error("Source drift")
        updates[name]=bytes
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    bytes=read(path)
    actual=file_session(bytes) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        dir=joinpath(root,"lint_revision_results");path=joinpath(dir,"conditions.json")
        (islink(root)||islink(dir)||islink(path))&&error("Linked registration")
        if isfile(path)
            read(path)==data||error("Preserve registration")
        else
            mkpath(dir);open(io->write(io,data),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==bytes2hex(sha256(bytes))||error("Condition transfer mismatch")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_lint_revision(file_session;directory=joinpath(@__DIR__,"lint_revision_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do names
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/lint_revision_results"
        islink(root)&&error("Linked outputs")
        found=Dict{String,Vector{UInt8}}()
        for id in names
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
