isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")
using JSON3,SHA
function deploy_boundary_transfer(file_session;directory=joinpath(@__DIR__,"boundary_transfer_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    parent=JSON3.read(read(joinpath(@__DIR__,"julia_stencil_transfer_results/conditions.json"),String))
    expected=union(Set(String.(keys(parent.source_sha256))),Set(("boundary_transfer_tasks.jl","boundary_transfer_study.jl",
        "stencil_boundary_contract.jl","boundary_example_synthesis.jl","julia_stencil_admission_v2.jl","julia_stencil_admission_v3.jl")))
    Set(String.(keys(p.source_sha256)))==expected||error("Unexpected public source closure")
    updates=Dict{String,Vector{UInt8}}()
    for (f,h) in pairs(p.source_sha256)
        name=String(f);basename(name)==name||error("Unsafe path")
        data=read(joinpath(@__DIR__,name));bytes2hex(sha256(data))==h||error("Source drift")
        updates[name]=data
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    bytes=read(path)
    result=file_session(bytes) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"boundary_transfer_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("Linked registration")
        if isfile(path)
            read(path)==data||error("Preserve registration")
        else
            mkpath(directory);open(io->write(io,data),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    result==bytes2hex(sha256(bytes))||error("Conditions mismatch")
    (;complete=true,conditions_sha256=result,sources)
end

function collect_boundary_transfer(file_session;directory=joinpath(@__DIR__,"boundary_transfer_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do names
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/boundary_transfer_results"
        islink(root)&&error("Linked results")
        found=Dict{String,Vector{UInt8}}()
        for id in names
            basename(id)==id||error("Unsafe job ID")
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
