isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")
using JSON3,SHA

function alias_transfer3_payload(conditions;source_root=@__DIR__)
    p=JSON3.read(read(conditions,String))
    parent=JSON3.read(read(joinpath(source_root,"julia_stencil_transfer_results/conditions.json"),String))
    expected=union(Set(String.(keys(parent.source_sha256))),Set(("alias_transfer3_study.jl","run_alias_transfer3.jl")))
    Set(String.(keys(p.source_sha256)))==expected||error("Unexpected inference source closure; no graders/reference data")
    updates=Dict{String,Vector{UInt8}}()
    for (name,hash) in pairs(p.source_sha256)
        name=String(name);basename(name)==name||error("Unsafe source path")
        data=read(joinpath(source_root,name));bytes2hex(sha256(data))==hash||error("Public source drift")
        updates[name]=data
    end
    (;root=DOCUMENT_STUDY_REMOTE_ROOT,updates)
end

function deploy_alias_transfer3(file_session;directory=joinpath(@__DIR__,"alias_transfer3_results"))
    conditions=joinpath(directory,"conditions.json")
    sources=deploy_document_study(file_session,alias_transfer3_payload(conditions))
    bytes=read(conditions)
    hash=file_session(bytes) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"alias_transfer3_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("Linked registration")
        if isfile(path)
            read(path)==data||error("Preserve registration")
        else
            mkpath(directory);open(io->write(io,data),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    hash==bytes2hex(sha256(bytes))||error("Condition transfer mismatch")
    (;complete=true,conditions_sha256=hash,sources)
end

function collect_alias_transfer3(file_session;directory=joinpath(@__DIR__,"alias_transfer3_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do names
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/alias_transfer3_results"
        islink(root)&&error("Linked results")
        found=Dict{String,Vector{UInt8}}()
        for id in names
            basename(id)==id||error("Unsafe job ID")
            output=id*".json";attempt=id*"_attempt.json"
            if isfile(joinpath(root,output))
                isfile(joinpath(root,attempt))||error("Unpaired result/attempt")
                (islink(joinpath(root,output))||islink(joinpath(root,attempt)))&&error("Linked receipt")
                found[output]=read(joinpath(root,output));found[attempt]=read(joinpath(root,attempt))
            end
        end
        found
    end
    save_immutable_example_outputs(directory,files,ids)
end
