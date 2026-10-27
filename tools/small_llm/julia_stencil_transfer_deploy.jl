isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
isdefined(@__MODULE__,:save_immutable_example_outputs)||include("collect_public_example_outputs.jl")

function deploy_julia_stencil_transfer(file_session;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    # Exact public-only source closure. Private graders/reference fixtures are
    # deliberately absent even though they live beside these source files.
    parent=JSON3.read(read(joinpath(@__DIR__,"laptop_example_identity_gate/conditions.json"),String))
    expected=union(Set(String.(keys(parent.sources))),Set(("julia_stencil_transfer_tasks.jl","julia_stencil_transfer_study.jl",
        "run_julia_stencil_transfer.jl","julia_stencil_interface.jl","public_example_synthesis.jl","public_example_fallback.jl")))
    Set(String.(keys(p.source_sha256)))==expected||error("Unexpected source closure")
    updates=Dict{String,Vector{UInt8}}()
    for (file,hash) in pairs(p.source_sha256)
        name=String(file);basename(name)==name||error("Unsafe source path")
        data=read(joinpath(@__DIR__,name));bytes2hex(sha256(data))==hash||error("Public source drift")
        updates[name]=data
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    bytes=read(conditionpath)
    received=file_session(bytes) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        dir=joinpath(root,"julia_stencil_transfer_results");path=joinpath(dir,"conditions.json")
        (islink(root)||islink(dir)||islink(path))&&error("Linked registration path")
        if isfile(path)
            read(path)==data||error("Preserve existing registration")
        else
            mkpath(dir);open(io->write(io,data),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    received==bytes2hex(sha256(bytes))||error("Registration transfer mismatch")
    (;complete=true,conditions_sha256=received,sources)
end

function collect_julia_stencil_transfer(file_session;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));ids=public_example_output_names(p.jobs)
    files=file_session(ids) do names
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/julia_stencil_transfer_results"
        islink(root)&&error("Linked result directory")
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
