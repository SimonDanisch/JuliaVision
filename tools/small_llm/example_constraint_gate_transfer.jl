isdefined(@__MODULE__,:EXAMPLE_GATE_SOURCES)||include("laptop_example_constraint_gate.jl")
isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")

function example_constraint_gate_payload(;source_root=@__DIR__)
    path=joinpath(source_root,"laptop_example_constraint_gate/conditions.json")
    p=JSON3.read(read(path,String));Set(String.(keys(p.sources)))==Set(EXAMPLE_GATE_SOURCES)||error("Source allowlist")
    updates=Dict{String,Vector{UInt8}}()
    for file in EXAMPLE_GATE_SOURCES
        basename(file)==file&&(endswith(file,".jl")||endswith(file,".py"))||error("Non-source file")
        data=read(joinpath(source_root,file));bytes2hex(sha256(data))==p.sources[file]||error("Source drift")
        updates[file]=data
    end
    (;root=DOCUMENT_STUDY_REMOTE_ROOT,updates,conditions=read(path))
end

function deploy_example_constraint_gate(file_session;source_root=@__DIR__)
    payload=example_constraint_gate_payload(;source_root)
    sources=deploy_document_study(file_session,(;payload.root,payload.updates))
    expected=bytes2hex(sha256(payload.conditions))
    actual=file_session(payload.conditions) do data
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"laptop_example_constraint_gate")
        path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("Linked destination")
        if isfile(path)
            read(path)==data||error("Preserve different registration")
        else
            mkpath(directory);open(io->write(io,data),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==expected||error("Registration transfer")
    (;complete=true,sources,conditions_sha256=actual)
end
