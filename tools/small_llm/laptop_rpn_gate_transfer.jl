isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")

function rpn_gate_source_payload(conditions;source_root=@__DIR__)
    p=JSON3.read(read(conditions,String));updates=Dict{String,Vector{UInt8}}()
    for (name,expected) in pairs(p.source_sha256)
        name=String(name)
        # Reuse the already-qualified public source allowlist BEFORE reading
        # anything. No private CPU semantic proof/report is in this packet.
        validate_document_study_payload((;root=DOCUMENT_STUDY_REMOTE_ROOT,updates=Dict(name=>UInt8[])))
        data=read(joinpath(source_root,name));bytes2hex(sha256(data))==expected||error("gate source drift")
        updates[name]=data
    end
    (;root=DOCUMENT_STUDY_REMOTE_ROOT,updates)
end

function deploy_rpn_gate(file_session,conditions;source_root=@__DIR__)
    payload=rpn_gate_source_payload(conditions;source_root)
    sources=deploy_document_study(file_session,payload)
    data=read(conditions);expected=bytes2hex(sha256(data))
    actual=file_session(data) do bytes
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"laptop_rpn_gate");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("linked gate destination")
        if isfile(path)
            read(path)==bytes||error("preserve differing gate conditions")
        else
            mkpath(directory);temporary,io=mktemp(directory)
            try
                write(io,bytes);close(io);mv(temporary,path;force=false)
            finally
                isopen(io)&&close(io)
                ispath(temporary)&&rm(temporary)
            end
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==expected||error("deployed gate conditions drift")
    (;complete=true,conditions_sha256=actual,sources)
end
