isdefined(@__MODULE__,:rpn_gate_source_payload)||include("laptop_rpn_gate_transfer.jl")

function rpn_contract_v2_source_payload(conditions;source_root=@__DIR__)
    p=JSON3.read(read(conditions,String))
    parentpath=joinpath(source_root,"rpn_stencil_results/conditions.json")
    bytes2hex(open(sha256,parentpath))==p.format_parent_conditions_sha256||error("v2 parent drift")
    parent=JSON3.read(read(parentpath,String))
    expected=union(Set(String.(keys(parent.source_sha256))),Set(("rpn_stencil_contract_v2.jl","rpn_stencil_contract_v2_study.jl")))
    Set(String.(keys(p.source_sha256)))==expected||error("unexpected v2 inference sources")
    rpn_gate_source_payload(conditions;source_root)
end

function deploy_rpn_contract_v2(file_session,conditions;source_root=@__DIR__)
    sources=deploy_document_study(file_session,rpn_contract_v2_source_payload(conditions;source_root))
    data=read(conditions);expected=bytes2hex(sha256(data))
    actual=file_session(data) do bytes
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"rpn_stencil_contract_v2_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("linked v2 destination")
        if isfile(path)
            read(path)==bytes||error("preserve differing v2 conditions")
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
    actual==expected||error("v2 deployment digest mismatch")
    (;complete=true,conditions_sha256=actual,sources)
end
