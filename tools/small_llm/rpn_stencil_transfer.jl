isdefined(@__MODULE__,:rpn_gate_source_payload)||include("laptop_rpn_gate_transfer.jl")

function rpn_stencil_source_payload(conditions;source_root=@__DIR__)
    p=JSON3.read(read(conditions,String))
    base=JSON3.read(read(joinpath(source_root,"laptop_rpn_gate/conditions.json"),String))
    expected=union(Set(String.(keys(base.source_sha256))),Set(("rpn_stencil_tasks.jl","rpn_stencil_study.jl")))
    Set(String.(keys(p.source_sha256)))==expected||error("unexpected stencil inference sources")
    # An explicit positive source set keeps ROOT-only references/scorer away,
    # rather than treating every .jl file as safe for this study.
    rpn_gate_source_payload(conditions;source_root)
end

function deploy_rpn_stencil_study(file_session,conditions;source_root=@__DIR__)
    payload=rpn_stencil_source_payload(conditions;source_root)
    sources=deploy_document_study(file_session,payload)
    data=read(conditions);expected=bytes2hex(sha256(data))
    actual=file_session(data) do bytes
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"rpn_stencil_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("linked stencil destination")
        if isfile(path)
            read(path)==bytes||error("preserve differing stencil conditions")
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
    actual==expected||error("deployed stencil conditions drift")
    (;complete=true,conditions_sha256=actual,sources)
end
