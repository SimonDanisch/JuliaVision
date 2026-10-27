isdefined(@__MODULE__,:rpn_gate_source_payload)||include("laptop_rpn_gate_transfer.jl")

function public_example_source_payload(conditions;source_root=@__DIR__)
    p=JSON3.read(read(conditions,String));parentpath=joinpath(source_root,"laptop_example_identity_gate/conditions.json")
    bytes2hex(sha256(read(parentpath)))==p.identity_conditions_sha256||error("Identity parent drift")
    parent=JSON3.read(read(parentpath,String))
    expected=union(Set(String.(keys(parent.sources))),Set(("rpn_public_example_tasks.jl","rpn_public_example_study.jl")))
    Set(String.(keys(p.source_sha256)))==expected||error("Unexpected inference sources; private grading excluded")
    rpn_gate_source_payload(conditions;source_root)
end

function deploy_public_example_study(file_session,conditions;source_root=@__DIR__)
    sources=deploy_document_study(file_session,public_example_source_payload(conditions;source_root))
    data=read(conditions);expected=bytes2hex(sha256(data))
    actual=file_session(data) do bytes
        S=Base.require(Main,:SHA)
        root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        directory=joinpath(root,"rpn_public_example_results");path=joinpath(directory,"conditions.json")
        (islink(root)||islink(directory)||islink(path))&&error("Linked destination")
        if isfile(path)
            read(path)==bytes||error("Preserve differing registration")
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
    actual==expected||error("Registration transfer mismatch")
    (;complete=true,conditions_sha256=actual,sources)
end
