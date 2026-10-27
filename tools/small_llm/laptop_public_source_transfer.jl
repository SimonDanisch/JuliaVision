using JSON3,SHA

"""Build an explicit public generation deployment, not a whole result-folder sync.
Only frozen source dependencies, the public policy and generation conditions
are included. Never include test datasets, scores, errors or evaluator traces.
"""
function laptop_public_revision_payload(conditions_path;
        remote_root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm")
    p=JSON3.read(read(conditions_path,String));updates=Dict{String,Vector{UInt8}}()
    for (name,expected) in pairs(p.source_sha256)
        file=String(name);basename(file)==file||error("source path traversal")
        public_json=("julia_namespace_contract.json","public_program_diagnostics_validation.json",
            "qualified_binding_diagnostics_validation.json","bigcodebench_x_secure_audit_validation.json")
        endswith(file,".jl")||endswith(file,".py")||file in public_json||error("not an approved public dependency")
        data=read(joinpath(@__DIR__,file));bytes2hex(sha256(data))==expected||error("source drift")
        updates[file]=data
    end
    policy="bigcodebench_x_revision_policy.json";data=read(joinpath(@__DIR__,policy))
    bytes2hex(sha256(data))==p.policy_sha256||error("policy drift");updates[policy]=data
    # Final condition packet contains original public task text/model code and
    # public analyzer reports, never evaluator test/reference/score contents.
    updates["bigcodebench_x_revision_results/conditions.json"]=read(conditions_path)
    (;root=remote_root,updates)
end

function deploy_public_revision_payload(file_session,payload)
    expected=Dict(file=>bytes2hex(sha256(data)) for (file,data) in payload.updates)
    receipt=file_session(payload) do packet
        S=Base.require(Main,:SHA);rows=Dict{String,String}()
        root=packet.root
        root=="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"||error("unexpected destination")
        islink(root)&&error("refuse linked deployment root")
        for (relative,data) in packet.updates
            (basename(relative)==relative||relative=="bigcodebench_x_revision_results/conditions.json")||error("unsafe path")
            path=joinpath(root,relative)
            islink(path)&&error("refuse symlink")
            islink(dirname(path))&&error("refuse linked destination directory")
            if isfile(path)
                read(path)==data||error("preserve differing remote source: $relative")
            else
                mkpath(dirname(path));temporary,io=mktemp(dirname(path))
                try
                    write(io,data);close(io);mv(temporary,path;force=false)
                finally
                    isopen(io)&&close(io)
                    ispath(temporary)&&rm(temporary)
                end
            end
            rows[relative]=bytes2hex(Base.invokelatest(S.sha256,read(path)))
        end
        rows
    end
    receipt==expected||error("deployed byte integrity")
    (;complete=true,files=length(receipt),sha256=receipt)
end
