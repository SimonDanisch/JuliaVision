using JSON3,SHA
const DOCUMENT_STUDY_REMOTE_ROOT="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
const DOCUMENT_STUDY_JSON_SOURCES=Set(("julia_namespace_contract.json",
    "public_program_diagnostics_validation.json","qualified_binding_diagnostics_validation.json",
    "bigcodebench_x_secure_audit_validation.json","public_document_retrieval_software_validation.json",
    "public_document_context_validation.json","bigcodebench_x_doc_scorer_validation.json","julia_public_docs.json"))

function document_study_payload(conditions;source_root=@__DIR__)
    p=JSON3.read(read(conditions,String));updates=Dict{String,Vector{UInt8}}()
    for (name,hash) in pairs(p.source_sha256)
        name=String(name)
        basename(name)==name&&(endswith(name,".jl")||endswith(name,".py")||name in DOCUMENT_STUDY_JSON_SOURCES)||error("not approved public source")
        data=read(joinpath(source_root,name));bytes2hex(sha256(data))==hash||error("source drift")
        updates[name]=data
    end
    updates["bigcodebench_x_doc_results/conditions.json"]=read(conditions)
    (;root=DOCUMENT_STUDY_REMOTE_ROOT,updates)
end

function validate_document_study_payload(packet)
    packet.root==DOCUMENT_STUDY_REMOTE_ROOT||error("unexpected destination")
    for name in keys(packet.updates)
        name=="bigcodebench_x_doc_results/conditions.json"||
            (basename(name)==name&&(endswith(name,".jl")||endswith(name,".py")||name in DOCUMENT_STUDY_JSON_SOURCES))||error("unsafe public deployment path")
    end
    nothing
end

function deploy_document_study(file_session,packet)
    validate_document_study_payload(packet)
    expected=Dict(name=>bytes2hex(sha256(data)) for (name,data) in packet.updates)
    receipt=file_session(packet) do p
        S=Base.require(Main,:SHA)
        p.root=="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"||error("unexpected remote root")
        islink(p.root)&&error("linked deployment root")
        rows=Dict{String,String}()
        for (relative,data) in p.updates
            # Host allowlist checked before transfer; enforce basename boundary
            # again in the remote namespace before constructing any path.
            basename(relative)==relative||relative=="bigcodebench_x_doc_results/conditions.json"||error("remote traversal")
            path=joinpath(p.root,relative)
            (islink(path)||islink(dirname(path)))&&error("linked destination")
            if isfile(path)
                read(path)==data||error("preserve different remote bytes: $relative")
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
    receipt==expected||error("deployed document-study integrity")
    (;complete=true,files=length(receipt),sha256=receipt)
end
