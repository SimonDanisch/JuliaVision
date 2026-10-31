isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
using JSON3,SHA

function freeze_finite_choice_model_gate()
    directory=joinpath(@__DIR__,"finite_choice_model_gate");ispath(directory)&&!isempty(readdir(directory))&&error("Preserve gate registration")
    digest(path)=bytes2hex(sha256(read(path)))
    parentpath=joinpath(@__DIR__,"bayes_extraction_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    nativepath=joinpath(@__DIR__,"finite_choice_native_gate/result.json");native=JSON3.read(read(nativepath,String))
    native.complete&&native.normal_final_waitidle&&native.host_checks==39&&native.device_checks==47||error("Physical fixture gate required")
    proofpath=joinpath(@__DIR__,"finite_choice_model_checks_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&proof.passed==12||error("Checker qualification required")
    for registry in (parent.source_sha256,native.source_sha256,proof.source_sha256),(f,h) in pairs(registry)
        digest(joinpath(@__DIR__,String(f)))==h||error("Qualified source drift")
    end
    source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict(String(k)=>String(v) for (k,v) in pairs(native.source_sha256)),
        Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("run_finite_choice_model_gate.jl","finite_choice_model_checks.jl","test_finite_choice_model_checks.jl")))
    p=(;ready=true,steps=8,seed=20261025,vocabulary=248320,choices=["Makie.lines!","Makie.scatter!"],
        messages=["system"=>"Return only the requested Julia function name.","user"=>"Choose exactly one of Makie.lines! or Makie.scatter! and return that qualified name only."],
        scope="One-load integration gate: baseline vs disabled hook raw-logit/token/seen identity, then canonical finite support. No correct-choice task or improved programming/semantic accuracy is tested.",
        policy="One first attempt, one prompt, three fresh-session controls on unchanged model/sampler; eight recorded decode steps each, no per-token host feedback. Preserve failures and final cleanup; no model-quality score.",
        source_sha256,portable_manifest_sha256=parent.portable_manifest_sha256,
        physical_fixture_report_sha256=digest(nativepath),checker_proof_sha256=digest(proofpath),transfer_source_sha256=digest(@__FILE__))
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end

function deploy_finite_choice_model_gate(file_session)
    path=joinpath(@__DIR__,"finite_choice_model_gate/conditions.json");p=JSON3.read(read(path,String))
    updates=Dict{String,Vector{UInt8}}()
    for (f,h) in pairs(p.source_sha256)
        f=String(f);basename(f)==f&&(endswith(f,".jl")||endswith(f,".py")||f in DOCUMENT_STUDY_JSON_SOURCES)||error("Public source only")
        bytes=read(joinpath(@__DIR__,f));bytes2hex(sha256(bytes))==h||error("Public source drift");updates[f]=bytes
    end
    sources=deploy_document_study(file_session,(;root=DOCUMENT_STUDY_REMOTE_ROOT,updates))
    actual=file_session(read(path)) do bytes
        S=Base.require(Main,:SHA);root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/finite_choice_model_gate"
        path=joinpath(root,"conditions.json");any(islink,(root,path))&&error("Linked registration")
        if isfile(path);read(path)==bytes||error("Preserve conditions")
        else;mkpath(root);open(io->write(io,bytes),path,"w")
        end
        bytes2hex(Base.invokelatest(S.sha256,read(path)))
    end
    actual==bytes2hex(sha256(read(path)))||error("Deployment mismatch")
    (;complete=true,conditions_sha256=actual,sources)
end

function collect_finite_choice_model_gate(file_session)
    bytes=file_session() do
        path="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/finite_choice_model_gate/result.json"
        islink(path)&&error("Linked result");read(path)
    end
    directory=joinpath(@__DIR__,"finite_choice_model_gate");path=joinpath(directory,"result.json")
    isfile(path)&&read(path)!=bytes&&error("Preserve first gate result")
    isfile(path)||open(io->write(io,bytes),path,"w")
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));r=JSON3.read(bytes)
    r.complete&&r.normal_final_waitidle&&r.result.passed||error("Full-model integration incomplete/failed")
    r.conditions_sha256==bytes2hex(sha256(read(joinpath(directory,"conditions.json"))))&&r.source_sha256==p.source_sha256||error("Gate provenance mismatch")
    (;complete=true,result=r.result,report_sha256=bytes2hex(sha256(bytes)))
end
