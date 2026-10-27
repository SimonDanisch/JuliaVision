isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")
using JSON3,SHA

const FINITE_CHOICE_PUBLIC_FILES=("finite_choice_constraint.jl","finite_choice_constraint_gpu.jl",
    "test_finite_choice_constraint.jl","test_finite_choice_constraint_gpu.jl","token_prefix_trie.jl",
    "safe_reasoning_budget.jl","reasoning_budget.jl","gpu_sampling.jl","device_finish.jl",
    "runtime_provenance.jl","run_finite_choice_gate.jl")

function finite_choice_gate_payload()
    proofpath=joinpath(@__DIR__,"finite_choice_software_validation.json");p=JSON3.read(read(proofpath,String))
    p.complete&&p.host_checks==39&&p.shared_kernel_checks==47||error("Software fixture gate required")
    for (f,h) in pairs(p.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(f)))))==h||error("Software source drift")
    end
    updates=Dict(f=>read(joinpath(@__DIR__,f)) for f in FINITE_CHOICE_PUBLIC_FILES)
    (;root=DOCUMENT_STUDY_REMOTE_ROOT,updates)
end

function deploy_finite_choice_gate(file_session)
    packet=finite_choice_gate_payload();sources=Dict(f=>bytes2hex(sha256(b)) for (f,b) in packet.updates)
    conditions=(;scope="Physical fixture qualification only; no model, no inference-quality claims",source_sha256=sources,
        software_validation_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"finite_choice_software_validation.json")))),
        transfer_source_sha256=bytes2hex(sha256(read(@__FILE__))))
    directory=joinpath(@__DIR__,"finite_choice_native_gate");mkpath(directory);path=joinpath(directory,"conditions.json")
    if isfile(path);JSON3.read(read(path,String))==JSON3.read(JSON3.write(conditions))||error("Preserve registration")
    else;open(io->JSON3.pretty(io,conditions),path,"w")
    end
    receipt=deploy_document_study(file_session,packet)
    file_session() do
        path="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/finite_choice_native_gate"
        islink(path)&&error("Linked report directory");mkpath(path);nothing
    end
    (;complete=true,sources,public_transfer=receipt,scope=conditions.scope)
end

function collect_finite_choice_gate(file_session)
    bytes=file_session() do
        path="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/finite_choice_native_gate/result.json"
        islink(path)&&error("Linked report");read(path)
    end
    directory=joinpath(@__DIR__,"finite_choice_native_gate");path=joinpath(directory,"result.json")
    isfile(path)&&read(path)!=bytes&&error("Preserve first result")
    isfile(path)||open(io->write(io,bytes),path,"w")
    report=JSON3.read(bytes);p=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    report.complete&&report.normal_final_waitidle&&report.host_checks==39&&report.device_checks==47||error("Native gate incomplete/failed")
    report.source_sha256==p.source_sha256||error("Native source mismatch")
    (;complete=true,host_checks=report.host_checks,device_checks=report.device_checks,report_sha256=bytes2hex(sha256(bytes)))
end
