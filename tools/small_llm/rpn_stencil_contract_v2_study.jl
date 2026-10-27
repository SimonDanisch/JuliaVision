include("rpn_stencil_study.jl")
include("rpn_stencil_contract_v2.jl")

function rpn_stencil_contract_v2_jobs(parent)
    tasks=rpn_stencil_tasks()
    [(;id=String(job.id)*"_contract_v2",task=job.task,arm=job.arm,enabled=job.enabled,
        messages=rpn_stencil_messages_v2(only(filter(t->t.id==job.task,tasks)))) for job in parent.jobs]
end

function freeze_rpn_stencil_contract_v2(;directory=joinpath(@__DIR__,"rpn_stencil_contract_v2_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("preserve v2 registration")
    parentdir=joinpath(@__DIR__,"rpn_stencil_results");s=rpn_stencil_state(parentdir);parent=s.packet
    proofpath=joinpath(@__DIR__,"rpn_stencil_contract_v2_validation.json")
    proof=JSON3.read(read(proofpath,String));proof.complete&&proof.passed==49||error("new prompt not qualified")
    for (file,hash) in pairs(proof.source_sha256)
        basename(String(file))==String(file)&&laptop_rpn_digest(joinpath(@__DIR__,String(file)))==hash||error("v2 prompt source drift")
    end
    unitpath=joinpath(@__DIR__,"rpn_stencil_contract_v2_study_validation.json")
    units=JSON3.read(read(unitpath,String));units.complete&&units.study_checks==58&&units.transfer_checks==7||error("v2 study/transfer qualification pending")
    for (file,hash) in pairs(units.source_sha256)
        basename(String(file))==String(file)&&laptop_rpn_digest(joinpath(@__DIR__,String(file)))==hash||error("v2 unit source drift")
    end
    observed=Dict(String(j.id)=>laptop_rpn_digest(joinpath(parentdir,String(j.id)*".json"))
        for j in parent.jobs if isfile(joinpath(parentdir,String(j.id)*".json")))
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict(f=>laptop_rpn_digest(joinpath(@__DIR__,f)) for f in ("rpn_stencil_contract_v2.jl","rpn_stencil_contract_v2_study.jl")))
    packet=merge(JSON3.read(JSON3.write(parent),Dict{String,Any}),Dict(
        "scope"=>"POSTHOC prompt-format development on SAME8 authored stencil tasks, after observing partial original outputs. Not independent transfer, external held-out, or model-only/frontier evidence. Original cohort retained unchanged.",
        "policy"=>"All16 follow-up conditions frozen before ANY follow-up generation. Same16 paired jobs/order/model/sampler/nonthinking512token cap/grammar/compiler/tests. Identical revised prompt in both arms: correct32-normal-plus-up-to7-closure wording, empty-stack explanation, ONE generic constant-only worked example. No task-answer example, hidden-test feedback, retries or seed search. All original generations must finish first.",
        "jobs"=>rpn_stencil_contract_v2_jobs(parent),"source_sha256"=>sources,
        "format_parent_conditions_sha256"=>laptop_rpn_digest(s.path),
        "observed_parent_outputs_at_registration"=>observed,
        "new_prompt_qualification_sha256"=>laptop_rpn_digest(proofpath),
        "new_study_qualification_sha256"=>laptop_rpn_digest(unitpath)))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function run_rpn_stencil_contract_v2(;directory=joinpath(@__DIR__,"rpn_stencil_contract_v2_results"))
    p=rpn_stencil_state(directory).packet
    parentdir=joinpath(@__DIR__,"rpn_stencil_results");path=joinpath(parentdir,"conditions.json")
    laptop_rpn_digest(path)==p.format_parent_conditions_sha256||error("original registration drift")
    parent=JSON3.read(read(path,String))
    all(isfile(joinpath(parentdir,String(j.id)*".json")) for j in parent.jobs)||error("finish ALL original outputs first; no device setup")
    run_rpn_stencil_study(;directory)
end
