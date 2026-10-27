include("laptop_pending_study.jl")

function freeze_laptop_shared_source(;directory=joinpath(@__DIR__,"laptop_shared_source_results"))
    original=joinpath(@__DIR__,"shared_source_kernel_results/conditions.json")
    p=JSON3.read(read(original,String))
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash || error("original source drift")
    end
    any(isfile(joinpath(dirname(original),String(j.id)*suffix)) for j in p.jobs for suffix in (".json","_attempt.json")) && error("original attempt exists")
    # This also verifies the exact portable hardware gate and all runtime sources.
    migration=JSON3.read(read(joinpath(@__DIR__,"laptop_pending_results/conditions.json"),String))
    gatepath=joinpath(@__DIR__,"laptop_portable_identity_gate.json")
    gate=JSON3.read(read(gatepath,String))
    gate.complete && gate.normal_final_waitidle && gate.full_model_passed==10 || error("portable gate incomplete")
    bytes2hex(open(sha256,gatepath))==migration.gate_sha256 || error("gate drift")
    ispath(directory) && !isempty(readdir(directory)) && error("refuse overwrite")
    packet=(;scope=p.scope,migration="All eight NEVER ATTEMPTED conditions migrated together to Laptop portable exact-tensor loader; direct GPU vs shared CPU-source arms both on SAME host/runtime. Not pooled with Bosgame results.",
        policy=p.policy,grading=p.grading,jobs=p.jobs,decoding=p.decoding,thinking_budget=p.thinking_budget,
        parent_conditions_sha256=bytes2hex(open(sha256,original)),gate_sha256=migration.gate_sha256,
        portable_manifest_sha256=gate.portable_manifest_sha256,
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(p.source_sha256)),
            Dict(String(k)=>String(v) for (k,v) in pairs(migration.source_sha256)),
            Dict("laptop_shared_source.jl"=>bytes2hex(open(sha256,@__FILE__)))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function laptop_shared_source_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash || error("source drift: $file")
    end
    bytes2hex(open(sha256,joinpath(@__DIR__,"shared_source_kernel_results/conditions.json")))==p.parent_conditions_sha256 || error("parent drift")
    bytes2hex(open(sha256,joinpath(@__DIR__,"laptop_portable_identity_gate.json")))==p.gate_sha256 || error("gate drift")
    (;path,packet=p)
end

function run_laptop_shared_source(;seconds=210,directory=joinpath(@__DIR__,"laptop_shared_source_results"))
    seconds>0 || throw(ArgumentError("time budget"))
    s=laptop_shared_source_state(directory);p=s.packet;b=p.decoding
    bytes2hex(open(sha256,"gen/qwen35-4b-portable/portable.json"))==p.portable_manifest_sha256 || error("tensor identity drift")
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending) && return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending) && error("unfinished attempt; no retries")
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
            for job in pending
                time()<started+seconds-75 || return (;complete=false,completed)
                output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
                (isfile(output)||isfile(attempt)) && error("refuse regeneration")
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
                receipt=(;id=job.id,conditions_sha256=bytes2hex(open(sha256,s.path)),started_unix_seconds=time())
                open(io->JSON3.pretty(io,receipt),attempt,"w")
                answer=try
                    sampled_safe_budget_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                        thinking_budget=Int(p.thinking_budget),seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),
                        top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                catch err
                    open(io->JSON3.pretty(io,(;id=job.id,inference_error=sprint(showerror,err),attempt_sha256=bytes2hex(open(sha256,attempt)))),output,"w")
                    rethrow()
                end
                row=(;answer...,id=job.id,task=job.task,arm=job.arm,messages,
                    conditions_sha256=bytes2hex(open(sha256,s.path)),attempt_sha256=bytes2hex(open(sha256,attempt)),
                    hardware="Laptop8060S",runtime_provenance=inference_runtime_provenance())
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,String(job.id))
            end
            (;complete=true,completed)
        finally
            try model===nothing || close(model)
            finally finish_device!(device) end
        end
    end
end
