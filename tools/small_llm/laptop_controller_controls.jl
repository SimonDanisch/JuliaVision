include("laptop_pending_study.jl")

"""Register all same-host controls before generating any; never alter the frozen pilot."""
function freeze_laptop_controller_controls(;directory=joinpath(@__DIR__,"laptop_controller_controls_results"))
    original=laptop_original_pending_jobs()
    pilot=joinpath(@__DIR__,"laptop_pending_results/conditions.json")
    summary=JSON3.read(read(joinpath(dirname(pilot),"summary.json"),String))
    summary.complete || error("finish the pilot before this explicitly posthoc control study")
    path=joinpath(directory,"conditions.json")
    ispath(directory) && !isempty(readdir(directory)) && error("refuse registration overwrite")
    packet=(;scope="Posthoc same-Laptop disabled-controller controls for ALL six frozen migrated outputs; not an independent held-out benchmark. No selective task/arm exclusions.",
        policy="Identical prompts, sampling, token budget and order. Disable only forced thinking closure; preserve every attempt, including failure. Never regenerate pilot outputs.",
        original...,thinking_limit=Int(typemax(UInt32)),pilot_conditions_sha256=bytes2hex(open(sha256,pilot)),
        pilot_output_sha256=Dict(String(j.id)=>bytes2hex(open(sha256,joinpath(dirname(pilot),String(j.id)*".json"))) for j in original.jobs),
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(JSON3.read(read(pilot,String)).source_sha256)),
            Dict("laptop_controller_controls.jl"=>bytes2hex(open(sha256,@__FILE__)))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),path,"w");packet
end

function laptop_control_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash || error("control source drift")
    end
    pilot=joinpath(@__DIR__,"laptop_pending_results/conditions.json")
    bytes2hex(open(sha256,pilot))==p.pilot_conditions_sha256 || error("pilot protocol drift")
    for (id,hash) in pairs(p.pilot_output_sha256)
        bytes2hex(open(sha256,joinpath(dirname(pilot),String(id)*".json")))==hash || error("pilot output drift")
    end
    (;path,packet=p)
end

function run_laptop_controller_controls(;seconds=210,directory=joinpath(@__DIR__,"laptop_controller_controls_results"))
    seconds>0 || throw(ArgumentError("time budget"))
    s=laptop_control_state(directory);p=s.packet;b=p.decoding
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending) && return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending) && error("unfinished attempt; do not regenerate")
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
                        thinking_budget=Int(p.thinking_limit),seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),
                        top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                catch err
                    open(io->JSON3.pretty(io,(;id=job.id,inference_error=sprint(showerror,err),attempt_sha256=bytes2hex(open(sha256,attempt)))),output,"w")
                    rethrow()
                end
                answer.forced_closures==0 || error("disabled controller forced a closure")
                row=(;answer...,id=job.id,task=job.task,arm=job.arm,messages,original_sha256=job.original_sha256,
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
