# Migrate two NEVER ATTEMPTED long-output conditions, not old short outputs.
include("laptop_shared_source.jl")
include("score_kernel_transfer_budget.jl")
long_budget_digest(path)=bytes2hex(open(sha256,path))

function freeze_laptop_long_budget(;directory=joinpath(@__DIR__,"laptop_long_budget_v2_results"))
    parent=transfer_budget_state();p=parent.packet
    for arm in p.arms, suffix in (".json","_attempt.json")
        ispath(joinpath(dirname(parent.path),String(p.task)*"_"*String(arm)*suffix)) && error("original long condition already attempted")
    end
    ispath(directory) && !isempty(readdir(directory)) && error("preserve registration")
    base=laptop_shared_source_state(joinpath(@__DIR__,"laptop_shared_source_results")).packet
    jobs=[(;id=String(p.task)*"_"*String(arm),task=p.task,arm,messages=p.messages[Symbol(arm)]) for arm in p.arms]
    packet=(;scope="Migration of two NEVER ATTEMPTED6144-token budget diagnostics to Laptop. Old short Bosgame failures retained; different host/capacity means NOT a strict budget-only causal comparison.",
        policy="One fixed-seed attempt per original arm, exact original prompts/6144 budget/sampling. No thinking closure, retries, feedback, seed search or hidden-answer selection. New immutable attempts and source/runtime receipts.",
        jobs,decoding=p.decoding,original_sha256=p.original_sha256,
        pre_attempt_amendment=(;reason="Before ANY GPU setup/generation attempt: replace original absolute-path API validation with same-byte package-relative validation on Laptop; original registration/source preserved",
            original_registration_sha256=long_budget_digest(joinpath(@__DIR__,"laptop_long_budget_results/conditions.json")),
            original_source_sha256=long_budget_digest(joinpath(@__DIR__,"laptop_long_budget_v1_prelaunch.jl"))),
        original_long_conditions_sha256=long_budget_digest(parent.path),
        parent_conditions_sha256=base.parent_conditions_sha256,gate_sha256=base.gate_sha256,
        portable_manifest_sha256=base.portable_manifest_sha256,
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.source_sha256)),
            Dict(f=>long_budget_digest(joinpath(@__DIR__,f)) for f in
                ("laptop_long_budget.jl","kernel_transfer_budget.jl","score_kernel_transfer_budget.jl","sampled_julia.jl"))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function laptop_long_budget_state(directory)
    s=laptop_shared_source_state(directory)
    long_budget_digest(joinpath(@__DIR__,"kernel_transfer_budget_results/conditions.json"))==s.packet.original_long_conditions_sha256 || error("original long registration drift")
    # Original transfer_budget_state uses captured Bosgame absolute source paths.
    # The existing migration validator checks SAME package version/source bytes
    # through pkgdir, not a symlink or relaxation of API provenance.
    laptop_original_pending_jobs()
    parentpath=joinpath(@__DIR__,"kernel_transfer_budget_results/conditions.json")
    parent=JSON3.read(read(parentpath,String))
    parent.source_sha256==long_budget_digest(joinpath(@__DIR__,"kernel_transfer_budget.jl")) || error("parent source drift")
    transfer=joinpath(@__DIR__,"kernel_transfer_results/conditions.json")
    parent.transfer_conditions_sha256==long_budget_digest(transfer) || error("original transfer drift")
    for arm in parent.arms
        source=joinpath(dirname(transfer),String(parent.task)*"_"*String(arm)*".json")
        long_budget_digest(source)==parent.original_sha256[Symbol(arm)]==s.packet.original_sha256[Symbol(arm)] || error("original prompt drift")
    end
    s
end

function run_laptop_long_budget(;seconds=270,directory=joinpath(@__DIR__,"laptop_long_budget_v2_results"))
    seconds>0 || error("time budget required")
    s=laptop_long_budget_state(directory);p=s.packet;b=p.decoding
    long_budget_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256 || error("model identity")
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending) && return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending) && error("no retry of interrupted attempt")
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
            for job in pending
                time()<started+seconds-180 || return (;complete=false,completed)
                path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
                ispath(path)||ispath(attempt) ? error("preserve attempt") : nothing
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
                open(io->JSON3.pretty(io,(;id=job.id,conditions_sha256=long_budget_digest(s.path),started_unix_seconds=time())),attempt,"w")
                answer=try
                    sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                        output_chunk=Int(b.output_chunk),seed=UInt32(b.seed),temperature=Float32(b.temperature),
                        top_p=Float32(b.top_p),topk=Int(b.topk),presence=Float32(b.presence))
                catch err
                    open(io->JSON3.pretty(io,(;id=job.id,inference_error=sprint(showerror,err),attempt_sha256=long_budget_digest(attempt))),path,"w")
                    rethrow()
                end
                row=(;answer...,id=job.id,task=job.task,arm=job.arm,messages,
                    conditions_sha256=long_budget_digest(s.path),attempt_sha256=long_budget_digest(attempt),
                    runtime_provenance=inference_runtime_provenance(),hardware="Laptop8060S")
                open(io->JSON3.pretty(io,row),path,"w");push!(completed,String(job.id))
            end
            (;complete=true,completed)
        finally
            try model===nothing || close(model)
            finally finish_device!(device) end
        end
    end
end

function score_laptop_long_budget(;directory=joinpath(@__DIR__,"laptop_long_budget_v2_results"))
    s=laptop_long_budget_state(directory);p=s.packet;b=p.decoding
    parent=transfer_budget_state();task=only(filter(t->t.id==first(p.jobs).task,parent.state.tasks));rows=Any[]
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        c=JSON3.read(read(path,String))
        c.attempt_sha256==long_budget_digest(attempt) || error("attempt drift")
        if hasproperty(c,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        c.conditions_sha256==long_budget_digest(s.path)&&c.messages==job.messages&&c.budget_tokens==b.max_tokens || error("generation drift")
        c.decode_steps_executed==c.decode_submissions || error("submission invariant")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            a=getproperty(c.sampling,field);v=getproperty(b,field)
            (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(v) : a==v) || error("sampling drift")
        end
        scorepath=replace(path,".json"=>"_score.json");code=String(c.code)
        result=if isfile(scorepath)
            prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(code)) || error("score drift");prior
        else
            JuliaTaskEval.evaluate(task,code;output=scorepath)
        end
        push!(rows,(;id=job.id,status=result.passed ? "passed" : "failed",generated_tokens=c.generated_tokens,
            truncated=c.truncated,finished_thinking=c.finished_thinking,generation_sha256=long_budget_digest(path)))
    end
    packet=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=long_budget_digest(s.path),scorer_sha256=long_budget_digest(@__FILE__))
    open(io->JSON3.pretty(io,packet),joinpath(directory,"summary.json"),"w");packet
end
