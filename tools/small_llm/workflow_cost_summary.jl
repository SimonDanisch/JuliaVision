using JSON3,SHA

"""Descriptive final costs; never confuse generation timer with decode speed.
All jobs, including errors/truncations, are retained. No best-of-N selection.
"""
function workflow_cost_totals(packet,primary,receipts)
    primary.complete||throw(ArgumentError("Finish all scoring before final cost comparison"))
    ids=String[j.id for j in packet.jobs]
    length(ids)==length(unique(ids))||throw(ArgumentError("Duplicate jobs"))
    length(primary.rows)==length(ids)&&Set(String(r.id) for r in primary.rows)==Set(ids)||throw(ArgumentError("Score coverage"))
    Set(String.(keys(receipts)))==Set(ids)||throw(ArgumentError("Receipt coverage"))
    all(r->r.status in ("passed","failed","inference_error","paired_context_capacity_failure"),primary.rows)||throw(ArgumentError("Incomplete scores"))
    rows=map(packet.jobs) do job
        receipt=receipts[String(job.id)]
        receipt.id==job.id&&get(receipt,:normal_final_waitidle,false)===true||throw(ArgumentError("Receipt identity/cleanup"))
        score=only(filter(r->r.id==job.id,primary.rows))
        known=!hasproperty(receipt,:inference_error)&&all(k->hasproperty(receipt,k),
            (:input_tokens,:generated_tokens,:decode_steps_executed,:decode_submissions,:elapsed_seconds,:truncated))
        if known
            all(k->getproperty(receipt,k) isa Integer&&getproperty(receipt,k)>=0,
                (:input_tokens,:generated_tokens,:decode_steps_executed,:decode_submissions))||throw(ArgumentError("Token counts"))
            isfinite(receipt.elapsed_seconds)&&receipt.elapsed_seconds>=0||throw(ArgumentError("Generation timer"))
            receipt.generated_tokens<=receipt.decode_steps_executed&&receipt.decode_submissions==receipt.decode_steps_executed||throw(ArgumentError("Decode accounting"))
        end
        (;id=String(job.id),arm=String(job.arm),status=String(score.status),cost_known=known,
            input_tokens=known ? Int(receipt.input_tokens) : nothing,
            generated_tokens=known ? Int(receipt.generated_tokens) : nothing,
            decode_steps=known ? Int(receipt.decode_steps_executed) : nothing,
            decode_submissions=known ? Int(receipt.decode_submissions) : nothing,
            generation_timer_seconds=known ? Float64(receipt.elapsed_seconds) : nothing,
            truncated=known ? receipt.truncated : nothing)
    end
    arms=map(unique(String(j.arm) for j in packet.jobs)) do arm
        allrows=filter(r->r.arm==arm,rows);known=filter(r->r.cost_known,allrows)
        n_pass=count(r->r.status=="passed",allrows)
        total(field)=sum((getproperty(r,field) for r in known);init=0)
        elapsed=total(:generation_timer_seconds)
        (;arm,label=hasproperty(packet,:condition_legend) ? String(getproperty(packet.condition_legend,Symbol(arm))) : arm,
            tasks=length(allrows),passed=n_pass,cost_known=length(known),cost_missing=length(allrows)-length(known),
            input_tokens=total(:input_tokens),generated_tokens=total(:generated_tokens),
            decode_steps=total(:decode_steps),decode_submissions=total(:decode_submissions),
            generation_timer_seconds=elapsed,truncated=count(r->r.truncated===true,allrows),
            timed_seconds_per_pass=length(known)==length(allrows)&&n_pass>0 ? elapsed/n_pass : nothing)
    end
    (;complete=true,rows,arms,
        scope="Descriptive incremental generation costs for exactly the registered jobs; not a pooled intelligence score, throughput benchmark or causal speed comparison.",
        timer="Existing generation timer includes prompt processing, decode and graph/JIT work after session creation; excludes model loading, session creation, worker startup/retirement, sandbox tests and external solvers. Token rate must NOT be reported as steady-state decode throughput.",
        baseline_cost="Revision arms exclude the shared original answer-generation cost; these are incremental costs, NOT end-to-end workflow costs. Missing error costs remain unknown, never zero.")
end

function summarize_workflow_costs(directory;destination=joinpath(directory,"incremental_costs.json"))
    isfile(destination)&&error("Preserve final cost receipt")
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(directory,"conditions.json");summarypath=joinpath(directory,"summary.json")
    packet=JSON3.read(read(conditionpath,String));primary=JSON3.read(read(summarypath,String))
    primary.conditions_sha256==digest(conditionpath)||error("Score provenance")
    receipts=Dict(String(job.id)=>JSON3.read(read(joinpath(directory,String(job.id)*".json"),String)) for job in packet.jobs)
    for (id,receipt) in receipts
        receipt.conditions_sha256==digest(conditionpath)||error("Generation provenance")
        score=only(filter(r->r.id==id,primary.rows))
        hasproperty(score,:generation_sha256)&&score.generation_sha256!=digest(joinpath(directory,id*".json"))&&error("Score generation drift")
    end
    result=(;workflow_cost_totals(packet,primary,receipts)...,
        conditions_sha256=digest(conditionpath),summary_sha256=digest(summarypath),
        generations=Dict(id=>digest(joinpath(directory,id*".json")) for id in keys(receipts)),
        source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
