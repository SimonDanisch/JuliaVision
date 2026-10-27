using JSON3,SHA
function identical_prompt_groups(receipts)
    groups=Dict{String,Vector{Any}}()
    for receipt in receipts
        hasproperty(receipt,:inference_error)&&throw(ArgumentError("Infrastructure error cannot certify identity"))
        receipt.normal_final_waitidle===true||throw(ArgumentError("No task synchronization receipt"))
        receipt.input_sha256==bytes2hex(sha256(JSON3.write(receipt.messages)))||throw(ArgumentError("Actual prompt hash drift"))
        settings=(;seed=receipt.sampling.seed,topk=receipt.sampling.topk,temperature=receipt.sampling.temperature,
            top_p=receipt.sampling.top_p,presence=receipt.sampling.presence,budget=receipt.budget_tokens,chunk=receipt.eos_output_poll_chunk)
        key=String(receipt.input_sha256)*":"*JSON3.write(settings)
        push!(get!(groups,key,Any[]),receipt)
    end
    rows=Any[]
    for (key,group) in sort!(collect(groups);by=first)
        length(group)>1||continue
        agreement=length(unique((r.answer,r.input_tokens,r.generated_tokens,r.truncated) for r in group))==1
        push!(rows,(;ids=String[r.id for r in group],same_prompt_and_sampling=true,answer_agreement=agreement))
    end
    (;groups=rows,group_count=length(rows),all_agree=all(r->r.answer_agreement,rows),
        comparisons=sum(length(r.ids)-1 for r in rows;init=0))
end

function audit_identical_prompt_outputs(directory;destination=joinpath(directory,"identical_prompt_audit.json"))
    isfile(destination)&&error("Preserve audit")
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    summarypath=joinpath(directory,"summary.json");s=JSON3.read(read(summarypath,String))
    s.complete&&s.conditions_sha256==digest(conditionpath)||error("Complete frozen study required")
    receipts=map(p.jobs) do job
        path=joinpath(directory,String(job.id)*".json");r=JSON3.read(read(path,String))
        r.conditions_sha256==digest(conditionpath)&&r.messages==job.messages||error("Receipt/conditions drift")
        score=only(filter(row->row.id==job.id,s.rows))
        score.generation_sha256==digest(path)||error("Score/receipt drift")
        r
    end
    audit=identical_prompt_groups(receipts)
    result=(;complete=true,audit...,conditions_sha256=digest(conditionpath),summary_sha256=digest(summarypath),
        scope="Within-study repeated exact prompt/sampling output agreement ONLY. Different contexts are not compared. Not FP32/reference logit parity, broad determinism, medical reliability or a changed score.",
        source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
