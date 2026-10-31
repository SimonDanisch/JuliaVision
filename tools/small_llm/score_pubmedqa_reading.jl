include("pubmedqa_reading_protocol.jl")
include("pubmedqa_metrics.jl")
using JSON3,SHA

function validate_pubmedqa_receipt(receipt,job,p;condition_hash,attempt_hash)
    receipt.id==job.id&&receipt.messages==job.messages&&receipt.normal_final_waitidle===true||error("Identity/prompt/task synchronization")
    receipt.conditions_sha256==condition_hash&&receipt.attempt_sha256==attempt_hash||error("Generation provenance")
    hasproperty(receipt,:inference_error)&&return nothing
    receipt.input_sha256==bytes2hex(sha256(JSON3.write(job.messages)))||error("Actual prompt drift")
    receipt.budget_tokens==p.decoding.max_tokens&&0<receipt.decode_steps_executed<=p.decoding.max_tokens||error("Token budget")
    receipt.decode_submissions==receipt.decode_steps_executed&&receipt.sampling.draws==receipt.decode_steps_executed+1||error("Decode accounting")
    receipt.eos_output_poll_chunk==p.decoding.output_chunk&&receipt.truncated isa Bool||error("Polling/completion")
    for field in (:seed,:topk,:temperature,:top_p,:presence)
        actual=getproperty(receipt.sampling,field);expected=getproperty(p.decoding,field)
        (field in (:temperature,:top_p,:presence) ? Float32(actual)==Float32(expected) : actual==expected)||error("Sampling drift")
    end
    nothing
end

function score_pubmedqa_reading(;directory=joinpath(@__DIR__,"pubmedqa_reading_results"),snapshot="gen/pubmedqa-reading-pilot-20261025")
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    digest(@__FILE__)==p.private_scorer_sha256||error("Private scorer drift")
    for (file,hash) in pairs(p.private_metric_sources)
        basename(String(file))==String(file)&&digest(joinpath(@__DIR__,String(file)))==hash||error("Metric source drift")
    end
    manifestpath=joinpath(snapshot,"manifest.json");manifest=JSON3.read(read(manifestpath,String))
    digest(manifestpath)==p.snapshot_manifest_sha256||error("Source manifest drift")
    goldpath=joinpath(snapshot,"private/gold.json")
    digest(goldpath)==manifest.gold_sha256||error("Gold drift")
    gold=Dict(String(k)=>String(v) for (k,v) in pairs(JSON3.read(read(goldpath,String))))
    Set(String(j.task) for j in p.jobs)==Set(keys(gold))||error("Cohort drift")
    rows=map(p.jobs) do job
        output=joinpath(directory,String(job.id)*".json");attempt=joinpath(directory,String(job.id)*"_attempt.json")
        if !isfile(output)
            return (;id=job.id,task=job.task,arm=job.arm,status=isfile(attempt) ? "interrupted_attempt" : "pending")
        end
        receipt=JSON3.read(read(output,String))
        validate_pubmedqa_receipt(receipt,job,p;condition_hash=digest(path),attempt_hash=digest(attempt))
        inference_failed=hasproperty(receipt,:inference_error)
        prediction=inference_failed||receipt.truncated ? "invalid" : PubMedQAReadingProtocol.decision(String(receipt.answer))
        status=inference_failed ? "inference_error" : prediction=="invalid" ? "invalid_answer" : prediction==gold[String(job.task)] ? "passed" : "failed"
        (;id=job.id,task=job.task,arm=job.arm,status,prediction,generation_sha256=digest(output),truncated=inference_failed ? nothing : receipt.truncated)
    end
    complete=all(r->r.status∉("pending","interrupted_attempt"),rows)
    result=(;complete,scope=p.scope,policy=p.policy,rows,
        arms=complete ? PubMedQAMetrics.summarize(rows,gold) : nothing,
        conditions_sha256=digest(path),scorer_sha256=digest(@__FILE__),
        caveats="16 public old papers, one seed, oracle source arm and unrelated-context control; not current medical consensus, patient advice, calibrated uncertainty or a complete retriever. Invalid/truncated/error answers retained in denominators; no frontier claim.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
