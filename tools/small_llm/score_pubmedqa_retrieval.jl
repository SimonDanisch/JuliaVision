isdefined(@__MODULE__,:validate_pubmedqa_receipt)||include("score_pubmedqa_reading.jl")

function score_pubmedqa_retrieval(;directory=joinpath(@__DIR__,"pubmedqa_retrieval_results"),snapshot="gen/pubmedqa-retrieval-20261025")
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    digest(@__FILE__)==p.private_scorer_sha256||error("Private scorer drift")
    for (file,hash) in pairs(p.private_metric_sources)
        basename(String(file))==String(file)&&digest(joinpath(@__DIR__,String(file)))==hash||error("Metric/receipt source drift")
    end
    manifestpath=joinpath(snapshot,"manifest.json");manifest=JSON3.read(read(manifestpath,String))
    digest(manifestpath)==p.snapshot_manifest_sha256||error("Snapshot drift")
    goldpath=joinpath(snapshot,"private/gold.json");digest(goldpath)==manifest.gold_sha256||error("Gold drift")
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
    arms=("question_only","corresponding_context","mismatched_context","retrieved_context")
    result=(;complete,scope=p.scope,policy=p.policy,rows,
        arms=complete ? PubMedQAMetrics.summarize(rows,gold;arms) : nothing,
        conditions_sha256=digest(path),scorer_sha256=digest(@__FILE__),
        caveats="New16 public papers; same-seed four-arm comparison, body-only BM25 over1000 matching-domain abstracts. Not a general web retriever, official leaderboard, contamination-free evaluation, medical consensus, calibrated uncertainty or patient advice.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
