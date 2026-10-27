using JSON3,SHA
bayes_thinking_digest(path)=bytes2hex(sha256(read(path)))

function freeze_bayes_thinking(;token_count,directory=joinpath(@__DIR__,"bayes_thinking_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    parentpath=joinpath(@__DIR__,"bayes_extraction_results/conditions.json")
    parent=JSON3.read(read(parentpath,String))
    summarypath=joinpath(dirname(parentpath),"summary.json")
    summary=JSON3.read(read(summarypath,String))
    summary.complete&&length(summary.rows)==16||error("Finish original study first")
    summary.conditions_sha256==bayes_thinking_digest(parentpath)||error("Parent score drift")
    gatepath=joinpath(@__DIR__,"laptop_portable_identity_gate.json")
    gate=JSON3.read(read(gatepath,String))
    gate.complete&&gate.normal_final_waitidle&&gate.hardware_fixture_passed==156&&gate.full_model_passed==10||error("Controller gate required")
    for registry in (parent.source_sha256,gate.source_sha256), (file,hash) in pairs(registry)
        bayes_thinking_digest(joinpath(@__DIR__,String(file)))==hash||error("Qualified source drift")
    end
    jobs=map(parent.jobs) do job
        tokens=token_count(job.messages)
        tokens isa Integer&&!(tokens isa Bool)&&tokens>0&&tokens+2048+32<=8192||error("Capacity; no task filtering")
        (;id=String(job.id),task=String(job.task),arm=String(job.arm),messages=job.messages,input_tokens=tokens,
            original_generation_sha256=bayes_thinking_digest(joinpath(dirname(parentpath),String(job.id)*".json")))
    end
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict(f=>bayes_thinking_digest(joinpath(@__DIR__,f)) for f in ("bayes_thinking_study.jl","run_bayes_thinking.jl")))
    p=(;ready_for_generation=true,jobs,
        scope="POST-HOC same8 authored fictional Bayesian problems; original nonthinking results already known. Not independent transfer, clinical guidance or frontier equivalence.",
        policy="One immutable answer per ALL16 original jobs/order, original public prompts unchanged. Thinking=true, total2048 output tokens, force one closing marker after1024 thinking steps if necessary using previously qualified raw-logit-validating controller. Same seed/sampler; no retries, seeds or favorable subset selection. Freeze new grader/extraction before generation; collect ALL16 before grading. Compare original1024 nonthinking as workflow diagnostic, NOT isolated mode effect or equal-compute trial. Retain truncation/controller/solver costs and all failures. Exact solver sees only generated network, never a corrected one.",
        decoding=(;pairs(parent.decoding)...,max_tokens=2048,thinking=true,thinking_budget=1024),
        source_sha256=sources,portable_manifest_sha256=parent.portable_manifest_sha256,
        project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,
        parent_conditions_sha256=bayes_thinking_digest(parentpath),parent_summary_sha256=bayes_thinking_digest(summarypath),
        controller_gate_sha256=bayes_thinking_digest(gatepath),
        private_source_sha256=Dict(f=>bayes_thinking_digest(joinpath(@__DIR__,f)) for f in
            ("bayes_thinking_admission.jl","score_bayes_thinking.jl","bayes_extraction_references.jl",
             "bayes_extraction_admission.jl","bayes_json.jl","portable_evidence.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end
