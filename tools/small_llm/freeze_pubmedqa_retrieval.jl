using JSON3,SHA
function freeze_pubmedqa_retrieval(;directory=joinpath(@__DIR__,"pubmedqa_retrieval_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    digest(path)=bytes2hex(sha256(read(path)))
    preparationpath=joinpath(@__DIR__,"pubmedqa_retrieval_preparation.json");p=JSON3.read(read(preparationpath,String))
    p.complete&&!p.ready_for_generation&&length(p.jobs)==64||error("Incomplete public-only preparation")
    for (file,hash) in pairs(p.sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Public preparation source drift")
    end
    proofpath=joinpath(@__DIR__,"pubmedqa_retrieval_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&proof.protocol_checks==26&&proof.runner_checks==9&&proof.scoring_checks==6||error("Host qualification incomplete")
    for (file,hash) in pairs(proof.sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Host source drift")
    end
    numericpath=joinpath(@__DIR__,"pubmedqa_retrieval_numeric_validation.json");numeric=JSON3.read(read(numericpath,String))
    numeric.complete&&numeric.software_checks==130&&numeric.corpus_sha256==p.corpus_sha256&&numeric.tasks_sha256==p.public_sha256||error("Shared retrieval math unqualified")
    for (file,hash) in pairs(numeric.sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Numeric source drift")
    end
    parentpath=joinpath(@__DIR__,"pubmedqa_reading_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    digest(parentpath)==p.prior_conditions_sha256||error("Prior runtime drift")
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict("run_pubmedqa_retrieval.jl"=>digest(joinpath(@__DIR__,"run_pubmedqa_retrieval.jl"))))
    for (file,hash) in sources
        digest(joinpath(@__DIR__,file))==hash||error("Inference runtime drift")
    end
    result=(;scope=p.scope,policy=p.policy,ready_for_generation=true,jobs=p.jobs,decoding=p.decoding,
        condition_legend=(;question_only="question_only_memorization_diagnostic",corresponding_context="oracle_abstract",mismatched_context="next_selected_item_control",retrieved_context="raw_question_body_only_BM25_top1_of1000"),
        source_sha256=sources,portable_manifest_sha256=parent.portable_manifest_sha256,
        project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,
        snapshot_manifest_sha256=p.snapshot_manifest_sha256,public_sha256=p.public_sha256,corpus_sha256=p.corpus_sha256,
        parent_runtime_conditions_sha256=digest(parentpath),preparation_sha256=digest(preparationpath),
        validation_sha256=digest(proofpath),numeric_validation_sha256=digest(numericpath),registration_source_sha256=digest(@__FILE__),
        private_scorer_sha256=digest(joinpath(@__DIR__,"score_pubmedqa_retrieval.jl")),
        private_metric_sources=Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("score_pubmedqa_reading.jl","pubmedqa_metrics.jl","pubmedqa_reading_protocol.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,result),joinpath(directory,"conditions.json"),"w");result
end
