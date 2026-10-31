using JSON3,SHA
function freeze_pubmedqa_reading(;directory=joinpath(@__DIR__,"pubmedqa_reading_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    digest(path)=bytes2hex(sha256(read(path)))
    preparationpath=joinpath(@__DIR__,"pubmedqa_reading_preparation.json")
    prepared=JSON3.read(read(preparationpath,String))
    prepared.complete&&!prepared.ready_for_generation&&length(prepared.jobs)==48||error("Incomplete public preparation")
    prepared.protocol_sha256==digest(joinpath(@__DIR__,"pubmedqa_reading_protocol.jl"))&&
        prepared.preparer_sha256==digest(joinpath(@__DIR__,"pubmedqa_reading_study.jl"))||error("Preparation source drift")
    proofpath=joinpath(@__DIR__,"pubmedqa_reading_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&proof.protocol_checks==59&&proof.metric_checks==13&&proof.receipt_checks==17&&
        proof.preparation_checks==16&&proof.runner_checks==9&&proof.scoring_checks==8||error("Host harness unqualified")
    for (file,hash) in pairs(proof.sources)
        basename(String(file))==String(file)&&digest(joinpath(@__DIR__,String(file)))==hash||error("Qualification source drift")
    end
    parentpath=joinpath(@__DIR__,"alias_transfer3_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict("run_pubmedqa_reading.jl"=>digest(joinpath(@__DIR__,"run_pubmedqa_reading.jl"))))
    for (file,hash) in sources
        basename(file)==file&&digest(joinpath(@__DIR__,file))==hash||error("Runtime source drift")
    end
    packet=(;scope=prepared.scope,policy=prepared.policy,ready_for_generation=true,jobs=prepared.jobs,decoding=prepared.decoding,
        condition_legend=(;question_only="question_only_memorization_diagnostic",corresponding_context="oracle_corresponding_abstract",mismatched_context="next_selected_item_abstract_control"),
        public_sha256=prepared.public_sha256,snapshot_manifest_sha256=prepared.snapshot_manifest_sha256,
        preparation_sha256=digest(preparationpath),validation_sha256=digest(proofpath),
        registration_source_sha256=digest(@__FILE__),parent_runtime_conditions_sha256=digest(parentpath),
        portable_manifest_sha256=parent.portable_manifest_sha256,source_sha256=sources,
        project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,
        private_scorer_sha256=digest(joinpath(@__DIR__,"score_pubmedqa_reading.jl")),
        private_metric_sources=Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("pubmedqa_metrics.jl","pubmedqa_reading_protocol.jl")),
        execution="Fresh Laptop worker, unchanged qualified Q4 model/arithmetic/sampler, <=4 jobs per one-load window, task waitidle receipts, final context waitidle/cache save. No gold/reference data on inference worker.")
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end
