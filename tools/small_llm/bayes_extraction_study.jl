include("bayes_extraction_tasks.jl")
using JSON3,SHA
function freeze_bayes_extraction(;token_count,directory=joinpath(@__DIR__,"bayes_extraction_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    digest(path)=bytes2hex(sha256(read(path)))
    referencepath=joinpath(@__DIR__,"bayes_extraction_reference_validation.json")
    reference=JSON3.read(read(referencepath,String))
    reference.complete&&length(reference.rows)==8&&reference.checks>=65||error("References/shared solver unqualified")
    for (file,hash) in pairs(reference.sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Reference/source drift")
    end
    jobs=Any[]
    for (i,task) in enumerate(bayes_extraction_tasks()),arm in (isodd(i) ? ("direct_probability","bayes_network") : ("bayes_network","direct_probability"))
        messages=bayes_extraction_messages(task,arm);tokens=token_count(messages)
        tokens isa Integer&&!(tokens isa Bool)&&tokens>0&&tokens+1024+32<=8192||error("Capacity failure; no task filtering")
        push!(jobs,(;id=task.id*"_"*arm,task=task.id,arm,messages,input_tokens=tokens))
    end
    parentpath=joinpath(@__DIR__,"plot_interface_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    for (file,hash) in pairs(parent.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Qualified runtime drift")
    end
    result=(;ready_for_generation=true,
        scope="NEW8 authored fictional finite Bayesian word problems; direct numeric answer vs model-extracted complete network plus exact SHARED CPU/GPU solver. Not medical guidance, a public held-out benchmark or frontier equivalence.",
        policy="ALL16 requests/references/schema/grader frozen before generation, one immutable sampled answer each, counterbalanced arms. Same entire public problem/schema in both arms, only requested representation differs, same1024 nonthinking allowance. No retries/private grade feedback; all workers retired and ALL16 outputs saved before grading. Solver sees model network, never a corrected/reference network. Report admission, numeric posterior accuracy and full network faithfulness separately. Primary tool success requires faithful network AND accurate posterior; direct success requires accurate numeric posterior. Additional solver compute is charged separately.",
        jobs,decoding=parent.decoding,source_sha256=parent.source_sha256,portable_manifest_sha256=parent.portable_manifest_sha256,
        project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,parent_runtime_conditions_sha256=digest(parentpath),
        condition_legend=(;direct_probability="model_numeric_posterior",bayes_network="model_network_plus_shared_exact_solver"),
        public_task_sha256=digest(joinpath(@__DIR__,"bayes_extraction_tasks.jl")),registration_source_sha256=digest(@__FILE__),
        private_reference_validation_sha256=digest(referencepath),private_scorer_sha256=digest(joinpath(@__DIR__,"score_bayes_extraction.jl")),
        private_sources=Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("bayes_extraction_references.jl","bayes_extraction_admission.jl","bayes_json.jl","portable_evidence.jl","score_pubmedqa_reading.jl","pubmedqa_metrics.jl","pubmedqa_reading_protocol.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,result),joinpath(directory,"conditions.json"),"w");result
end
