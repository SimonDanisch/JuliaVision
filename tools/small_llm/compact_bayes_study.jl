include("compact_bayes_tasks.jl")
using JSON3,SHA

function freeze_compact_bayes(;token_count,directory=joinpath(@__DIR__,"compact_bayes_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    digest(path)=bytes2hex(sha256(read(path)))
    parentpath=joinpath(@__DIR__,"bayes_extraction_results/conditions.json")
    parent=JSON3.read(read(parentpath,String));summarypath=joinpath(dirname(parentpath),"summary.json")
    summary=JSON3.read(read(summarypath,String));summary.complete&&length(summary.rows)==16||error("Parent incomplete")
    summary.conditions_sha256==digest(parentpath)||error("Parent score drift")
    proofpath=joinpath(@__DIR__,"compact_bayes_reference_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&proof.checks>=86&&length(proof.rows)==8||error("Shared solver/reference gate")
    for registry in (parent.source_sha256,proof.source_sha256),(file,hash) in pairs(registry)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Qualified source drift")
    end
    tasks=bayes_extraction_tasks();jobs=map(tasks) do task
        messages=compact_bayes_messages(task);tokens=token_count(messages)
        tokens isa Integer&&!(tokens isa Bool)&&tokens>0&&tokens+1024+32<=8192||error("Capacity; no task filtering")
        original=joinpath(dirname(parentpath),task.id*"_bayes_network.json")
        (;id=task.id*"_compact_network",task=task.id,arm="compact_network",messages,input_tokens=tokens,
            original_generation_sha256=digest(original))
    end
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("compact_bayes_tasks.jl","compact_bayes_study.jl")))
    p=(;ready_for_generation=true,jobs,decoding=parent.decoding,source_sha256=sources,
        scope="POSTHOC SAME8 fictional Bayesian problems with a shorter conditional-table interface. Original full-schema network grades already known; not independent transfer or clinical guidance.",
        policy="ALL8 original problems retained in original order; one immutable nonthinking1024-token sampled answer, original sampler/seed/model. No retries, examples, reference answers or hidden grade feedback in prompts. Different public interface and prompt/output lengths: workflow comparison, not isolated tool arithmetic, equal end-to-end compute, weight improvement or frontier evidence. Packer ONLY computes declared binary complements and expands explicit rows; unchanged shared solver runs the generated network, never a corrected/reference one. Freeze all sources/grades before generation, ALL8 before grading.",
        portable_manifest_sha256=parent.portable_manifest_sha256,project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,
        parent_conditions_sha256=digest(parentpath),parent_summary_sha256=digest(summarypath),reference_validation_sha256=digest(proofpath),
        public_tasks_sha256=digest(joinpath(@__DIR__,"compact_bayes_tasks.jl")),
        private_source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("score_compact_bayes.jl","compact_bayes_admission.jl",
            "compact_bayes_json.jl","bayes_extraction_references.jl","bayes_extraction_admission.jl","bayes_json.jl","portable_evidence.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end
