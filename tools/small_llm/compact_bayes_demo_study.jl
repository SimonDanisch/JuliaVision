include("compact_bayes_study.jl")
include("compact_bayes_demonstrations.jl")

function freeze_compact_bayes_demo(;token_count,directory=joinpath(@__DIR__,"compact_bayes_demo_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    digest(path)=bytes2hex(sha256(read(path)))
    barepath=joinpath(@__DIR__,"compact_bayes_results/conditions.json")
    bare=JSON3.read(read(barepath,String));summarypath=joinpath(dirname(barepath),"summary.json")
    summary=JSON3.read(read(summarypath,String))
    summary.complete&&length(summary.rows)==8&&summary.conditions_sha256==digest(barepath)||error("Bare study incomplete/drift")
    proofpath=joinpath(@__DIR__,"compact_bayes_demonstrations_validation.json")
    proof=JSON3.read(read(proofpath,String));proof.complete&&proof.passed>=35||error("Public examples unqualified")
    for registry in (bare.source_sha256,proof.source_sha256),(f,h) in pairs(registry)
        digest(joinpath(@__DIR__,String(f)))==h||error("Qualified source drift")
    end
    base=mktempdir() do temporary
        freeze_compact_bayes(;token_count,directory=temporary)
    end
    jobs=map(base.jobs) do job
        messages=compact_bayes_demonstration_messages(job.messages);tokens=token_count(messages)
        tokens isa Integer&&!(tokens isa Bool)&&tokens>0&&tokens+1024+32<=8192||error("Capacity; no filtering")
        (;job...,id=job.id*"_demonstrations",messages,input_tokens=tokens,
            bare_generation_sha256=digest(joinpath(dirname(barepath),job.id*".json")))
    end
    sources=merge(base.source_sha256,Dict(f=>digest(joinpath(@__DIR__,f)) for f in
        ("compact_bayes_demo_study.jl","compact_bayes_demonstrations.jl")))
    p=(;base...,jobs,source_sha256=sources,
        scope="POSTHOC SAME8 compact Bayesian extraction tasks with two public unrelated format examples; not independent transfer or medical guidance.",
        policy="ALL8 original tasks/order, same nonthinking1024 sampler/seed/model/unchanged grader and shared exact solver. Two fixed unrelated network examples, no posterior answers or hidden task feedback. Earlier failures known; longer input changes compute. No retries or selected tasks; finish and retire ALL8 before grading. This is an interface demonstration comparison, not weight improvement or frontier intelligence.",
        bare_conditions_sha256=digest(barepath),bare_summary_sha256=digest(summarypath),
        demonstrations_validation_sha256=digest(proofpath))
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end
