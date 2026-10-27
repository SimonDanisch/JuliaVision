isdefined(@__MODULE__,:PublicExampleFallback)||include("public_example_fallback.jl")
using JSON3,SHA

function freeze_public_example_hybrid(;directory=joinpath(@__DIR__,"public_example_hybrid_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve hybrid registration")
    modeldir=joinpath(@__DIR__,"rpn_public_example_results");modelpath=joinpath(modeldir,"conditions.json")
    symbolicdir=joinpath(@__DIR__,"public_example_synthesis_results");symbolicpath=joinpath(symbolicdir,"conditions.json")
    p=JSON3.read(read(modelpath,String));s=JSON3.read(read(symbolicpath,String))
    jobs=[j for j in p.jobs if j.arm=="disabled"]
    length(jobs)==4&&length(unique(j.task for j in jobs))==4||error("Retain all four baseline tasks")
    [j.task for j in jobs]==[c.task for c in s.cases]||error("Symbolic cohort order")
    all(j.public_cases==c.public_cases for (j,c) in zip(jobs,s.cases))||error("Public examples drift")
    proofpath=joinpath(@__DIR__,"public_example_fallback_validation.json")
    proof=JSON3.read(read(proofpath,String));proof.complete&&proof.passed==11||error("Fallback unqualified")
    for (file,hash) in pairs(proof.sources)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Fallback source drift")
    end
    observed=Dict(String(j.id)=>bytes2hex(sha256(read(joinpath(modeldir,String(j.id)*".json")))) for j in p.jobs if isfile(joinpath(modeldir,String(j.id)*".json")))
    packet=(;scope="POSTHOC combined workflow on SAME4 authored development tasks after partial model outputs and symbolic-only scoring. No new neural generations; NOT independent held-out evidence or model-only intelligence.",
        policy="Uniform rule: preserve original numeric-disabled baseline if raw infrastructure is valid and its complete program fits public examples. Otherwise use the frozen first-match symbolic result only if it fits the SAME public examples. Never use numeric-enabled outputs, hidden tests, task-ID rules, private correctness labels, retries or extra candidates for selection. Infrastructure errors remain failures.",
        jobs,model_conditions_sha256=bytes2hex(sha256(read(modelpath))),symbolic_conditions_sha256=bytes2hex(sha256(read(symbolicpath))),
        observed_model_outputs_at_registration=observed,
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("public_example_fallback.jl","public_example_hybrid_experiment.jl","rpn_example_semantics.jl","rpn_example_constraint.jl","rpn_prefix_grammar.jl","rpn_pointwise_interface.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function select_public_example_hybrid(;directory=joinpath(@__DIR__,"public_example_hybrid_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.sources)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Frozen policy source drift")
    end
    modeldir=joinpath(@__DIR__,"rpn_public_example_results");symbolicdir=joinpath(@__DIR__,"public_example_synthesis_results")
    bytes2hex(sha256(read(joinpath(modeldir,"conditions.json"))))==p.model_conditions_sha256||error("Model protocol drift")
    bytes2hex(sha256(read(joinpath(symbolicdir,"conditions.json"))))==p.symbolic_conditions_sha256||error("Symbolic protocol drift")
    all(isfile(joinpath(modeldir,String(j.id)*".json")) for j in p.jobs)||error("Wait for ALL four baselines before selection")
    all(isfile(joinpath(symbolicdir,String(j.task)*".json")) for j in p.jobs)||error("Wait for ALL symbolic results")
    # Commit every selection before any independent correctness grading.
    for job in p.jobs
        output=joinpath(directory,String(job.task)*".json");isfile(output)&&continue
        modelpath=joinpath(modeldir,String(job.id)*".json");m=JSON3.read(read(modelpath,String))
        m.id==job.id&&m.conditions_sha256==p.model_conditions_sha256&&m.messages==job.messages&&m.normal_final_waitidle||error("Baseline receipt drift")
        symbolicpath=joinpath(symbolicdir,String(job.task)*".json");s=JSON3.read(read(symbolicpath,String))
        s.task==job.task&&s.conditions_sha256==p.symbolic_conditions_sha256||error("Symbolic receipt drift")
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        ok=!hasproperty(m,:inference_error)&&m.constraint.raw_status==0&&m.constraint.accepted
        selected=PublicExampleFallback.choose(ok ? String(m.answer) : "",bank,s.result;
            infrastructure_ok=!hasproperty(m,:inference_error)&&m.constraint.raw_status==0)
        row=(;task=job.task,conditions_sha256=bytes2hex(sha256(read(path))),model_sha256=bytes2hex(sha256(read(modelpath))),
            symbolic_sha256=bytes2hex(sha256(read(symbolicpath))),selected)
        open(io->JSON3.pretty(io,row),output,"w")
    end
    (;complete=true)
end
