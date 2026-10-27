isdefined(@__MODULE__,:PublicExampleSynthesis)||include("public_example_synthesis.jl")
using JSON3,SHA

function freeze_public_example_synthesis(;directory=joinpath(@__DIR__,"public_example_synthesis_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve symbolic baseline registration")
    modeldir=joinpath(@__DIR__,"rpn_public_example_results");modelpath=joinpath(modeldir,"conditions.json")
    model=JSON3.read(read(modelpath,String))
    cases=Any[]
    for task in unique(j.task for j in model.jobs)
        pair=filter(j->j.task==task,model.jobs)
        length(pair)==2&&pair[1].public_cases==pair[2].public_cases||error("Public case mismatch")
        push!(cases,(;task,public_cases=pair[1].public_cases))
    end
    length(cases)==4||error("Retain all four tasks")
    proofpath=joinpath(@__DIR__,"public_example_synthesis_validation.json")
    proof=JSON3.read(read(proofpath,String));proof.complete&&proof.passed==21||error("Solver unqualified")
    for (file,hash) in pairs(proof.sources)
        basename(String(file))==String(file)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Source drift")
    end
    sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
        ("public_example_synthesis.jl","public_example_synthesis_experiment.jl","rpn_example_semantics.jl","rpn_prefix_grammar.jl"))
    observed=Dict(String(j.id)=>bytes2hex(sha256(read(joinpath(modeldir,String(j.id)*".json"))))
        for j in model.jobs if isfile(joinpath(modeldir,String(j.id)*".json")))
    p=(;scope="POSTHOC symbolic synthesis baseline on SAME4 authored tasks, after partial neural outputs. No new neural attempts, not independent transfer/model intelligence or proof beyond public examples.",
        policy="All4 retained; fixed candidate/state/cost/time limits before search; first public-example match only. No task formula, reference, hidden inputs, model output or grading feedback enters the solver. Independent scoring only after all search outputs committed; no retries/hidden-score selection.",
        cases,options=(;max_cost=8,candidate_budget=2_000_000,state_budget=100_000,seconds=10.0),
        max_depth=8,model_conditions_sha256=bytes2hex(sha256(read(modelpath))),observed_model_outputs_at_registration=observed,
        solver_proof_sha256=bytes2hex(sha256(read(proofpath))),sources)
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end

function run_public_example_synthesis(;directory=joinpath(@__DIR__,"public_example_synthesis_results"))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.sources)
        basename(String(file))==String(file)&&bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash||error("Frozen solver source drift")
    end
    digest=bytes2hex(sha256(read(path)))
    for case in p.cases
        output=joinpath(directory,String(case.task)*".json");attempt=replace(output,".json"=>"_attempt.json")
        isfile(output)&&continue
        ispath(attempt)&&error("Interrupted search; do not repeat")
        open(io->JSON3.pretty(io,(;task=case.task,conditions_sha256=digest,pid=getpid())),attempt,"w")
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in case.public_cases];max_depth=Int(p.max_depth))
        o=p.options;started=time()
        result=PublicExampleSynthesis.solve(bank;max_cost=Int(o.max_cost),candidate_budget=Int(o.candidate_budget),state_budget=Int(o.state_budget),seconds=Float64(o.seconds))
        row=(;task=case.task,conditions_sha256=digest,attempt_sha256=bytes2hex(sha256(read(attempt))),
            total_call_seconds_including_jit=time()-started,result)
        open(io->JSON3.pretty(io,row),output,"w")
        println(case.task," search: ",result.reason," candidates=",result.candidates," states=",result.unique_states);flush(stdout)
        yield()
    end
    (;complete=all(isfile(joinpath(directory,String(c.task)*".json")) for c in p.cases))
end
