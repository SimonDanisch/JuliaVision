# Posthoc SAME8 workflow extension. Original frozen6/8 transfer stays unchanged.
include("julia_stencil_transfer_tasks.jl")
include("boundary_example_synthesis.jl")
include("public_example_fallback.jl")
using JSON3,SHA
boundary_digest(path)=bytes2hex(sha256(read(path)))

function freeze_stencil_boundary_development(;directory=joinpath(@__DIR__,"stencil_boundary_development_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    parent=joinpath(@__DIR__,"julia_stencil_transfer_results")
    summarypath=joinpath(parent,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.complete&&summary.total==8||error("Primary transfer incomplete")
    tasks=julia_stencil_transfer_tasks();jobs=Any[]
    for task in tasks
        selection=joinpath(parent,task.id*"_selection.json");s=JSON3.read(read(selection,String))
        generation=joinpath(parent,task.id*".json");m=JSON3.read(read(generation,String))
        s.generation_sha256==boundary_digest(generation)||error("Parent receipt drift")
        push!(jobs,(;id=task.id,specification=task.specification,public_cases=task.public_cases,
            parent_selection_sha256=boundary_digest(selection),parent_generation_sha256=boundary_digest(generation)))
    end
    packet=(;scope="POSTHOC SAME8 authored Julia tasks after observing6/8 primary transfer and boundary mistakes. No new neural generations; NOT independent validation/model-only intelligence.",
        policy="Apply uniform literal affirmative boundary-phrase recognizer to EVERY public specification. Unknown/negated/conflicting phrases impose no feature restriction. Retain original neural IR only if it fits public examples AND conservative boundary feature subset. Otherwise first bounded symbolic fit with prohibited boundary pushes excluded. No task-ID matching, hidden values, regenerated neural answers, or private-score candidate selection. Commit ALL8 decisions before grading.",
        semantics="Conservative source-feature admission, not a proof that a program using the other boundary mode is mathematically wrong; valid cancelling programs may be excluded. All arithmetic reuses the same CPU/GPU evaluator.",
        jobs,synthesis=(;max_cost=8,candidate_budget=2_000_000,state_budget=100_000,seconds=10.0),
        parent_summary_sha256=boundary_digest(summarypath),
        observed_parent_results=(;direct_passed=summary.direct_passed,hybrid_passed=summary.hybrid_passed,total=summary.total),
        sources=Dict(f=>boundary_digest(joinpath(@__DIR__,f)) for f in
            ("stencil_boundary_contract.jl","boundary_example_synthesis.jl","stencil_boundary_development.jl",
             "public_example_synthesis.jl","public_example_fallback.jl","rpn_example_semantics.jl","rpn_example_constraint.jl","rpn_prefix_grammar.jl","rpn_pointwise_interface.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function select_stencil_boundary_development(;directory=joinpath(@__DIR__,"stencil_boundary_development_results"))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    for (f,h) in pairs(p.sources)
        boundary_digest(joinpath(@__DIR__,String(f)))==h||error("Development source drift")
    end
    parent=joinpath(@__DIR__,"julia_stencil_transfer_results")
    boundary_digest(joinpath(parent,"summary.json"))==p.parent_summary_sha256||error("Parent summary drift")
    for job in p.jobs
        output=joinpath(directory,String(job.id)*".json");isfile(output)&&continue
        selection=joinpath(parent,String(job.id)*"_selection.json");generation=joinpath(parent,String(job.id)*".json")
        boundary_digest(selection)==job.parent_selection_sha256&&boundary_digest(generation)==job.parent_generation_sha256||error("Parent output drift")
        s=JSON3.read(read(selection,String));m=JSON3.read(read(generation,String))
        mode=StencilBoundaryContract.recognize(String(job.specification))
        program=String(s.direct.program);ok=!hasproperty(m,:inference_error)&&m.normal_final_waitidle
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        fits=PublicExampleFallback.matches(program,bank)&&StencilBoundaryContract.accepts(program,mode)
        proposal=if ok&&!fits
            b=p.synthesis
            BoundaryExampleSynthesis.solve(bank,mode;max_cost=Int(b.max_cost),candidate_budget=Int(b.candidate_budget),state_budget=Int(b.state_budget),seconds=Float64(b.seconds))
        else
            (;found=false,program="",reason="not_needed_or_infrastructure_failure",elapsed_seconds=0.0,candidates=0)
        end
        selected=if !ok
            (;selected="none",program="",reason="infrastructure_failure")
        elseif fits
            (;selected="neural_baseline",program,reason="public_examples_and_boundary_match")
        elseif proposal.found&&PublicExampleFallback.matches(proposal.program,bank)&&StencilBoundaryContract.accepts(proposal.program,mode)
            (;selected="symbolic_public_boundary",program=String(proposal.program),reason="public_boundary_contract_failure")
        else
            (;selected="none",program="",reason="no_matching_proposal")
        end
        row=(;id=job.id,conditions_sha256=boundary_digest(conditionpath),mode=String(mode),parent_selection_sha256=job.parent_selection_sha256,proposal,selected)
        open(io->JSON3.pretty(io,row),output,"w")
    end
    (;complete=true)
end
