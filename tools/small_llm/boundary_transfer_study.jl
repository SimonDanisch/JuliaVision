include("julia_stencil_transfer_study.jl")
include("boundary_transfer_tasks.jl")
isdefined(@__MODULE__,:JuliaStencilAdmissionV3)||include("julia_stencil_admission_v3.jl")
isdefined(@__MODULE__,:BoundaryExampleSynthesis)||include("boundary_example_synthesis.jl")

function boundary_transfer_decision(program,bank,mode;infrastructure_ok=true,
        max_cost=8,candidate_budget=2_000_000,state_budget=100_000,seconds=10.0)
    infrastructure_ok isa Bool||throw(ArgumentError("Explicit infrastructure flag"))
    mode in (:unknown,:clamped,:periodic)||throw(ArgumentError("Invalid boundary contract"))
    settings=(;max_cost,candidate_budget,state_budget,seconds)
    empty_proposal=(;found=false,program="",reason="not_needed_or_infrastructure_failure",elapsed_seconds=0.0,candidates=0)
    plain_match=PublicExampleFallback.matches(program,bank)
    plain_proposal=infrastructure_ok&&!plain_match ? PublicExampleSynthesis.solve(bank;settings...) : empty_proposal
    example=PublicExampleFallback.choose(program,bank,plain_proposal;infrastructure_ok)
    boundary_proposal=empty_proposal
    boundary=if !infrastructure_ok
        (;selected="none",program="",reason="infrastructure_failure")
    elseif mode==:unknown
        example
    elseif plain_match&&StencilBoundaryContract.accepts(program,mode)
        (;selected="neural_baseline",program=String(program),reason="public_examples_and_boundary_match")
    else
        boundary_proposal=BoundaryExampleSynthesis.solve(bank,mode;settings...)
        if boundary_proposal.found&&PublicExampleFallback.matches(boundary_proposal.program,bank)&&StencilBoundaryContract.accepts(boundary_proposal.program,mode)
            (;selected="symbolic_public_boundary",program=String(boundary_proposal.program),reason="public_example_or_boundary_failure")
        else
            (;selected="none",program="",reason="no_matching_proposal")
        end
    end
    (;example,boundary,plain_proposal,boundary_proposal)
end

function freeze_boundary_transfer(;directory=joinpath(@__DIR__,"boundary_transfer_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    proofpath=joinpath(@__DIR__,"boundary_transfer_reference_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&length(proof.rows)==8&&all(r->r.reference.passed&&!r.mutant.passed,proof.rows)||error("References unqualified")
    for (f,h) in pairs(proof.sources)
        julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Reference source drift")
    end
    parentpath=joinpath(@__DIR__,"julia_stencil_transfer_results/conditions.json");parent=JSON3.read(read(parentpath,String))
    developmentpath=joinpath(@__DIR__,"stencil_boundary_development_results/conditions.json")
    development=JSON3.read(read(developmentpath,String))
    for (f,h) in pairs(development.sources)
        julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Developed policy drift")
    end
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
        Dict(f=>julia_transfer_digest(joinpath(@__DIR__,f)) for f in
            ("boundary_transfer_tasks.jl","boundary_transfer_study.jl","stencil_boundary_contract.jl",
             "boundary_example_synthesis.jl","julia_stencil_admission_v2.jl","julia_stencil_admission_v3.jl")))
    packet=(;scope="Prospective NEW8 authored related-family boundary-contract transfer tasks, not external/unseen-family/frontier evidence.",
        policy="One frozen Julia-source neural answer per task. Score same answer direct, plus public-example-only fallback, plus public-example/explicit-boundary fallback. Retain every task/failed/interrupted attempt. No regeneration, seed search, task-ID rules, hidden feedback, or private candidate selection. Literal boundary recognizer and solver unchanged from prior development. ALL8 choices in ALL arms committed before independent grading.",
        jobs=[(;id=t.id,task=t.id,specification=t.specification,public_cases=t.public_cases,messages=julia_stencil_transfer_messages(t)) for t in boundary_transfer_tasks()],
        decoding=parent.decoding,synthesis=parent.synthesis,
        admission="V3 safe source admission used uniformly in all arms, with exact same canonical numerical CPU/GPU IR; prior primary V1 results retained separately.",
        comparison="Workflow comparison, not equal compute or changed model weights. Both solver timings/candidate counts retained. Finite examples and conservative feature subset are not a correctness proof.",
        portable_manifest_sha256=parent.portable_manifest_sha256,source_sha256=sources,
        parent_development_conditions_sha256=julia_transfer_digest(developmentpath),
        private_reference_sha256=julia_transfer_digest(proofpath),private_scorer_sha256=julia_transfer_digest(joinpath(@__DIR__,"score_boundary_transfer.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function select_boundary_transfer(;directory=joinpath(@__DIR__,"boundary_transfer_results"))
    state=julia_stencil_transfer_state(directory);p=state.packet
    all(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)||error("ALL8 receipts before selection")
    for job in p.jobs
        output=joinpath(directory,String(job.id)*"_selection.json");isfile(output)&&continue
        path=joinpath(directory,String(job.id)*".json");m=JSON3.read(read(path,String))
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=julia_transfer_digest(state.path),attempt_sha256=julia_transfer_digest(joinpath(directory,String(job.id)*"_attempt.json")))
        ok=!hasproperty(m,:inference_error)
        lower=JuliaStencilAdmissionV3.lower(ok&&!m.truncated ? String(m.code) : "")
        program=lower.accepted ? lower.program : ""
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        mode=StencilBoundaryContract.recognize(String(job.specification));b=p.synthesis
        choices=boundary_transfer_decision(program,bank,mode;infrastructure_ok=ok,max_cost=Int(b.max_cost),
            candidate_budget=Int(b.candidate_budget),state_budget=Int(b.state_budget),seconds=Float64(b.seconds))
        row=(;id=job.id,conditions_sha256=julia_transfer_digest(state.path),generation_sha256=julia_transfer_digest(path),
            mode=String(mode),direct=(;accepted=lower.accepted,reason=lower.reason,program),choices)
        open(io->JSON3.pretty(io,row),output,"w")
    end
    (;complete=true)
end
