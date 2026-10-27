# ROOT ONLY; ALL public-only decisions must exist before any private scoring.
include("boundary_transfer_study.jl")
include("qualify_boundary_transfer.jl")
function score_boundary_transfer(device;directory=joinpath(@__DIR__,"boundary_transfer_results"))
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only grading")
    state=julia_stencil_transfer_state(directory);p=state.packet
    julia_transfer_digest(@__FILE__)==p.private_scorer_sha256||error("Scorer drift")
    proofpath=joinpath(@__DIR__,"boundary_transfer_reference_validation.json")
    julia_transfer_digest(proofpath)==p.private_reference_sha256||error("Reference receipt drift")
    proof=JSON3.read(read(proofpath,String))
    for (f,h) in pairs(proof.sources)
        julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Private source drift")
    end
    all(isfile(joinpath(directory,String(j.id)*"_selection.json")) for j in p.jobs)||error("ALL arm decisions before grading")
    rows=Any[]
    for (job,(_,formula)) in zip(p.jobs,private_boundary_transfer_fixtures())
        path=joinpath(directory,String(job.id)*"_selection.json");s=JSON3.read(read(path,String))
        source=joinpath(directory,String(job.id)*".json");m=JSON3.read(read(source,String))
        s.conditions_sha256==julia_transfer_digest(state.path)&&s.generation_sha256==julia_transfer_digest(source)||error("Selection provenance")
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=julia_transfer_digest(state.path),attempt_sha256=julia_transfer_digest(joinpath(directory,String(job.id)*"_attempt.json")))
        ok=!hasproperty(m,:inference_error)
        lower=JuliaStencilAdmissionV3.lower(ok&&!m.truncated ? String(m.code) : "")
        (lower.accepted ? lower.program : "")==s.direct.program||error("Direct compilation drift")
        mode=StencilBoundaryContract.recognize(String(job.specification));String(mode)==s.mode||error("Contract drift")
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        example=PublicExampleFallback.choose(String(s.direct.program),bank,s.choices.plain_proposal;infrastructure_ok=ok)
        JSON3.write(example)==JSON3.write(s.choices.example)||error("Public-example policy drift")
        chosen=s.choices.boundary
        expected=if !ok
            (;selected="none",program="",reason="infrastructure_failure")
        elseif mode==:unknown
            example
        elseif PublicExampleFallback.matches(String(s.direct.program),bank)&&StencilBoundaryContract.accepts(String(s.direct.program),mode)
            (;selected="neural_baseline",program=String(s.direct.program),reason="public_examples_and_boundary_match")
        elseif s.choices.boundary_proposal.found&&PublicExampleFallback.matches(String(s.choices.boundary_proposal.program),bank)&&StencilBoundaryContract.accepts(String(s.choices.boundary_proposal.program),mode)
            (;selected="symbolic_public_boundary",program=String(s.choices.boundary_proposal.program),reason="public_example_or_boundary_failure")
        else
            (;selected="none",program="",reason="no_matching_proposal")
        end
        JSON3.write(expected)==JSON3.write(chosen)||error("Boundary policy drift")
        outcomes=Dict{String,Any}();cases=private_boundary_transfer_cases(formula)
        for (arm,program) in (("direct",String(s.direct.program)),("example",String(example.program)),("boundary",String(chosen.program)))
            outcomes[arm]=isempty(program) ? (;admitted=false,passed=false) : audit_rpn_kernel_device(device,program,cases)
        end
        push!(rows,(;id=job.id,selection_sha256=julia_transfer_digest(path),example_selection=example.selected,boundary_selection=chosen.selected,outcomes))
    end
    passed=Dict(arm=>count(r->r.outcomes[arm].passed,rows) for arm in ("direct","example","boundary"))
    result=(;complete=true,scope=p.scope,policy=p.policy,conditions_sha256=julia_transfer_digest(state.path),total=length(rows),passed,rows,
        limitation="Small authored related-family prospective transfer, one seed, extra symbolic compute; not frontier-level or external holdout evidence.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
