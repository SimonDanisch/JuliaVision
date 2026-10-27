# ROOT ONLY, after ALL neural outputs and public-only choices are immutable.
include("julia_stencil_transfer_study.jl")
include("qualify_julia_stencil_transfer.jl")

function score_julia_stencil_transfer(device;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only grading")
    state=julia_stencil_transfer_state(directory);p=state.packet
    julia_transfer_digest(@__FILE__)==p.private_scorer_sha256||error("Scorer drift")
    julia_transfer_digest(joinpath(@__DIR__,"julia_stencil_transfer_reference_validation.json"))==p.private_reference_sha256||error("Reference drift")
    proof=JSON3.read(read(joinpath(@__DIR__,"julia_stencil_transfer_reference_validation.json"),String))
    for (f,h) in pairs(proof.sources)
        julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Private source drift")
    end
    all(isfile(joinpath(directory,String(j.id)*"_selection.json")) for j in p.jobs)||error("Commit ALL8 selections BEFORE grading")
    rows=Any[]
    for (job,(_,formula)) in zip(p.jobs,private_julia_stencil_transfer_fixtures())
        path=joinpath(directory,String(job.id)*"_selection.json");s=JSON3.read(read(path,String))
        s.conditions_sha256==julia_transfer_digest(state.path)||error("Selection drift")
        generation=joinpath(directory,String(job.id)*".json")
        s.generation_sha256==julia_transfer_digest(generation)||error("Generation drift")
        m=JSON3.read(read(generation,String));ok=!hasproperty(m,:inference_error)
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=julia_transfer_digest(state.path),
            attempt_sha256=julia_transfer_digest(joinpath(directory,String(job.id)*"_attempt.json")))
        lower=JuliaStencilInterface.lower(ok&&!m.truncated ? String(m.code) : "")
        (lower.accepted ? lower.program : "")==s.direct.program||error("Direct translation drift")
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        choice=PublicExampleFallback.choose(String(s.direct.program),bank,s.symbolic;infrastructure_ok=ok)
        JSON3.write(choice)==JSON3.write(s.selected)||error("Selection policy drift")
        cases=private_julia_stencil_transfer_cases(formula)
        outcomes=Dict{String,Any}()
        for (arm,program) in (("direct",String(s.direct.program)),("hybrid",String(s.selected.program)))
            outcomes[arm]=isempty(program) ? (;admitted=false,passed=false) : audit_rpn_kernel_device(device,program,cases)
        end
        push!(rows,(;id=job.id,selected=s.selected.selected,selection_sha256=julia_transfer_digest(path),outcomes))
    end
    direct=count(r->r.outcomes["direct"].passed,rows);hybrid=count(r->r.outcomes["hybrid"].passed,rows)
    result=(;complete=true,scope=p.scope,conditions_sha256=julia_transfer_digest(state.path),direct_passed=direct,hybrid_passed=hybrid,total=length(rows),rows,
        interpretation="Authored related-family transfer of a frozen public-example fallback. Additional symbolic compute, not improved model weights or frontier intelligence.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
