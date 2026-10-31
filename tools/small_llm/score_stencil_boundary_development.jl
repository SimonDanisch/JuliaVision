# ROOT ONLY; every public-only selection must already exist before grading.
include("stencil_boundary_development.jl")
include("qualify_julia_stencil_transfer.jl")
function score_stencil_boundary_development(device;directory=joinpath(@__DIR__,"stencil_boundary_development_results"))
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only")
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    all(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)||error("ALL selections BEFORE grading")
    for (f,h) in pairs(p.sources)
        boundary_digest(joinpath(@__DIR__,String(f)))==h||error("Frozen source drift")
    end
    parent=joinpath(@__DIR__,"julia_stencil_transfer_results")
    boundary_digest(joinpath(parent,"summary.json"))==p.parent_summary_sha256||error("Parent drift")
    proof=JSON3.read(read(joinpath(@__DIR__,"julia_stencil_transfer_reference_validation.json"),String))
    for (f,h) in pairs(proof.sources)
        boundary_digest(joinpath(@__DIR__,String(f)))==h||error("Private reference drift")
    end
    rows=Any[]
    for (job,(_,formula)) in zip(p.jobs,private_julia_stencil_transfer_fixtures())
        path=joinpath(directory,String(job.id)*".json");s=JSON3.read(read(path,String))
        s.conditions_sha256==boundary_digest(conditionpath)&&s.parent_selection_sha256==job.parent_selection_sha256||error("Selection receipt drift")
        mode=StencilBoundaryContract.recognize(String(job.specification));String(mode)==s.mode||error("Mode drift")
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        program=String(s.selected.program)
        isempty(program)||(PublicExampleFallback.matches(program,bank)&&StencilBoundaryContract.accepts(program,mode))||error("Public constraints disagree")
        result=isempty(program) ? (;admitted=false,passed=false) : audit_rpn_kernel_device(device,program,private_julia_stencil_transfer_cases(formula))
        push!(rows,(;id=job.id,selected=s.selected.selected,status=result.passed ? "passed" : "failed",selection_sha256=boundary_digest(path),result))
    end
    result=(;complete=true,scope=p.scope,policy=p.policy,conditions_sha256=boundary_digest(conditionpath),passed=count(r->r.status=="passed",rows),total=length(rows),rows,
        limitation="Posthoc workflow development after observing boundary errors; no independent transfer evidence. Original prospective6/8 direct/fallback results unchanged.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
