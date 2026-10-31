# ROOT-only posthoc diagnostic, never a replacement primary score/selector.
include("stencil_boundary_max_proof.jl")
using JSON3,SHA
function audit_boundary_transfer_replacements(;directory=joinpath(@__DIR__,"boundary_transfer_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    summary=JSON3.read(read(joinpath(directory,"summary.json"),String));summary.complete||error("Complete primary grading first")
    rows=Any[]
    for job in p.jobs
        path=joinpath(directory,String(job.id)*"_selection.json");s=JSON3.read(read(path,String))
        primary=only(filter(r->r.id==job.id,summary.rows))
        proof=StencilBoundaryMaxProof.certify(String(s.direct.program),Symbol(s.mode))
        push!(rows,(;id=job.id,replaced=s.choices.boundary.selected=="symbolic_public_boundary",
            original_passed=primary.outcomes.direct.passed,example_passed=primary.outcomes.example.passed,
            boundary_passed=primary.outcomes.boundary.passed,proof,
            selection_sha256=bytes2hex(sha256(read(path)))))
    end
    result=(;complete=true,scope="Supplementary POSTHOC explanation of conservative contract rejections; no primary score changes/neural regeneration or new quality cohort",rows,
        primary_summary_sha256=bytes2hex(sha256(read(joinpath(directory,"summary.json")))),
        observed_replacements=count(r->r.replaced,rows),
        replacements_of_originally_passing_programs=count(r->r.replaced&&r.original_passed,rows),
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("stencil_boundary_max_proof.jl","audit_boundary_transfer_replacements.jl")))
    open(io->JSON3.pretty(io,result),joinpath(directory,"replacement_diagnostic.json"),"w");result
end
