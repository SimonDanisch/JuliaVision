include("bayes_thinking_admission.jl")
using JSON3,SHA

function audit_bayes_json_extent(;output=joinpath(@__DIR__,"bayes_json_extent_audit.json"))
    rows=Any[]
    for (folder,arm_override) in (("bayes_extraction_results",nothing),("bayes_cpu_reference_results","direct_probability"))
        directory=joinpath(@__DIR__,folder)
        summary=JSON3.read(read(joinpath(directory,"summary.json"),String))
        summary.complete||error("Only audit completed studies")
        for row in summary.rows
            path=joinpath(directory,String(row.id)*".json");m=JSON3.read(read(path,String))
            arm=arm_override===nothing ? String(row.arm) : arm_override
            raw=hasproperty(m,:answer) ? String(m.answer) : ""
            loose=admit_bayes_extraction(raw,arm)
            whole=try bayes_whole_json_object(bayes_extraction_payload(raw)) catch;false end
            push!(rows,(;study=folder,id=row.id,legacy_admitted=loose.accepted,
                whole_object=whole,strict_admitted=loose.accepted&&whole,
                admission_changed=loose.accepted&&!whole,original_status=row.status,
                generation_sha256=bytes2hex(sha256(read(path)))))
        end
    end
    result=(;complete=true,scope="Posthoc parser-extent audit only; original registrations/results unchanged. JSON3 can ignore trailing bytes. New thinking grader independently requires entire object before parsing.",
        rows,changed_admissions=count(r->r.admission_changed,rows),
        prior_passes_invalidated=count(r->r.admission_changed&&r.original_status=="passed",rows),
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("audit_bayes_json_extent.jl","bayes_thinking_admission.jl","bayes_extraction_admission.jl")))
    open(io->JSON3.pretty(io,result),output,"w");result
end
