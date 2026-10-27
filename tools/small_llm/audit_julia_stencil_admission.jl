# ROOT-only supplementary source admission audit. Never changes primary scores.
include("julia_stencil_admission_v3.jl")
using JSON3,SHA
function audit_julia_stencil_admission(;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    rows=Any[]
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");isfile(path)||continue
        m=JSON3.read(read(path,String))
        if hasproperty(m,:inference_error)
            push!(rows,(;id=job.id,status="infrastructure_failure"));continue
        end
        source=m.truncated ? "" : String(m.code)
        v1=JuliaStencilInterface.lower(source);v3=JuliaStencilAdmissionV3.lower(source)
        push!(rows,(;id=job.id,status="audited",v1_accepted=v1.accepted,v3_accepted=v3.accepted,
            reason=v3.reason,canonical_identity=!v3.accepted||(v1.program==v3.program&&v1.code==v3.code),
            generation_sha256=bytes2hex(sha256(read(path)))))
    end
    result=(;scope="Supplementary uniform admission hardening audit, NOT a replacement for frozen V1 primary scores; no neural regeneration/arithmetic repair",complete=length(rows)==length(p.jobs),
        conditions_sha256=bytes2hex(sha256(read(conditionpath))),rows,
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("julia_stencil_interface.jl","julia_stencil_admission_v2.jl","julia_stencil_admission_v3.jl","audit_julia_stencil_admission.jl")))
    open(io->JSON3.pretty(io,result),joinpath(directory,"supplementary_admission_audit.json"),"w");result
end
