include("run_julia_stencil_transfer.jl")
include("alias_transfer3_study.jl")
function run_alias_transfer3(;directory=joinpath(@__DIR__,"alias_transfer3_results"))
    state=julia_stencil_transfer_state(directory);p=state.packet
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    job=first(pending)
    if !job.eligible
        output=joinpath(directory,String(job.id)*".json");attempt=joinpath(directory,String(job.id)*"_attempt.json")
        (ispath(output)||ispath(attempt))&&error("Preserve skipped receipt")
        open(io->JSON3.pretty(io,(;id=job.id,conditions_sha256=alias3_digest(state.path),skipped_without_device=true)),attempt,"w")
        open(io->JSON3.pretty(io,(;id=job.id,conditions_sha256=alias3_digest(state.path),attempt_sha256=alias3_digest(attempt),
            not_generated=true,reason="paired_context_capacity_failure",gpu_setup=false)),output,"w")
        return (;not_generated=true,id=job.id)
    end
    run_julia_stencil_transfer(;directory)
end
