isdefined(@__MODULE__,:source_channel_jobs) || include("source_channel_control.jl")
isdefined(@__MODULE__,:api_runtime_feedback_jobs) || include("api_runtime_feedback.jl")

function api_channel_combined_jobs()
    # Source-context completion first; subsequent jobs retain their distinct
    # unchanged conditions paths, public-only diagnostics and result locations.
    jobs=vcat(source_channel_jobs(),api_runtime_feedback_jobs())
    length(unique(job.output for job in jobs))==length(jobs) || error("duplicate frozen output")
    all(job->hasproperty(job,:conditions_path)&&isfile(job.conditions_path),jobs) || error("missing per-study conditions")
    jobs
end

function run_api_channel_combined(;seconds=210)
    run_humaneval_repairs(;seconds,jobs=api_channel_combined_jobs(),extra_source_files=
        ("run_api_channel_combined.jl","source_channel_control.jl","api_runtime_feedback.jl","api_public_smoke.jl"))
end
