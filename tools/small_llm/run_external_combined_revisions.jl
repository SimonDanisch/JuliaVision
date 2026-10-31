# One model load/window, but separate unchanged conditions and result locations.
include("humaneval_public_feedback.jl")

function combined_external_revision_jobs()
    static=external_repair_jobs()
    public_conditions=joinpath(@__DIR__,"humaneval_public_feedback_results","conditions.json")
    public_jobs=[merge(job,(;conditions_path=public_conditions)) for job in external_public_feedback_jobs()]
    vcat(static,public_jobs)
end

function run_external_combined_revisions(;seconds=240)
    run_humaneval_repairs(;seconds,jobs=combined_external_revision_jobs(),
        extra_source_files=("run_external_combined_revisions.jl","humaneval_public_feedback.jl",
            "public_julia_smoke.jl","public_julia_doctest.jl"))
end
