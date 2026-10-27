using JSON3,SHA
isdefined(@__MODULE__,:plot_validator_revision_messages)||include("plot_validator_policy.jl")
function freeze_plot_validator_study(;preparation=joinpath(@__DIR__,"plot_validator_preparation.json"),directory=joinpath(@__DIR__,"plot_validator_results"),parent=joinpath(@__DIR__,"plot_interface_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    digest(path)=bytes2hex(sha256(read(path)))
    prepared=JSON3.read(read(preparation,String));prepared.complete&&length(prepared.decisions)==8||error("Keep ALL8 originals")
    digest(joinpath(@__DIR__,"plot_validator_policy.jl"))==prepared.source_sha256||error("Preparation drift")
    parentpath=joinpath(parent,"conditions.json");parent_packet=JSON3.read(read(parentpath,String))
    digest(parentpath)==prepared.parent_conditions_sha256||error("Parent drift")
    length(unique(d.task for d in prepared.decisions))==8||error("Unique original tasks required")
    length(unique(j.id for j in prepared.jobs))==length(prepared.jobs)||error("Duplicate revision IDs")
    for row in prepared.decisions
        baseline_path=joinpath(parent,row.baseline_id*".json")
        digest(baseline_path)==row.baseline_generation_sha256||error("Original changed")
        original=only(filter(j->j.id==row.baseline_id&&j.arm=="plot_spec"&&j.task==row.task,parent_packet.jobs))
        receipt=JSON3.read(read(baseline_path,String));trigger=plot_validator_trigger(receipt)
        trigger.triggered==row.triggered&&trigger.reason==row.public_reason||error("Public trigger mismatch")
        selected=filter(j->j.task==row.task,prepared.jobs)
        if row.triggered
            length(selected)==2&&Set(String(j.arm) for j in selected)==Set(("self_review","validator_feedback"))||error("Both conditions required")
            for job in selected
                job.id==row.task*"_"*job.arm&&basename(job.id)==job.id||error("Unsafe/mismatched revision ID")
                expected=plot_validator_revision_messages(original.messages,receipt,String(job.arm))
                JSON3.write(expected)==JSON3.write(job.messages)||error("Revision contains nonpublic/drifted context")
                0<job.input_tokens&&job.input_tokens+1024+32<=8192||error("Input capacity")
            end
        else
            isempty(selected)||error("Admitted originals must NOT be revised")
        end
    end
    length(prepared.jobs)==2count(d->d.triggered,prepared.decisions)||error("Both paired revisions for EVERY public trigger")
    result=(;ready_for_generation=true,scope="POSTHOC SAME8 Makie JSON requests; conditional single public-validator revision vs conditional single self-review. Not independent transfer, unrestricted Julia or model-weight/frontier evidence.",
        policy=prepared.policy,jobs=prepared.jobs,decisions=prepared.decisions,decoding=parent_packet.decoding,
        source_sha256=parent_packet.source_sha256,portable_manifest_sha256=parent_packet.portable_manifest_sha256,
        project_sha256=parent_packet.project_sha256,manifest_sha256=parent_packet.manifest_sha256,
        parent_runtime_conditions_sha256=digest(parentpath),preparation_sha256=digest(preparation),
        observed_parent_summary_sha256=digest(joinpath(parent,"summary.json")),
        condition_legend=(;self_review="conditional_one_self_review",validator_feedback="conditional_one_public_schema_feedback_revision"),
        registration_source_sha256=digest(@__FILE__),private_scorer_sha256=digest(joinpath(@__DIR__,"score_plot_validator_study.jl")),
        private_sources=Dict(f=>digest(joinpath(@__DIR__,f)) for f in ("plot_validator_policy.jl","plot_interface_references.jl","plot_interface_admission.jl","safe_plot_spec.jl","safe_plot_julia_source.jl","score_pubmedqa_reading.jl","pubmedqa_metrics.jl","pubmedqa_reading_protocol.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,result),joinpath(directory,"conditions.json"),"w");result
end
