using JSON3,SHA
lint_revision_digest(path)=bytes2hex(sha256(read(path)))
function freeze_lint_revision(;directory=joinpath(@__DIR__,"lint_revision_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve revision registration")
    preparedpath=joinpath(@__DIR__,"lint_revision_preparation.json");p=JSON3.read(read(preparedpath,String))
    p.complete&&!p.ready_for_generation&&length(p.jobs)==16||error("Incomplete public-only preparation")
    lint_revision_digest(joinpath(@__DIR__,"lint_revision_protocol.jl"))==p.protocol_sha256||error("Protocol drift")
    parentdir=joinpath(@__DIR__,"alias_transfer3_results");parentpath=joinpath(parentdir,"conditions.json")
    lint_revision_digest(parentpath)==p.parent_conditions_sha256||error("Parent conditions drift")
    lint_revision_digest(joinpath(parentdir,"summary.json"))==p.observed_parent_summary_sha256||error("Parent observed summary drift")
    parent=JSON3.read(read(parentpath,String))
    for job in p.jobs
        original=joinpath(parentdir,String(job.original_id)*".json")
        lint_revision_digest(original)==job.original_generation_sha256||error("Original code drift")
    end
    corpuspath=joinpath(@__DIR__,"julia_public_binding_docs_v2.json")
    lint_revision_digest(corpuspath)==p.expanded_corpus_sha256||error("Expanded corpus drift")
    lint_revision_digest(joinpath(@__DIR__,"julia_docs_v2_validation.json"))==p.expanded_validation_sha256||error("Expanded validation drift")
    preparation_sources=Dict(f=>lint_revision_digest(joinpath(@__DIR__,f)) for f in
        ("lint_revision_study.jl","lint_revision_protocol.jl","lint_document_context.jl","qualified_binding_diagnostics_v2.jl",
         "qualified_binding_diagnostics.jl","alias_document_context.jl","alias_document_retrieval.jl","public_document_retrieval.jl"))
    packet=(;scope=p.scope,policy=p.policy,ready_for_generation=true,jobs=p.jobs,decoding=p.decoding,
        portable_manifest_sha256=parent.portable_manifest_sha256,source_sha256=parent.source_sha256,
        preparation_sha256=lint_revision_digest(preparedpath),preparation_sources,
        parent_conditions_sha256=p.parent_conditions_sha256,observed_parent_summary_sha256=p.observed_parent_summary_sha256,
        expanded_corpus_sha256=p.expanded_corpus_sha256,expanded_validation_sha256=p.expanded_validation_sha256,
        snapshot_manifest_sha256=parent.snapshot_manifest_sha256,reference_validation_sha256=parent.reference_validation_sha256,
        project_sha256=parent.project_sha256,manifest_sha256=parent.manifest_sha256,
        private_scorer_sha256=parent.private_scorer_sha256,secure_runtime_validation_sha256=parent.secure_runtime_validation_sha256,
        secure_runtime_sources=parent.secure_runtime_sources,
        condition_legend=(;direct="self_review",public_docs="static_inventory_and_expanded_public_docs"),
        backend_reuse="Unchanged validated inference and test-blind scorer; internal arm/file IDs reused ONLY in this separate directory. Original results are immutable; no baseline is replaced.")
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end
