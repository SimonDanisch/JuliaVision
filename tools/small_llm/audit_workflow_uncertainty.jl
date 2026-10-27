include("paired_workflow_uncertainty.jl")
using JSON3,SHA

function audit_workflow_uncertainty(;include_plots=false)
    comparisons=[("pubmedqa_reading_results","question_only","corresponding_context"),
        ("pubmedqa_retrieval_results","question_only","retrieved_context"),
        ("pubmedqa_retrieval_results","mismatched_context","retrieved_context"),
        ("lint_revision_results","direct","public_docs")]
    include_plots&&push!(comparisons,("plot_interface_results","julia_source","plot_spec"))
    rows=map(comparisons) do (folder,before,after)
        path=joinpath(@__DIR__,folder,"summary.json")
        summary=JSON3.read(read(path,String))
        (;folder,summary_sha256=bytes2hex(sha256(read(path))),paired_summary_uncertainty(summary,before,after)...)
    end
    result=(;scope="POSTHOC descriptive sidecar, not a new study/primary score or a multiplicity-adjusted confirmatory analysis",
        interpretation="Do not pool tasks across studies. Wilson intervals assume binomial sampling and exact paired tests assume exchangeability of discordant outcomes; neither assumption is established for these fixed authored/public development cohorts. Many previous experiments and outcome-informed development prevent confirmatory significance claims. Extra context/tool compute differs between arms.",
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in ("paired_workflow_uncertainty.jl","audit_workflow_uncertainty.jl")),rows)
    output=joinpath(@__DIR__,include_plots ? "workflow_uncertainty_with_plots.json" : "workflow_uncertainty.json")
    open(io->JSON3.pretty(io,result),output,"w");result
end
