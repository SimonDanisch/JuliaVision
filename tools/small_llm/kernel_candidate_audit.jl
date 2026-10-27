isdefined(@__MODULE__,:JuliaKernelLint) || include("julia_kernel_lint.jl")
isdefined(@__MODULE__,:KernelBoundsAdvisory) || include("kernel_bounds_advisory.jl")
using JSON3, SHA

"""Post-hoc source advisories, separate from frozen numerical scores.

No macro expansion or generated-code execution. A clean report is not a proof:
arbitrary aliasing, macro bindings, non-Atomix atomics and other races are unknown.
"""
function audit_kernel_candidates(;directories=("vulkan_kernel_results","kernel_runtime_feedback_results"))
    rows=Any[]
    for directory in directories
        path=joinpath(@__DIR__,directory);isdir(path) || continue
        for file in sort(readdir(path;join=true))
            endswith(file,".json") || continue
            candidate=JSON3.read(read(file,String))
            hasproperty(candidate,:code) && hasproperty(candidate,:task) || continue
            code=String(candidate.code);parse_problem=false;findings=Any[];bounds=Any[];index_calls=0
            try
                append!(findings,JuliaKernelLint.atomic_extrema_assignments(code))
                append!(bounds,KernelBoundsAdvisory.direct_global_stores(code))
                index_calls=KernelBoundsAdvisory.global_index_calls(code)
                tree=Meta.parseall(code)
                function contains_parse_error(x)
                    x isa Expr && (x.head in (:error,:incomplete) || any(contains_parse_error,x.args))
                end
                parse_problem=contains_parse_error(tree)
            catch err
                err isa InterruptException && rethrow()
                parse_problem=true
            end
            push!(rows,(;task=candidate.task,file=relpath(file,@__DIR__),code_sha256=bytes2hex(sha256(code)),
                parse_problem,has_executable_code=!isempty(strip(code)),atomic_advisories=findings,bounds_advisories=bounds,
                global_index_calls=index_calls,
                partitioning_review=!isempty(strip(code))&&index_calls==0,
                partitioning_limitation="No recognized global-index call merits inspection, not automatic failure: helper calls/aliases or manually derived indices may partition work correctly. Numerical agreement alone does not exclude racing serial writes by every invocation."))
        end
    end
    report=(;scope="post-hoc AST advisories; does NOT alter frozen functional scores or select revisions",
        limitations="No warnings does not establish valid RMW semantics, race freedom or general correctness",
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("kernel_candidate_audit.jl","kernel_bounds_advisory.jl","julia_kernel_lint.jl")),rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"kernel_candidate_audit.json"),"w");report
end
