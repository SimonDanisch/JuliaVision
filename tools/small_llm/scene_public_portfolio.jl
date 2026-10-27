using JSON3, SHA
const SCENE_PUBLIC_PROBES=Dict(
    "scene_ratio"=>"r=makeplot([0.,1.],[4.,9.],[2.,3.]); @assert r isa Tuple && length(r)==2; @assert r[1] isa Makie.Figure; @assert r[2] isa Makie.Axis",
    "scene_shared_y"=>"r=makeplot([0.,1.],[-1.,2.]); @assert r isa Tuple && length(r)==3; @assert r[1] isa Makie.Figure; @assert all(a->a isa Makie.Axis,r[2:3])")

function evaluate_scene_public_portfolio()
    rows=Any[];outdir=joinpath(@__DIR__,"scene_public_portfolio_results");mkpath(outdir)
    for task in SCENE_GENERALIZATION_TASKS
        attempts=Any[];selected=nothing;code=""
        for arm in ("direct","adapter")
            source=joinpath(@__DIR__,"scene_generalization_"*arm,task.id*"_baseline.json")
            isfile(source) || error("missing preregistered candidate")
            candidate=JSON3.read(read(source,String))
            code=String(candidate.code)
            probe=if hasproperty(candidate,:validation_error) && candidate.validation_error!==nothing
                (;passed=false,diagnostic=String(candidate.validation_error),elapsed_seconds=0.0)
            else
                public_task=(;id=task.id,category="makie",tests=SCENE_PUBLIC_PROBES[task.id])
                JuliaTaskEval.evaluate(public_task,code)
            end
            push!(attempts,(;arm,probe,generation_seconds=candidate.elapsed_seconds,input_tokens=candidate.input_tokens,
                output_budget=candidate.budget_tokens,code_sha256=bytes2hex(sha256(code))))
            if probe.passed
                selected=arm
                break
            end
        end
        # Only AFTER selection apply the private scoring contract. The selector
        # never reads earlier hidden score files or their exception diagnostics.
        score=selected===nothing ? (;passed=false,diagnostic="no public-smoke-passing candidate") : JuliaTaskEval.evaluate(task,code)
        row=(;task=task.id,selected,passed=score.passed,attempts,private_score=score)
        open(io->JSON3.pretty(io,row),joinpath(outdir,task.id*".json"),"w");push!(rows,row)
        println(task.id," public selector=",selected," private pass=",score.passed);flush(stdout)
    end
    report=(;scope="posthoc DEVELOPMENT policy after inspecting candidate outputs; not an unbiased held-out portfolio result",
        policy="fixed direct-first, scene fallback only after ordinary public type/arity/runtime smoke failure; no private-score or semantic-oracle selection",
        budget="up to two 512-token attempts plus sandbox smoke costs; not equal compute to a single direct attempt",
        generation_cost_policy="paired run generated all candidates; attempts field reports hypothetical lazy-policy costs, not a fresh lazy generation run",
        passed=count(r->r.passed,rows),total=length(rows),rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"scene_public_portfolio_summary.json"),"w")
    report
end
