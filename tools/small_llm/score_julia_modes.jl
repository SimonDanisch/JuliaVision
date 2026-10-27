isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA

function score_julia_modes()
    conditions=Any[]
    for (directory,scope) in (
        ("julia_thinking_results","greedy thinking, 2048-token cap; more compute than original baseline, not a tool gain"),
        ("julia_sampled_results","one predeclared seed, thinking, 2048-token cap; no seed selection"),
        ("julia_scene_results","non-thinking scene schema plus trusted compiler, 512-token cap; constrained workflow, not intrinsic coding"),
        ("julia_scene_v2_results","clarified schema after v1 format failures, 512-token cap; development result, not held-out"),
        ("julia_api_v2_results","one-shot scoped/diverse public API-doc retrieval, 512-token cap; revised development policy"))
        rows=Any[];dir=joinpath(@__DIR__,directory)
        tasks=filter(t->t.category!="kernel" && (!(startswith(directory,"julia_scene")||directory=="julia_api_v2_results") || t.category=="makie"),JuliaTaskEval.JULIA_TASKS)
        # Only the grouping task was predeclared for the sampled control; do
        # not manufacture four pending experiments or expand its denominator.
        directory=="julia_sampled_results" && (tasks=filter(t->t.id=="stable_groups",tasks))
        for task in tasks
            source=joinpath(dir,task.id*"_baseline.json");score=joinpath(dir,task.id*"_baseline_score.json")
            if !isfile(source)
                push!(rows,(;task=task.id,status="pending"));continue
            end
            candidate=JSON3.read(read(source,String))
            candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift: "*task.id)
            code=String(candidate.code)
            if isfile(score)
                result=JSON3.read(read(score,String))
                hasproperty(result,:code_sha256) && result.code_sha256!=bytes2hex(sha256(code)) && error("stale score")
            else
                if hasproperty(candidate,:validation_error) && candidate.validation_error!==nothing
                    result=(;id=task.id,passed=false,diagnostic=String(candidate.validation_error),code_sha256=bytes2hex(sha256(code)))
                else
                    result=JuliaTaskEval.evaluate(task,code)
                end
                open(io->JSON3.pretty(io,result),score,"w")
            end
            push!(rows,(;task=task.id,status=result.passed ? "passed" : "failed",
                generated_tokens=candidate.generated_tokens,budget_tokens=candidate.budget_tokens,
                truncated=candidate.truncated,finished_thinking=candidate.finished_thinking,
                generation_seconds=candidate.elapsed_seconds))
            println(directory," ",task.id," ",result.passed ? "PASS" : "FAIL");flush(stdout)
        end
        evaluated=count(r->r.status!="pending",rows);passed=count(r->r.status=="passed",rows)
        push!(conditions,(;directory,scope,passed,evaluated,planned=length(tasks),pending=length(tasks)-evaluated,rows))
    end
    report=(;scope="authored diagnostic, not held-out/frontier evidence; pending cases are not scored failures",
        comparison="original non-thinking direct code and all three revision arms solved 0/5 qualified contracts; see julia_tasks_results/summary.json",
        limitations=["schema and prompt are extra workflow resources even when output cap matches",
            "Makie contracts test construction/data/style attributes, not rendered aesthetics",
            "sampled and greedy modes have different output lengths and compute costs"],conditions)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"julia_modes_summary.json"),"w")
    report
end
