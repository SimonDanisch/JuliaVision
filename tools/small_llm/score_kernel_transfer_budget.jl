isdefined(@__MODULE__,:transfer_budget_state) || include("kernel_transfer_budget.jl")

function score_kernel_transfer_budget()
    s=transfer_budget_state();p=s.packet
    task=only(filter(t->t.id==p.task,s.state.tasks));rows=Any[]
    for arm in p.arms
        output=joinpath(dirname(s.path),String(p.task)*"_"*String(arm)*".json")
        if !isfile(output)
            push!(rows,(;task=p.task,arm,status="pending"));continue
        end
        candidate=JSON3.read(read(output,String))
        candidate.task==p.task && candidate.arm==arm && candidate.conditions_sha256==vulkan_benchmark_hash(s.path) &&
            candidate.original_sha256==p.original_sha256[Symbol(arm)] || error("budget candidate provenance drift")
        code=String(candidate.code);scorepath=replace(output,".json"=>"_score.json")
        if isfile(scorepath)
            result=JSON3.read(read(scorepath,String))
            result.code_sha256==bytes2hex(sha256(code)) || error("budget score drift")
        else
            result=JuliaTaskEval.evaluate(task,code;output=scorepath)
        end
        push!(rows,(;task=p.task,arm,status=result.passed ? "passed" : "failed",
            budget_tokens=candidate.budget_tokens,generated_tokens=candidate.generated_tokens,
            truncated=candidate.truncated,finished_thinking=candidate.finished_thinking,
            exact_original_text_prefix=candidate.exact_original_text_prefix,
            original_2048_empty_truncation=true,generation_sha256=vulkan_benchmark_hash(output)))
    end
    report=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status!="pending",rows),
        conditions_sha256=vulkan_benchmark_hash(s.path),scoring_source_sha256=vulkan_benchmark_hash(@__FILE__))
    open(io->JSON3.pretty(io,report),joinpath(dirname(s.path),"summary.json"),"w");report
end
