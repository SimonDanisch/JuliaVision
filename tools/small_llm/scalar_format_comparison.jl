using JSON3,SHA

function scalar_format_comparison()
    base=joinpath(@__DIR__,"compiler_transfer2_results")
    follow=joinpath(@__DIR__,"scalar_format_followup_results")
    digest(path)=bytes2hex(open(sha256,path))
    conditions=JSON3.read(read(joinpath(base,"conditions.json"),String))
    amended=JSON3.read(read(joinpath(follow,"conditions.json"),String))
    original=JSON3.read(read(joinpath(base,"summary.json"),String))
    revised=JSON3.read(read(joinpath(follow,"summary.json"),String))
    original.complete && revised.complete || error("unfinished study")
    amended.format_parent_conditions_sha256==digest(joinpath(base,"conditions.json")) || error("parent conditions")
    amended.format_parent_summary_sha256==digest(joinpath(base,"summary.json")) || error("parent scores")
    conditions.decoding==amended.decoding && conditions.thinking_budget==amended.thinking_budget || error("budget/sampling mismatch")
    conditions.frozen_compiler_sha256==amended.frozen_compiler_sha256 || error("compiler mismatch")
    tasks=String[j.task for j in conditions.jobs if j.arm=="scalar_program"]
    length(tasks)==length(unique(tasks))==4 || error("cohort drift")
    rows=map(tasks) do task
        a=only(filter(r->r.task==task && r.arm=="scalar_program",original.rows))
        b=only(filter(r->r.task==task,revised.rows))
        direct=only(filter(r->r.task==task && r.arm=="direct_gpu",original.rows))
        (;task,original_scalar=a.status,format_followup=b.status,direct_gpu=direct.status,
            change=a.status==b.status ? "unchanged" : b.status=="passed" ? "improved" : "worsened")
    end
    report=(;complete=true,scope="POSTHOC development comparison on four now-used tasks; not independent held-out transfer, statistical evidence of general intelligence, or a frontier comparison",
        policy="Original outputs/scores retained. Same arithmetic specs/compiler/total and thinking budget/sampling. Prompt wording/position/system instruction changed; no compiler repair or regeneration within a condition.",
        rows,original_scalar_passed=count(r->r.original_scalar=="passed",rows),
        format_followup_passed=count(r->r.format_followup=="passed",rows),direct_gpu_passed=count(r->r.direct_gpu=="passed",rows),
        improved=count(r->r.change=="improved",rows),worsened=count(r->r.change=="worsened",rows),
        receipts=Dict(relpath(path,@__DIR__)=>digest(path) for path in
            (joinpath(base,"conditions.json"),joinpath(base,"summary.json"),joinpath(follow,"conditions.json"),joinpath(follow,"summary.json"))),
        source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"scalar_format_comparison.json"),"w");report
end
