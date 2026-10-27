using JSON3, SHA

function public_feedback_pair_audit()
    directory=joinpath(@__DIR__,"humaneval_public_feedback_results")
    summary=JSON3.read(read(joinpath(directory,"summary.json"),String));rows=Any[]
    tasks=unique(String(row.task) for row in summary.rows)
    for task in tasks
        control=only(filter(row->row.task==task&&row.arm=="self_review",summary.rows))
        feedback=only(filter(row->row.task==task&&row.arm=="public_feedback",summary.rows))
        paths=[joinpath(directory,task*"_"*arm*".json") for arm in ("self_review","public_feedback")]
        candidates=all(isfile,paths) ? map(path->JSON3.read(read(path,String)),paths) : nothing
        identity=candidates===nothing ? "retained identical baseline" :
            all(candidate->hasproperty(candidate,:runtime_provenance),candidates) ?
            (candidates[1].runtime_provenance==candidates[2].runtime_provenance ? "recorded same engine" : "recorded different engines") :
            "engine identity not recorded in both artifacts"
        push!(rows,(;task,self_review=control.status,public_feedback=feedback.status,
            improvement=control.status!="passed"&&feedback.status=="passed",
            regression=control.status=="passed"&&feedback.status!="passed",engine_identity=identity))
    end
    report=(;scope="paired provenance audit; descriptive, NOT statistical or frontier-level evidence",rows,
        limitations=["Twelve public basic-test tasks; contamination possible",
            "Development extension motivated by earlier observed failures",
            "No retroactive invention of engine hashes for older outputs",
            "Do not pool this result with the separate static JET policy or authored pilots"],
        summary_sha256=bytes2hex(sha256(read(joinpath(directory,"summary.json")))))
    open(io->JSON3.pretty(io,report),joinpath(directory,"pair_audit.json"),"w");report
end
