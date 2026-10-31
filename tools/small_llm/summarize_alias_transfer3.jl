using JSON3,SHA
function alias3_exception_kind(stderr::AbstractString)
    matched=match(r"(?m)^ERROR: (?:LoadError: )?([A-Za-z][A-Za-z0-9]*Error):",stderr)
    matched===nothing ? "test_failure_without_classified_julia_exception" : matched.captures[1]
end

function summarize_alias_transfer3(;directory=joinpath(@__DIR__,"alias_transfer3_results"))
    conditions=joinpath(directory,"conditions.json");p=JSON3.read(read(conditions,String))
    path=joinpath(directory,"summary.json");s=JSON3.read(read(path,String));s.complete||error("Complete ALL16 before final paired interpretation")
    s.conditions_sha256==bytes2hex(sha256(read(conditions)))||error("Summary provenance mismatch")
    rows=Any[];taxonomies=Dict(arm=>Dict{String,Int}() for arm in ("direct","public_docs"))
    for job in p.jobs
        primary=only(filter(r->r.id==job.id,s.rows));candidate=JSON3.read(read(joinpath(directory,String(job.id)*".json"),String))
        kind=if primary.status=="passed"
            "passed"
        elseif primary.status in ("inference_error","paired_context_capacity_failure")
            String(primary.status)
        else
            score=JSON3.read(read(joinpath(directory,String(job.id)*"_score.json"),String))
            alias3_exception_kind(String(score.stderr_tail))
        end
        counts=taxonomies[String(job.arm)];counts[kind]=get(counts,kind,0)+1
        push!(rows,(;id=job.id,arm=job.arm,status=primary.status,exception_category=kind,
            truncated=hasproperty(candidate,:truncated) ? candidate.truncated : nothing))
    end
    tasks=unique(String(j.task) for j in p.jobs)
    passed=Dict(arm=>count(r->r.arm==arm&&r.status=="passed",rows) for arm in ("direct","public_docs"))
    pair_rows=[(;task,direct=only(filter(r->r.id=="alias3_"*last(split(task,'/'))*"_direct",rows)).status,
        public_docs=only(filter(r->r.id=="alias3_"*last(split(task,'/'))*"_public_docs",rows)).status) for task in tasks]
    result=(;complete=true,scope=p.scope,conditions_sha256=s.conditions_sha256,summary_sha256=bytes2hex(sha256(read(path))),
        passed,total_per_arm=8,exception_categories=taxonomies,truncated_outputs=count(r->r.truncated===true,rows),pairs=pair_rows,
        interpretation="Descriptive SAME-COHORT outcomes only. Extra context did not establish frontier capability. Exception labels are trace categories, not a proof of root cause or private-test feedback for future prompts.",
        summarizer_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,result),joinpath(directory,"paired_comparison.json"),"w");result
end
