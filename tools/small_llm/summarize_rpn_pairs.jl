using JSON3,SHA

function summarize_rpn_pairs(jobs,rows)
    length(unique(j.id for j in jobs))==length(jobs)||error("duplicate jobs")
    [r.id for r in rows]==[j.id for j in jobs]||error("score cohort/order drift")
    all(j->j.arm in ("enabled","disabled")&&j.enabled==(j.arm=="enabled"),jobs)||error("arm/control drift")
    all(r->r.status in ("passed","failed","inference_error","pending","interrupted_attempt"),rows)||error("unknown score status")
    all(r->r.status!="passed"||get(r,:syntax_accepted,false),rows)||error("mathematical pass without syntax admission")
    by_arm=Dict{String,Any}()
    terminal(r)=r.status in ("passed","failed","inference_error")
    for arm in ("enabled","disabled")
        selected=[rows[i] for i in eachindex(jobs) if jobs[i].arm==arm]
        by_arm[arm]=(;conditions=length(selected),completed=count(terminal,selected),
            syntax_accepted=count(r->get(r,:syntax_accepted,false),selected),
            mathematical_passes=count(r->r.status=="passed",selected),inference_errors=count(r->r.status=="inference_error",selected))
    end
    pairs=Any[]
    for task in unique(j.task for j in jobs)
        indices=findall(j->j.task==task,jobs)
        length(indices)==2&&Set(jobs[i].arm for i in indices)==Set(("enabled","disabled"))||error("incomplete/duplicate task pair")
        enabled=rows[only(filter(i->jobs[i].arm=="enabled",indices))]
        disabled=rows[only(filter(i->jobs[i].arm=="disabled",indices))]
        outcome=if !terminal(enabled)||!terminal(disabled);"pending"
            elseif enabled.status=="passed"&&disabled.status=="passed";"both_pass"
            elseif enabled.status=="passed";"enabled_only"
            elseif disabled.status=="passed";"disabled_only"
            else;"neither_pass"
        end
        push!(pairs,(;task,outcome,enabled=enabled.status,disabled=disabled.status))
    end
    (;complete=all(terminal,rows),by_arm,pairs,
        paired_counts=Dict(s=>count(r->r.outcome==s,pairs) for s in ("both_pass","enabled_only","disabled_only","neither_pass","pending")))
end

function write_rpn_pair_summary(directory)
    digest(path)=bytes2hex(open(sha256,path))
    conditions=joinpath(directory,"conditions.json");primary=joinpath(directory,"summary.json")
    p=JSON3.read(read(conditions,String));s=JSON3.read(read(primary,String))
    s.conditions_sha256==digest(conditions)||error("primary conditions drift")
    counts=summarize_rpn_pairs(p.jobs,s.rows)
    packet=(;scope=p.scope,policy=p.policy,counts...,
        conditions_sha256=digest(conditions),primary_summary_sha256=digest(primary),analyst_sha256=digest(@__FILE__),
        limits="Small authored related-family cohort, one seed/attempt each. Syntax admission is not arithmetic success; no frontier/equivalence/significance or independent transfer claim. Original prompt has a separate literal32-operation audit; corrected-prompt follow-up is posthoc development.")
    open(io->JSON3.pretty(io,packet),joinpath(directory,"paired_comparison.json"),"w");packet
end
