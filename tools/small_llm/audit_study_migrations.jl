using JSON3,SHA

"""Provenance closure only. Never copy new results into old frozen folders.
Old pending rows can be intentional after migration, not permission to generate
them again. A completed migration is NOT proof of same-engine equivalence.
"""
function audit_study_migration(old_directory,new_directory,parent_field;source_root=@__DIR__)
    digest(path)=bytes2hex(sha256(read(path)))
    readjson(path)=JSON3.read(read(path,String))
    oldpath=joinpath(old_directory,"conditions.json");newpath=joinpath(new_directory,"conditions.json")
    old=readjson(oldpath);new=readjson(newpath)
    getproperty(new,parent_field)==digest(oldpath)||error("Migration parent drift")
    oldids=if hasproperty(old,:jobs)
        String[j.id for j in old.jobs]
    elseif hasproperty(old,:order)
        String[j.id for j in old.order]
    else
        [String(old.task)*"_"*String(arm) for arm in old.arms]
    end
    ids=String[j.id for j in new.jobs]
    length(ids)==length(unique(ids))&&Set(ids)==Set(oldids)||error("Migration must cover entire original pending cohort")
    summarypath=joinpath(new_directory,"summary.json");summary=readjson(summarypath)
    summary.complete&&summary.conditions_sha256==digest(newpath)||error("Migration not completely scored")
    all(r->r.status∉("pending","interrupted_attempt"),summary.rows)||error("Unfinished migrated scores")
    Set(String(r.id) for r in summary.rows)==Set(ids)||error("Scored ID coverage")
    rows=map(new.jobs) do job
        id=String(job.id)
        basename(id)==id||error("Unsafe ID")
        all(!ispath(joinpath(old_directory,id*suffix)) for suffix in (".json","_attempt.json"))||error("Original slot was already attempted; not a clean never-attempted migration")
        output=joinpath(new_directory,id*".json");attempt=joinpath(new_directory,id*"_attempt.json")
        receipt=readjson(output)
        receipt.conditions_sha256==digest(newpath)&&receipt.attempt_sha256==digest(attempt)||error("Migration receipt drift")
        receipt.messages==job.messages||error("Actual migrated prompt drift")
        source=if hasproperty(old,:jobs)
            original=only(filter(j->j.id==id,old.jobs));original.messages
        elseif hasproperty(old,:messages)
            getproperty(old.messages,Symbol(job.arm))
        else
            path=joinpath(source_root,"kernel_transfer_results",id*".json")
            digest(path)==job.original_sha256||error("Original short baseline drift")
            readjson(path).messages
        end
        source==job.messages||error("Original prompt not preserved")
        (;id,generation_sha256=digest(output),attempt_sha256=digest(attempt),
            old_slot="unattempted_preserved",new_slot="scored_on_different_qualified_runtime")
    end
    (;complete=true,old_conditions_sha256=digest(oldpath),new_conditions_sha256=digest(newpath),
        new_summary_sha256=digest(summarypath),rows,
        scope="Migration bookkeeping/provenance, not new generations or a model-quality improvement. Do not pool old/new runtime scores or regenerate historical slots.")
end

function audit_study_migrations(;destination=joinpath(@__DIR__,"study_migration_audit.json"))
    isfile(destination)&&error("Preserve audit")
    pairs=(("safe_budget_transfer_extension_results","laptop_pending_results",:extension_sha256),
        ("shared_source_kernel_results","laptop_shared_source_results",:parent_conditions_sha256),
        ("kernel_transfer_budget_results","laptop_long_budget_v2_results",:original_long_conditions_sha256))
    rows=[(;original,migrated,audit=audit_study_migration(joinpath(@__DIR__,original),joinpath(@__DIR__,migrated),field))
        for (original,migrated,field) in pairs]
    result=(;complete=true,scope="Historical generation-slot closure only. Separate posthoc compiler analyses can still remain unfinished; consult their own summaries.",
        rows,source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
