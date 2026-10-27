include("lint_document_context.jl")
using JSON3,SHA

function lint_revision_messages(original_messages,code::AbstractString,condition::Symbol,index,bindings)
    condition in (:self_review,:lint_docs)||throw(ArgumentError("Unknown revision condition"))
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in original_messages]
    length(messages)==2&&first(messages[1])=="system"&&first(messages[2])=="user"||throw(ArgumentError("Original task-only messages required"))
    request=last(messages[2])
    push!(messages,"assistant"=>String(code))
    instruction="Revise the preceding complete standalone Julia program against the original public specification and package contract. Return ONLY the complete corrected program. Check API names, imports, argument order, algorithm and stdin/stdout format. No benchmark tests, expected outputs or grading traces are supplied.\n\n"
    context=condition==:lint_docs ? LintDocumentContext.build(index,bindings,request,code;documents=4,characters_per_document=1200,name_weight=2f0,max_references=16) : nothing
    instruction*=context===nothing ? "Self-review control: no tool observations or additional documentation are supplied." :
        context.diagnostic_text*"\n\n"*context.text
    push!(messages,"user"=>instruction)
    (;messages,context)
end

"""POSTHOC SAME8 public-code follow-up preparation; not a new held-out cohort.
All baselines retained. Reads original CODE only, never score traces/test suites.
Summary hash documents prior observation; it is not used to choose tasks/queries.
"""
function prepare_lint_revision(;token_count,destination=joinpath(@__DIR__,"lint_revision_preparation.json"))
    isfile(destination)&&error("Preserve revision preparation")
    digest(path)=bytes2hex(sha256(read(path)))
    parent=joinpath(@__DIR__,"alias_transfer3_results");conditionpath=joinpath(parent,"conditions.json")
    p=JSON3.read(read(conditionpath,String));summarypath=joinpath(parent,"summary.json")
    summary=JSON3.read(read(summarypath,String));summary.complete||error("Finish ALL16 original conditions first")
    summary.conditions_sha256==digest(conditionpath)||error("Parent summary drift")
    corpuspath=joinpath(@__DIR__,"julia_public_binding_docs_v2.json");corpus=JSON3.read(read(corpuspath,String))
    proofpath=joinpath(@__DIR__,"julia_docs_v2_validation.json");proof=JSON3.read(read(proofpath,String))
    proof.complete&&proof.corpus_sha256==digest(corpuspath)&&proof.corpus_checks==22&&proof.software_checks==96||error("Expanded corpus unqualified")
    for (f,h) in pairs(proof.sources)
        digest(joinpath(@__DIR__,String(f)))==h||error("Expanded source drift")
    end
    for file in ("qualified_binding_diagnostics_v2_validation.json","lint_document_context_validation.json")
        unit=JSON3.read(read(joinpath(@__DIR__,file),String));unit.complete||error("Public frontend unqualified")
        for (f,h) in pairs(unit.sources)
            digest(joinpath(@__DIR__,String(f)))==h||error("Frontend source drift")
        end
    end
    index=AliasDocumentRetrieval.Index(corpus.rows)
    tasks=unique(String(j.task) for j in p.jobs);length(tasks)==8||error("Retain entire cohort")
    jobs=Any[]
    for (position,task) in enumerate(tasks)
        original=only(filter(j->j.task==task&&j.arm=="direct",p.jobs))
        path=joinpath(parent,String(original.id)*".json");c=JSON3.read(read(path,String))
        c.conditions_sha256==digest(conditionpath)&&c.id==original.id&&c.messages==original.messages&&c.normal_final_waitidle||error("Parent receipt drift")
        hasproperty(c,:inference_error)&&error("Infrastructure failures require separate handling, never silently drop a task")
        messages=Dict{String,Any}();packs=Dict{String,Any}();sizes=Dict{String,Int}()
        for (arm,mode) in (("direct",:self_review),("public_docs",:lint_docs))
            pack=lint_revision_messages(original.messages,String(c.code),mode,index,corpus.bindings)
            messages[arm]=pack.messages;packs[arm]=pack.context;sizes[arm]=token_count(pack.messages)
        end
        eligible=all(n->n+4096+32<=8192,values(sizes))
        for arm in (isodd(position) ? ("direct","public_docs") : ("public_docs","direct"))
            context=packs[arm]
            push!(jobs,(;id="alias3_"*last(split(task,'/'))*"_"*arm,task,arm,
                condition_name=arm=="direct" ? "self_review" : "lint_and_expanded_docs",eligible,
                skip_reason=eligible ? nothing : "paired_context_capacity_failure",messages=messages[arm],prompt_tokens=sizes[arm],
                original_id=original.id,original_generation_sha256=digest(path),
                retrieved_names=context===nothing ? String[] : [r.qualified for r in context.rows],
                query_sha256=context===nothing ? nothing : bytes2hex(sha256(context.query))))
        end
    end
    packet=(;complete=true,ready_for_generation=false,
        scope="POSTHOC SAME8 external tasks after negative original direct/docs study. Not held-out or a replacement score. All baselines retained, all tasks used.",
        policy="One counterbalanced revision per task/arm from SAME original DIRECT code: self-review vs static unlisted-name observations + automatically retrieved expanded public docs. Same4096 nonthinking token cap/seed/sampler. Internal arm direct MEANS self-review; public_docs MEANS lint-plus-expanded-doc revision, not new baselines. No private tests/expected outputs/grading traces in messages or query; no task-ID rules, retries or score-selected tasks. Bundled tool/context change, not isolated causal evidence for any one component.",
        jobs,decoding=p.decoding,parent_conditions_sha256=digest(conditionpath),observed_parent_summary_sha256=digest(summarypath),
        expanded_corpus_sha256=digest(corpuspath),expanded_validation_sha256=digest(proofpath),
        protocol_sha256=digest(@__FILE__),
        remaining="Finalize and qualify immutable generation/scoring receipts; freeze final conditions BEFORE any revision generation. This preparation generates no answers.")
    open(io->JSON3.pretty(io,packet),destination,"w");packet
end
