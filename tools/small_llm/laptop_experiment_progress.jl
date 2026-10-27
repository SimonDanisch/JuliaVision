using JSON3,SHA,Dates
isdefined(@__MODULE__,:compact_bayes_demo_lost)||include("compact_bayes_demo_interruption.jl")

"""Infrastructure receipts are not generation results or intelligence scores."""
function laptop_infrastructure_progress(directory=@__DIR__)
    gate=joinpath(directory,"laptop_rpn_gate")
    conditions=joinpath(gate,"conditions.json")
    isfile(conditions)||return [(;id="rpn_grammar",status="not_registered",task_quality_evidence=false,phases=Any[])]
    digest=bytes2hex(open(sha256,conditions));rows=Any[];previous=nothing
    for (phase,expected_checks) in (("kernels",557),("identity",14))
        path=joinpath(gate,phase*".json")
        if !isfile(path)
            push!(rows,(;phase,status="pending",expected_checks));continue
        end
        receipt=try JSON3.read(read(path,String)) catch; nothing end
        valid=receipt!==nothing&&all(k->haskey(receipt,k),
            (:phase,:conditions_sha256,:complete,:pid))&&
            receipt.phase==phase&&receipt.conditions_sha256==digest&&receipt.pid isa Integer&&receipt.pid>0
        complete=valid&&receipt.complete===true&&get(receipt,:normal_final_waitidle,false)===true&&
            get(receipt,:passed,0)==expected_checks
        # Identity only counts after a normally completed kernel gate in a different worker.
        eligible=phase=="kernels"||(valid&&previous!==nothing&&previous.pid!=receipt.pid)
        status=!valid ? "invalid_receipt" : complete&&eligible ? "passed" : "incomplete"
        push!(rows,(;phase,status,expected_checks,receipt_sha256=bytes2hex(open(sha256,path))))
        phase=="kernels"&&status=="passed"&&(previous=receipt)
    end
    [(;id="rpn_grammar",status=all(r->r.status=="passed",rows) ? "qualified" : "unfinished",
        task_quality_evidence=false,conditions_sha256=digest,phases=rows)]
end

"""Durable work inventory, not a task-completion signal or pooled quality score."""
function laptop_experiment_progress()
    studies=(("migrated_controller_pilot","laptop_pending_results"),
        ("matched_disabled_controls","laptop_controller_controls_results"),
        ("shared_source_interface","laptop_shared_source_results"),
        ("native_hidden_probability_memory","laptop_native_memory_results"),
        ("native_source_scaled_temperature","laptop_native_temperature_results"),
        ("scalar_kernel_interface","laptop_scalar_kernel_results"),
        ("fresh_frozen_compiler_transfer","compiler_transfer2_results"),
        ("posthoc_scalar_format_followup","scalar_format_followup_results"),
        ("migrated_long_budget_diagnostic","laptop_long_budget_v2_results"),
        ("external_bigcodebench_x_subset","bigcodebench_x_results"),
        ("external_bigcodebench_x_paired_revisions","bigcodebench_x_revision_results"),
        ("prospective_bigcodebench_x_public_docs","bigcodebench_x_doc_results"),
        ("prospective_authored_rpn_stencil_grammar","rpn_stencil_results"),
        ("posthoc_rpn_corrected_contract_example","rpn_stencil_contract_v2_results"),
        ("prospective_numeric_public_example_constraint","rpn_public_example_results"),
        ("prospective_julia_source_symbolic_transfer","julia_stencil_transfer_results"),
        ("prospective_boundary_contract_transfer","boundary_transfer_results"),
        ("prospective_alias_aware_external_transfer","alias_transfer3_results"),
        ("posthoc_external_lint_document_revision","lint_revision_results"),
        ("prospective_pubmedqa_source_reading_pilot","pubmedqa_reading_results"),
        ("prospective_pubmedqa_automatic_retrieval","pubmedqa_retrieval_results"),
        ("prospective_makie_construction_interface","plot_interface_results"),
        ("prospective_bayesian_network_extraction","bayes_extraction_results"),
        ("posthoc_makie_conditional_validator_policy","plot_validator_results"),
        ("posthoc_bayesian_guarded_thinking","bayes_thinking_results"),
        ("posthoc_compact_bayesian_interface","compact_bayes_results"),
        ("posthoc_compact_bayesian_public_demonstrations","compact_bayes_demo_results"))
    rows=Any[]
    for (id,folder) in studies
        directory=joinpath(@__DIR__,folder);path=joinpath(directory,"conditions.json")
        if !isfile(path)
            push!(rows,(;id,status="not_registered"));continue
        end
        p=JSON3.read(read(path,String));completed=String[];pending=String[];interrupted=String[];lost=String[]
        for job in p.jobs
            output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
            loss=compact_bayes_demo_lost(directory,job,bytes2hex(open(sha256,path)))
            push!(loss ? lost : isfile(output) ? completed : isfile(attempt) ? interrupted : pending,String(job.id))
        end
        summarypath=joinpath(directory,"summary.json")
        summary=isfile(summarypath) ? JSON3.read(read(summarypath,String)) : nothing
        scores=summary===nothing ? Any[] : summary.rows
        terminal=isempty(pending)&&isempty(interrupted)
        push!(rows,(;id,status=terminal ? isempty(lost) ? "generation_complete" : "terminal_with_attrition" : "unfinished",
            completed,pending,interrupted,lost,all_slots_terminal=terminal,generation_complete=terminal&&isempty(lost),
            scoring_complete=summary!==nothing&&summary.complete,
            passed_rows=count(r->hasproperty(r,:status)&&r.status=="passed",scores),
            failed_rows=count(r->hasproperty(r,:status)&&r.status=="failed",scores),
            paired_workflow=summary!==nothing&&hasproperty(summary,:direct_passed) ?
                (;direct_passed=summary.direct_passed,hybrid_passed=summary.hybrid_passed,total=summary.total) : nothing,
            workflow_passed=summary!==nothing&&hasproperty(summary,:passed)&&!(summary.passed isa Number) ? summary.passed : nothing,
            classification_arms=summary!==nothing&&hasproperty(summary,:arms) ? summary.arms : nothing,
            conditions_sha256=bytes2hex(open(sha256,path))))
    end
    next_study=findfirst(r->hasproperty(r,:pending)&&!isempty(r.pending),rows)
    packet=(;recorded_utc=string(now(UTC)),not_complete=true,
        scope="Laptop continuation inventory. Original Bosgame recovery checkpoint remains a historical, machine-specific observation, NOT a global work blocker.",
        machines=(;Bosgame="Radeon 8060S",Laptop="Independent fallback; final waitidle per run",Desktop="Not used by this study"),
        studies=rows,infrastructure_gates=laptop_infrastructure_progress(),
        pending_preparation=[r.id for r in rows if r.status=="not_registered"],
        unfinished_workflows=isfile(joinpath(@__DIR__,"bigcodebench_x_revision_policy.json"))&&!isfile(joinpath(@__DIR__,"bigcodebench_x_revision_results/conditions.json")) ?
            ["External paired public-static/self-review: policy qualified/registered, final conditions/runner/grader and all revisions NOT complete"] : String[],
        next_generation=next_study===nothing ? nothing : (;study=rows[next_study].id,id=first(rows[next_study].pending)),
        rule="Never regenerate completed or interrupted attempts. Finish scoring separately. Completing this inventory does not establish frontier intelligence or complete the research program.")
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"laptop_experiment_progress.json"),"w");packet
end
