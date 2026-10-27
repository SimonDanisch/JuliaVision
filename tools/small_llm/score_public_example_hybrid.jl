# ROOT ONLY: selection is public-only; this file independently grades committed choices.
include("score_public_example_synthesis.jl")
include("public_example_hybrid_experiment.jl")

function score_public_example_hybrid(device;directory=joinpath(@__DIR__,"public_example_hybrid_results"))
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software grading only")
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    all(isfile(joinpath(directory,String(j.task)*".json")) for j in p.jobs)||error("Commit ALL choices before independent grading")
    for (file,hash) in pairs(p.sources)
        example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Policy source drift")
    end
    modeldir=joinpath(@__DIR__,"rpn_public_example_results");symbolicdir=joinpath(@__DIR__,"public_example_synthesis_results")
    modelpath=joinpath(modeldir,"conditions.json");example_study_digest(modelpath)==p.model_conditions_sha256||error("Parent model drift")
    model=JSON3.read(read(modelpath,String));baselinepath=joinpath(modeldir,"summary.json")
    baseline=JSON3.read(read(baselinepath,String));baseline.complete||error("Finish all model grading first")
    proofpath=joinpath(@__DIR__,"rpn_public_example_reference_validation.json")
    example_study_digest(proofpath)==model.private_reference_sha256||error("Reference drift")
    proof=JSON3.read(read(proofpath,String))
    for (file,hash) in pairs(proof.sources)
        example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Independent source drift")
    end
    tasks=rpn_public_example_tasks();fixtures=private_public_example_fixtures();rows=Any[]
    for job in p.jobs
        choicepath=joinpath(directory,String(job.task)*".json");choice=JSON3.read(read(choicepath,String))
        choice.task==job.task&&choice.conditions_sha256==example_study_digest(path)||error("Choice receipt drift")
        neuralpath=joinpath(modeldir,String(job.id)*".json");m=JSON3.read(read(neuralpath,String))
        choice.model_sha256==example_study_digest(neuralpath)||error("Original model output drift")
        validate_public_example_receipt(m,job,model;conditions_sha256=p.model_conditions_sha256,attempt_sha256=example_study_digest(replace(neuralpath,".json"=>"_attempt.json")))
        symbolicpath=joinpath(symbolicdir,String(job.task)*".json");s=JSON3.read(read(symbolicpath,String))
        choice.symbolic_sha256==example_study_digest(symbolicpath)||error("Symbolic output drift")
        bank=public_example_bank(job,model)
        ok=!hasproperty(m,:inference_error)&&m.constraint.raw_status==0&&m.constraint.accepted
        expected=PublicExampleFallback.choose(ok ? String(m.answer) : "",bank,s.result;
            infrastructure_ok=!hasproperty(m,:inference_error)&&m.constraint.raw_status==0)
        all(getproperty(choice.selected,f)==getproperty(expected,f) for f in (:selected,:program,:reason))||error("Choice violates public-only policy")
        base=only(filter(r->r.id==job.id,baseline.rows))
        if choice.selected.selected=="none"
            push!(rows,(;task=job.task,status="failed",baseline_status=base.status,selected="none",reason=choice.selected.reason));continue
        end
        index=only(findall(t->t.id==job.task,tasks));cases=private_public_example_cases(fixtures[index].formula)
        cases_hash=bytes2hex(sha256(JSON3.write(cases)));scorepath=replace(choicepath,".json"=>"_score.json")
        result=if isfile(scorepath)
            cached=JSON3.read(read(scorepath,String))
            cached.choice_sha256==example_study_digest(choicepath)&&cached.cases_sha256==cases_hash||error("Cached hybrid score drift")
            cached.result
        else
            audit=audit_rpn_kernel_device(device,String(choice.selected.program),cases)
            open(io->JSON3.pretty(io,(;choice_sha256=example_study_digest(choicepath),cases_sha256=cases_hash,result=audit)),scorepath,"w");audit
        end
        push!(rows,(;task=job.task,status=result.passed ? "passed" : "failed",baseline_status=base.status,
            selected=choice.selected.selected,choice_sha256=example_study_digest(choicepath)))
    end
    report=(;complete=true,scope=p.scope,policy=p.policy,rows,passed=count(r->r.status=="passed",rows),total=length(rows),
        baseline_passed=count(r->r.baseline_status=="passed",rows),
        paired_wins=count(r->r.status=="passed"&&r.baseline_status!="passed",rows),
        paired_losses=count(r->r.status!="passed"&&r.baseline_status=="passed",rows),
        conditions_sha256=example_study_digest(path),scorer_sha256=example_study_digest(@__FILE__),
        baseline_summary_sha256=example_study_digest(baselinepath),
        limits="POSTHOC system/workflow comparison on four related development tasks, not equal-compute, new neural attempts, independent transfer, frontier intelligence or a model-only reasoning gain. Symbolic search was already run for all4; retain its full recorded cost, including compilation.")
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");println("Hybrid ",report.passed,"/",report.total," vs baseline ",report.baseline_passed,"/",report.total);report
end
