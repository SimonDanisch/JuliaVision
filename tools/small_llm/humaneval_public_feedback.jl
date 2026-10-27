isdefined(@__MODULE__,:run_humaneval_repairs) || include("humaneval_jet_policy.jl")
isdefined(@__MODULE__,:diagnose_public_smoke) || include("public_julia_smoke.jl")
isdefined(@__MODULE__,:diagnose_public_doctest) || include("public_julia_doctest.jl")
using TOML

function public_smoke_trigger(diagnosis)
    diagnosis.analysis_completed && return any(row->!row.completed ||
        (hasproperty(row,:matches_public_example)&&!row.matches_public_example),diagnosis.payload)
    text=diagnosis.loading_or_harness_error
    # Only a reported failure in candidate loading, not a timeout or unknown
    # sandbox/tool failure. Never use hidden test results to choose revisions.
    text!==nothing && occursin("ERROR:",text) && occursin("/input/candidate.jl",text)
end

function freeze_public_feedback_conditions()
    baseline_dir=joinpath(@__DIR__,"humaneval_subset_results")
    qualification_path=joinpath(baseline_dir,"public_doctest_qualification.json")
    qualification=JSON3.read(read(qualification_path,String))
    qualification.qualified==qualification.selected==12 || error("all public reference doctests must qualify")
    qualification.source_sha256==bytes2hex(sha256(read(joinpath(@__DIR__,"public_julia_doctest.jl")))) || error("doctest source drift")
    directory=joinpath(@__DIR__,"humaneval_public_feedback_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("public feedback conditions already frozen")
    rows=Any[]
    for task in load_frozen_humaneval_tasks()
        source=joinpath(baseline_dir,task.id*"_baseline.json")
        isfile(source) || error("complete the unchanged twelve baselines first")
        diagnostic_path=joinpath(baseline_dir,task.id*"_public_doctest.json")
        isfile(diagnostic_path) || error("prepare public diagnostics first")
        baseline=JSON3.read(read(source,String));diagnosis=JSON3.read(read(diagnostic_path,String))
        validate_external_candidate(task,baseline)
        diagnosis.code_sha256==bytes2hex(sha256(String(baseline.code))) || error("public diagnostic drift")
        push!(rows,(;task=task.id,baseline_sha256=bytes2hex(sha256(read(source))),
            diagnostic_sha256=bytes2hex(sha256(read(diagnostic_path))),eligible=public_smoke_trigger(diagnosis)))
    end
    project=joinpath(@__DIR__,"../../../../Project.toml")
    packages=sort!(collect(keys(TOML.parsefile(project)["deps"])))
    report=(;scope="development extension motivated by observed loading/empty-input failures; NOT a preregistered held-out transfer result",
        trigger="public candidate-loading rejection, runtime error, or mismatch with a prompt example's displayed output; no hidden scores",
        arms=["self_review","public_feedback"],thinking=true,max_tokens=2048,decoding="greedy",attempts_per_arm=1,
        retention="retain original baseline when no public failure; revisions replace baseline even if they regress",
        installed_direct_packages=packages,project_sha256=bytes2hex(sha256(read(project))),rows,
        public_doctest_qualification_sha256=bytes2hex(sha256(read(qualification_path))),
        source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,report),path,"w")
    report
end

function external_public_feedback_jobs()
    directory=joinpath(@__DIR__,"humaneval_public_feedback_results")
    conditions=JSON3.read(read(joinpath(directory,"conditions.json"),String));jobs=Any[]
    conditions.source_sha256==bytes2hex(sha256(read(@__FILE__))) || error("public policy source drift")
    project=joinpath(@__DIR__,"../../../../Project.toml")
    conditions.project_sha256==bytes2hex(sha256(read(project))) || error("installed environment drift")
    baseline_dir=joinpath(@__DIR__,"humaneval_subset_results")
    for (i,task) in enumerate(load_frozen_humaneval_tasks())
        registration=only(filter(row->row.task==task.id,conditions.rows))
        source=joinpath(baseline_dir,task.id*"_baseline.json")
        diagnostic_path=joinpath(baseline_dir,task.id*"_public_doctest.json")
        registration.baseline_sha256==bytes2hex(sha256(read(source))) || error("baseline drift")
        registration.diagnostic_sha256==bytes2hex(sha256(read(diagnostic_path))) || error("diagnostic drift")
        registration.eligible || continue
        baseline=JSON3.read(read(source,String));diagnosis=JSON3.read(read(diagnostic_path,String))
        for arm in (isodd(i) ? ("self_review","public_feedback") : ("public_feedback","self_review"))
            output=joinpath(directory,task.id*"_"*arm*".json");isfile(output) && continue
            request="Review the complete prior solution for compliance with the original request. Fix any problems and return the complete executable Julia solution."
            if arm=="public_feedback"
                evidence=diagnosis.analysis_completed ? JSON3.write(diagnosis.payload) : String(diagnosis.loading_or_harness_error)
                request*="\n\nSandboxed execution of ONLY the examples already in the public prompt reports:\n"*first(evidence,12000)*
                    "\nAll expected values above were already in your original public prompt. No private tests or values were used. Installed direct packages: "*join(conditions.installed_direct_packages,", ")*
                    ". Do not install packages. Use Base/standard libraries or available packages."
            end
            messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
                "user"=>task.prompt,"assistant"=>String(baseline.answer),"user"=>request]
            push!(jobs,(;task,arm,output,messages,diagnostic_path))
        end
    end
    jobs
end

function run_external_public_feedback(;seconds=240)
    run_humaneval_repairs(;seconds,jobs=external_public_feedback_jobs(),
        conditions_path=joinpath(@__DIR__,"humaneval_public_feedback_results","conditions.json"),
        extra_source_files=("humaneval_public_feedback.jl","public_julia_smoke.jl","public_julia_doctest.jl"))
end

function score_external_public_feedback()
    directory=joinpath(@__DIR__,"humaneval_public_feedback_results")
    conditions_path=joinpath(directory,"conditions.json")
    conditions=JSON3.read(read(conditions_path,String));rows=Any[]
    baseline_dir=joinpath(@__DIR__,"humaneval_subset_results")
    for task in load_frozen_humaneval_tasks()
        registration=only(filter(row->row.task==task.id,conditions.rows))
        baseline_path=joinpath(baseline_dir,task.id*"_baseline.json")
        registration.baseline_sha256==bytes2hex(sha256(read(baseline_path))) || error("baseline drift")
        baseline=JSON3.read(read(baseline_path,String))
        # Evaluate all tasks; neither these scores nor earlier hidden scores
        # determine eligibility or which revision to retain.
        baseline_score=JuliaTaskEval.evaluate(task,String(baseline.code))
        for arm in ("self_review","public_feedback")
            path=joinpath(directory,task.id*"_"*arm*".json")
            if !registration.eligible
                push!(rows,(;task=task.id,arm,status=baseline_score.passed ? "passed" : "failed",
                    action="no_public_failure_retain_baseline",baseline_passed=baseline_score.passed))
            elseif !isfile(path)
                push!(rows,(;task=task.id,arm,status="pending",baseline_passed=baseline_score.passed))
            else
                candidate=JSON3.read(read(path,String))
                candidate.task==task.id && candidate.arm==arm || error("revision identity drift")
                candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
                candidate.conditions_sha256==bytes2hex(sha256(read(conditions_path))) || error("conditions drift")
                candidate.diagnostic_sha256==registration.diagnostic_sha256 || error("diagnostic drift")
                score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(path,".json"=>"_score.json"))
                push!(rows,(;task=task.id,arm,status=score.passed ? "passed" : "failed",action="single_revision",
                    baseline_passed=baseline_score.passed,truncated=candidate.truncated,generation_seconds=candidate.elapsed_seconds))
            end
        end
    end
    totals=Dict(arm=>(;passed=count(r->r.arm==arm&&r.status=="passed",rows),
        evaluated=count(r->r.arm==arm&&r.status!="pending",rows),pending=count(r->r.arm==arm&&r.status=="pending",rows),
        improvements=count(r->r.arm==arm&&!r.baseline_passed&&r.status=="passed",rows),
        regressions=count(r->r.arm==arm&&r.baseline_passed&&r.status=="failed",rows)) for arm in ("self_review","public_feedback"))
    report=(;scope=conditions.scope,policy=conditions.trigger,rows,totals)
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w")
    report
end
