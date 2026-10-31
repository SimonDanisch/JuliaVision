# ROOT ONLY: independent grading AFTER all public-only search outputs exist.
include("score_rpn_public_example_study.jl")

function score_public_example_synthesis(device;directory=joinpath(@__DIR__,"public_example_synthesis_results"))
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software grading only")
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.sources)
        basename(String(file))==String(file)&&example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Solver source drift")
    end
    modelpath=joinpath(@__DIR__,"rpn_public_example_results/conditions.json")
    example_study_digest(modelpath)==p.model_conditions_sha256||error("Parent model-study drift")
    model=JSON3.read(read(modelpath,String))
    proofpath=joinpath(@__DIR__,"rpn_public_example_reference_validation.json")
    example_study_digest(proofpath)==model.private_reference_sha256||error("Reference drift")
    proof=JSON3.read(read(proofpath,String));proof.complete||error("Reference incomplete")
    for (file,hash) in pairs(proof.sources)
        example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Private source drift")
    end
    all(isfile(joinpath(directory,String(case.task)*".json")) for case in p.cases)||error("Commit ALL search outputs before private grading")
    fixtures=private_public_example_fixtures();tasks=rpn_public_example_tasks();rows=Any[]
    for case in p.cases
        candidatepath=joinpath(directory,String(case.task)*".json");candidate=JSON3.read(read(candidatepath,String))
        candidate.task==case.task&&candidate.conditions_sha256==example_study_digest(path)||error("Search receipt drift")
        attempt=replace(candidatepath,".json"=>"_attempt.json")
        candidate.attempt_sha256==example_study_digest(attempt)||error("Attempt receipt drift")
        r=candidate.result
        for field in (:max_cost,:candidate_budget,:state_budget,:seconds)
            getproperty(r,field)==getproperty(p.options,field)||error("Search option drift")
        end
        0<=r.candidates<=p.options.candidate_budget&&0<=r.unique_states<=p.options.state_budget||error("Search budget exceeded")
        if !r.found
            isempty(r.program)||error("Failed search has a program")
            push!(rows,(;task=case.task,status="failed",stage="bounded_search",reason=r.reason,generation_sha256=example_study_digest(candidatepath)));continue
        end
        program=String(r.program);parsed=RPNPointwiseInterface.parse_program(program)
        parsed.accepted&&length(parsed.actions)-1==r.operations<=p.options.max_cost||error("Invalid synthesized program")
        job=(;public_cases=case.public_cases)
        bank=public_example_bank(job,model)
        recompute_public_example_match(program,bank)||error("Symbolic output does not fit public examples")
        index=only(findall(t->t.id==case.task,tasks));cases=private_public_example_cases(fixtures[index].formula)
        scorepath=replace(candidatepath,".json"=>"_score.json");cases_hash=bytes2hex(sha256(JSON3.write(cases)))
        result=if isfile(scorepath)
            cached=JSON3.read(read(scorepath,String))
            cached.candidate_sha256==example_study_digest(candidatepath)&&cached.cases_sha256==cases_hash||error("Cached synthesis score drift")
            cached.result
        else
            result=audit_rpn_kernel_device(device,program,cases)
            open(io->JSON3.pretty(io,(;candidate_sha256=example_study_digest(candidatepath),cases_sha256=cases_hash,result)),scorepath,"w");result
        end
        push!(rows,(;task=case.task,status=result.passed ? "passed" : "failed",stage="independent_mathematical_execution",
            public_examples_matched=true,generation_sha256=example_study_digest(candidatepath),operations=r.operations,
            candidates=r.candidates,total_call_seconds_including_jit=candidate.total_call_seconds_including_jit))
    end
    report=(;complete=true,scope=p.scope,policy=p.policy,rows,passed=count(r->r.status=="passed",rows),total=length(rows),
        conditions_sha256=example_study_digest(path),scorer_sha256=example_study_digest(@__FILE__),
        limits="Symbolic-only baseline, zero neural calls; not an equal-compute comparison or an improvement in model intelligence. Public-example fits may fail unseen inputs. Related authored development tasks, not external frontier evidence.")
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");println("Symbolic independent correctness ",report.passed,"/",report.total);report
end
