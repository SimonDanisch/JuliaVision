# ROOT-only scorer: PRIVATE formulas/test inputs must never be an inference input.
include("rpn_stencil_study.jl")
include("qualify_rpn_stencil_tasks.jl")

function validate_rpn_stencil_receipt(candidate,job,p;conditions_sha256,attempt_sha256)
    candidate.id==job.id&&candidate.task==job.task&&candidate.arm==job.arm||error("job identity drift")
    candidate.conditions_sha256==conditions_sha256&&candidate.attempt_sha256==attempt_sha256||error("receipt drift")
    candidate.messages==job.messages&&candidate.normal_final_waitidle===true||error("prompt/final-idle drift")
    hasproperty(candidate,:inference_error)&&return nothing
    candidate.input_sha256==bytes2hex(sha256(JSON3.write(job.messages)))||error("actual input digest drift")
    candidate.budget_tokens==p.decoding.max_tokens&&candidate.decode_submissions==candidate.decode_steps_executed||error("budget/submission drift")
    0<candidate.decode_steps_executed<=p.decoding.max_tokens||error("invalid execution length")
    candidate.sampling.draws==candidate.decode_steps_executed+1||error("sampler draw drift")
    for field in (:seed,:topk,:temperature,:top_p,:presence)
        a=getproperty(candidate.sampling,field);b=getproperty(p.decoding,field)
        (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b)||error("sampling drift")
    end
    candidate.grammar.enabled==job.enabled&&candidate.grammar.operation_budget==p.grammar.operation_budget&&candidate.grammar.max_depth==p.grammar.max_depth||error("grammar control drift")
    candidate.grammar.rejected==0||error("device transition rejected")
    job.enabled&&(candidate.grammar.done!=1||candidate.truncated)&&error("enabled grammar did not finish")
    nothing
end

function score_rpn_stencil_study(;directory=joinpath(@__DIR__,"rpn_stencil_results"))
    state=rpn_stencil_state(directory);p=state.packet
    laptop_rpn_digest(@__FILE__)==p.private_scorer_sha256||error("frozen scorer drift")
    proofpath=joinpath(@__DIR__,"rpn_stencil_reference_validation.json")
    laptop_rpn_digest(proofpath)==p.private_reference_sha256||error("reference qualification drift")
    proof=JSON3.read(read(proofpath,String));proof.complete||error("unqualified grader")
    for (file,hash) in pairs(proof.source_sha256)
        basename(String(file))==String(file)&&laptop_rpn_digest(joinpath(@__DIR__,String(file)))==hash||error("private grader drift")
    end
    rows=Any[];tasks=rpn_stencil_tasks();fixtures=private_rpn_stencil_fixtures()
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        validate_rpn_stencil_receipt(candidate,job,p;conditions_sha256=laptop_rpn_digest(state.path),attempt_sha256=laptop_rpn_digest(attempt))
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status="inference_error",stage="generation",syntax_accepted=false));continue
        end
        # Strict RAW answer admission: no code-fence stripping or posthoc repair.
        lowered=RPNPointwiseInterface.lower(String(candidate.answer);budget=Int(p.grammar.operation_budget),max_depth=Int(p.grammar.max_depth))
        if !lowered.accepted
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status="failed",stage="syntax_admission",syntax_accepted=false,reason=lowered.reason));continue
        end
        index=only(findall(t->t.id==job.task,tasks));task=private_rpn_stencil_task(tasks[index],fixtures[index])
        code=lowered.serial_code*"\n"*lowered.code;scorepath=replace(path,".json"=>"_score.json")
        result=if isfile(scorepath)
            prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(code))||error("cached score drift");prior
        else
            JuliaTaskEval.evaluate(task,code;output=scorepath)
        end
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=result.passed ? "passed" : "failed",
            stage="mathematical_execution",syntax_accepted=true,truncated=candidate.truncated,
            generation_sha256=laptop_rpn_digest(path),certificate=lowered.certificate))
    end
    packet=(;scope=p.scope,policy=p.policy,rows,
        complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=laptop_rpn_digest(state.path),scorer_sha256=laptop_rpn_digest(@__FILE__),
        scoring_policy="Raw strict postfix admission, then independent numerical CPU/software-Vulkan tests; syntax and math reported separately. No timing/frontier inference.")
    open(io->JSON3.pretty(io,packet),joinpath(directory,"summary.json"),"w");packet
end
