# ROOT ONLY. Do not deploy this scorer or its imported private formulas.
include("rpn_public_example_study.jl")
include("qualify_rpn_public_example_tasks.jl")

function public_example_bank(job,p)
    RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases];
        max_depth=Int(p.grammar.max_depth),atol=Float32(p.public_example_tolerance.atol),rtol=Float32(p.public_example_tolerance.rtol))
end
public_example_bank_digest(bank)=bytes2hex(sha256(JSON3.write((;bank.features,bank.expected,bank.case_offsets,bank.max_depth,bank.atol,bank.rtol))))

function validate_public_example_receipt(c,job,p;conditions_sha256,attempt_sha256)
    c.id==job.id&&c.task==job.task&&c.arm==job.arm||error("Job mismatch")
    c.conditions_sha256==conditions_sha256&&c.attempt_sha256==attempt_sha256||error("Receipt mismatch")
    c.messages==job.messages&&c.normal_final_waitidle===true||error("Prompt/idle mismatch")
    hasproperty(c,:inference_error)&&return nothing
    c.input_sha256==bytes2hex(sha256(JSON3.write(job.messages)))||error("Actual input mismatch")
    c.budget_tokens==p.decoding.max_tokens&&0<c.decode_steps_executed<=p.decoding.max_tokens||error("Budget mismatch")
    c.decode_submissions==c.decode_steps_executed&&c.sampling.draws==c.decode_steps_executed+1||error("Submission/draw mismatch")
    c.eos_output_poll_chunk==p.decoding.output_chunk||error("Polling mismatch")
    for field in (:seed,:topk,:temperature,:top_p,:presence)
        a=getproperty(c.sampling,field);b=getproperty(p.decoding,field)
        (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b)||error("Sampling mismatch")
    end
    k=c.constraint
    k.enabled===job.enabled&&k.operation_budget==p.grammar.operation_budget&&k.max_depth==p.grammar.max_depth||error("Constraint mismatch")
    k.examples_sha256==public_example_bank_digest(public_example_bank(job,p))||error("Public example mismatch")
    all(v->v in (0,1),(k.raw_status,k.sampler_status,k.rejected,k.done))||error("Invalid device flags")
    parsed=RPNPointwiseInterface.parse_program(String(c.answer);budget=Int(p.grammar.operation_budget),max_depth=Int(p.grammar.max_depth))
    expected_accepted=k.raw_status==k.sampler_status==k.rejected==0&&k.done==1&&!c.truncated&&parsed.accepted&&(!job.enabled||k.public_examples_matched)
    k.accepted===expected_accepted||error("Acceptance metadata mismatch")
    if k.accepted
        k.raw_status==k.sampler_status==k.rejected==0&&k.done==1&&!c.truncated||error("False completion")
        !job.enabled||k.public_examples_matched||error("Accepted examples disagree")
    end
    nothing
end

function recompute_public_example_match(program,bank;budget=32)
    parsed=RPNPointwiseInterface.parse_program(program;budget,max_depth=Int(bank.max_depth))
    parsed.accepted||return false
    n=length(bank.expected);stack=zeros(Float32,n*Int(bank.max_depth));status=zeros(Int32,n);depth=Int32(0);operations=Int32(0)
    for action in parsed.actions
        for example in 1:n
            status[example]==0&&(status[example]=RPNExampleSemantics.step!(stack,bank.features,Int32(example),action,depth,bank.max_depth))
        end
        depth,operations,_,invalid=RPNPrefixGrammar.action_state(RPNPrefixGrammar.ARITIES[action],depth,operations,bank.max_depth,Int32(budget))
        invalid==0||return false
    end
    RPNExampleConstraint.finish_allowed(stack,bank.expected,status,depth,bank.max_depth,Int32(n),bank.atol,bank.rtol)
end

function score_rpn_public_example_study(device;directory=joinpath(@__DIR__,"rpn_public_example_results"))
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software grading only")
    state=rpn_public_example_state(directory);p=state.packet
    example_study_digest(@__FILE__)==p.private_scorer_sha256||error("Scorer drift")
    proofpath=joinpath(@__DIR__,"rpn_public_example_reference_validation.json")
    example_study_digest(proofpath)==p.private_reference_sha256||error("Reference drift")
    proof=JSON3.read(read(proofpath,String));proof.complete||error("Unqualified references")
    for (file,hash) in pairs(proof.sources)
        example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Private source drift")
    end
    tasks=rpn_public_example_tasks();fixtures=private_public_example_fixtures();rows=Any[]
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        c=JSON3.read(read(path,String))
        validate_public_example_receipt(c,job,p;conditions_sha256=example_study_digest(state.path),attempt_sha256=example_study_digest(attempt))
        if hasproperty(c,:inference_error)||c.constraint.raw_status!=0
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status="inference_error",stage="generation",syntax_accepted=false));continue
        end
        lowered=RPNPointwiseInterface.lower(String(c.answer);budget=Int(p.grammar.operation_budget),max_depth=Int(p.grammar.max_depth))
        if !c.constraint.accepted||!lowered.accepted
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status="failed",stage="constraint_completion_or_syntax",syntax_accepted=lowered.accepted,
                reason=c.constraint.reason,generation_sha256=example_study_digest(path)));continue
        end
        public_match=recompute_public_example_match(String(c.answer),public_example_bank(job,p);budget=Int(p.grammar.operation_budget))
        public_match==c.constraint.public_examples_matched||error("Numeric state disagrees with independent CPU replay")
        index=only(findall(t->t.id==job.task,tasks));cases=private_public_example_cases(fixtures[index].formula)
        cases_hash=bytes2hex(sha256(JSON3.write(cases)));scorepath=replace(path,".json"=>"_score.json")
        code_hash=bytes2hex(sha256(lowered.serial_code*"\n"*lowered.code))
        result=if isfile(scorepath)
            cached=JSON3.read(read(scorepath,String))
            cached.generation_sha256==example_study_digest(path)&&cached.cases_sha256==cases_hash&&cached.result.code_sha256==code_hash||error("Cached score drift")
            cached.result
        else
            audit=audit_rpn_kernel_device(device,String(c.answer),cases)
            open(io->JSON3.pretty(io,(;generation_sha256=example_study_digest(path),cases_sha256=cases_hash,result=audit)),scorepath,"w");audit
        end
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=result.passed ? "passed" : "failed",stage="independent_mathematical_execution",
            syntax_accepted=true,public_examples_matched=public_match,generation_sha256=example_study_digest(path)))
    end
    packet=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=example_study_digest(state.path),scorer_sha256=example_study_digest(@__FILE__),
        limits="Finite public-example agreement is not mathematical success. Small authored related-family cohort/one seed, not frontier evidence; all failed/incomplete attempts retained.")
    open(io->JSON3.pretty(io,packet),joinpath(directory,"summary.json"),"w");packet
end
