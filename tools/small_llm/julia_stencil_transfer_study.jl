include("julia_stencil_transfer_tasks.jl")
isdefined(@__MODULE__,:JuliaStencilInterface)||include("julia_stencil_interface.jl")
isdefined(@__MODULE__,:PublicExampleSynthesis)||include("public_example_synthesis.jl")
isdefined(@__MODULE__,:PublicExampleFallback)||include("public_example_fallback.jl")
using JSON3,SHA
julia_transfer_digest(path)=bytes2hex(sha256(read(path)))

function freeze_julia_stencil_transfer(;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    proofpath=joinpath(@__DIR__,"julia_stencil_transfer_reference_validation.json")
    proof=JSON3.read(read(proofpath,String))
    proof.complete&&length(proof.rows)==8&&all(r->r.correct.passed&&!r.mutant.passed,proof.rows)||error("References unqualified")
    for (f,h) in pairs(proof.sources)
        julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Reference drift")
    end
    base=JSON3.read(read(joinpath(@__DIR__,"laptop_example_identity_gate/conditions.json"),String))
    extra=("julia_stencil_transfer_tasks.jl","julia_stencil_transfer_study.jl","run_julia_stencil_transfer.jl",
        "julia_stencil_interface.jl","public_example_synthesis.jl","public_example_fallback.jl")
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.sources)),
        Dict(f=>julia_transfer_digest(joinpath(@__DIR__,f)) for f in extra))
    packet=(;scope="Prospective NEW8 authored related-family Julia-source transfer tasks. Not an external/unseen-family/frontier benchmark.",
        policy="All eight prompts, decoding and uniform fallback frozen before ANY generated output. One unmodified neural answer per task, no retries or hidden feedback. Direct arm uses only admitted complete Julia source. Hybrid retains that same program if it fits public examples, else runs bounded first-match symbolic search on the SAME public examples only. Infrastructure failure is not repaired. No task-ID/specification/hidden-answer access by solver. Commit ALL eight choices before independent grading.",
        jobs=[(;id=t.id,task=t.id,public_cases=t.public_cases,messages=julia_stencil_transfer_messages(t)) for t in julia_stencil_transfer_tasks()],
        decoding=(;max_tokens=512,thinking=false,seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64),
        synthesis=(;max_cost=8,candidate_budget=2_000_000,state_budget=100_000,seconds=10.0),
        public_tolerance=(;atol=0f0,rtol=0f0),
        comparison="Paired program workflow, not equal compute or model-weight improvement. Synthesis timing/search work retained; finite example fits need not generalize. Safe subset translation may reject valid unrestricted Julia.",
        portable_manifest_sha256=base.portable_manifest_sha256,source_sha256=sources,
        private_reference_sha256=julia_transfer_digest(proofpath),
        private_scorer_sha256=julia_transfer_digest(joinpath(@__DIR__,"score_julia_stencil_transfer.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function julia_stencil_transfer_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (f,h) in pairs(p.source_sha256)
        basename(String(f))==String(f)&&julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Frozen source drift")
    end
    (;path,packet=p)
end

function validate_julia_stencil_receipt(m,job,p;conditions_sha256,attempt_sha256)
    m.id==job.id&&m.messages==job.messages&&m.normal_final_waitidle===true||error("Receipt identity/prompt/idle mismatch")
    m.conditions_sha256==conditions_sha256&&m.attempt_sha256==attempt_sha256||error("Receipt provenance mismatch")
    hasproperty(m,:inference_error)&&return nothing
    m.input_sha256==bytes2hex(sha256(JSON3.write(job.messages)))||error("Actual prompt mismatch")
    m.budget_tokens==p.decoding.max_tokens&&0<m.decode_steps_executed<=p.decoding.max_tokens||error("Decode budget mismatch")
    m.decode_submissions==m.decode_steps_executed&&m.sampling.draws==m.decode_steps_executed+1||error("Submission/draw mismatch")
    m.eos_output_poll_chunk==p.decoding.output_chunk||error("Polling mismatch")
    m.truncated isa Bool||error("Invalid completion flag")
    for field in (:seed,:topk,:temperature,:top_p,:presence)
        a=getproperty(m.sampling,field);b=getproperty(p.decoding,field)
        (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b)||error("Sampling mismatch")
    end
    nothing
end

function select_julia_stencil_transfer(;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    state=julia_stencil_transfer_state(directory);p=state.packet
    all(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)||error("Wait for ALL8 neural receipts")
    for job in p.jobs
        output=joinpath(directory,String(job.id)*"_selection.json");isfile(output)&&continue
        path=joinpath(directory,String(job.id)*".json");m=JSON3.read(read(path,String))
        attempt=joinpath(directory,String(job.id)*"_attempt.json")
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=julia_transfer_digest(state.path),attempt_sha256=julia_transfer_digest(attempt))
        bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases])
        ok=!hasproperty(m,:inference_error)
        lower=JuliaStencilInterface.lower(ok&&!m.truncated ? String(m.code) : "")
        program=lower.accepted ? lower.program : ""
        symbolic=if ok&&!PublicExampleFallback.matches(program,bank)
            b=p.synthesis
            PublicExampleSynthesis.solve(bank;max_cost=Int(b.max_cost),candidate_budget=Int(b.candidate_budget),
                state_budget=Int(b.state_budget),seconds=Float64(b.seconds))
        else
            (;found=false,program="",reason="not_needed_or_infrastructure_failure",elapsed_seconds=0.0,candidates=0)
        end
        selected=PublicExampleFallback.choose(program,bank,symbolic;infrastructure_ok=ok)
        row=(;id=job.id,conditions_sha256=julia_transfer_digest(state.path),generation_sha256=julia_transfer_digest(path),
            direct=(;accepted=lower.accepted,reason=lower.reason,program),symbolic,selected)
        open(io->JSON3.pretty(io,row),output,"w")
    end
    (;complete=true)
end
