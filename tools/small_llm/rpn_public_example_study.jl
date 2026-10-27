include("laptop_example_identity_gate.jl")
include("rpn_public_example_tasks.jl")
example_study_digest(path)=bytes2hex(sha256(read(path)))

function freeze_rpn_public_example_study(;directory=joinpath(@__DIR__,"rpn_public_example_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("Preserve registration")
    gate=joinpath(@__DIR__,"laptop_example_identity_gate")
    base=JSON3.read(read(joinpath(gate,"conditions.json"),String))
    identity=JSON3.read(read(joinpath(gate,"result.json"),String))
    identity.complete&&identity.normal_final_waitidle&&identity.passed==12||error("Identity gate pending")
    identity.conditions_sha256==example_study_digest(joinpath(gate,"conditions.json"))||error("Gate provenance")
    proofpath=joinpath(@__DIR__,"rpn_public_example_reference_validation.json")
    proof=JSON3.read(read(proofpath,String))
    proof.complete&&length(proof.rows)==4&&all(r->r.correct.passed&&!r.incorrect.passed,proof.rows)||error("Reference qualification pending")
    for (file,hash) in pairs(proof.sources)
        basename(String(file))==String(file)&&example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Private qualification drift")
    end
    unitpath=joinpath(@__DIR__,"rpn_public_example_study_validation.json")
    units=JSON3.read(read(unitpath,String));units.complete||error("Study units pending")
    for (file,hash) in pairs(units.sources)
        example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Unit source drift")
    end
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.sources)),
        Dict(f=>example_study_digest(joinpath(@__DIR__,f)) for f in ("rpn_public_example_tasks.jl","rpn_public_example_study.jl")))
    packet=(;scope="Four NEW authored related-family public-example tasks after prior RPN development. Prospective numeric-mask comparison, NOT external held-out/frontier evidence.",
        policy="ALL8 jobs frozen before generation. Both arms grammar-enabled and receive IDENTICAL public examples/prompts, model, sampling and512-token cap. One numeric-mask flag differs; numeric tracking and graph declarations identical. No retries/seed search/private feedback/arithmetic repair. Dead ends count as failures. Success requires independent CPU/software-Vulkan mathematical tests, not just public-example agreement.",
        jobs=rpn_public_example_jobs(),decoding=(;max_tokens=512,seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,output_chunk=64),
        grammar=(;operation_budget=32,max_depth=8),public_example_tolerance=(;atol=0f0,rtol=0f0),
        identity_conditions_sha256=example_study_digest(joinpath(gate,"conditions.json")),
        identity_receipt_sha256=example_study_digest(joinpath(gate,"result.json")),
        private_reference_sha256=example_study_digest(proofpath),
        private_scorer_sha256=example_study_digest(joinpath(@__DIR__,"score_rpn_public_example_study.jl")),
        private_sources_not_deployed="Private reference programs/formulas/hidden input arrays/scorer remain ROOT-only. Only proof digests committed.",
        portable_manifest_sha256=base.portable_manifest_sha256,source_sha256=sources)
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function rpn_public_example_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        basename(String(file))==String(file)&&example_study_digest(joinpath(@__DIR__,String(file)))==hash||error("Study source drift")
    end
    gate=joinpath(@__DIR__,"laptop_example_identity_gate")
    example_study_digest(joinpath(gate,"conditions.json"))==p.identity_conditions_sha256||error("Identity conditions drift")
    example_study_digest(joinpath(gate,"result.json"))==p.identity_receipt_sha256||error("Identity receipt drift")
    (;path,packet=p)
end

function run_rpn_public_example_study(;directory=joinpath(@__DIR__,"rpn_public_example_results"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only; no setup")
    state=rpn_public_example_state(directory);p=state.packet
    example_study_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("Model drift")
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending)&&error("Interrupted attempt; no regeneration")
    job=first(pending);b=p.decoding
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
    length(SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking=false))+b.max_tokens+32<=8192||error("Context capacity")
    bank=RPNExampleSemantics.Bank([(;input=Float32.(c.input),output=Float32.(c.output)) for c in job.public_cases];
        max_depth=Int(p.grammar.max_depth),atol=Float32(p.public_example_tolerance.atol),rtol=Float32(p.public_example_tolerance.rtol))
    let
        let
            output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
            (ispath(output)||ispath(attempt))&&error("Preserve attempt")
            open(io->JSON3.pretty(io,(;id=job.id,pid=getpid(),conditions_sha256=example_study_digest(state.path),started_unix_seconds=time())),attempt,"w")
            device=Mantle.Device("radeon");model=nothing
            row=try
                model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
                answer=sampled_example_rpn_program(model,tokenizer,messages,bank;enabled=Bool(job.enabled),max_tokens=Int(b.max_tokens),
                    seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),top_p=Float32(b.top_p),presence=Float32(b.presence),
                    operation_budget=Int(p.grammar.operation_budget),output_chunk=Int(b.output_chunk))
                (;answer...,id=job.id,task=job.task,arm=job.arm,messages,conditions_sha256=example_study_digest(state.path),
                    attempt_sha256=example_study_digest(attempt),runtime_provenance=inference_runtime_provenance())
            catch err
                (;id=job.id,task=job.task,arm=job.arm,messages,inference_error=sprint(showerror,err),
                    conditions_sha256=example_study_digest(state.path),attempt_sha256=example_study_digest(attempt))
            finally
                try model===nothing||close(model)
                finally finish_device!(device) end
            end
            row=(;row...,normal_final_waitidle=true)
            open(io->JSON3.pretty(io,row),output,"w")
            (;complete=false,id=job.id,inference_error=hasproperty(row,:inference_error))
        end
    end
end
