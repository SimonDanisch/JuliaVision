include("laptop_rpn_gate.jl")
include("rpn_stencil_tasks.jl")

function rpn_stencil_jobs()
    jobs=Any[]
    for (i,task) in enumerate(rpn_stencil_tasks())
        for arm in (isodd(i) ? ("disabled","enabled") : ("enabled","disabled"))
            push!(jobs,(;id=task.id*"_"*arm,task=task.id,arm,enabled=arm=="enabled",messages=rpn_stencil_messages(task)))
        end
    end
    jobs
end

function freeze_rpn_stencil_study(;directory=joinpath(@__DIR__,"rpn_stencil_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("preserve prior stencil registration")
    gate=joinpath(@__DIR__,"laptop_rpn_gate")
    gatepath=joinpath(gate,"conditions.json");base=JSON3.read(read(gatepath,String))
    receipts=Dict{String,String}();previous=nothing
    for (phase,count) in (("kernels",557),("identity",14))
        path=joinpath(gate,phase*".json");isfile(path)||error("hardware gate pending: $phase")
        r=JSON3.read(read(path,String))
        r.complete&&r.normal_final_waitidle&&r.passed==count&&r.conditions_sha256==laptop_rpn_digest(gatepath)||error("hardware gate not qualified: $phase")
        phase=="identity"&&r.pid==previous.pid&&error("identity gate must use a fresh worker")
        previous=r;receipts[phase]=laptop_rpn_digest(path)
    end
    proofpath=joinpath(@__DIR__,"rpn_stencil_reference_validation.json")
    proof=JSON3.read(read(proofpath,String));proof.complete&&length(proof.rows)==8&&all(r->r.correct.passed&&!r.incorrect.passed,proof.rows)||error("reference qualification pending")
    for (name,hash) in pairs(proof.source_sha256)
        basename(String(name))==String(name)&&laptop_rpn_digest(joinpath(@__DIR__,String(name)))==hash||error("reference source drift")
    end
    unitpath=joinpath(@__DIR__,"rpn_stencil_study_validation.json")
    units=JSON3.read(read(unitpath,String))
    units.complete&&units.public_job_checks==36&&units.scoring_checks==22&&units.transfer_checks==9||error("study unit qualification pending")
    for (name,hash) in pairs(units.source_sha256)
        basename(String(name))==String(name)&&laptop_rpn_digest(joinpath(@__DIR__,String(name)))==hash||error("unit-qualified source drift")
    end
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.source_sha256)),
        Dict(f=>laptop_rpn_digest(joinpath(@__DIR__,f)) for f in ("rpn_stencil_tasks.jl","rpn_stencil_study.jl")))
    packet=(;scope="Prospective authored related-family stencil ablation, NOT external held-out/frontier evidence; no speed/aesthetic/general Julia claim.",
        policy="All8 new tasks and16 counterbalanced jobs frozen before generation. Identical prompts/model/sampler/nonthinking512token cap/seed; only grammar enabled device flag changes. No retries, hidden feedback, arithmetic lookup/repair or compiler changes. Success requires BOTH raw postfix admission and independent CPU/software-Vulkan mathematical tests. Syntax-only improvement is reported separately from mathematical success.",
        jobs=rpn_stencil_jobs(),decoding=(;max_tokens=512,seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0),
        grammar=(;operation_budget=32,max_depth=8),thinking=false,
        gate_conditions_sha256=laptop_rpn_digest(gatepath),gate_receipts_sha256=receipts,
        private_reference_sha256=laptop_rpn_digest(proofpath),
        private_scorer_sha256=laptop_rpn_digest(joinpath(@__DIR__,"score_rpn_stencil_study.jl")),
        private_unit_qualification_sha256=laptop_rpn_digest(unitpath),
        private_sources_not_deployed="Reference formulas/programs/mutants/tests and scorer remain ROOT-only; only their proof digest is committed.",
        portable_manifest_sha256=base.portable_manifest_sha256,source_sha256=sources)
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function rpn_stencil_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        basename(String(file))==String(file)&&laptop_rpn_digest(joinpath(@__DIR__,String(file)))==hash||error("study source drift: $file")
    end
    gate=joinpath(@__DIR__,"laptop_rpn_gate")
    laptop_rpn_digest(joinpath(gate,"conditions.json"))==p.gate_conditions_sha256||error("gate registration drift")
    for (phase,hash) in pairs(p.gate_receipts_sha256)
        String(phase) in ("kernels","identity")||error("unknown phase")
        laptop_rpn_digest(joinpath(gate,String(phase)*".json"))==hash||error("gate receipt drift")
    end
    (;path,packet=p)
end

function run_rpn_stencil_study(;directory=joinpath(@__DIR__,"rpn_stencil_results"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only; no device setup")
    state=rpn_stencil_state(directory);p=state.packet
    laptop_rpn_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("model drift")
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending)&&error("interrupted attempt; no regeneration")
    job=first(pending);b=p.decoding
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
    length(SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking=false))+b.max_tokens+32<=8192||error("context capacity")
    let
        output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
        (ispath(output)||ispath(attempt))&&error("preserve attempt")
        receipt=(;id=job.id,pid=getpid(),conditions_sha256=laptop_rpn_digest(state.path),started_unix_seconds=time())
        open(io->JSON3.pretty(io,receipt),attempt,"w")
        device=Mantle.Device("radeon");model=nothing
        row=try
            model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
            answer=sampled_rpn_program(model,tokenizer,messages;enabled=Bool(job.enabled),max_tokens=Int(b.max_tokens),
                seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),top_p=Float32(b.top_p),presence=Float32(b.presence),
                operation_budget=Int(p.grammar.operation_budget),max_depth=Int(p.grammar.max_depth),output_chunk=64)
            (;answer...,id=job.id,task=job.task,arm=job.arm,messages,conditions_sha256=laptop_rpn_digest(state.path),
                attempt_sha256=laptop_rpn_digest(attempt),runtime_provenance=inference_runtime_provenance())
        catch err
            (;id=job.id,task=job.task,arm=job.arm,messages,inference_error=sprint(showerror,err),
                conditions_sha256=laptop_rpn_digest(state.path),attempt_sha256=laptop_rpn_digest(attempt))
        finally
            try model===nothing||close(model)
            finally finish_device!(device) end
        end
        # A normal-idle receipt is only written after final synchronization succeeds.
        row=(;row...,normal_final_waitidle=true)
        open(io->JSON3.pretty(io,row),output,"w")
        (;complete=false,id=job.id,inference_error=hasproperty(row,:inference_error))
    end
end
