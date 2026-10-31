isdefined(@__MODULE__,:load_vulkan_kernel_conditions) || include("vulkan_kernel_benchmark.jl")
isdefined(@__MODULE__,:diagnose_kernel_public_smoke) || include("kernel_public_smoke.jl")

function freeze_kernel_runtime_feedback()
    baseline=load_vulkan_kernel_conditions()
    any(isfile(joinpath(dirname(baseline.path),t.id*"_baseline.json")) for t in baseline.tasks) &&
        error("register this extension BEFORE generating any kernel baselines")
    qualification=joinpath(@__DIR__,"kernel_public_smoke_validation.json")
    validation=JSON3.read(read(qualification,String))
    validation.complete && validation.passed==14 || error("unqualified public fixture")
    for (file,hash) in pairs(validation.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("public fixture drift")
    end
    directory=joinpath(@__DIR__,"kernel_runtime_feedback_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("repair policy already frozen")
    report=(;scope="pre-registered development extension on four authored kernel tasks, NOT held-out/frontier evidence",
        trigger="public smoke reports syntax/compiler/runtime exception; no private assertion, reference output or observed-value correctness used for eligibility",
        arms=["self_review","runtime_feedback"],attempts_per_arm=1,
        retention="no public exception or unavailable diagnostic: retain baseline; positive exception: one revision replaces baseline even if worse",
        context="original prompt plus prior executable code only, never prior private thinking trace",
        decoding=baseline.packet.baseline,
        baseline_conditions_sha256=vulkan_benchmark_hash(baseline.path),
        qualification_sha256=vulkan_benchmark_hash(qualification),
        source_sha256=Dict(file=>vulkan_benchmark_hash(joinpath(@__DIR__,file)) for file in
            ("kernel_runtime_feedback.jl","kernel_public_smoke.jl")))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function kernel_feedback_state()
    baseline=load_vulkan_kernel_conditions()
    directory=joinpath(@__DIR__,"kernel_runtime_feedback_results")
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    packet.baseline_conditions_sha256==vulkan_benchmark_hash(baseline.path) || error("baseline drift")
    packet.qualification_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"kernel_public_smoke_validation.json")) || error("fixture qualification drift")
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("repair source drift")
    end
    (;baseline,directory,path,packet)
end

function kernel_feedback_jobs()
    state=kernel_feedback_state();jobs=Any[]
    for (i,task) in enumerate(state.baseline.tasks)
        source=joinpath(dirname(state.baseline.path),task.id*"_baseline.json")
        isfile(source) || continue
        candidate=JSON3.read(read(source,String))
        candidate.task==task.id && candidate.conditions_sha256==state.packet.baseline_conditions_sha256 || error("candidate drift")
        diagnostic=joinpath(state.directory,task.id*"_diagnostic.json")
        if !isfile(diagnostic)
            observation=diagnose_kernel_public_smoke((;id=task.id,prompt=task.prompt),String(candidate.code))
            open(io->JSON3.pretty(io,observation),diagnostic,"w")
        end
        observation=JSON3.read(read(diagnostic,String))
        observation.code_sha256==bytes2hex(sha256(String(candidate.code))) || error("diagnostic drift")
        observation.analysis_completed && !observation.payload.completed || continue
        for arm in (isodd(i) ? ("self_review","runtime_feedback") : ("runtime_feedback","self_review"))
            output=joinpath(state.directory,task.id*"_"*arm*".json");isfile(output) && continue
            request="Review your prior kernel against the original request. Fix any issues and return the complete executable kernel code."
            arm=="runtime_feedback" && (request*="\n\nPublic software-Vulkan compilation/execution reported this exception (no private tests or oracle answers):\n"*String(observation.payload.error))
            messages=["system"=>"You are a Julia GPU programmer. Return only the requested executable kernel code.",
                "user"=>task.prompt,"assistant"=>String(candidate.code),"user"=>request]
            push!(jobs,(;task,arm,output,messages,source,diagnostic))
        end
    end
    jobs
end

function run_kernel_runtime_feedback(;seconds=210,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    state=kernel_feedback_state();jobs=kernel_feedback_jobs()
    isempty(jobs) && return (;complete=true,completed=String[])
    b=state.packet.decoding;cache=qualified_qwen_cache()
    tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            runtime=inference_runtime_provenance()
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device,quantcache=cache)
            for job in jobs
                time()<deadline-75 || return (;complete=false,completed)
                answer=sampled_julia_answer(model,tokenizer,job.messages;max_tokens=Int(b.max_tokens),thinking=Bool(b.thinking),
                    output_chunk=Int(b.output_chunk),seed=UInt32(b.seed),temperature=Float32(b.temperature),
                    top_p=Float32(b.top_p),topk=Int(b.topk),presence=Float32(b.presence))
                row=(;task=job.task.id,arm=job.arm,messages=job.messages,runtime_provenance=runtime,
                    conditions_sha256=vulkan_benchmark_hash(state.path),baseline_sha256=vulkan_benchmark_hash(job.source),
                    diagnostic_sha256=vulkan_benchmark_hash(job.diagnostic),answer...)
                isfile(job.output) && error("refuse regeneration")
                open(io->JSON3.pretty(io,row),job.output,"w");push!(completed,job.output)
                println(job.task.id," ",job.arm," kernel revision saved");flush(stdout)
            end
            (;complete=true,completed,elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end

function score_kernel_runtime_feedback()
    state=kernel_feedback_state();rows=Any[]
    for task in state.baseline.tasks
        source=joinpath(dirname(state.baseline.path),task.id*"_baseline.json")
        diagnostic=joinpath(state.directory,task.id*"_diagnostic.json")
        for arm in state.packet.arms
            if !isfile(source)||!isfile(diagnostic)
                push!(rows,(;task=task.id,arm,status="pending"));continue
            end
            original=JSON3.read(read(source,String));observation=JSON3.read(read(diagnostic,String))
            original.task==task.id && original.arm=="sampled_baseline" &&
                original.conditions_sha256==state.packet.baseline_conditions_sha256 || error("baseline identity drift")
            observation.code_sha256==bytes2hex(sha256(String(original.code))) || error("diagnostic drift")
            eligible=observation.analysis_completed && !observation.payload.completed
            candidate=original;output=joinpath(state.directory,task.id*"_"*String(arm)*".json")
            if eligible
                if !isfile(output)
                    push!(rows,(;task=task.id,arm,status="pending"));continue
                end
                candidate=JSON3.read(read(output,String))
                candidate.task==task.id && candidate.arm==arm || error("revision identity drift")
                candidate.conditions_sha256==vulkan_benchmark_hash(state.path) &&
                    candidate.baseline_sha256==vulkan_benchmark_hash(source) &&
                    candidate.diagnostic_sha256==vulkan_benchmark_hash(diagnostic) || error("revision provenance drift")
            end
            result=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(output,".json"=>"_score.json"))
            push!(rows,(;task=task.id,arm,status=result.passed ? "passed" : "failed",
                action=eligible ? "single_revision" : "retain_baseline",truncated=candidate.truncated))
        end
    end
    report=(;scope=state.packet.scope,policy=state.packet.trigger,rows,
        totals=Dict(String(arm)=>(;passed=count(r->r.arm==arm&&r.status=="passed",rows),
            pending=count(r->r.arm==arm&&r.status=="pending",rows)) for arm in state.packet.arms))
    open(io->JSON3.pretty(io,report),joinpath(state.directory,"summary.json"),"w");report
end
