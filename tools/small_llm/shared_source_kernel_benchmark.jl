isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:safe_reasoning_budget_study_state) || include("safe_reasoning_budget_study.jl")
isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")

function freeze_shared_source_kernel_benchmark()
    parent=safe_reasoning_budget_study_state(;require_hardware=true)
    qualification=joinpath(@__DIR__,"shared_source_kernel_reference_validation.json")
    q=JSON3.read(read(qualification,String));q.complete && length(q.rows)==4 || error("reference qualification incomplete")
    for (file,hash) in pairs(q.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("reference source drift")
    end
    directory=joinpath(@__DIR__,"shared_source_kernel_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("already registered")
    api=load_kernel_api_pack();jobs=Any[]
    for (i,task) in enumerate(shared_source_kernel_tasks())
        for arm in (isodd(i) ? ("cpu_source","gpu_source") : ("gpu_source","cpu_source"))
            prompt=arm=="cpu_source" ? task.cpu_prompt : task.gpu_prompt
            messages=["system"=>"You are a Julia programmer. Return only the requested executable code.",
                "user"=>prompt*"\n\nIdentical installed public API/source context for both interface conditions:\n"*api.text]
            push!(jobs,(;id=task.id*"_"*arm,task=task.id,arm,messages))
        end
    end
    packet=(;scope="Prospective authored interface comparison after development; tasks are related to prior kernel families, NOT independent public held-out/frontier evidence",
        design="same4B checkpoint, fixed seed/API pack/2048 total output/1024 thinking cap; direct GPU source vs typed CPU source compiled by the fixed restricted lowerer",
        policy="all four tasks/botharms; counterbalanced arm order; immutable attempt before each generation; no retries, seed search, hidden feedback or best-output selection",
        grading="same software-Vulkan functional target; common void-tail normalization. CPU arm must be admitted by frozen lowerer AND pass original CPU tests AND compiled GPU tests. Direct arm must pass GPU tests. Rejection is pipeline failure, not backend inference failure.",
        execution="No generated code runs in the privileged worker; all candidate execution is sandboxed, no /dev/dri",
        jobs,decoding=parent.packet.decoding,thinking_budget=1024,
        parent_conditions_sha256=vulkan_benchmark_hash(parent.path),api_pack_sha256=vulkan_benchmark_hash(api.path),
        qualification_sha256=vulkan_benchmark_hash(qualification),
        source_sha256=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in
            ("shared_source_kernel_tasks.jl","shared_source_kernel_benchmark.jl","typed_pointwise_kernel_lowering.jl","kernel_void_tail.jl")))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function shared_source_kernel_state(;require_hardware=false)
    parent=safe_reasoning_budget_study_state(;require_hardware)
    path=joinpath(@__DIR__,"shared_source_kernel_results","conditions.json")
    p=JSON3.read(read(path,String));api=load_kernel_api_pack()
    p.parent_conditions_sha256==vulkan_benchmark_hash(parent.path) || error("parent protocol drift")
    p.api_pack_sha256==vulkan_benchmark_hash(api.path) || error("API pack drift")
    p.qualification_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"shared_source_kernel_reference_validation.json")) || error("qualification drift")
    for (file,hash) in pairs(p.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("shared-source protocol drift")
    end
    (;parent,path,packet=p,tasks=shared_source_kernel_tasks())
end

function run_shared_source_kernel_benchmark(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    s=shared_source_kernel_state(;require_hardware=true);p=s.packet;directory=dirname(s.path)
    pending=filter(job->!isfile(joinpath(directory,String(job.id)*".json")),p.jobs)
    isempty(pending) && return (;complete=true)
    any(isfile(joinpath(directory,String(job.id)*"_attempt.json")) for job in pending) &&
        error("unfinished prior attempt exists; preserve it, do not regenerate")
    cache=qualified_qwen_cache();tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b");b=p.decoding
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            for job in pending
                time()<started+seconds-75 || return (;complete=false,completed)
                output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
                (isfile(output)||isfile(attempt)) && error("refuse regeneration")
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
                receipt=(;id=job.id,conditions_sha256=vulkan_benchmark_hash(s.path),seed=b.seed,
                    budget_tokens=b.max_tokens,thinking_budget=p.thinking_budget,started_unix_seconds=time())
                open(io->JSON3.pretty(io,receipt),attempt,"w")
                answer=try
                    sampled_safe_budget_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                        thinking_budget=Int(p.thinking_budget),seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),
                        top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                catch err
                    failure=(;id=job.id,task=job.task,arm=job.arm,conditions_sha256=vulkan_benchmark_hash(s.path),
                        attempt_sha256=vulkan_benchmark_hash(attempt),inference_error=sprint(showerror,err),
                        runtime_provenance=inference_runtime_provenance())
                    open(io->JSON3.pretty(io,failure),output,"w");rethrow()
                end
                row=(;answer...,id=job.id,task=job.task,arm=job.arm,messages,
                    conditions_sha256=vulkan_benchmark_hash(s.path),attempt_sha256=vulkan_benchmark_hash(attempt),
                    runtime_provenance=inference_runtime_provenance())
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,String(job.id))
            end
            (;complete=true,completed)
        finally
            try model===nothing || close(model)
            finally finish_device!(device) end
        end
    end
end

function score_shared_source_kernel_benchmark()
    s=shared_source_kernel_state();rows=Any[];directory=dirname(s.path)
    cpu_setup="function __bench_launch__(f,initial,args...;threads);out=copy(initial);f(out,args...);out;end"
    for job in s.packet.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==vulkan_benchmark_hash(s.path) || error("generation protocol drift")
        candidate.attempt_sha256==vulkan_benchmark_hash(attempt) || error("attempt receipt drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error",error=String(candidate.inference_error)));continue
        end
        task=only(filter(t->t.id==job.task,s.tasks));code=String(candidate.code)
        lowered=job.arm=="cpu_source" ? CheckedTypedPointwiseLowering.lower(code) : nothing
        admitted=lowered===nothing || lowered.accepted
        if !admitted
            push!(rows,(;id=job.id,status="failed",stage="compiler_admission",reason=lowered.reason,
                generation_sha256=vulkan_benchmark_hash(path)));continue
        end
        target=lowered===nothing ? code : lowered.code
        tail=KernelVoidTail.normalize(target);selected=tail.accepted ? tail.code : target
        function evaluate_once(task,source,suffix)
            scorepath=replace(path,".json"=>"_"*suffix*"_score.json")
            if isfile(scorepath)
                prior=JSON3.read(read(scorepath,String))
                prior.code_sha256==bytes2hex(sha256(source)) || error("score drift")
                return prior
            end
            JuliaTaskEval.evaluate(task,source;output=scorepath)
        end
        cpu=lowered===nothing ? nothing : evaluate_once((;id=task.id,category="base",setup=cpu_setup,tests=task.tests),code,"cpu")
        gpu=evaluate_once(task,selected,"gpu")
        passed=gpu.passed && (cpu===nothing || cpu.passed)
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=passed ? "passed" : "failed",stage="execution",
            cpu_passed=cpu===nothing ? nothing : cpu.passed,gpu_passed=gpu.passed,
            generation_sha256=vulkan_benchmark_hash(path),forced_closures=candidate.forced_closures,truncated=candidate.truncated))
    end
    report=(;scope=s.packet.scope,policy=s.packet.policy,rows,
        complete=all(row->row.status in ("passed","failed","inference_error"),rows),conditions_sha256=vulkan_benchmark_hash(s.path))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
