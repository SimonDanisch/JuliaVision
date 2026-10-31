include("laptop_shared_source.jl")
isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")
isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")

function score_laptop_shared_source(;directory=joinpath(@__DIR__,"laptop_shared_source_results"))
    s=laptop_shared_source_state(directory);p=s.packet;rows=Any[];tasks=shared_source_kernel_tasks()
    digest(path)=bytes2hex(open(sha256,path))
    cpu_setup="function __bench_launch__(f,initial,args...;threads);out=copy(initial);f(out,args...);out;end"
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.attempt_sha256==digest(attempt) || error("attempt drift")
        JSON3.read(read(attempt,String)).conditions_sha256==digest(s.path) || error("attempt protocol drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        candidate.conditions_sha256==digest(s.path) && candidate.messages==job.messages || error("generation drift")
        candidate.thinking_budget==p.thinking_budget && candidate.budget_tokens==p.decoding.max_tokens || error("budget drift")
        candidate.decode_steps_executed==candidate.decode_submissions || error("submission invariant")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            a=getproperty(candidate.sampling,field);b=getproperty(p.decoding,field)
            (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b) || error("sampling drift")
        end
        task=only(filter(t->t.id==job.task,tasks));code=String(candidate.code)
        lowered=job.arm=="cpu_source" ? CheckedTypedPointwiseLowering.lower(code) : nothing
        if lowered!==nothing && !lowered.accepted
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status="failed",stage="compiler_admission",reason=lowered.reason,
                generation_sha256=digest(path)));continue
        end
        target=lowered===nothing ? code : lowered.code
        tail=KernelVoidTail.normalize(target);selected=tail.accepted ? tail.code : target
        function evaluate_once(task,source,suffix)
            scorepath=replace(path,".json"=>"_"*suffix*"_score.json")
            if isfile(scorepath)
                previous=JSON3.read(read(scorepath,String));previous.code_sha256==bytes2hex(sha256(source)) || error("score drift");previous
            else
                JuliaTaskEval.evaluate(task,source;output=scorepath)
            end
        end
        cpu=lowered===nothing ? nothing : evaluate_once((;id=task.id,category="base",setup=cpu_setup,tests=task.tests),code,"cpu")
        gpu=evaluate_once(task,selected,"gpu")
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=gpu.passed && (cpu===nothing || cpu.passed) ? "passed" : "failed",
            stage="execution",cpu_passed=cpu===nothing ? nothing : cpu.passed,gpu_passed=gpu.passed,
            truncated=candidate.truncated,forced_closures=candidate.forced_closures,generation_sha256=digest(path)))
    end
    report=(;scope=p.scope,migration=p.migration,grading=p.grading,rows,
        complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=digest(s.path),scorer_sha256=digest(@__FILE__),execution="Sandboxed CPU/software Vulkan only; no generated privileged code")
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
