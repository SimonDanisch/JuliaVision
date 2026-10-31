include("laptop_shared_source.jl")
isdefined(@__MODULE__,:verify_transfer2_frozen_compiler) || include("qualify_compiler_transfer2.jl")
isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")

function freeze_compiler_transfer2_benchmark(;directory=joinpath(@__DIR__,"compiler_transfer2_results"))
    frozen=verify_transfer2_frozen_compiler()
    validation=joinpath(@__DIR__,"compiler_transfer2_reference_validation.json")
    v=JSON3.read(read(validation,String));v.complete&&v.passed==20||error("reference gate incomplete")
    for (file,hash) in pairs(v.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash||error("reference source drift")
    end
    v.frozen_compiler_sha256==bytes2hex(open(sha256,frozen))||error("compiler freeze drift")
    base=laptop_shared_source_state(joinpath(@__DIR__,"laptop_shared_source_results"))
    old=first(base.packet.jobs);user=String(only(values(last(old.messages))))
    marker="\n\nIdentical installed public API/source context for both interface conditions:\n"
    parts=split(user,marker;limit=2);length(parts)==2||error("API delimiter")
    context=last(parts);jobs=Any[]
    for (i,task) in enumerate(compiler_transfer2_tasks())
        for arm in (isodd(i) ? ("scalar_program","direct_gpu") : ("direct_gpu","scalar_program"))
            public=ScalarKernelInterface.INTERFACES[task.interface_id]
            instruction=if arm=="scalar_program"
                "Return a pure scalar Julia PROGRAM computing out[i]: optional local scalar assignments, followed by ONE final value expression (or return value). The framework supplies signature, one-dimensional valid1-based index i, bounds guard and unique output store. Do not emit functions, loops, imports or writes. Do not mutate arguments or i. Model supplies ALL arithmetic/indexing; compiler does not repair them. Public signature: "*public.signature*"; valid i in1:("*public.limit*"). Pure functions: +,-,*,/,div,mod,mod1,min,max,clamp,muladd,Float32,Int32,Int."
            else
                "Return a Julia GPU function kernel!, optionally import KernelInterface as KI. Arguments are device arrays, NOT host Vector annotations. Use one1-based global KI.get_global_id().x with rounded-launch bounds guard. Return nothing; no host launcher, device creation, KernelAbstractions or loop over the whole array."
            end
            common=" All arrays distinct, Float32 except indices::Int32. Preserve out padding outside valid length. Array indices must be integers.\n\n"
            messages=["system"=>"You are a Julia programmer. Return only the requested executable code.",
                "user"=>instruction*common*task.specification*marker*context]
            push!(jobs,(;id=task.id*"_"*arm,task=task.id,arm,interface_id=task.interface_id,messages))
        end
    end
    ispath(directory)&&!isempty(readdir(directory))&&error("refuse overwrite")
    compiler=JSON3.read(read(frozen,String))
    packet=(;scope="PROSPECTIVE authored related-family transfer after compiler development; four NEW arithmetic/indexing tasks. Not independent public held-out/frontier evidence.",
        policy="Frozen compiler and ALL eight conditions before generation; no compiler/task/grader changes during study, retries, seed search or hidden feedback. Counterbalanced direct-GPU vs scalar-program arms; same model, sampling, total/thinking cap, API pack. Scalar arm has extra compiler/scaffolding; not model-only comparison.",
        jobs,decoding=base.packet.decoding,thinking_budget=base.packet.thinking_budget,
        parent_conditions_sha256=base.packet.parent_conditions_sha256,gate_sha256=base.packet.gate_sha256,portable_manifest_sha256=base.packet.portable_manifest_sha256,
        frozen_compiler_sha256=bytes2hex(open(sha256,frozen)),qualification_sha256=bytes2hex(open(sha256,validation)),
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.packet.source_sha256)),
            Dict(String(k)=>String(v) for (k,v) in pairs(compiler.source_sha256)),
            Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("compiler_transfer2_tasks.jl","compiler_transfer2_benchmark.jl","qualify_compiler_transfer2.jl"))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function score_compiler_transfer2(;directory=joinpath(@__DIR__,"compiler_transfer2_results"))
    frozen=verify_transfer2_frozen_compiler();s=laptop_shared_source_state(directory);p=s.packet;tasks=compiler_transfer2_tasks();rows=Any[]
    digest(path)=bytes2hex(open(sha256,path))
    digest(frozen)==p.frozen_compiler_sha256||error("compiler drift")
    digest(joinpath(@__DIR__,"compiler_transfer2_reference_validation.json"))==p.qualification_sha256||error("qualification drift")
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==digest(s.path)&&candidate.attempt_sha256==digest(attempt)||error("receipt drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        candidate.messages==job.messages&&candidate.budget_tokens==p.decoding.max_tokens&&candidate.thinking_budget==p.thinking_budget||error("prompt/budget drift")
        candidate.decode_submissions==candidate.decode_steps_executed||error("submission invariant")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            a=getproperty(candidate.sampling,field);b=getproperty(p.decoding,field)
            (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b)||error("sampling drift")
        end
        task=only(filter(t->t.id==job.task,tasks));code=String(candidate.code)
        lowered=job.arm=="scalar_program" ? CanonicalScalarInterface.lower(code,String(job.interface_id)) : nothing
        if lowered!==nothing&&!lowered.accepted
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,status="failed",stage="compiler_admission",reason=lowered.reason));continue
        end
        target=lowered===nothing ? code : lowered.code
        tail=KernelVoidTail.normalize(target);selected=tail.accepted ? tail.code : target
        scorepath=replace(path,".json"=>"_score.json")
        result=if isfile(scorepath)
            prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(selected))||error("score drift");prior
        else
            JuliaTaskEval.evaluate(task,selected;output=scorepath)
        end
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=result.passed ? "passed" : "failed",stage="execution",truncated=candidate.truncated,
            certificate=lowered===nothing ? nothing : lowered.certificate,generation_sha256=digest(path)))
    end
    packet=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),conditions_sha256=digest(s.path),scorer_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,packet),joinpath(directory,"summary.json"),"w");packet
end
