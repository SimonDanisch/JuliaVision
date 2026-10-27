include("laptop_shared_source.jl")
isdefined(@__MODULE__,:ScalarKernelInterface) || include("scalar_kernel_interface.jl")
isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")

function freeze_laptop_scalar_kernel_study(;directory=joinpath(@__DIR__,"laptop_scalar_kernel_results"))
    baseline=joinpath(@__DIR__,"laptop_shared_source_results");s=laptop_shared_source_state(baseline)
    summary=JSON3.read(read(joinpath(baseline,"summary.json"),String));summary.complete||error("finish all baseline scoring")
    validation=joinpath(@__DIR__,"scalar_kernel_interface_validation.json");v=JSON3.read(read(validation,String));v.complete||error("adapter unqualified")
    for (file,hash) in pairs(v.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash||error("adapter validation drift")
    end
    jobs=Any[]
    for task in shared_source_kernel_tasks()
        old=only(filter(j->j.task==task.id&&j.arm=="cpu_source",s.packet.jobs))
        olduser=String(only(values(last(old.messages))))
        separator="\n\nIdentical installed public API/source context for both interface conditions:\n"
        pieces=split(olduser,separator;limit=2);length(pieces)==2||error("API context delimiter")
        interface=ScalarKernelInterface.INTERFACES[task.id]
        instruction="Return exactly ONE Julia scalar VALUE EXPRESSION computing out[i]. The framework supplies the public signature, valid1-based i, bounds guard, unique output store and device launch. Do NOT emit functions, assignments, loops, imports, return statements or prose. You must supply ALL arithmetic and indexing; the compiler does not fix it. Pure operations available: +,-,*,/,div,mod,mod1,min,max,clamp,muladd,Float32,Int32,Int. Input arrays are distinct from out. Public signature: "*interface.signature*". Valid indices:1:("*interface.limit*").\n\nTask specification (signature is supplied by framework, do not repeat it):\n"*task.specification
        messages=["system"=>"You are a Julia programmer. Return only the requested executable code.","user"=>instruction*separator*last(pieces)]
        push!(jobs,(;id=task.id*"_scalar_source",task=task.id,arm="scalar_source",messages))
    end
    ispath(directory)&&!isempty(readdir(directory))&&error("refuse overwrite")
    packet=(;scope="POSTHOC interface simplification on the SAME four authored development tasks, after observing function-level failures; not independent held-out evidence",
        policy="All four tasks, exact original sampling/2048 total/1024 thinking limit/API context. One immutable attempt each; no baseline regeneration or seed selection. Model supplies full expression; adapter only supplies public scaffolding.",
        jobs,decoding=s.packet.decoding,thinking_budget=s.packet.thinking_budget,
        parent_conditions_sha256=s.packet.parent_conditions_sha256,gate_sha256=s.packet.gate_sha256,portable_manifest_sha256=s.packet.portable_manifest_sha256,
        baseline_conditions_sha256=bytes2hex(open(sha256,s.path)),baseline_summary_sha256=bytes2hex(open(sha256,joinpath(baseline,"summary.json"))),
        validation_sha256=bytes2hex(open(sha256,validation)),
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(s.packet.source_sha256)),
            Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("scalar_kernel_interface.jl","laptop_scalar_kernel_study.jl"))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function score_laptop_scalar_kernel_study(;directory=joinpath(@__DIR__,"laptop_scalar_kernel_results"))
    s=laptop_shared_source_state(directory);p=s.packet;rows=Any[];tasks=shared_source_kernel_tasks()
    digest(path)=bytes2hex(open(sha256,path))
    digest(joinpath(@__DIR__,"scalar_kernel_interface_validation.json"))==p.validation_sha256||error("adapter validation drift")
    digest(joinpath(@__DIR__,"laptop_shared_source_results/summary.json"))==p.baseline_summary_sha256||error("baseline score drift")
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==digest(s.path)&&candidate.attempt_sha256==digest(attempt)||error("generation drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        candidate.messages==job.messages&&candidate.budget_tokens==p.decoding.max_tokens&&candidate.thinking_budget==p.thinking_budget||error("prompt/budget drift")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            a=getproperty(candidate.sampling,field);b=getproperty(p.decoding,field)
            (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b)||error("sampling drift")
        end
        lowered=ScalarKernelInterface.lower(String(candidate.code),String(job.task))
        if !lowered.accepted
            push!(rows,(;id=job.id,status="failed",stage="adapter_admission",reason=lowered.reason));continue
        end
        task=only(filter(t->t.id==job.task,tasks));scorepath=replace(path,".json"=>"_score.json")
        score=if isfile(scorepath)
            prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(lowered.code))||error("score drift");prior
        else
            JuliaTaskEval.evaluate(task,lowered.code;output=scorepath)
        end
        push!(rows,(;id=job.id,task=job.task,status=score.passed ? "passed" : "failed",stage="execution",
            certificate=lowered.certificate,truncated=candidate.truncated,generation_sha256=digest(path)))
    end
    report=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),conditions_sha256=digest(s.path),scorer_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
