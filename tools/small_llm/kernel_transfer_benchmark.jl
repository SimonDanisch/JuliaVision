isdefined(@__MODULE__,:kernel_transfer_tasks) || include("kernel_transfer_tasks.jl")
isdefined(@__MODULE__,:load_vulkan_kernel_conditions) || include("vulkan_kernel_benchmark.jl")
isdefined(@__MODULE__,:load_kernel_api_pack) || include("capture_kernel_api_pack.jl")
isdefined(@__MODULE__,:CheckedPointwiseLowering) || include("checked_pointwise_lowering.jl")
isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")

const TRANSFER_SOURCES=("kernel_transfer_benchmark.jl","kernel_transfer_tasks.jl",
    "checked_pointwise_lowering.jl","pointwise_kernel_lowering.jl","kernel_void_tail.jl",
    "capture_kernel_api_pack.jl",VULKAN_BENCHMARK_SOURCES...)

function freeze_kernel_transfer()
    directory=joinpath(@__DIR__,"kernel_transfer_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("already frozen")
    qualification=joinpath(@__DIR__,"kernel_transfer_validation.json")
    q=JSON3.read(read(qualification,String));q.complete || error("unqualified harness")
    length(q.rows)==8 || error("incomplete reference/mutant qualification")
    for (file,hash) in pairs(q.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("qualification drift")
    end
    api=load_kernel_api_pack();tasks=kernel_transfer_tasks()
    packet=(;scope="prospective authored validation after four-task development; not independent public/held-out/frontier evidence",
        design="paired no-addon vs installed API pack; raw and uniformly compiler-transformed scores from SAME outputs",
        policy="one attempt per task/arm; fixed seed; counterbalanced arm order; no retries, best-of selection or hidden feedback",
        execution="sandboxed software Vulkan, not Radeon performance or formal correctness",
        decoding=(;thinking=true,max_tokens=2048,output_chunk=128,seed=20261024,temperature=0.6,top_p=0.95,topk=20,presence=0.0),
        qualification_sha256=vulkan_benchmark_hash(qualification),api_pack_sha256=vulkan_benchmark_hash(api.path),
        tasks=[(;id=t.id,prompt_sha256=bytes2hex(sha256(t.prompt)),setup_sha256=bytes2hex(sha256(t.setup)),tests_sha256=bytes2hex(sha256(t.tests))) for t in tasks],
        source_sha256=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in TRANSFER_SOURCES))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function kernel_transfer_state()
    path=joinpath(@__DIR__,"kernel_transfer_results","conditions.json")
    packet=JSON3.read(read(path,String));tasks=kernel_transfer_tasks();api=load_kernel_api_pack()
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("source drift: $file")
    end
    packet.qualification_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"kernel_transfer_validation.json")) || error("qualification drift")
    packet.api_pack_sha256==vulkan_benchmark_hash(api.path) || error("API pack drift")
    length(tasks)==length(packet.tasks) || error("task count drift")
    for (t,row) in zip(tasks,packet.tasks)
        t.id==row.id || error("task order drift")
        all(bytes2hex(sha256(getproperty(t,f)))==getproperty(row,Symbol(f,"_sha256")) for f in (:prompt,:setup,:tests)) || error("task drift")
    end
    (;path,packet,tasks,api)
end

function run_kernel_transfer(;seconds=210,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    state=kernel_transfer_state();jobs=Any[];directory=dirname(state.path)
    for (i,t) in enumerate(state.tasks), arm in (isodd(i) ? ("no_addon","api_pack") : ("api_pack","no_addon"))
        output=joinpath(directory,t.id*"_"*arm*".json")
        isfile(output) || push!(jobs,(;task=t,arm,output))
    end
    isempty(jobs) && return (;complete=true,completed=String[])
    cache=qualified_qwen_cache();tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir);b=state.packet.decoding
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            runtime=inference_runtime_provenance()
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device,quantcache=cache)
            for job in jobs
                time()<started+seconds-75 || return (;complete=false,completed)
                context=job.arm=="api_pack" ? "\n\nInstalled public API/source context:\n"*state.api.text : ""
                messages=["system"=>"You are a Julia GPU programmer. Return only the requested executable kernel code.","user"=>job.task.prompt*context]
                answer=sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=Bool(b.thinking),
                    output_chunk=Int(b.output_chunk),seed=UInt32(b.seed),temperature=Float32(b.temperature),top_p=Float32(b.top_p),topk=Int(b.topk),presence=Float32(b.presence))
                isfile(job.output) && error("refuse regeneration")
                row=(;task=job.task.id,arm=job.arm,messages,runtime_provenance=runtime,conditions_sha256=vulkan_benchmark_hash(state.path),answer...)
                open(io->JSON3.pretty(io,row),job.output,"w");push!(completed,job.output)
                println(job.task.id," ",job.arm," saved");flush(stdout)
            end
            (;complete=true,completed)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end

function score_kernel_transfer()
    state=kernel_transfer_state();directory=dirname(state.path);rows=Any[]
    for task in state.tasks, arm in ("no_addon","api_pack")
        path=joinpath(directory,task.id*"_"*arm*".json")
        if !isfile(path)
            push!(rows,(;task=task.id,arm,status="pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.task==task.id && candidate.arm==arm && candidate.conditions_sha256==vulkan_benchmark_hash(state.path) || error("candidate drift")
        code=String(candidate.code)
        pointwise=CheckedPointwiseLowering.lower(code)
        intermediate=pointwise.accepted ? pointwise.code : code
        tail=KernelVoidTail.normalize(intermediate);compiled=tail.accepted ? tail.code : intermediate
        for (variant,selected) in (("raw",code),("compiler",compiled))
            scorepath=replace(path,".json"=>"_"*variant*"_score.json")
            if isfile(scorepath)
                result=JSON3.read(read(scorepath,String))
                result.code_sha256==bytes2hex(sha256(selected)) || error("score source drift")
            else
                result=JuliaTaskEval.evaluate(task,selected;output=scorepath)
            end
            push!(rows,(;task=task.id,arm,variant,status=result.passed ? "passed" : "failed",
                truncated=candidate.truncated,pointwise=pointwise.accepted,void_tail=tail.accepted,
                generation_sha256=vulkan_benchmark_hash(path),code_sha256=bytes2hex(sha256(selected))))
        end
    end
    totals=Dict(arm*"_"*variant=>(;passed=count(r->r.arm==arm&&hasproperty(r,:variant)&&r.variant==variant&&r.status=="passed",rows),
        pending=count(r->r.arm==arm&&r.status=="pending",rows)) for arm in ("no_addon","api_pack"),variant in ("raw","compiler"))
    report=(;scope=state.packet.scope,design=state.packet.design,rows,totals,
        complete=all(r->r.status!="pending",rows),conditions_sha256=vulkan_benchmark_hash(state.path))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
