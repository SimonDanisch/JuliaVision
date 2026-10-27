# Authored development benchmark. References/assertions never enter model prompts.
isdefined(@__MODULE__,:vulkan_kernel_tasks) || include("vulkan_kernel_tasks.jl")
isdefined(@__MODULE__,:run_humaneval_baselines) || include("run_humaneval_subset.jl")
isdefined(@__MODULE__,:sampled_julia_answer) || include("sampled_julia.jl")

vulkan_benchmark_hash(path)=bytes2hex(sha256(read(path)))
const VULKAN_BENCHMARK_SOURCES=("vulkan_kernel_benchmark.jl","vulkan_kernel_tasks.jl",
    "julia_eval.jl","julia_sandbox.sh","sampled_julia.jl","gpu_sampling.jl",
    "run_julia_tasks.jl","run_humaneval_subset.jl","qwen35.jl","qwen_weight_cache.jl",
    "qwen_text.jl","prefill_qwen35.jl","device_finish.jl")

function freeze_vulkan_kernel_benchmark()
    directory=joinpath(@__DIR__,"vulkan_kernel_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("kernel conditions already frozen")
    goldpath=joinpath(@__DIR__,"vulkan_kernel_harness_validation_v2.json")
    mutantpath=joinpath(@__DIR__,"vulkan_kernel_mutant_validation.json")
    gold=JSON3.read(read(goldpath,String));mutants=JSON3.read(read(mutantpath,String))
    tasks=vulkan_kernel_tasks()
    length(gold.references)==length(tasks)==4 && all(row->row.passed,gold.references) || error("unqualified references")
    mutants.complete && length(mutants.rows)==4 || error("unqualified numerical rejection")
    mutants.task_source_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"vulkan_kernel_tasks.jl")) || error("task drift")
    mutants.sandbox_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"julia_sandbox.sh")) || error("sandbox drift")
    for task in tasks
        only(filter(row->row.id==task.id,gold.references)).code_sha256==bytes2hex(sha256(task.reference)) || error("gold drift")
    end
    report=(;scope="four authored development GPU-kernel bodies, not held-out/frontier evidence",
        execution="sandboxed llvmpipe software Vulkan, no Radeon candidate execution; neither hardware performance nor formal race freedom",
        policy="one sampled attempt per task; fixed seed, no retries/seed search; checkpoint each output",
        baseline=(;thinking=true,max_tokens=2048,output_chunk=128,seed=20261024,
            temperature=0.6,top_p=0.95,topk=20,presence=0.0),
        limitations=["2048 output cap can truncate reasoning; not a long-budget model benchmark",
            "new tasks and sampled decoding: cannot attribute differences from older greedy tasks to decoding",
            "private numerical assertions test examples, not arbitrary-input correctness or proof of atomic safety"],
        qualifications=Dict(basename(p)=>vulkan_benchmark_hash(p) for p in (goldpath,mutantpath)),
        tasks=[(;id=t.id,prompt_sha256=bytes2hex(sha256(t.prompt)),
            setup_sha256=bytes2hex(sha256(t.setup)),tests_sha256=bytes2hex(sha256(t.tests))) for t in tasks],
        source_sha256=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in VULKAN_BENCHMARK_SOURCES))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function load_vulkan_kernel_conditions()
    path=joinpath(@__DIR__,"vulkan_kernel_results","conditions.json")
    packet=JSON3.read(read(path,String))
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("frozen kernel source drift: $file")
    end
    for (file,hash) in pairs(packet.qualifications)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("qualification drift")
    end
    tasks=vulkan_kernel_tasks()
    String.(getproperty.(packet.tasks,:id))==getproperty.(tasks,:id) || error("task order drift")
    for (task,row) in zip(tasks,packet.tasks)
        all(bytes2hex(sha256(getproperty(task,field)))==getproperty(row,Symbol(field,"_sha256"))
            for field in (:prompt,:setup,:tests)) || error("contract drift")
    end
    (;path,packet,tasks)
end

function run_vulkan_kernel_baselines(;seconds=210,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    path,packet,tasks=load_vulkan_kernel_conditions();directory=dirname(path)
    pending=filter(t->!isfile(joinpath(directory,t.id*"_baseline.json")),tasks)
    isempty(pending) && return (;complete=true,completed=String[])
    cache=qualified_qwen_cache();tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
    b=packet.baseline
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            runtime=inference_runtime_provenance()
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device,quantcache=cache)
            for task in pending
                time()<deadline-75 || return (;complete=false,completed)
                messages=["system"=>"You are a Julia GPU programmer. Return only the requested executable kernel code.","user"=>task.prompt]
                answer=sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=Bool(b.thinking),
                    output_chunk=Int(b.output_chunk),seed=UInt32(b.seed),temperature=Float32(b.temperature),
                    top_p=Float32(b.top_p),topk=Int(b.topk),presence=Float32(b.presence))
                output=joinpath(directory,task.id*"_baseline.json");isfile(output) && error("refuse regeneration")
                row=(;task=task.id,arm="sampled_baseline",messages,runtime_provenance=runtime,
                    conditions_sha256=vulkan_benchmark_hash(path),answer...)
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,output)
                println(task.id," sampled kernel saved");flush(stdout)
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

function score_vulkan_kernel_baselines()
    path,packet,tasks=load_vulkan_kernel_conditions();directory=dirname(path);rows=Any[]
    for task in tasks
        output=joinpath(directory,task.id*"_baseline.json")
        if !isfile(output)
            push!(rows,(;task=task.id,status="pending"));continue
        end
        candidate=JSON3.read(read(output,String))
        candidate.task==task.id && candidate.arm=="sampled_baseline" || error("candidate identity drift")
        candidate.conditions_sha256==vulkan_benchmark_hash(path) || error("candidate conditions drift")
        result=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(output,".json"=>"_score.json"))
        push!(rows,(;task=task.id,status=result.passed ? "passed" : "failed",truncated=candidate.truncated))
    end
    report=(;scope=packet.scope,execution=packet.execution,rows,passed=count(r->r.status=="passed",rows),
        evaluated=count(r->r.status!="pending",rows),pending=count(r->r.status=="pending",rows))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
