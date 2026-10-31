isdefined(@__MODULE__,:kernel_transfer_state) || include("kernel_transfer_benchmark.jl")
isdefined(@__MODULE__,:sampled_safe_budget_julia_answer) || include("sampled_safe_budget_julia.jl")

function freeze_safe_reasoning_budget_study()
    state=kernel_transfer_state();task=first(state.tasks)
    any(isfile(joinpath(@__DIR__,"reasoning_budget_results",task.id*"_"*arm*".json")) for arm in ("no_addon","api_pack")) &&
        error("guard amendment must precede ALL prototype task outputs")
    directory=joinpath(@__DIR__,"safe_reasoning_budget_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("already registered")
    qualification=joinpath(@__DIR__,"safe_reasoning_budget_software_validation.json")
    q=JSON3.read(read(qualification,String));q.complete || error("software gate missing")
    for (file,hash) in pairs(q.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("software qualification drift")
    end
    sources=Dict{String,String}()
    for arm in ("no_addon","api_pack")
        source=joinpath(dirname(state.path),task.id*"_"*arm*".json")
        p=JSON3.read(read(source,String))
        p.truncated && isempty(p.code) && p.conditions_sha256==vulkan_benchmark_hash(state.path) || error("not the registered capped-empty pair")
        sources[arm]=vulkan_benchmark_hash(source)
    end
    packet=(;superseded_prototype_conditions_sha256=vulkan_benchmark_hash(joinpath(@__DIR__,"reasoning_budget_results","conditions.json")),
        amendment="Before ANY controller task output: add raw-logit validation so forced closure cannot conceal invalid neural inference; prototype preserved separately",scope="POST-HOC first-task budget-controller diagnostic after both original arms truncated empty; not independent held-out or knowledge-gain evidence",
        task=task.id,arms=["no_addon","api_pack"],thinking_budget=1024,
        rule="same original messages/seed/sampling/2048 total allowance; reserve answer space by forcing the native closing-think token once after1024 thinking draws if not already closed",
        policy="one fixed attempt per arm; immutable attempt receipt BEFORE generation; interrupted attempts never regenerated; inference errors reported separately, not language failures; original failures unchanged",
        decoding=state.packet.decoding,transfer_conditions_sha256=vulkan_benchmark_hash(state.path),original_sha256=sources,
        software_qualification_sha256=vulkan_benchmark_hash(qualification),
        required_hardware_gate="safe_reasoning_budget_full_model_validation.json",
        source_sha256=Dict(f=>vulkan_benchmark_hash(joinpath(@__DIR__,f)) for f in
            ("safe_reasoning_budget_study.jl","reasoning_budget.jl","safe_reasoning_budget.jl","sampled_safe_budget_julia.jl","qualify_safe_reasoning_budget_full_model.jl")))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function safe_reasoning_budget_study_state(;require_hardware=false)
    state=kernel_transfer_state();path=joinpath(@__DIR__,"safe_reasoning_budget_results","conditions.json")
    p=JSON3.read(read(path,String))
    p.transfer_conditions_sha256==vulkan_benchmark_hash(state.path) || error("transfer protocol drift")
    for (file,hash) in pairs(p.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("budget protocol source drift")
    end
    p.software_qualification_sha256==vulkan_benchmark_hash(joinpath(@__DIR__,"safe_reasoning_budget_software_validation.json")) || error("software receipt drift")
    for arm in p.arms
        source=joinpath(dirname(state.path),String(p.task)*"_"*String(arm)*".json")
        p.original_sha256[Symbol(arm)]==vulkan_benchmark_hash(source) || error("original generation drift")
    end
    if require_hardware
        gate=JSON3.read(read(joinpath(@__DIR__,String(p.required_hardware_gate)),String))
        gate.complete && gate.normal_final_waitidle && gate.hardware_fixture_passed==156 && gate.full_model_passed==9 || error("hardware identity/activation gate incomplete")
        for (file,hash) in pairs(gate.source_sha256)
            vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("hardware qualification drift")
        end
    end
    (;state,path,packet=p)
end

function run_safe_reasoning_budget_study(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    s=safe_reasoning_budget_study_state(;require_hardware=true);p=s.packet;directory=dirname(s.path)
    arms=filter(a->!isfile(joinpath(directory,String(p.task)*"_"*String(a)*".json")),p.arms)
    isempty(arms) && return (;complete=true)
    any(isfile(joinpath(directory,String(p.task)*"_"*String(a)*"_attempt.json")) for a in arms) &&
        error("unfinished prior attempt exists; preserve it, do not regenerate")
    cache=qualified_qwen_cache();tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b");b=p.decoding
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            for arm in arms
                time()<started+seconds-75 || return (;complete=false,completed)
                source=joinpath(dirname(s.state.path),String(p.task)*"_"*String(arm)*".json")
                original=JSON3.read(read(source,String))
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in original.messages]
                output=joinpath(directory,String(p.task)*"_"*String(arm)*".json");isfile(output) && error("refuse regeneration")
                attempt=replace(output,".json"=>"_attempt.json");isfile(attempt) && error("prior attempt exists")
                receipt=(;task=p.task,arm,conditions_sha256=vulkan_benchmark_hash(s.path),original_sha256=vulkan_benchmark_hash(source),
                    seed=b.seed,budget_tokens=b.max_tokens,thinking_budget=p.thinking_budget,started_unix_seconds=time())
                open(io->JSON3.pretty(io,receipt),attempt,"w")
                answer=try
                    sampled_safe_budget_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,thinking_budget=Int(p.thinking_budget),
                        seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                catch err
                    failure=(;task=p.task,arm,messages,conditions_sha256=vulkan_benchmark_hash(s.path),original_sha256=vulkan_benchmark_hash(source),
                        attempt_sha256=vulkan_benchmark_hash(attempt),runtime_provenance=inference_runtime_provenance(),
                        inference_error=sprint(showerror,err),scope="inference failure, not a measured language-task failure")
                    open(io->JSON3.pretty(io,failure),output,"w")
                    rethrow()
                end
                row=(;task=p.task,arm,messages,conditions_sha256=vulkan_benchmark_hash(s.path),original_sha256=vulkan_benchmark_hash(source),
                    attempt_sha256=vulkan_benchmark_hash(attempt),
                    runtime_provenance=inference_runtime_provenance(),hardware_gate_sha256=vulkan_benchmark_hash(joinpath(@__DIR__,String(p.required_hardware_gate))),answer...)
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,output)
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

function score_safe_reasoning_budget_study()
    s=safe_reasoning_budget_study_state();p=s.packet
    task=only(filter(t->t.id==p.task,s.state.tasks));rows=Any[]
    for arm in p.arms
        path=joinpath(dirname(s.path),String(p.task)*"_"*String(arm)*".json")
        if !isfile(path)
            interrupted=isfile(replace(path,".json"=>"_attempt.json"))
            push!(rows,(;task=p.task,arm,status=interrupted ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==vulkan_benchmark_hash(s.path) && candidate.original_sha256==p.original_sha256[Symbol(arm)] || error("controlled generation drift")
        candidate.attempt_sha256==vulkan_benchmark_hash(replace(path,".json"=>"_attempt.json")) || error("attempt receipt drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;task=p.task,arm,status="inference_error",error=String(candidate.inference_error)));continue
        end
        code=String(candidate.code);pointwise=CheckedPointwiseLowering.lower(code)
        intermediate=pointwise.accepted ? pointwise.code : code
        tail=KernelVoidTail.normalize(intermediate);compiled=tail.accepted ? tail.code : intermediate
        for (variant,selected) in (("raw",code),("compiler",compiled))
            scorepath=replace(path,".json"=>"_"*variant*"_score.json")
            if isfile(scorepath)
                result=JSON3.read(read(scorepath,String));result.code_sha256==bytes2hex(sha256(selected)) || error("score drift")
            else
                result=JuliaTaskEval.evaluate(task,selected;output=scorepath)
            end
            push!(rows,(;task=p.task,arm,variant,status=result.passed ? "passed" : "failed",truncated=candidate.truncated,
                finished_thinking=candidate.finished_thinking,forced_closures=candidate.forced_closures,
                generation_sha256=vulkan_benchmark_hash(path)))
        end
    end
    report=(;scope=p.scope,policy=p.policy,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),conditions_sha256=vulkan_benchmark_hash(s.path))
    open(io->JSON3.pretty(io,report),joinpath(dirname(s.path),"summary.json"),"w");report
end
