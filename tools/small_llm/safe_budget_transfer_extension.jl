isdefined(@__MODULE__,:safe_reasoning_budget_study_state) || include("safe_reasoning_budget_study.jl")

# A separately registered development follow-up, never a replacement for the
# original eight capped outputs or the first-task controller diagnostic.
function freeze_safe_budget_transfer_extension()
    parent=safe_reasoning_budget_study_state(;require_hardware=true)
    pilotpath=joinpath(dirname(parent.path),"summary.json")
    pilot=JSON3.read(read(pilotpath,String))
    pilot.complete || error("score the first-task pilot before registering extension")
    tasks=filter(t->t.id!=parent.packet.task,parent.state.tasks)
    length(tasks)==3 || error("expected all three remaining tasks, no selection")
    directory=joinpath(@__DIR__,"safe_budget_transfer_extension_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("already registered")
    originals=Dict{String,String}()
    order=Any[]
    for (index,task) in enumerate(tasks)
        arms=isodd(index) ? ["api_pack","no_addon"] : ["no_addon","api_pack"]
        for arm in arms
            id=task.id*"_"*arm
            source=joinpath(dirname(parent.state.path),id*".json")
            row=JSON3.read(read(source,String))
            row.conditions_sha256==vulkan_benchmark_hash(parent.state.path) || error("original protocol drift")
            originals[id]=vulkan_benchmark_hash(source)
            push!(order,(;task=task.id,arm,id))
        end
    end
    packet=(;scope="POST-HOC development follow-up on ALL remaining three authored transfer tasks; original outputs already observed, not independent held-out evidence",
        rationale="Test fixed1024 thinking reservation beyond the initial gather diagnostic, without selecting favorable remaining tasks",
        policy="one immutable attempt per registered condition; no regeneration or seed search; retain original outputs and separate backend failures",
        order,thinking_budget=1024,decoding=parent.packet.decoding,
        parent_conditions_sha256=vulkan_benchmark_hash(parent.path),pilot_summary_sha256=vulkan_benchmark_hash(pilotpath),
        original_sha256=originals,
        source_sha256=Dict("safe_budget_transfer_extension.jl"=>vulkan_benchmark_hash(@__FILE__)))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function safe_budget_transfer_extension_state(;require_hardware=false)
    parent=safe_reasoning_budget_study_state(;require_hardware)
    path=joinpath(@__DIR__,"safe_budget_transfer_extension_results","conditions.json")
    packet=JSON3.read(read(path,String))
    packet.parent_conditions_sha256==vulkan_benchmark_hash(parent.path) || error("parent protocol drift")
    packet.pilot_summary_sha256==vulkan_benchmark_hash(joinpath(dirname(parent.path),"summary.json")) || error("pilot score drift")
    for (file,hash) in pairs(packet.source_sha256)
        vulkan_benchmark_hash(joinpath(@__DIR__,String(file)))==hash || error("extension source drift")
    end
    for row in packet.order
        source=joinpath(dirname(parent.state.path),String(row.id)*".json")
        vulkan_benchmark_hash(source)==packet.original_sha256[Symbol(row.id)] || error("original output drift")
    end
    (;parent,path,packet)
end

function run_safe_budget_transfer_extension(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    s=safe_budget_transfer_extension_state(;require_hardware=true);p=s.packet;directory=dirname(s.path)
    pending=filter(row->!isfile(joinpath(directory,String(row.id)*".json")),p.order)
    isempty(pending) && return (;complete=true)
    any(isfile(joinpath(directory,String(row.id)*"_attempt.json")) for row in pending) &&
        error("unfinished prior attempt exists; preserve it, do not regenerate")
    cache=qualified_qwen_cache();tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b");b=p.decoding
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            for condition in pending
                time()<started+seconds-75 || return (;complete=false,completed)
                source=joinpath(dirname(s.parent.state.path),String(condition.id)*".json")
                original=JSON3.read(read(source,String))
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in original.messages]
                output=joinpath(directory,String(condition.id)*".json")
                attempt=joinpath(directory,String(condition.id)*"_attempt.json")
                (isfile(output)||isfile(attempt)) && error("refuse regeneration")
                receipt=(;id=condition.id,conditions_sha256=vulkan_benchmark_hash(s.path),
                    original_sha256=vulkan_benchmark_hash(source),seed=b.seed,budget_tokens=b.max_tokens,
                    thinking_budget=p.thinking_budget,started_unix_seconds=time())
                open(io->JSON3.pretty(io,receipt),attempt,"w")
                answer=try
                    sampled_safe_budget_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                        thinking_budget=Int(p.thinking_budget),seed=UInt32(b.seed),topk=Int(b.topk),
                        temperature=Float32(b.temperature),top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                catch err
                    failure=(;id=condition.id,conditions_sha256=vulkan_benchmark_hash(s.path),
                        original_sha256=vulkan_benchmark_hash(source),attempt_sha256=vulkan_benchmark_hash(attempt),
                        inference_error=sprint(showerror,err),runtime_provenance=inference_runtime_provenance())
                    open(io->JSON3.pretty(io,failure),output,"w");rethrow()
                end
                result=(;id=condition.id,task=condition.task,arm=condition.arm,messages,
                    conditions_sha256=vulkan_benchmark_hash(s.path),original_sha256=vulkan_benchmark_hash(source),
                    attempt_sha256=vulkan_benchmark_hash(attempt),runtime_provenance=inference_runtime_provenance(),answer...)
                open(io->JSON3.pretty(io,result),output,"w");push!(completed,String(condition.id))
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

function score_safe_budget_transfer_extension()
    s=safe_budget_transfer_extension_state();rows=Any[]
    for condition in s.packet.order
        path=joinpath(dirname(s.path),String(condition.id)*".json")
        attempt=joinpath(dirname(s.path),String(condition.id)*"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=condition.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==vulkan_benchmark_hash(s.path) || error("conditions drift")
        candidate.original_sha256==s.packet.original_sha256[Symbol(condition.id)] || error("original drift")
        candidate.attempt_sha256==vulkan_benchmark_hash(attempt) || error("attempt receipt drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=condition.id,status="inference_error",error=String(candidate.inference_error)));continue
        end
        task=only(filter(t->t.id==condition.task,s.parent.state.tasks))
        code=String(candidate.code);pointwise=CheckedPointwiseLowering.lower(code)
        intermediate=pointwise.accepted ? pointwise.code : code
        tail=KernelVoidTail.normalize(intermediate);compiled=tail.accepted ? tail.code : intermediate
        for (variant,selected) in (("raw",code),("compiler",compiled))
            scorepath=replace(path,".json"=>"_"*variant*"_score.json")
            result=if isfile(scorepath)
                prior=JSON3.read(read(scorepath,String))
                prior.code_sha256==bytes2hex(sha256(selected)) || error("score drift")
                prior
            else
                JuliaTaskEval.evaluate(task,selected;output=scorepath)
            end
            push!(rows,(;id=condition.id,task=condition.task,arm=condition.arm,variant,
                status=result.passed ? "passed" : "failed",truncated=candidate.truncated,
                finished_thinking=candidate.finished_thinking,forced_closures=candidate.forced_closures,
                generation_sha256=vulkan_benchmark_hash(path)))
        end
    end
    result=(;scope=s.packet.scope,policy=s.packet.policy,rows,
        complete=all(r->r.status in ("passed","failed","inference_error"),rows),conditions_sha256=vulkan_benchmark_hash(s.path))
    open(io->JSON3.pretty(io,result),joinpath(dirname(s.path),"summary.json"),"w");result
end
