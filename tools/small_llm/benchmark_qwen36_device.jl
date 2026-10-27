# Include after qwen36_moe_model.jl and qwen36_gguf.jl. No native backend
# selection, compiler flags, or inference arithmetic belongs in this harness.
using JSON3, SHA

function benchmark_qwen36_device(device, checkpoint, config, output;
                                 tokens=64, repeats=3, work_seconds=240, variants=nothing,
                                 interleave=false,observe=()->nothing)
    ispath(output) && error("Preserve existing benchmark receipt: $output")
    1 <= tokens <= 248 && 1 <= repeats <= 10 && 0 < work_seconds <= 600 ||
        throw(ArgumentError("Bounded decode benchmark required"))
    R = Qwen36MoEExperiment
    G = Qwen36GGUF
    cases=variants===nothing ? [("default",R.DEFAULT_OPTIMIZATIONS)] : variants
    isempty(cases)&&throw(ArgumentError("At least one benchmark condition required"))
    prompt = Int[27175,667,17839,958,3145,63368,5638,13]
    report = Dict{String,Any}(
        "complete" => false, "normal_final_waitidle" => false,
        "scope" => "Short-context full-model GPU decode, not task quality or long-context performance",
        "project" => Base.active_project(), "machine" => Sys.MACHINE,
        "device" => string(device), "ram_bytes" => Sys.total_memory(),
        "checkpoint_sha256" => G.CHECKPOINT_SHA256,
        "prompt" => prompt, "decode_tokens" => tokens, "repeats" => repeats,
        "capacity" => 256, "kv_type" => "Float32", "head_layout" => "tiled",
        "feedback" => "device-resident greedy", "automatic_sync" => true,
        "optimizations" => string.(R.DEFAULT_OPTIMIZATIONS),
        "soft_work_budget_seconds" => work_seconds,
        "samples" => Any[], "interleave" => interleave,
        "conditions" => [(;name,flags) for (name,flags) in cases])
    src = dirname(pathof(Mantle))
    report["mantle_source_sha256"] = Dict(
        relpath(joinpath(root,f),src) => bytes2hex(sha256(read(joinpath(root,f))))
        for (root,_,files) in walkdir(src) for f in files if endswith(f,".jl"))
    report["runner_source_sha256"] = Dict(f => bytes2hex(sha256(read(joinpath(@__DIR__,f))))
        for f in ("qwen36_moe_model.jl", "qwen36_optim.jl", "qwen36_gguf.jl",
                  "qwen36_gguf_kernels.jl", "benchmark_qwen36_device.jl"))
    save() = open(io -> JSON3.pretty(io, report), output, "w")
    save()
    model = nothing
    session = nothing
    try
        started = time()
        model = G.loadmodel(checkpoint, config, device; verify_hash=true)
        R.prepare_shared_gateup!(model)
        report["load_seconds"] = time() - started
        started = time()
        session = R.session(model; capacity=256, head_layout=:tiled, kv_type=Float32)
        for token in prompt
            R.step!(session,token)
        end
        R.decode!(session,1)
        Mantle.waitidle(device)
        report["warmup_seconds"] = time() - started
        close(session)
        session = nothing
        save()
        # The bounded measured phase starts after checkpoint verification and
        # cold compilation. Neither contributes to steady-state tokens/second.
        work_start = time()
        first_predictions = nothing
        first_logits = nothing
        schedule=interleave ? [(name,flags,trial) for trial in 1:repeats for (name,flags) in cases] :
                              [(name,flags,trial) for (name,flags) in cases for trial in 1:repeats]
        for (condition,flags,trial) in schedule
            time()-work_start < work_seconds || break
            session = R.session(model; capacity=256, head_layout=:tiled, kv_type=Float32,optimizations=flags)
            report["passes"] = length(session.plan.passes)
            report["barrier_passes"] = count(p -> !isempty(p.pre), session.plan.passes)
            report["transient_peak_bytes"] = session.plan.peak
            for token in prompt
                R.step!(session,token)
            end
            Mantle.waitidle(device)
            observation_before=observe()
            before = Mantle.submissions(device)
            started_ns = time_ns()
            R.decode!(session,tokens)
            queued = (time_ns()-started_ns)/1e9
            Mantle.waitidle(device)
            elapsed = (time_ns()-started_ns)/1e9
            submissions = Mantle.submissions(device)-before
            observation_after=observe()
            predictions = Int.(Array(session.predictions)[length(prompt)+1:length(prompt)+tokens])
            logits = Array(session.logits)
            first_predictions === nothing && (first_predictions=copy(predictions))
            first_logits === nothing && (first_logits=copy(logits))
            sample = (; condition,flags,trial,passes=length(session.plan.passes),
                      barrier_passes=count(p->!isempty(p.pre),session.plan.passes),
                      elapsed_seconds=elapsed, queue_seconds=queued,observation_before,observation_after,
                      tokens_per_second=tokens/elapsed, submissions, predictions,
                      finite_logits=all(isfinite,logits),
                      repeat_tokens_identical=predictions==first_predictions,
                      repeat_logits_bitidentical=reinterpret(UInt32,logits)==reinterpret(UInt32,first_logits))
            push!(report["samples"],sample)
            save()
            println(sample)
            flush(stdout)
            submissions==tokens && sample.finite_logits &&
                sample.repeat_tokens_identical && sample.repeat_logits_bitidentical ||
                error("Decode consistency/submission check failed")
            close(session)
            session = nothing
        end
        report["complete"] = length(report["samples"])==repeats*length(cases)
    catch err
        report["error"] = sprint(showerror,err)
        rethrow()
    finally
        session === nothing || close(session)
        model === nothing || close(model)
        try
            Mantle.waitidle(device)
            report["normal_final_waitidle"] = true
            GC.gc(true)
            Mantle.reclaim!(Mantle.pool(device),device; wait=true)
            Mantle.trim!(Mantle.pool(device),device)
            report["pool_trim"] = true
        finally
            save()
        end
    end
    report
end
