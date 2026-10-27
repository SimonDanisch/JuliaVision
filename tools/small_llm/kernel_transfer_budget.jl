isdefined(@__MODULE__,:kernel_transfer_state) || include("kernel_transfer_benchmark.jl")

function freeze_kernel_transfer_budget()
    state=kernel_transfer_state();directory=joinpath(@__DIR__,"kernel_transfer_budget_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("already frozen")
    task=first(state.tasks);sources=Dict{String,String}();messages=Dict{String,Any}()
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b")
    for arm in ("no_addon","api_pack")
        source=joinpath(dirname(state.path),task.id*"_"*arm*".json")
        candidate=JSON3.read(read(source,String))
        candidate.conditions_sha256==vulkan_benchmark_hash(state.path) && candidate.truncated && isempty(candidate.code) || error("not a capped-empty diagnostic case")
        sources[arm]=vulkan_benchmark_hash(source);messages[arm]=candidate.messages
        pairs=[String(first(keys(m)))=>String(first(values(m))) for m in candidate.messages]
        length(SmallLLMChecks.QwenText.chat_ids(tokenizer,pairs;thinking=true))+6144+32<=8192 || error("long-budget capacity exceeds qualified harness")
    end
    report=(;scope="POST-HOC budget diagnostic on first authored transfer task after BOTH arms truncated empty; not held-out evidence or a replacement score",
        policy="one new fixed-seed attempt per original arm at6144 tokens; retain original2048 failures and all long-budget outcomes; no seed search",
        limitation="cache capacity can differ from2048-token run, so do not assume identical sampled prefixes; record exact text-prefix agreement",
        task=task.id,arms=["no_addon","api_pack"],decoding=(;state.packet.decoding...,max_tokens=6144),messages,
        transfer_conditions_sha256=vulkan_benchmark_hash(state.path),original_sha256=sources,
        source_sha256=vulkan_benchmark_hash(@__FILE__))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function transfer_budget_state()
    state=kernel_transfer_state();path=joinpath(@__DIR__,"kernel_transfer_budget_results","conditions.json")
    packet=JSON3.read(read(path,String))
    packet.source_sha256==vulkan_benchmark_hash(@__FILE__) && packet.transfer_conditions_sha256==vulkan_benchmark_hash(state.path) || error("budget protocol drift")
    for arm in packet.arms
        source=joinpath(dirname(state.path),String(packet.task)*"_"*String(arm)*".json")
        packet.original_sha256[Symbol(arm)]==vulkan_benchmark_hash(source) || error("original generation drift")
    end
    (;state,path,packet)
end

function run_kernel_transfer_budget(;seconds=240,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    s=transfer_budget_state();p=s.packet;directory=dirname(s.path)
    arms=filter(a->!isfile(joinpath(directory,String(p.task)*"_"*String(a)*".json")),p.arms)
    isempty(arms) && return (;complete=true)
    arm=String(first(arms));b=p.decoding
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in p.messages[Symbol(arm)]]
    cache=qualified_qwen_cache();tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
    let
        started=time();device=Mantle.Device("radeon");model=nothing
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device,quantcache=cache)
            time()<started+seconds-170 || return (;complete=false,reason="insufficient one-attempt cleanup margin")
            answer=sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                output_chunk=Int(b.output_chunk),seed=UInt32(b.seed),temperature=Float32(b.temperature),top_p=Float32(b.top_p),topk=Int(b.topk),presence=Float32(b.presence))
            source=joinpath(dirname(s.state.path),String(p.task)*"_"*arm*".json")
            original=JSON3.read(read(source,String))
            row=(;task=p.task,arm,messages,conditions_sha256=vulkan_benchmark_hash(s.path),
                runtime_provenance=inference_runtime_provenance(),original_sha256=vulkan_benchmark_hash(source),
                exact_original_text_prefix=startswith(answer.answer,String(original.answer)),answer...)
            output=joinpath(directory,String(p.task)*"_"*arm*".json");isfile(output) && error("refuse regeneration")
            open(io->JSON3.pretty(io,row),output,"w");(;complete=length(arms)==1,output)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end
