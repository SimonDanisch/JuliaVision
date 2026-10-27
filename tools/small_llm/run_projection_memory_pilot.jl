isdefined(@__MODULE__,:generate_projection_memory_answer) || include("projection_memory_generation.jl")
isdefined(@__MODULE__,:JULIA_ADVANCED_TASKS) || include("julia_advanced_tasks.jl")

function freeze_projection_memory_pilot()
    directory=joinpath(@__DIR__,"projection_memory_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("conditions already frozen")
    datastore=joinpath(@__DIR__,"projection_datastore","registration.json")
    tasks=filter(task->task.id in ("ki_launch","makie_update"),JULIA_ADVANCED_TASKS)
    report=(;scope="two authored development tasks selected after prior API failures; NOT held-out intelligence evidence",
        tasks=[(;id=task.id,prompt=task.prompt,contract_sha256=bytes2hex(sha256(task.tests))) for task in tasks],
        arms=[(;id="zero",weight=0f0,shuffled=false),(;id="memory_10",weight=.1f0,shuffled=false),
              (;id="memory_25",weight=.25f0,shuffled=false),(;id="shuffled_25",weight=.25f0,shuffled=true)],
        policy="one greedy thinking2048 attempt per arm/task; no regeneration or best-arm selection; all outcomes reported",
        budget_tokens=2048,thinking=true,temperature=32f0,topk=8,shuffle_seed=20261024,
        datastore_registration_sha256=bytes2hex(sha256(read(datastore))),
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("run_projection_memory_pilot.jl","projection_memory_generation.jl","projection_datastore.jl",
             "run_julia_tasks.jl","knn_memory.jl","logit_features.jl")))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function run_projection_memory_pilot(;seconds=240)
    seconds>0 || throw(ArgumentError("time budget"))
    require_post_extent_gate()
    directory=joinpath(@__DIR__,"projection_memory_results");path=joinpath(directory,"conditions.json")
    packet=JSON3.read(read(path,String));conditions_sha=bytes2hex(sha256(read(path)))
    for (file,hash) in pairs(packet.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("pilot source drift")
    end
    data=load_projection_datastore()
    data.registration_sha256==packet.datastore_registration_sha256 || error("datastore drift")
    cache=qualified_qwen_cache();Q=SmallLLMChecks.Qwen35Experiment
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b")
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=Q.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            for (i,task) in enumerate(packet.tasks),arm in (isodd(i) ? packet.arms : reverse(packet.arms))
                output=joinpath(directory,String(task.id)*"_"*String(arm.id)*".json")
                isfile(output) && continue
                time()<deadline-90 || return (;complete=false,completed)
                messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>String(task.prompt)]
                result=generate_projection_memory_answer(model,tokenizer,messages,data;
                    weight=Float32(arm.weight),shuffled=Bool(arm.shuffled),max_tokens=Int(packet.budget_tokens),thinking=Bool(packet.thinking))
                row=(;task=task.id,arm=arm.id,conditions_sha256=conditions_sha,
                    contract_sha256=task.contract_sha256,runtime_provenance=inference_runtime_provenance(),result...)
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,output)
                println(task.id," ",arm.id," saved");flush(stdout)
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
