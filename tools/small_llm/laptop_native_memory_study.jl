include("laptop_native_datastore.jl")
isdefined(@__MODULE__,:generate_native_hidden_answer) || include("native_hidden_generation.jl")
isdefined(@__MODULE__,:read_native_hidden_records) || include("native_hidden_records.jl")
isdefined(@__MODULE__,:JULIA_ADVANCED_TASKS) || include("julia_advanced_tasks.jl")

function load_laptop_native_data()
    directory=joinpath(@__DIR__,LAPTOP_NATIVE_DIRECTORY);s=laptop_native_datastore_state(directory)
    gate=JSON3.read(read(joinpath(directory,"head_gate.json"),String))
    gate.complete && gate.checks==8 && gate.registration_sha256==bytes2hex(open(sha256,s.path)) || error("native head gate missing")
    read_native_hidden_records(directory,s.packet,Int(s.packet.vocabulary);registration_sha256=bytes2hex(open(sha256,s.path)))
end

function freeze_laptop_native_memory_study(;directory=joinpath(@__DIR__,"laptop_native_memory_results"))
    data=load_laptop_native_data()
    source=laptop_native_datastore_state(joinpath(@__DIR__,LAPTOP_NATIVE_DIRECTORY))
    tasks=filter(t->t.id in ("ki_launch","makie_update"),JULIA_ADVANCED_TASKS)
    arms=[(;id="zero",weight=0f0,shuffled=false),(;id="memory_10",weight=.1f0,shuffled=false),
        (;id="memory_25",weight=.25f0,shuffled=false),(;id="shuffled_25",weight=.25f0,shuffled=true),
        (;id="source_context",weight=0f0,shuffled=false)]
    jobs=Any[]
    for (i,task) in enumerate(tasks),arm in (isodd(i) ? arms : reverse(arms))
        row=only(filter(r->r.id==(task.id=="ki_launch" ? "kernel_call" : "makie_update"),source.packet.rows))
        user=String(task.prompt)
        arm.id=="source_context" && (user*="\n\nPublished installed method source:\n"*String(row.source))
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>user]
        push!(jobs,(;id=task.id*"_"*arm.id,task=task.id,arm=arm.id,weight=arm.weight,shuffled=arm.shuffled,
            messages,contract_sha256=bytes2hex(sha256(task.tests))))
    end
    ispath(directory)&&!isempty(readdir(directory))&&error("refuse overwrite")
    packet=(;scope="Two authored DEVELOPMENT API tasks already used in earlier pilots. Same Laptop runtime and exact published-source corpus; NOT independent held-out/frontier evidence.",
        design="Native final-RMSNorm kNN probability mixture vs zero/shuffled labels/source-in-context; all ten fixed conditions, no arm/seed selection. No tool request or textual memory feedback inside decode.",
        policy="One immutable attempt per condition; failures/interruption retained; greedy non-thinking512-token cap for short API functions. No regeneration. Zero-mixture identity gate on unrelated fixed prompt BEFORE task generation.",
        jobs,budget_tokens=512,thinking=false,temperature=32f0,topk=8,shuffle_seed=20261024,
        datastore_registration_sha256=data.registration_sha256,records=length(data.labels),width=data.width,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("laptop_native_memory_study.jl","native_hidden_generation.jl","native_hidden_records.jl","head_input_tap.jl",
                "knn_memory.jl","memory_status_history.jl","julia_advanced_tasks.jl","run_julia_tasks.jl","portable_qwen35.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function laptop_native_memory_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash||error("memory study source drift")
    end
    data=load_laptop_native_data();data.registration_sha256==p.datastore_registration_sha256||error("datastore drift")
    (;path,packet=p,data)
end

function run_laptop_native_memory_study(;seconds=210,directory=joinpath(@__DIR__,"laptop_native_memory_results"))
    seconds>0||throw(ArgumentError("time budget"))
    s=laptop_native_memory_state(directory);p=s.packet
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending)&&error("unfinished attempt; no retries")
    digest(path)=bytes2hex(open(sha256,path))
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
            gatepath=joinpath(directory,"zero_gate.json")
            if !isfile(gatepath)
                messages=["system"=>"You are a Julia programmer.","user"=>"Write a function that adds two numbers."]
                open(io->JSON3.pretty(io,(;complete=false,conditions_sha256=digest(s.path))),gatepath,"w")
                direct=generate_julia_answer(model,tokenizer,messages;max_tokens=32,thinking=false,output_chunk=128)
                zero=generate_native_hidden_answer(model,tokenizer,messages,s.data;weight=0f0,max_tokens=32,thinking=false)
                direct.answer==zero.answer && direct.generated_tokens==zero.generated_tokens && direct.decode_submissions==zero.decode_submissions==32||error("zero mixture changed output")
                zero.final_memory_status==zero.worst_memory_status==0||error("zero mixture validity")
                open(io->JSON3.pretty(io,(;complete=true,conditions_sha256=digest(s.path),checks=4,runtime_provenance=inference_runtime_provenance())),gatepath,"w")
            end
            gate=JSON3.read(read(gatepath,String));gate.complete&&gate.conditions_sha256==digest(s.path)||error("zero gate incomplete")
            for job in pending
                time()<started+seconds-60||return (;complete=false,completed)
                output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
                (isfile(output)||isfile(attempt))&&error("refuse regeneration")
                open(io->JSON3.pretty(io,(;id=job.id,conditions_sha256=digest(s.path),started_unix_seconds=time())),attempt,"w")
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
                result=try
                    answer=generate_native_hidden_answer(model,tokenizer,messages,s.data;weight=Float32(job.weight),temperature=Float32(p.temperature),
                        shuffled=Bool(job.shuffled),max_tokens=Int(p.budget_tokens),thinking=Bool(p.thinking))
                    answer.final_memory_status==answer.worst_memory_status==0||error("invalid memory query")
                    answer
                catch err
                    open(io->JSON3.pretty(io,(;id=job.id,inference_error=sprint(showerror,err),conditions_sha256=digest(s.path),attempt_sha256=digest(attempt))),output,"w")
                    rethrow()
                end
                row=(;result...,id=job.id,task=job.task,arm=job.arm,messages,contract_sha256=job.contract_sha256,
                    conditions_sha256=digest(s.path),attempt_sha256=digest(attempt),runtime_provenance=inference_runtime_provenance())
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,String(job.id))
            end
            (;complete=true,completed)
        finally
            try model===nothing||close(model)
            finally finish_device!(device) end
        end
    end
end
