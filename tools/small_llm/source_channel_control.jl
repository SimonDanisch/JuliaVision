isdefined(@__MODULE__,:JULIA_ADVANCED_TASKS) || include("julia_advanced_tasks.jl")
isdefined(@__MODULE__,:run_humaneval_repairs) || include("humaneval_jet_policy.jl")

function freeze_source_channel_control()
    directory=joinpath(@__DIR__,"source_channel_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("source channel control already frozen")
    datastore=joinpath(@__DIR__,"projection_datastore","registration.json")
    memory=joinpath(@__DIR__,"projection_memory_results","conditions.json")
    packet=JSON3.read(read(datastore,String));pilot=JSON3.read(read(memory,String))
    evidence=join((String(row.source) for row in packet.rows),"\n\n")
    report=(;scope="two authored development API tasks; same published sources as probability-space memory, now in prompt",
        evidence,thinking=true,budget_tokens=2048,decoding="greedy",attempts=1,
        rows=pilot.tasks,retention="all two attempts retained, no regeneration or best-result selection",
        datastore_registration_sha256=bytes2hex(sha256(read(datastore))),
        memory_conditions_sha256=bytes2hex(sha256(read(memory))),
        source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function source_channel_jobs()
    directory=joinpath(@__DIR__,"source_channel_results");path=joinpath(directory,"conditions.json")
    packet=JSON3.read(read(path,String));packet.source_sha256==bytes2hex(sha256(read(@__FILE__))) || error("source control drift")
    datastore=joinpath(@__DIR__,"projection_datastore","registration.json")
    packet.datastore_registration_sha256==bytes2hex(sha256(read(datastore))) || error("source evidence drift")
    jobs=Any[]
    for row in packet.rows
        task=only(filter(task->task.id==row.id,JULIA_ADVANCED_TASKS))
        bytes2hex(sha256(task.tests))==row.contract_sha256 || error("task contract drift")
        output=joinpath(directory,String(row.id)*"_source_context.json");isfile(output) && continue
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
            "user"=>String(row.prompt)*"\n\nPublished installed method definitions, for reference only:\n```julia\n"*String(packet.evidence)*"\n```"]
        push!(jobs,(;task,arm="source_context",output,messages,diagnostic_path=datastore,conditions_path=path))
    end
    jobs
end

function run_source_channel_control(;seconds=210)
    run_humaneval_repairs(;seconds,jobs=source_channel_jobs(),extra_source_files=("source_channel_control.jl",))
end
