include("api_public_smoke.jl")
isdefined(@__MODULE__,:JULIA_ADVANCED_TASKS) || include("julia_advanced_tasks.jl")
isdefined(@__MODULE__,:run_humaneval_repairs) || include("humaneval_jet_policy.jl")

function freeze_api_runtime_feedback()
    directory=joinpath(@__DIR__,"api_runtime_feedback_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("runtime policy already frozen")
    baseline=joinpath(@__DIR__,"projection_memory_results","conditions.json")
    validation=joinpath(@__DIR__,"api_public_smoke_validation.json")
    JSON3.read(read(validation,String)).passed==14 || error("public diagnostic infrastructure unqualified")
    packet=JSON3.read(read(baseline,String))
    report=(;scope="development follow-up motivated by observed API errors; two authored tasks, NOT held-out evidence",
        trigger="completed public-input diagnostic reports runtime exception; NEVER hidden scores or observed-value correctness judgments",
        baseline="unchanged zero-weight arm of frozen projection-memory pilot",
        assistant_context="prior executable code ONLY, not the potentially long private reasoning trace",
        arms=["self_review","runtime_feedback"],thinking=true,budget_tokens=2048,attempts=1,
        retention="retain baseline if no public runtime exception or tool unavailable; otherwise single revision replaces baseline even if worse",
        rows=packet.tasks,baseline_conditions_sha256=bytes2hex(sha256(read(baseline))),
        qualification_sha256=bytes2hex(sha256(read(validation))),
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("api_runtime_feedback.jl","api_public_smoke.jl")))
    open(io->JSON3.pretty(io,report),path,"w");report
end

function api_runtime_feedback_jobs()
    directory=joinpath(@__DIR__,"api_runtime_feedback_results");conditions=joinpath(directory,"conditions.json")
    packet=JSON3.read(read(conditions,String));jobs=Any[]
    for (file,hash) in pairs(packet.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("runtime policy drift")
    end
    baseline_dir=joinpath(@__DIR__,"projection_memory_results")
    packet.baseline_conditions_sha256==bytes2hex(sha256(read(joinpath(baseline_dir,"conditions.json")))) || error("baseline policy drift")
    for (i,row) in enumerate(packet.rows)
        source=joinpath(baseline_dir,String(row.id)*"_zero.json");isfile(source) || continue
        candidate=JSON3.read(read(source,String));task=only(filter(task->task.id==row.id,JULIA_ADVANCED_TASKS))
        candidate.task==row.id && candidate.arm=="zero" || error("baseline identity drift")
        candidate.conditions_sha256==packet.baseline_conditions_sha256 && candidate.contract_sha256==row.contract_sha256 || error("baseline contract drift")
        diagnostic=joinpath(directory,String(row.id)*"_diagnostic.json")
        if !isfile(diagnostic)
            observation=diagnose_api_public_smoke((;id=task.id,prompt=task.prompt),String(candidate.code))
            open(io->JSON3.pretty(io,observation),diagnostic,"w")
        end
        observation=JSON3.read(read(diagnostic,String))
        observation.code_sha256==bytes2hex(sha256(String(candidate.code))) || error("diagnostic drift")
        observation.analysis_completed && !observation.payload.completed || continue
        for arm in (isodd(i) ? ("self_review","runtime_feedback") : ("runtime_feedback","self_review"))
            output=joinpath(directory,String(row.id)*"_"*arm*".json");isfile(output) && continue
            request="Review the prior executable solution against the original request. Fix any problems and return the complete executable Julia solution."
            arm=="runtime_feedback" && (request*="\n\nSandboxed public-spec execution reported a runtime exception. This contains no private assertions or expected answers:\n"*String(observation.payload.error))
            messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
                "user"=>task.prompt,"assistant"=>String(candidate.code),"user"=>request]
            push!(jobs,(;task,arm,output,messages,diagnostic_path=diagnostic,conditions_path=conditions))
        end
    end
    jobs
end

function run_api_runtime_feedback(;seconds=210)
    run_humaneval_repairs(;seconds,jobs=api_runtime_feedback_jobs(),extra_source_files=("api_runtime_feedback.jl","api_public_smoke.jl"))
end
