# Re-evaluate immutable candidates after fixing the sandbox's missing system
# depots. Do not overwrite legacy scores, change extraction, or regenerate code.
using JSON3, SHA
isdefined(@__MODULE__,:SCENE_GENERALIZATION_TASKS) || include("julia_scene_generalization_tasks.jl")

function rescore_system_harness(;workers=2)
    workers in 1:2 || throw(ArgumentError("limit CPU subprocess concurrency"))
    qualification=JSON3.read(read(joinpath(@__DIR__,"julia_harness_system_validation.json"),String))
    length(qualification.references)==6 && all(r->r.passed,qualification.references) && qualification.mutant_rejected || error("unqualified harness")
    isolation=JSON3.read(read(joinpath(@__DIR__,"sandbox_system_isolation_validation.json"),String))
    isolation.passed || error("sandbox boundary check failed")
    collections=[
        ("julia_tasks_results",("baseline","review","feedback","feedback_docs","native_cpu"),JuliaTaskEval.JULIA_TASKS),
        ("julia_thinking_results",("baseline",),JuliaTaskEval.JULIA_TASKS),
        ("julia_sampled_results",("baseline",),JuliaTaskEval.JULIA_TASKS),
        ("julia_scene_results",("baseline",),JuliaTaskEval.JULIA_TASKS),
        ("julia_scene_v2_results",("baseline",),JuliaTaskEval.JULIA_TASKS),
        ("julia_api_v2_results",("baseline",),JuliaTaskEval.JULIA_TASKS),
        ("scene_generalization_direct",("baseline",),SCENE_GENERALIZATION_TASKS),
        ("scene_generalization_adapter",("baseline",),SCENE_GENERALIZATION_TASKS),
        ("julia_jet_v4_results",("repair",),JuliaTaskEval.JULIA_TASKS),
        ("julia_kernel_results",("repair",),JuliaTaskEval.JULIA_TASKS)]
    jobs=Any[];outdir=joinpath(@__DIR__,"system_harness_rescores");mkpath(outdir)
    for (collection,arms,tasks) in collections,task in tasks,arm in arms
        source=joinpath(@__DIR__,collection,task.id*"_"*arm*".json")
        isfile(source) || continue
        push!(jobs,(;collection,task,arm,source))
    end
    queue=Channel{Any}(length(jobs))
    foreach(job->put!(queue,job),jobs);close(queue)
    rows=Any[];row_lock=ReentrantLock()
    function consume()
        for job in queue
            candidate=JSON3.read(read(job.source,String))
            hasproperty(candidate,:contract_sha256) && candidate.contract_sha256!=bytes2hex(sha256(job.task.tests)) && error("contract drift")
            code=hasproperty(candidate,:code) ? String(candidate.code) : JuliaTaskEval.extract_code(String(candidate.answer))
            oldpath=replace(job.source,".json"=>"_score.json")
            previous=isfile(oldpath) ? JSON3.read(read(oldpath,String)) : nothing
            result=if hasproperty(candidate,:validation_error) && candidate.validation_error!==nothing
                (;id=job.task.id,passed=false,diagnostic=String(candidate.validation_error),elapsed_seconds=0.0,
                    code_sha256=bytes2hex(sha256(code)),depot_order="withsystem")
            else
                JuliaTaskEval.evaluate(job.task,code)
            end
            row=(;collection=job.collection,task=job.task.id,arm=job.arm,
                source_sha256=bytes2hex(sha256(read(job.source))),
                previous_passed=previous===nothing ? nothing : previous.passed,
                changed=previous===nothing ? nothing : previous.passed!=result.passed,result...)
            dir=joinpath(outdir,job.collection);mkpath(dir)
            open(io->JSON3.pretty(io,row),joinpath(dir,job.task.id*"_"*job.arm*"_score.json"),"w")
            lock(row_lock) do;push!(rows,row);end
            println("Rescore ",job.collection,"/",job.task.id,"/",job.arm," ",result.passed ? "PASS" : "FAIL");flush(stdout)
        end
    end
    tasks=[Threads.@spawn consume() for _ in 1:workers]
    foreach(fetch,tasks)
    sort!(rows;by=r->(r.collection,r.arm,r.task))
    groups=Any[]
    for (collection,arm) in sort!(unique([(r.collection,r.arm) for r in rows]))
        selected=filter(r->r.collection==collection&&r.arm==arm,rows)
        push!(groups,(;collection,arm,passed=count(r->r.passed,selected),total=length(selected)))
    end
    report=(;scope="same retained model candidates, corrected sandbox system-depot configuration; no regeneration or extraction changes",
        legacy_policy="original scores retained; missing old scores are not counted as changes",
        changed=[(;collection=r.collection,task=r.task,arm=r.arm,previous=r.previous_passed,current=r.passed) for r in rows if r.changed===true],
        workers,groups,rows,
        harness_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in ("julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),joinpath(outdir,"summary.json"),"w")
    report
end
