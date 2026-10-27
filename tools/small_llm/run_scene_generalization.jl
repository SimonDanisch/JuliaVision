include("run_makie_scene.jl")
include("julia_scene_generalization_tasks.jl")
include("makie_scene_schema_v2.jl")

function direct_scene_controls(model,tokenizer,deadline)
    outdir=joinpath(@__DIR__,"scene_generalization_direct");mkpath(outdir);completed=String[]
    for task in SCENE_GENERALIZATION_TASKS
        path=joinpath(outdir,task.id*"_baseline.json");isfile(path) && continue
        time()<deadline-95 || return (;complete=false,completed)
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>task.prompt]
        answer=generate_julia_answer(model,tokenizer,messages;thinking=false,max_tokens=512,output_chunk=128)
        row=(;task=task.id,model="qwen35-4b",thinking=false,scope="fresh authored combination tasks; paired direct-code control, single greedy attempt",
            contract_sha256=bytes2hex(sha256(task.tests)),messages,answer...,
            generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                ("run_scene_generalization.jl","run_julia_tasks.jl","qwen35.jl","prefill_qwen35.jl","qwen_text.jl","models.toml")))
        open(io->JSON3.pretty(io,row),path,"w");push!(completed,path)
        println(task.id," direct control saved");flush(stdout)
    end
    (;complete=true,completed)
end

function register_scene_generalization()
    qualified=JSON3.read(read(joinpath(@__DIR__,"scene_generalization_harness_validation.json"),String))
    length(qualified.references)==2 && all(r->r.passed,qualified.references) && qualified.mutant_rejected || error("unqualified cases")
    row=(;scope="two fresh authored combinations, not a broad held-out benchmark; freeze v2 before generation",
        policy="one non-thinking greedy attempt per case per arm, 512 output tokens; same model/cache; schema supplies extra input tokens/compiler; no retuning after results",
        schema_sha256=bytes2hex(sha256(SCENE_SCHEMA_V2)),
        cases=[(;id=t.id,prompt_sha256=bytes2hex(sha256(t.prompt)),contract_sha256=bytes2hex(sha256(t.tests))) for t in SCENE_GENERALIZATION_TASKS])
    path=joinpath(@__DIR__,"scene_generalization_registration.json")
    if isfile(path)
        JSON3.write(JSON3.read(read(path,String)))==JSON3.write(row) || error("registration drift")
    else
        open(io->JSON3.pretty(io,row),path,"w")
    end
    row
end

function run_scene_generalization_window(;seconds=240)
    register_scene_generalization()
    run_makie_scene(;seconds,version="v2",schema=SCENE_SCHEMA_V2,tasks=SCENE_GENERALIZATION_TASKS,
        output_dir=joinpath(@__DIR__,"scene_generalization_adapter"),after_jobs=direct_scene_controls)
end

function score_scene_generalization()
    rows=Any[]
    for arm in ("direct","adapter"),task in SCENE_GENERALIZATION_TASKS
        dir=joinpath(@__DIR__,"scene_generalization_"*arm)
        path=joinpath(dir,task.id*"_baseline.json");score=replace(path,".json"=>"_score.json")
        if !isfile(path)
            push!(rows,(;arm,task=task.id,status="pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        result=if hasproperty(candidate,:validation_error) && candidate.validation_error!==nothing
            (;passed=false,diagnostic=candidate.validation_error)
        else
            JuliaTaskEval.evaluate(task,String(candidate.code))
        end
        open(io->JSON3.pretty(io,result),score,"w")
        push!(rows,(;arm,task=task.id,status=result.passed ? "passed" : "failed",input_tokens=candidate.input_tokens,
            generated_tokens=candidate.generated_tokens,budget_tokens=candidate.budget_tokens,generation_seconds=candidate.elapsed_seconds))
        println(arm," ",task.id," ",result.passed ? "PASS" : "FAIL");flush(stdout)
    end
    report=(;scope="small authored generalization diagnostic, frozen schema, no task-specific compiler rule; not frontier or aesthetic quality",rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"scene_generalization_summary.json"),"w")
    report
end
