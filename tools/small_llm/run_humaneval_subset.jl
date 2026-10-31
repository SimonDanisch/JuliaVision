isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")
isdefined(@__MODULE__,:load_frozen_humaneval_tasks) || include("humaneval_subset.jl")
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")

function prepare_humaneval_conditions()
    directory=joinpath(@__DIR__,"humaneval_subset_results")
    output=joinpath(directory,"conditions.json")
    isfile(output) && error("conditions already frozen")
    qualification=JSON3.read(read(joinpath(directory,"qualification_v2.json"),String))
    qualification.selected==12 && qualification.qualified==12 || error("all frozen references must qualify")
    report=(;dataset_revision=HUMANEVAL_REVISION,
        selection_sha256=bytes2hex(sha256(read(joinpath(directory,"registration.json")))),
        qualification_sha256=bytes2hex(sha256(read(joinpath(directory,"qualification_v2.json")))),
        baseline=(;thinking=true,decoding="greedy",max_tokens=2048,output_poll_chunk=128),
        planned_revision=(;trigger="new positive parser/dispatch/JET diagnostics at public docstring signatures only",
            arms=["self_review","jet_feedback"],thinking=true,max_tokens=2048,decoding="greedy",attempts_per_arm=1),
        constraints=["no replacing tasks, no regenerating completed candidates, no private assertions or reference bodies in prompts",
            "public benchmark, possible training contamination; fixed12-task basic-test diagnostic, NOT full HumanEval/EvalPlus or frontier evidence",
            "greedy2048-token development setting is not Qwen's recommended sampled/long-output benchmark configuration"],
        source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,report),output,"w")
    report
end

function qualified_qwen_cache()
    path=joinpath(@__DIR__,"qwen_quantcache_full_validation.json")
    qualification=JSON3.read(read(path,String))
    qualification.complete && qualification.checks.all_weight_bytes_equal && qualification.checks.exact_greedy ||
        error("full-model cache qualification incomplete")
    for file in ("qwen35.jl","qwen_weight_cache.jl")
        bytes2hex(sha256(read(joinpath(@__DIR__,file))))==qualification.source_sha256[Symbol(file)] ||
            error("cache runner changed since qualification")
    end
    String(qualification.cachebase)
end

function run_humaneval_baselines(;seconds=240,model_dir="gen/qwen35-4b",
        directory=joinpath(@__DIR__,"humaneval_subset_results"),generate_answer=generate_julia_answer,
        extra_source_files=())
    seconds>0 || throw(ArgumentError("time budget"))
    conditions_path=joinpath(directory,"conditions.json")
    conditions=JSON3.read(read(conditions_path,String))
    conditions.baseline.max_tokens==2048 && conditions.baseline.thinking || error("condition drift")
    conditions.selection_sha256==bytes2hex(sha256(read(joinpath(@__DIR__,"humaneval_subset_results","registration.json")))) || error("selection drift")
    (conditions.baseline.decoding=="greedy")===(generate_answer===generate_julia_answer) || error("decoding provider drift")
    tasks=load_frozen_humaneval_tasks();cache=qualified_qwen_cache()
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            runtime=inference_runtime_provenance()
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device,quantcache=cache)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
            for task in tasks
                path=joinpath(directory,task.id*"_baseline.json")
                isfile(path) && continue
                time()<deadline-75 || return (;complete=false,completed)
                messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>task.prompt]
                answer=generate_answer(model,tokenizer,messages;max_tokens=2048,thinking=true,output_chunk=128)
                row=(;task=task.id,arm="baseline",messages,thinking=true,runtime_provenance=runtime,
                    contract_sha256=bytes2hex(sha256(task.tests)),
                    conditions_sha256=bytes2hex(sha256(read(conditions_path))),
                    generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                        ("run_humaneval_subset.jl","humaneval_subset.jl","run_julia_tasks.jl","qwen35.jl",
                         "qwen_weight_cache.jl","qwen_text.jl","prefill_qwen35.jl","device_finish.jl",extra_source_files...)),answer...)
                open(io->JSON3.pretty(io,row),path,"w");push!(completed,path)
                println(task.id," external baseline saved");flush(stdout)
            end
            (;complete=true,completed,elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end

function score_humaneval_baselines(;directory=joinpath(@__DIR__,"humaneval_subset_results"))
    rows=Any[]
    for task in load_frozen_humaneval_tasks()
        path=joinpath(directory,task.id*"_baseline.json")
        if !isfile(path)
            push!(rows,(;task=task.id,status="pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(path,".json"=>"_score.json"))
        push!(rows,(;task=task.id,status=score.passed ? "passed" : "failed",truncated=candidate.truncated,
            finished_thinking=candidate.finished_thinking,generation_seconds=candidate.elapsed_seconds))
    end
    report=(;scope="fixed12-task public HumanEval.jl subset, basic tests only, possible training contamination",
        rows,passed=count(r->r.status=="passed",rows),evaluated=count(r->r.status!="pending",rows),
        pending=count(r->r.status=="pending",rows))
    open(io->JSON3.pretty(io,report),joinpath(directory,"baseline_summary.json"),"w")
    report
end
