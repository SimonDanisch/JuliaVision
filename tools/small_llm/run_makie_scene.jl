# Model-generated scene JSON is validated before compilation, and compiled code
# still runs only in the same sandbox as direct Julia candidates.
isdefined(@__MODULE__,:MakieSceneAdapter) || include("makie_scene_adapter.jl")
isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")

function run_makie_scene(;seconds=240,model_dir="gen/qwen35-4b",thinking=false,
        schema=MakieSceneAdapter.SCHEMA,version="v1",tasks=JuliaTaskEval.JULIA_TASKS,
        output_dir=nothing,after_jobs=nothing)
    seconds>0 || throw(ArgumentError("time budget"))
    qualification=JSON3.read(read(joinpath(@__DIR__,"makie_scene_adapter_validation.json"),String))
    length(qualification.execution)==2 && all(r->r.passed,qualification.execution) || error("adapter execution unqualified")
    qualification.adapter_sha256==bytes2hex(sha256(read(joinpath(@__DIR__,"makie_scene_adapter.jl")))) || error("adapter changed since qualification")
    occursin(r"^v[0-9]+$",version) || throw(ArgumentError("invalid version"))
    outdir=output_dir===nothing ? joinpath(@__DIR__,version=="v1" ? "julia_scene_results" : "julia_scene_"*version*"_results") : output_dir
    mkpath(outdir)
    let
        started=time();deadline=started+seconds
        device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
            sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                ("run_makie_scene.jl","run_julia_tasks.jl","qwen35.jl","qwen_text.jl","prefill_qwen35.jl","models.toml"))
            for task in tasks
                task.category=="makie" || continue
                path=joinpath(outdir,task.id*(thinking ? "_thinking" : "_baseline")*".json")
                isfile(path) && continue
                time()<deadline-95 || return (;complete=false,completed)
                inputs=hasproperty(task,:inputs) ? task.inputs : task.id=="makie_link" ? ["x","y"] : ["x","mu","sigma"]
                messages=["system"=>"Describe the requested plot using this declarative schema. "*schema,
                    "user"=>task.prompt*"\nInstead of Julia code, return scene JSON. Available input names: "*join(inputs,", ")]
                answer=generate_julia_answer(model,tokenizer,messages;thinking,max_tokens=thinking ? 2048 : 512,output_chunk=128)
                # Use final response only; do not search for a later valid object.
                raw=answer.answer
                if thinking
                    closing=findlast("</think>",raw)
                    raw=closing===nothing ? "" : raw[nextind(raw,last(closing)):end]
                end
                raw=strip(raw)
                fenced=match(r"^```(?:json)?\s*\n(.*?)\n```\s*$"s,raw)
                fenced===nothing || (raw=fenced.captures[1])
                code="";validation_error=nothing
                try
                    code=MakieSceneAdapter.compile_scene(raw;inputs)
                catch err
                    err isa ArgumentError || rethrow()
                    validation_error=sprint(showerror,err)
                end
                row=(;task=task.id,scope="constrained scene workflow; adapter supplies API syntax, not intrinsic coding gain",version,
                    model=basename(model_dir),thinking,inputs,messages,answer...,code,scene_json=raw,validation_error,
                    contract_sha256=bytes2hex(sha256(task.tests)),adapter_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"makie_scene_adapter.jl")))),
                    generation_source_sha256=sources)
                open(io->JSON3.pretty(io,row),path,"w");push!(completed,path)
                println(task.id," scene valid=",validation_error===nothing);flush(stdout)
            end
            extra=after_jobs===nothing ? nothing : after_jobs(model,tokenizer,deadline)
            (;complete=true,completed,extra)
        finally
            try
                model===nothing || close(model)
            finally
                Mantle.waitidle(device)
            end
        end
    end
end

function score_makie_scenes(;thinking=false)
    rows=Any[];outdir=joinpath(@__DIR__,"julia_scene_results")
    for task in JuliaTaskEval.JULIA_TASKS
        task.category=="makie" || continue
        path=joinpath(outdir,task.id*(thinking ? "_thinking" : "_baseline")*".json")
        isfile(path) || continue
        candidate=JSON3.read(read(path,String))
        result=candidate.validation_error===nothing ? JuliaTaskEval.evaluate(task,String(candidate.code)) :
            (;id=task.id,passed=false,diagnostic=String(candidate.validation_error))
        push!(rows,result)
        open(io->JSON3.pretty(io,result),replace(path,".json"=>"_score.json"),"w")
    end
    rows
end
