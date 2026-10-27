# Trusted model generation only. Generated code is NEVER evaluated in this
# process: julia_eval.jl executes it using the separate sandbox runner.
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")
isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:JuliaDocsRetrieval) || include("julia_docs_retrieval.jl")
isdefined(@__MODULE__,:textsession) || include("text_backbone.jl")
using JSON3, SHA
import Mantle

"""Generate with device-resident token feedback.

The fixed-budget default reads output only once. Optional output_chunk polls
completed OUTPUT batches to stop at EOS; it never reads a token to feed the next
model step. Every step inside a batch still uses the same recorded decode graph.
Callbacks support opt-in evidence/sampling policies without changing defaults.
"""
function generate_julia_answer(model,tokenizer,messages;max_tokens=512,thinking=false,
        output_chunk=nothing,logit_hook=nothing,before_last_prompt=nothing,after_output=nothing)
    T=SmallLLMChecks.QwenText
    ids=T.chat_ids(tokenizer,messages;thinking)
    capacity=nextpow(2,length(ids)+max_tokens+32)
    capacity<=maxcapacity(model) || error("prompt exceeds experiment capacity")
    output_chunk===nothing || output_chunk>0 || throw(ArgumentError("output chunk must be positive"))
    s=textsession(model;capacity,logit_hook)
    started=time()
    try
        if before_last_prompt===nothing
            ingest!(s,ids)
        else
            # A sampler must not consume RNG draws on intermediate prompt-chunk
            # predictions. Activate it only for the final known input token.
            ingest!(s,ids[1:end-1])
            before_last_prompt()
            feed!(s,last(ids))
        end
        generated=Int[];executed=0;decode_submissions=0
        while executed<max_tokens
            count=output_chunk===nothing ? max_tokens : min(output_chunk,max_tokens-executed)
            before=Mantle.submissions(model.device)
            generate!(s,count)
            decode_submissions+=Mantle.submissions(model.device)-before
            # ResourceView is a shader argument, not an AbstractArray with a
            # host Array conversion. Read the small owned history buffer using
            # the supported API and select this completed output batch on CPU.
            first_output=length(ids)+executed
            append!(generated,Int.(Array(s.predictions)[first_output:first_output+count-1]))
            executed+=count
            output_chunk===nothing && break
            any(t->t==tokenizer.bpe.eos||t==tokenizer.bpe.bos,generated) && break
        end
        after_output===nothing || after_output()
        decode_submissions==executed || error("expected one recorded GPU submission per decode step")
        stop=findfirst(t->t==tokenizer.bpe.eos||t==tokenizer.bpe.bos,generated)
        answer_ids=stop===nothing ? generated : generated[1:stop-1]
        answer=T.decode(tokenizer,answer_ids;skipspecial=false)
        (;answer,code=JuliaTaskEval.extract_code(answer;thinking),input_tokens=length(ids),
          generated_tokens=length(answer_ids),budget_tokens=max_tokens,
          decode_steps_executed=executed,decode_submissions,eos_output_poll_chunk=output_chunk,
          finished_thinking=!thinking||occursin("</think>",answer),
          truncated=stop===nothing,elapsed_seconds=time()-started,
          input_sha256=bytes2hex(sha256(JSON3.write(messages))))
    finally
        close(s)
    end
end

function run_julia_baseline(;seconds=270,max_tokens=512,thinking=false,model_dir="gen/qwen35-4b",tasks=JuliaTaskEval.JULIA_TASKS,
                            output_dir=joinpath(@__DIR__,"julia_tasks_results"))
    mkpath(output_dir)
    let
        started=time(); deadline=started+seconds
        device=Mantle.Device("radeon"); model=nothing
        try
            model=loadbackbone(model_dir;device)
            tokenizer=loadtokenizer(model_dir)
            for task in tasks
                time()<deadline-25 || error("bounded generation slot exhausted; saved completed tasks")
                path=joinpath(output_dir,task.id*"_baseline.json")
                isfile(path) && continue
                messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>task.prompt]
                answer=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking)
                result=(;task=task.id,category=task.category,arm="baseline",model=basename(model_dir),describe(model)...,thinking,
                    contract_sha256=bytes2hex(sha256(task.tests)),prompt_sha256=bytes2hex(sha256(task.prompt)),messages,answer...)
                open(io->JSON3.pretty(io,result),path,"w")
                println(task.id," saved; ",answer.generated_tokens," tokens; truncated=",answer.truncated); flush(stdout)
            end
            (;elapsed_seconds=time()-started,output_dir)
        finally
            try
                model===nothing || close(model)
            finally
                Mantle.waitidle(device)
            end
        end
    end
end

function score_julia_arm(arm="baseline";tasks=JuliaTaskEval.JULIA_TASKS,
                         validation_file=joinpath(@__DIR__,"julia_harness_validation.json"),
                         output_dir=joinpath(@__DIR__,"julia_tasks_results"))
    rows=Any[]
    validation=JSON3.read(read(validation_file,String))
    for task in tasks
        source=joinpath(output_dir,task.id*"_"*arm*".json")
        isfile(source) || continue
        reference=only(filter(r->r.id==task.id,validation.references))
        if !reference.passed
            println(task.id," UNSCORED: reference infrastructure not qualified");flush(stdout)
            continue
        end
        generated=JSON3.read(read(source,String))
        code=hasproperty(generated,:code) ? String(generated.code) : JuliaTaskEval.extract_code(String(generated.answer))
        row=JuliaTaskEval.evaluate(task,code;
            output=joinpath(output_dir,task.id*"_"*arm*"_score.json"))
        if arm=="baseline"
            probe=JuliaTaskEval.diagnose(task,code)
            open(io->JSON3.pretty(io,probe),joinpath(output_dir,task.id*"_public_probe.json"),"w")
        end
        push!(rows,row)
        println(task.id," ",row.passed ? "PASS" : "FAIL"); flush(stdout)
    end
    rows
end

function task_api_evidence(task)
    root="/home/sim/Programmieren/VulkanDev"
    sources=if task.category=="kernel"
        [(joinpath(root,"dev/KernelAbstractions/src/KernelAbstractions.jl"),1:78)]
    elseif task.category=="makie"
        task.id=="makie_band" ? [
            (joinpath(root,"dev/Makie/docs/src/reference/plots/band.md"),1:70),
            (joinpath(root,"dev/Makie/docs/src/reference/blocks/legend.md"),1:80)] : [
            (joinpath(root,"dev/Makie/docs/src/reference/blocks/axis.md"),1:70)]
    else
        Tuple{String,UnitRange{Int}}[]
    end
    packets=String[]
    for (path,range) in sources
        isfile(path) || continue
        lines=readlines(path); excerpt=join(lines[intersect(range,eachindex(lines))],"\n")
        push!(packets,"Local API documentation: $(relpath(path,root)); sha256=$(bytes2hex(sha256(read(path))))\n$excerpt")
    end
    if task.id=="alias_stencil"
        push!(packets,sprint(show,MIME("text/plain"),Base.Docs.doc(Base.mightalias)))
        push!(packets,sprint(show,MIME("text/plain"),Base.Docs.doc(Base.copy)))
    elseif task.id=="missing_windows"
        push!(packets,sprint(show,MIME("text/plain"),Base.Docs.doc(Base.skipmissing)))
    elseif task.id=="stable_groups"
        push!(packets,sprint(show,MIME("text/plain"),Base.Docs.doc(Base.Dict)))
    end
    join(packets,"\n\n")
end

function run_julia_repair(;seconds=270,max_tokens=512,model_dir="gen/qwen35-4b",
                          tasks=JuliaTaskEval.JULIA_TASKS,
                          arms=("review","feedback","feedback_docs"),
                          after_jobs=nothing,
                          output_dir=joinpath(@__DIR__,"julia_tasks_results"))
    # Prepare retrieval on CPU before GPU work.
    index=JuliaDocsRetrieval.build_index()
    let
        started=time();deadline=started+seconds
        device=Mantle.Device("radeon");model=nothing
        try
            model=loadbackbone(model_dir;device)
            tokenizer=loadtokenizer(model_dir)
            for task in tasks
                scorepath=joinpath(output_dir,task.id*"_baseline_score.json")
                isfile(scorepath) || continue
                JSON3.read(read(scorepath,String)).passed && continue
                baseline=JSON3.read(read(joinpath(output_dir,task.id*"_baseline.json"),String))
                probe=JSON3.read(read(joinpath(output_dir,task.id*"_public_probe.json"),String))
                for arm in arms
                    arm in ("review","feedback","feedback_docs") || error("unknown arm")
                    path=joinpath(output_dir,task.id*"_"*arm*".json")
                    isfile(path) && continue
                    time()<deadline-30 || return (;complete=false,reason="bounded repair slot exhausted; completed arms saved",elapsed_seconds=time()-started,output_dir)
                    request="Review your previous solution against the original requirements. Return a complete corrected solution, code only."
                    if arm!="review"
                        # Public probe only: no private test assertions or expected
                        # values enter the prompt, even after failed scoring.
                        request*="\nA public execution probe was run:\n"*JuliaTaskEval.PUBLIC_PROBES[task.id]*"\nJulia output:\n"*first(String(probe.diagnostic),4000)
                    end
                    if arm=="feedback_docs"
                        evidence=task.category=="base" ? task_api_evidence(task) : ""
                        isempty(evidence) && (evidence=JuliaDocsRetrieval.format_evidence(
                            JuliaDocsRetrieval.retrieve(index,task.prompt*"\n"*first(String(probe.diagnostic),2000))))
                        request*="\nRelevant installed API documentation:\n"*evidence
                    end
                    messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle specified edge cases.",
                              "user"=>task.prompt,"assistant"=>String(baseline.answer),"user"=>request]
                    answer=generate_julia_answer(model,tokenizer,messages;max_tokens)
                    result=(;task=task.id,category=task.category,arm,model=basename(model_dir),thinking=false,
                        describe(model)...,
                        baseline_source=task.id*"_baseline.json",messages,answer...)
                    open(io->JSON3.pretty(io,result),path,"w")
                    println(task.id," ",arm," saved; truncated=",answer.truncated);flush(stdout)
                end
            end
            followup=after_jobs===nothing ? nothing : after_jobs(model,tokenizer,deadline)
            (;complete=true,elapsed_seconds=time()-started,output_dir,followup)
        finally
            try
                model===nothing || close(model)
            finally
                Mantle.waitidle(device)
            end
        end
    end
end

"""Separate reasoning-budget control, never pooled with the 512-token repair arms.

Only runs if a conservative margin remains in the SAME reservation/model load.
Private tests and reference implementations are not used in the prompt.
"""
function julia_thinking_probe(model,tokenizer,deadline;
        task=first(JuliaTaskEval.JULIA_TASKS),max_tokens=2048,model_name="qwen35-4b",
        output_dir=joinpath(@__DIR__,"julia_thinking_results"))
    path=joinpath(output_dir,task.id*"_baseline.json")
    isfile(path) && return (;status="already saved",path)
    time()+max_tokens/30+25<deadline || return (;status="deferred: insufficient time budget remains")
    messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>task.prompt]
    answer=generate_julia_answer(model,tokenizer,messages;max_tokens,thinking=true)
    mkpath(output_dir)
    result=(;task=task.id,category=task.category,arm="baseline",model=model_name,thinking=true,
        sampling="greedy",scope="separate reasoning/output-budget control, not matched compute or a tool-assisted gain",
        describe(model)...,
        contract_sha256=bytes2hex(sha256(task.tests)),prompt_sha256=bytes2hex(sha256(task.prompt)),messages,answer...)
    open(io->JSON3.pretty(io,result),path,"w")
    println(task.id," thinking control saved; finished_thinking=",answer.finished_thinking);flush(stdout)
    (;status="saved",path)
end
