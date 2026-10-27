# Include from the root environment. Does not create a device until called.
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")
include("run_julia_tasks.jl")
include("test_gpu_sampling.jl")
include("sampled_julia.jl")

function run_julia_modes_window(;seconds=240,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    let
        started=time();deadline=started+seconds
        device=Mantle.Device("radeon");model=nothing
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("qwen35.jl","prefill_qwen35.jl","qwen_text.jl","gpu_sampling.jl","test_gpu_sampling.jl",
             "sampled_julia.jl","run_julia_tasks.jl","run_julia_modes_window.jl","models.toml"))
        completed=String[]
        function save(path,row)
            mkpath(dirname(path))
            open(io->JSON3.pretty(io,(;row...,generation_source_sha256=sources)),path,"w")
            push!(completed,path)
        end
        function baseline_messages(task)
            ["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>task.prompt]
        end
        function answer_row(task,answer;thinking=true,scope)
            (;task=task.id,category=task.category,arm="baseline",model=basename(model_dir),thinking,scope,
              cache="Float32 split",quantization="Q4_K/Q8 activations; Q6_K head",
              contract_sha256=bytes2hex(sha256(task.tests)),prompt_sha256=bytes2hex(sha256(task.prompt)),
              messages=baseline_messages(task),answer...)
        end
        try
            arithmetic=test_gpu_sampling(device)
            save(joinpath(@__DIR__,"sampling_gpu_validation.json"),(;device="Radeon8060S",arithmetic...))
            time()<deadline-100 || return (;complete=false,reason="arithmetic qualification used window",completed)
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
            task=first(JuliaTaskEval.JULIA_TASKS)
            paritypath=joinpath(@__DIR__,"julia_thinking_results",task.id*"_chunk128.json")
            if !isfile(paritypath)
                time()<deadline-95 || return (;complete=false,reason="not enough time for parity",completed)
                answer=generate_julia_answer(model,tokenizer,baseline_messages(task);
                    thinking=true,max_tokens=2048,output_chunk=128)
                previous=JSON3.read(read(joinpath(@__DIR__,"julia_thinking_results",task.id*"_baseline.json"),String))
                parity=answer.answer==String(previous.answer)
                save(paritypath,(;answer_row(task,answer;scope="batch-output EOS parity; not a new accuracy attempt")...,
                    matches_fixed_budget_answer=parity,fixed_budget_elapsed_seconds=previous.elapsed_seconds))
                parity || error("batch-output polling changed greedy answer; investigate before evaluating modes")
                println("Batched EOS matches fixed output: ",answer.decode_steps_executed," steps; ",answer.elapsed_seconds,"s");flush(stdout)
            end
            sampledpath=joinpath(@__DIR__,"julia_sampled_results",task.id*"_baseline.json")
            if !isfile(sampledpath)
                time()<deadline-95 || return (;complete=false,reason="not enough time for sampled control",completed)
                answer=sampled_julia_answer(model,tokenizer,baseline_messages(task);thinking=true,
                    max_tokens=2048,output_chunk=128,seed=UInt32(1),topk=20,temperature=0.6f0,top_p=0.95f0)
                save(sampledpath,answer_row(task,answer;scope="one predeclared-seed thinking/sampling control; not a tool gain or seed search"))
                println("Sampled grouping saved; finished_thinking=",answer.finished_thinking);flush(stdout)
            end
            for task in JuliaTaskEval.JULIA_TASKS[[5,6,2,3]]
                path=joinpath(@__DIR__,"julia_thinking_results",task.id*"_baseline.json")
                isfile(path) && continue
                time()<deadline-95 || return (;complete=false,reason="bounded thinking window exhausted",completed)
                answer=generate_julia_answer(model,tokenizer,baseline_messages(task);thinking=true,
                    max_tokens=2048,output_chunk=128)
                save(path,answer_row(task,answer;scope="greedy thinking/output-budget control; output-batch EOS polling"))
                println(task.id," thinking saved; finished_thinking=",answer.finished_thinking);flush(stdout)
            end
            (;complete=true,completed,elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                Mantle.waitidle(device)
            end
        end
    end
end
