# Caller owns the model. Fill only predeclared missing cases.
function finish_julia_thinking(model,tokenizer,deadline)
    completed=String[]
    for task in JuliaTaskEval.JULIA_TASKS
        task.category=="kernel" && continue
        path=joinpath(@__DIR__,"julia_thinking_results",task.id*"_baseline.json")
        isfile(path) && continue
        time()<deadline-95 || return (;complete=false,completed)
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.","user"=>task.prompt]
        answer=generate_julia_answer(model,tokenizer,messages;thinking=true,max_tokens=2048,output_chunk=128)
        row=(;task=task.id,category=task.category,thinking=true,arm="baseline",model="qwen35-4b",
            scope="greedy thinking/output-budget control; output-batch EOS polling; not a tool gain",
            contract_sha256=bytes2hex(sha256(task.tests)),messages,answer...,
            generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                ("finish_julia_thinking.jl","run_julia_tasks.jl","qwen35.jl","prefill_qwen35.jl","qwen_text.jl","models.toml")))
        open(io->JSON3.pretty(io,row),path,"w");push!(completed,path)
        println(task.id," missing thinking control saved");flush(stdout)
    end
    (;complete=true,completed)
end
