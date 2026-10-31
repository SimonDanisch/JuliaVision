# One-shot docs augmentation, not repair and not a hidden-test selection loop.
# Caller owns the model. Index contains public docs only.
function run_api_docs_v2(model,tokenizer,deadline,index)
    outdir=joinpath(@__DIR__,"julia_api_v2_results");mkpath(outdir);completed=String[]
    for task in JuliaTaskEval.JULIA_TASKS
        task.category=="makie" || continue
        path=joinpath(outdir,task.id*"_baseline.json");isfile(path) && continue
        time()<deadline-95 || return (;complete=false,completed)
        evidence=JuliaDocsRetrieval.retrieve_api(index,task.prompt)
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
            "user"=>task.prompt*"\n\nPublic local package documentation follows. Use the documented APIs rather than inventing helpers.\n"*JuliaDocsRetrieval.format_evidence(evidence)]
        answer=generate_julia_answer(model,tokenizer,messages;thinking=false,max_tokens=512,output_chunk=128)
        row=(;task=task.id,category=task.category,thinking=false,model="qwen35-4b",arm="one_shot_api_docs_v2",
            scope="revised retrieval after original negative result; authored development cases, not held-out",
            contract_sha256=bytes2hex(sha256(task.tests)),messages,evidence,answer...,
            generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                ("run_api_docs_v2.jl","julia_docs_retrieval.jl","run_julia_tasks.jl","qwen35.jl","qwen_text.jl","models.toml")))
        open(io->JSON3.pretty(io,row),path,"w");push!(completed,path)
        println(task.id," one-shot API-docs v2 saved");flush(stdout)
    end
    (;complete=true,completed)
end
