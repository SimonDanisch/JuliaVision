include("qwen_text.jl")
using JSON3,SHA
function prepare_bayes_cpu_reference(;directory=joinpath(@__DIR__,"bayes_extraction_results"),destination=joinpath(@__DIR__,"bayes_cpu_reference_requests.json"))
    isfile(destination)&&error("Preserve public native-control packet")
    digest(path)=bytes2hex(sha256(read(path)))
    conditions=joinpath(directory,"conditions.json");p=JSON3.read(read(conditions,String))
    p.source_sha256[Symbol("qwen_text.jl")]==digest(joinpath(@__DIR__,"qwen_text.jl"))||error("Original tokenizer source changed")
    jobs=filter(j->j.arm=="direct_probability",p.jobs)
    length(jobs)==8&&length(unique(j.task for j in jobs))==8||error("Keep ALL8 original direct prompts")
    tokenizer=QwenText.Tokenizer(joinpath(@__DIR__,"../../../../gen/qwen35-4b"))
    rows=map(jobs) do job
        pairs=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
        messages=[(;role=first(m),content=last(m)) for m in pairs]
        input_ids=QwenText.chat_ids(tokenizer,pairs;thinking=false)
        (;id=job.id,task=job.task,messages,input_ids)
    end
    result=(;scope="POSTHOC ALL8 unchanged direct requests, native FP32/greedy CPU diagnostic. Different precision AND sampler, NOT isolated Q4 ablation or replacement primary score.",
        original_conditions_sha256=digest(conditions),original_primary_summary_sha256=digest(joinpath(directory,"summary.json")),
        source_sha256=digest(@__FILE__),jobs=rows,private_answers_included=false)
    open(io->JSON3.pretty(io,result),destination,"w");(;complete=true,jobs=length(rows),requests_sha256=digest(destination))
end
