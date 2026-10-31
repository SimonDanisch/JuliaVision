include("run_pubmedqa_reading.jl")

function pubmedqa_retrieval_pending(directory,jobs)
    all(j->occursin(r"^pubmed_[0-9]+_(question_only|corresponding_context|mismatched_context|retrieved_context)$",String(j.id)),jobs)||throw(ArgumentError("Safe four-arm IDs"))
    length(unique(String(j.id) for j in jobs))==length(jobs)||throw(ArgumentError("Duplicate IDs"))
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),jobs)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending)&&error("Interrupted attempt; no regeneration")
    pending
end

"""Four-arm extension of the frozen source-reading host runner.
Same model/session/sampler arithmetic; new selector accepts the retrieval arm.
"""
function run_pubmedqa_retrieval(;directory=joinpath(@__DIR__,"pubmedqa_retrieval_results"),seconds=210,max_jobs=4)
    seconds>0&&1<=max_jobs<=4||throw(ArgumentError("time budget"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    state=julia_stencil_transfer_state(directory);p=state.packet;b=p.decoding
    b.max_tokens==128&&!b.thinking||error("Frozen label budget")
    julia_transfer_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("Model drift")
    pending=pubmedqa_retrieval_pending(directory,p.jobs)
    isempty(pending)&&return (;complete=true,completed=String[])
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    let
        let
            started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
            try
                model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
                for job in pending[1:min(end,max_jobs)]
                    time()<started+seconds-75||break
                    messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
                    length(SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking=false))+b.max_tokens+32<=8192||error("Context overflow; no filtering/retry")
                    output=joinpath(directory,String(job.id)*".json");attempt=joinpath(directory,String(job.id)*"_attempt.json")
                    (ispath(output)||ispath(attempt))&&error("Preserve attempt")
                    open(io->JSON3.pretty(io,(;id=job.id,pid=getpid(),conditions_sha256=julia_transfer_digest(state.path),started_unix_seconds=time())),attempt,"w")
                    answer=try
                        sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=false,
                            seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),top_p=Float32(b.top_p),
                            presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                    catch err
                        (;inference_error=sprint(showerror,err))
                    end
                    Mantle.waitidle(device)
                    row=(;answer...,id=job.id,messages,conditions_sha256=julia_transfer_digest(state.path),
                        attempt_sha256=julia_transfer_digest(attempt),normal_final_waitidle=true,
                        runtime_provenance=inference_runtime_provenance())
                    open(io->JSON3.pretty(io,row),output,"w");push!(completed,String(job.id))
                    hasproperty(answer,:inference_error)&&break
                end
                (;complete=length(completed)==length(pending),completed)
            finally
                try model===nothing||close(model)
                finally finish_device!(device) end
            end
        end
    end
end
