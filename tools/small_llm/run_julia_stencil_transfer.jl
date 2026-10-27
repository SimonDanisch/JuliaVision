include("laptop_pending_study.jl")
include("julia_stencil_transfer_study.jl")

function run_julia_stencil_transfer(;directory=joinpath(@__DIR__,"julia_stencil_transfer_results"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    state=julia_stencil_transfer_state(directory);p=state.packet;b=p.decoding
    julia_transfer_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("Model drift")
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    any(isfile(joinpath(directory,String(j.id)*"_attempt.json")) for j in pending)&&error("Interrupted attempt; no regeneration")
    job=first(pending);messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    length(SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking=false))+b.max_tokens+32<=8192||error("Context overflow")
    let
        let
            output=joinpath(directory,String(job.id)*".json");attempt=joinpath(directory,String(job.id)*"_attempt.json")
            (ispath(output)||ispath(attempt))&&error("Preserve attempt")
            open(io->JSON3.pretty(io,(;id=job.id,pid=getpid(),conditions_sha256=julia_transfer_digest(state.path),started_unix_seconds=time())),attempt,"w")
            device=Mantle.Device("radeon");model=nothing
            row=try
                model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
                answer=sampled_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=false,
                    seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),top_p=Float32(b.top_p),
                    presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                (;answer...,id=job.id,messages,conditions_sha256=julia_transfer_digest(state.path),
                    attempt_sha256=julia_transfer_digest(attempt),runtime_provenance=inference_runtime_provenance())
            catch err
                (;id=job.id,messages,inference_error=sprint(showerror,err),conditions_sha256=julia_transfer_digest(state.path),attempt_sha256=julia_transfer_digest(attempt))
            finally
                try model===nothing||close(model)
                finally finish_device!(device) end
            end
            row=(;row...,normal_final_waitidle=true)
            open(io->JSON3.pretty(io,row),output,"w");(;complete=false,id=job.id)
        end
    end
end
