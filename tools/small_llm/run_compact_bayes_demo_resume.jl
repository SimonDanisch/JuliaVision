include("run_julia_stencil_transfer.jl")
include("compact_bayes_demo_interruption.jl")

function run_compact_bayes_demo_resume(;directory=joinpath(@__DIR__,"compact_bayes_demo_results"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    state=julia_stencil_transfer_state(directory);p=state.packet;b=p.decoding
    amendment=JSON3.read(read(joinpath(directory,"interruption_amendment.json"),String))
    amendment.conditions_sha256==julia_transfer_digest(state.path)||error("Amendment registration drift")
    for (f,h) in pairs(amendment.runner_source_sha256)
        basename(String(f))==String(f)&&julia_transfer_digest(joinpath(@__DIR__,String(f)))==h||error("Resume source drift")
    end
    gate=joinpath(directory,"recovery_gate.json");g=JSON3.read(read(gate,String))
    g.complete&&g.normal_final_waitidle&&g.host_checks==39&&g.device_checks==47&&julia_transfer_digest(gate)==amendment.recovery_gate_sha256||error("Fresh device recovery gate required")
    julia_transfer_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("Model drift")
    pending=compact_bayes_demo_pending(directory,p,julia_transfer_digest(state.path))
    isempty(pending)&&return (;complete=true)
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
