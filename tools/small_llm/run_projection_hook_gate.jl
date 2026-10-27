include("verify_projection_model_hook.jl")

function run_projection_hook_gate(data;seconds=240)
    seconds>0 || throw(ArgumentError("time budget"))
    require_post_extent_gate()
    output=joinpath(@__DIR__,"projection_model_hook_validation.json")
    isfile(output) && error("preserve existing hook gate attempt")
    Q=SmallLLMChecks.Qwen35Experiment;cache=qualified_qwen_cache()
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b")
    let
        started=time();device=Mantle.Device("radeon");model=nothing;report=nothing
        try
            model=Q.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            time()<started+seconds-90 || error("hook gate cleanup margin")
            result=verify_projection_model_hook(model,tokenizer,data)
            report=(;complete=false,result,datastore_registration_sha256=data.registration_sha256,
                source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
                    ("run_projection_hook_gate.jl","verify_projection_model_hook.jl","projection_memory_generation.jl")),
                elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
        # A gate is complete only AFTER successful final synchronization.
        report=merge(report,(;complete=true,normal_final_waitidle=true))
        open(io->JSON3.pretty(io,report),output,"w");report
    end
end
