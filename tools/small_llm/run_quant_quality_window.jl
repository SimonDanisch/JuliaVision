# Load in the workspace root environment. Inclusion does not create a device.
include("device_finish.jl")
isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
using JSON3
import Mantle

function run_quant_quality_window(;seconds=270,model_dir="gen/qwen35-4b",output_dir=@__DIR__)
    seconds>0 || throw(ArgumentError("time budget"))
    let
        started=time();deadline=started+seconds
        device=Mantle.Device("radeon")
        model=nothing
        try
            quant=SmallLLMChecks.test_qwen35_quant_experts(device)
            open(io->JSON3.pretty(io,(device="Radeon 8060S",scope="selected-expert and complete synthetic Q4 block",
                rows=quant,source_sha256=SmallLLMChecks.source_hashes())),joinpath(output_dir,"moe_quant_gpu_validation.json"),"w")
            time()<deadline || error("GPU budget exhausted after kernel checks")
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device)
            corpus=JSON3.read(read(joinpath(output_dir,"attention_quality_corpus.json"),String))
            completed=String[]
            for (i,segment) in enumerate(corpus.segments)
                path=joinpath(output_dir,"attention_quality_segment_$i.json")
                SmallLLMChecks.compare_qwen35_attention_quality(model,Int.(segment.input_ids);output=path,deadline)
                push!(completed,path)
            end
            result=(;scope="fixed local short-context diagnostic, not general reasoning quality",
                completed_segments=completed,elapsed_seconds=time()-started,
                corpus="attention_quality_corpus.json",source_sha256=SmallLLMChecks.source_hashes())
            open(io->JSON3.pretty(io,result),joinpath(output_dir,"attention_quality_window.json"),"w")
            result
        finally
            try
                model===nothing || close(model)
            finally
                Mantle.waitidle(device)
            end
        end
    end
end
