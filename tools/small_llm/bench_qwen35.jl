using JSON3, Statistics, Test
import Mantle, KernelAbstractions
using .Qwen35Experiment

module QwenTelemetry
include(joinpath(@__DIR__,"..","measure.jl"))
end

function reset_qwen35!(s)
    for a in s.owned
        a isa Mantle.Buffer && fill!(a,zero(eltype(a)))
    end
    s.token[]=Int32(0);s.posref[]=Int32(0);s.position=0
    KernelAbstractions.synchronize(s.model.device)
    s
end

function bench_qwen35(model,metadata;newtokens=128,samples=5,capacity=4096,prefill_chunk=nothing,
                      kv_type=Float32,attention=:split,deadline=Inf,output=nothing)
    # Load/record/compile and one warm generation are excluded from timing.
    s=Qwen35Experiment.session(model;capacity,kv_type,attention)
    rows=Any[];reference=nothing
    try
        println("Warming recorded Qwen decode");flush(stdout)
        if prefill_chunk===nothing
            for token in metadata.chat_ids; Qwen35Experiment.step!(s,token); end
        else
            Qwen35Experiment.prefill!(s,metadata.chat_ids;chunk=prefill_chunk)
        end
        Qwen35Experiment.decode!(s,32)
        KernelAbstractions.synchronize(model.device)
        println("Warmup complete");flush(stdout)
        @testset "Qwen repeated quantized decode" begin
        for rep in 1:samples
            time()<deadline || error("Time budget exhausted before next sample")
            reset_qwen35!(s)
                prefill=@elapsed begin
                    if prefill_chunk===nothing
                        for token in metadata.chat_ids; Qwen35Experiment.step!(s,token); end
                    else
                        Qwen35Experiment.prefill!(s,metadata.chat_ids;chunk=prefill_chunk)
                    end
                    KernelAbstractions.synchronize(model.device)
                end
                before=QwenTelemetry.gpustate()
                # The final prompt step already predicted the first output.
                first_prediction=s.position
                submissions_before=Mantle.submissions(model.device)
                timing=@timed begin
                    Qwen35Experiment.decode!(s,newtokens)
                    KernelAbstractions.synchronize(model.device)
                end
                decode_submissions=Mantle.submissions(model.device)-submissions_before
                ids=Int.(Array(s.predictions)[first_prediction:first_prediction+newtokens-1])
                after=QwenTelemetry.gpustate()
                @test all(isfinite,Array(s.logits))
                @test decode_submissions==newtokens
                reference===nothing ? (reference=ids) : (@test ids==reference)
                row=(;repetition=rep,prefill_seconds=prefill,decode_seconds=timing.time,
                       decode_gc_seconds=timing.gctime,decode_tps=newtokens/timing.time,
                       decode_submissions,before,after,token_ids=ids)
                push!(rows,row);println((repetition=rep,tps=row.decode_tps));flush(stdout)
        end
        end
    finally
        close(s)
    end
    plateau=median([min(r.before.sm,r.after.sm) for r in rows])
    accepted=[r for r in rows if min(r.before.sm,r.after.sm)>=0.9plateau]
    @assert !isempty(accepted)
    result=(;width=Int(model.config.hidden_size),layers=Int(model.config.num_hidden_layers),
             quantize=string(model.quantize),prompt_tokens=length(metadata.chat_ids),newtokens,capacity,prefill_chunk,
             kv_type=string(kv_type),attention=string(attention),
             median_decode_tps=median([r.decode_tps for r in accepted]),
             decode_tps_range=extrema([r.decode_tps for r in accepted]),
             accepted_samples=length(accepted),rejected_samples=length(rows)-length(accepted),
             project=Base.active_project(),scope="recorded Mantle decode; fixed work including EOS; prefill mode recorded separately; shared desktop",
             samples=rows)
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end
