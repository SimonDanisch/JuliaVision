using Test,LinearAlgebra,JSON3
import Mantle,KernelAbstractions

function verify_qwen35_attention_modes(model,metadata;output=nothing)
    Q=Qwen35Experiment
    baseline=Any[];rows=Any[]
    @testset "Full-model cache/attention ablation" begin
        for (T,mode) in ((Float32,:split),(Float32,:grouped),(Float16,:split),(Float16,:grouped))
            s=Q.session(model;capacity=256,kv_type=T,attention=mode)
            try
                for (i,token) in enumerate(metadata.input_ids)
                    got=Array(Q.step!(s,token))
                    @test all(isfinite,got)
                    if T===Float32 && mode===:split
                        push!(baseline,copy(got))
                    else
                        want=baseline[i]
                        maxabs=maximum(abs,got-want);relrms=norm(got-want)/norm(want)
                        # Strict numerical qualification uses unquantized
                        # weights/activations. Q8 activation bins can change
                        # discontinuously after reordered FP32 reductions;
                        # report that deviation, do not claim Q4 equivalence.
                        if model.quantize===:none
                            @test relrms < (T===Float32 ? 2f-5 : 3f-3)
                            @test maxabs < (T===Float32 ? 2f-4 : 0.1f0)
                        end
                        @test argmax(got)==argmax(want)
                        push!(rows,(;step=i,cache=string(T),attention=string(mode),maxabs,relrms,
                            greedy=argmax(got)-1,reference=argmax(want)-1))
                    end
                end
                first=s.position
                before=Mantle.submissions(model.device)
                Q.decode!(s,4);KernelAbstractions.synchronize(model.device)
                @test Mantle.submissions(model.device)-before==4
                @test all(isfinite,Array(s.logits))
                @test length(Array(s.predictions)[first:first+3])==4
            finally
                close(s)
            end
        end
    end
    result=(;scope="short teacher-forced logits and token agreement, not general quality or long-context recall",
        quantize=string(model.quantize),numeric_equivalence_asserted=model.quantize===:none,rows)
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end

function bench_qwen35_attention_modes(model,metadata;context=4096,samples=3,newtokens=128,
                                     output_dir=@__DIR__,deadline=Inf)
    ids=repeat(Int.(metadata.chat_ids),cld(context,length(metadata.chat_ids)))[1:context]
    prompt=(;chat_ids=ids)
    rows=Any[]
    for (T,mode) in ((Float32,:split),(Float32,:grouped),(Float16,:split),(Float16,:grouped))
        time()<deadline || error("Time budget exhausted before next ablation variant")
        path=joinpath(output_dir,"qwen4b_$(lowercase(string(T)))_$(mode)_$(context)_benchmark.json")
        result=bench_qwen35(model,prompt;newtokens,samples,capacity=nextpow(2,context+newtokens+32),
            prefill_chunk=32,kv_type=T,attention=mode,deadline,output=path)
        push!(rows,(;cache=string(T),attention=string(mode),context,
            median_decode_tps=result.median_decode_tps,
            median_prefill_seconds=median([s.prefill_seconds for s in result.samples]),file=path))
        println(last(rows));flush(stdout)
    end
    open(io->JSON3.pretty(io,(;scope="four-way cache and kernel ablation; raw samples in linked files",rows)),
        joinpath(output_dir,"attention_cache_ablation.json"),"w")
    rows
end
