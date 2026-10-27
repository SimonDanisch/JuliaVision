include("run_qualified_projection_pilot.jl")
using Statistics, Test

function benchmark_projection_memory(data;seconds=210,repeats=3)
    seconds>0 && repeats==3 || throw(ArgumentError("fixed bounded benchmark"))
    require_projection_hook_gate();require_post_extent_gate()
    output=joinpath(@__DIR__,"projection_memory_throughput.json")
    isfile(output) && error("preserve earlier benchmark")
    native=JSON3.read(read(joinpath(@__DIR__,"conversation_native_reference.json"),String))
    ids=Int.(only(filter(row->row.id=="baseline_short",native.rows)).julia_ids)
    Q=SmallLLMChecks.Qwen35Experiment;cache=qualified_qwen_cache()
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing
        memory=nothing;features=nothing;query=nothing;rows=Any[]
        save(complete=false)=begin
            report=(;complete,scope="fixed short-context steady decode wall timing, NOT intelligence",
                warmup_steps=32,timed_steps=128,capacity=512,repeats,rows,
                runtime_provenance=inference_runtime_provenance(),
                source_sha256=bytes2hex(sha256(read(@__FILE__))))
            open(io->JSON3.pretty(io,report),output,"w");report
        end
        try
            model=Q.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            features=LogitFeatures.Features(device,data.vocabulary,data.feature_ids)
            query=Mantle.Buffer(device,zeros(Float32,length(data.feature_ids)))
            for (arm,weight) in (("native",nothing),("zero",0f0),("memory_10",.1f0))
                weight===nothing || (memory=KNNMemory.Memory(device,data.keys,data.labels,data.vocabulary;
                    topk=8,temperature=32f0,weight))
                hook=weight===nothing ? nothing : (graph,logits)->begin
                    LogitFeatures.declare!(graph,query,logits,features)
                    KNNMemory.declare!(graph,logits,query,memory)
                end
                try
                    for repeat in 1:repeats
                        time()<deadline-40 || return save()
                        session=Q.session(model;capacity=512,kv_type=Float32,attention=:split,logit_hook=hook)
                        try
                            Q.prefill!(session,ids;chunk=8);Q.decode!(session,32);Mantle.waitidle(device)
                            submissions=Mantle.submissions(device);t=time_ns()
                            Q.decode!(session,128);Mantle.waitidle(device)
                            elapsed=(time_ns()-t)/1e9
                            @test Mantle.submissions(device)-submissions==128
                            @test all(isfinite,Array(session.logits))
                            memory===nothing || (@test only(Array(memory.status))==0)
                            push!(rows,(;arm,repeat,seconds=elapsed,tokens_per_second=128/elapsed,
                                input_tokens=length(ids)+32,submissions=128))
                            save()
                        finally
                            close(session)
                        end
                    end
                finally
                    memory===nothing || close(memory);memory=nothing
                end
            end
        finally
            try
                features===nothing || close(features)
                query===nothing || Mantle.free!(query)
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
        save(true)
    end
end
