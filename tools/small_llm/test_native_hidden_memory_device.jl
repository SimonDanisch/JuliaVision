isdefined(@__MODULE__,:HeadInputTap) || include("head_input_tap.jl")
isdefined(@__MODULE__,:KNNMemory) || include("knn_memory.jl")
using Test

function test_native_hidden_memory_device(device)
    Q=SmallLLMChecks.Qwen35Experiment
    @testset "Native head to memory, shared recorded graph" begin
        width=512;vocabulary=7
        keys=zeros(Float32,width,2);keys[1,1]=1f0;keys[1,2]=-1f0
        labels=Int32[1,5]
        for weight in (0f0,1f0)
            owned=Any[];plan=nothing;memory=nothing
            upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b end
            weights=Q.DenseWeight(upload(zeros(Float32,width,vocabulary)),vocabulary,width)
            input=upload(zeros(Float32,width));query=upload(zeros(Float32,width))
            logits=upload(zeros(Float32,vocabulary));selected=upload(Int32[-1])
            model=(;config=(;hidden_size=width),weights=Dict{String,Any}("lm_head"=>weights))
            try
                memory=KNNMemory.Memory(device,keys,labels,vocabulary;topk=1,temperature=1f0,weight)
                with_head_input_tap(model,query) do
                    graph=Mantle.Graph(device)
                    Q.linear!(graph,logits,model.weights["lm_head"],input;name="native_head")
                    KNNMemory.declare!(graph,logits,query,memory)
                    Q.BR.declare_argmax!(graph,selected,logits;name="fixture_selection")
                    plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                    # Replay one plan with changed input. A stale captured key
                    # would select the wrong token on the second replay.
                    for (column,label) in ((1,1),(2,5),(1,1))
                        values=keys[:,column];input[:]=values
                        before=Mantle.submissions(device);Mantle.run!(plan)
                        @test Mantle.submissions(device)-before==1
                        @test Array(query)==values
                        @test only(Array(memory.status))==0
                        @test only(Array(selected))==(weight==0 ? 0 : label)
                        if weight==0
                            @test Array(logits)==zeros(Float32,vocabulary)
                        else
                            expected=zeros(Float32,vocabulary);expected[label+1]=1f0
                            @test exp.(Array(logits))==expected
                        end
                    end
                end
                @test model.weights["lm_head"]===weights
            finally
                plan===nothing || Mantle.free!(plan)
                memory===nothing || close(memory)
                foreach(Mantle.free!,reverse(owned))
            end
        end
    end
end
