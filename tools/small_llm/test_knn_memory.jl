using Test
import Mantle
isdefined(@__MODULE__,:KNNMemory) || include("knn_memory.jl")

function test_knn_memory(device)
    function reference(keys,labels,query,logits;k,temperature,weight)
        distances=vec(sum(abs2,Float64.(keys).-Float64.(query);dims=1))
        order=sortperm(eachindex(distances);by=i->(distances[i],i))[1:k]
        weights=exp.(-(distances[order].-minimum(distances[order]))./temperature)
        weights./=sum(weights)
        retrieved=zeros(Float64,length(logits))
        for (i,p) in zip(order,weights);retrieved[labels[i]+1]+=p;end
        neural=exp.(Float64.(logits).-maximum(logits));neural./=sum(neural)
        (1-weight).*neural.+weight.*retrieved
    end
    @testset "Shared representation-keyed memory arithmetic" begin
        base=Float32[0 1 1 -1;0 0 1 1;0 0 0 0]
        labels=Int32[0,1,1,3];original=Float32[1000,999,-Inf,-1000,997]
        for T in (Float16,Float32), repetitions in (1,100), k in (1,3), weight in (0f0,0.25f0,1f0)
            keys=repeat(T.(base),repetitions,1);query=repeat(Float32[0.1,0.2,0],repetitions)
            memory=KNNMemory.Memory(device,keys,labels,5;topk=k,temperature=0.3f0,weight)
            logits=Mantle.Buffer(device,original);q=Mantle.Buffer(device,query);plan=nothing
            try
                graph=Mantle.Graph(device);KNNMemory.declare!(graph,logits,q,memory)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                for point in (query,repeat(Float32[1,0.7,0],repetitions))
                    logits[:]=original;q[:]=point
                    before=Mantle.submissions(device);Mantle.run!(plan)
                    @test Mantle.submissions(device)-before==1
                    actual=Array(logits)
                    @test only(Array(memory.status))==0
                    if weight==0f0
                        @test isequal(actual,original)
                    else
                        expected=reference(keys,labels,point,original;k,temperature=0.3f0,weight)
                        @test exp.(actual) ≈ expected atol=2e-5 rtol=2e-4
                        @test sum(exp.(actual)) ≈ 1 atol=2e-5
                    end
                end
            finally
                plan===nothing || Mantle.free!(plan)
                close(memory);Mantle.free!(logits);Mantle.free!(q)
            end
        end
        # Preserve tiny neural log-probabilities, and abstain on invalid query
        # or neural values rather than converting uncertainty to a hard mask.
        memory=KNNMemory.Memory(device,zeros(Float32,2,1),Int32[2],3;topk=1,weight=0.25f0)
        original=Float32[-1000,0,-Inf]
        logits=Mantle.Buffer(device,original);q=Mantle.Buffer(device,zeros(Float32,2));plan=nothing
        try
            graph=Mantle.Graph(device);KNNMemory.declare!(graph,logits,q,memory)
            plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan)
            @test Array(logits)[1] ≈ -1000f0+log(0.75f0) atol=1e-4
            for query in (Float32[NaN,0],Float32[1e30,0])
                q[:]=query;logits[:]=original;Mantle.run!(plan)
                @test only(Array(memory.status))==2
                @test isequal(Array(logits),original)
            end
            q[:]=zeros(Float32,2)
            for invalid in (Float32[NaN,0,1],Float32[Inf,0,1],fill(-Inf32,3))
                logits[:]=invalid;Mantle.run!(plan)
                @test only(Array(memory.status))==1
                @test isequal(Array(logits),invalid)
            end
        finally
            plan===nothing || Mantle.free!(plan)
            close(memory);Mantle.free!(logits);Mantle.free!(q)
        end
        @test_throws ArgumentError KNNMemory.Memory(device,zeros(Float32,2,0),Int32[],3)
        @test_throws ArgumentError KNNMemory.Memory(device,zeros(Float32,2,1),Int32[3],3;topk=1)
        @test_throws ArgumentError KNNMemory.Memory(device,fill(NaN32,2,1),Int32[0],3;topk=1)
        @test_throws ArgumentError KNNMemory.Memory(device,zeros(Float32,2,1),Int32[0],3;topk=1,weight=1.1)
        for vocabulary in (257,248320)
            keys=Float32[0 1;0 1];labels=Int32[0,vocabulary-1];query=Float32[0.2,0.1]
            original=fill(-1000f0,vocabulary);original[3]=0f0;original[end]=-2f0
            memory=KNNMemory.Memory(device,keys,labels,vocabulary;topk=2,weight=0.4f0)
            logits=Mantle.Buffer(device,original);q=Mantle.Buffer(device,query);plan=nothing
            try
                graph=Mantle.Graph(device);KNNMemory.declare!(graph,logits,q,memory)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.run!(plan)
                expected=reference(keys,labels,query,original;k=2,temperature=1f0,weight=0.4f0)
                actual=Array(logits)
                @test only(Array(memory.status))==0
                @test exp.(actual) ≈ expected atol=2e-5 rtol=2e-4
                @test sum(exp.(actual)) ≈ 1 atol=2e-5
                @test actual[2] < -900f0 && isfinite(actual[2])
                @test isfinite(actual[end])
            finally
                plan===nothing || Mantle.free!(plan)
                close(memory);Mantle.free!(logits);Mantle.free!(q)
            end
        end
    end
end
