using Test
import Mantle
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")

function test_gpu_sampling(device)
    S=GPUSampling
    @testset "shared top-k/nucleus sampling arithmetic" begin
        @test all(i->0f0<S.uniform23(UInt32(7),UInt32(i))<1f0,0:2048)
        for topk in (1,3,7), top_p in (0.2f0,0.8f0,1f0), presence in (0f0,1.5f0)
            original=Float32[0.1,-0.7,1.2,-Inf,1.2,-2,0.6]
            logits=Mantle.Buffer(device,original)
            state=nothing;plan=nothing
            try
                state=S.State(device,length(original);topk,top_p,presence,temperature=0.7f0,seed=UInt32(234),fallback=2)
                graph=Mantle.Graph(device);S.declare!(graph,logits,state)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                seen=zeros(Int32,length(original))
                for counter in UInt32(0):UInt32(7)
                    expected=S.reference(original;topk,top_p,presence,temperature=0.7f0,
                        seed=UInt32(234),counter,seen,fallback=2)
                    logits[:]=original;Mantle.run!(plan)
                    @test only(Array(state.selected))==expected.selected
                    @test only(Array(state.status))==0
                    @test only(Array(state.counter))==counter+UInt32(1)
                    scores=Array(logits)
                    @test findall(==(0f0),scores)==[Int(expected.selected)+1]
                    @test count(==(-Inf32),scores)==length(original)-1
                    seen[Int(expected.selected)+1]+=1
                    @test Array(state.seen)==seen
                end
                S.activate!(state,false)
                logits[:]=original;Mantle.run!(plan)
                @test isequal(Array(logits),original)
                @test only(Array(state.counter))==8
                @test Array(state.seen)==seen
            finally
                plan===nothing || Mantle.free!(plan)
                state===nothing || close(state)
                Mantle.free!(logits)
            end
        end
        for original in (fill(-Inf32,5),Float32[0,NaN,1,2,3],Float32[0,Inf,1,2,3])
            logits=Mantle.Buffer(device,original);state=nothing;plan=nothing
            try
                state=S.State(device,5;topk=3,fallback=4)
                graph=Mantle.Graph(device);S.declare!(graph,logits,state)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                Mantle.run!(plan)
                @test only(Array(state.status))==1
                @test only(Array(state.selected))==4
                @test Array(logits)==Float32[-Inf,-Inf,-Inf,-Inf,0]
                @test S.reference(original;topk=3,fallback=4).status==1
            finally
                plan===nothing || Mantle.free!(plan)
                state===nothing || close(state)
                Mantle.free!(logits)
            end
        end
        for topk in (20,40)
            original=fill(-80f0,248320)
            for j in 1:60;original[mod(j*1009,length(original))+1]=Float32(60-j)*0.37f0;end
            logits=Mantle.Buffer(device,original);state=nothing;plan=nothing
            try
                state=S.State(device,length(original);topk,temperature=0.6f0,top_p=0.95f0,seed=UInt32(51))
                graph=Mantle.Graph(device);S.declare!(graph,logits,state)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                for counter in UInt32(0):UInt32(1)
                    expected=S.reference(original;topk,temperature=0.6f0,top_p=0.95f0,seed=UInt32(51),counter)
                    logits[:]=original;Mantle.run!(plan)
                    @test only(Array(state.selected))==expected.selected
                    @test only(Array(state.status))==0
                end
            finally
                plan===nothing || Mantle.free!(plan)
                state===nothing || close(state)
                Mantle.free!(logits)
            end
        end
        @test_throws ArgumentError S.State(device,7;temperature=1e-100)
        @test_throws ArgumentError S.State(device,7;topk=8)
        @test_throws ArgumentError S.State(device,7;top_p=0)
    end
    (;scope="exact top-k/nucleus arithmetic and GPU-state feedback, not model quality",cases=20,draws=148,full_vocabulary=248320)
end
