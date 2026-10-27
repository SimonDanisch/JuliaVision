isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")
isdefined(@__MODULE__,:ReasoningBudget) || include("reasoning_budget.jl")
import BonsaiRunner
using Test

function test_reasoning_budget(device)
    @testset "GPU-resident thinking budget, shared kernels" begin
        for (limit,favorite,expected,forced) in ((0,7,[3,7,7,7],1),(2,7,[7,7,3,7],1),
                (100,7,[7,7,7,7],0),(2,3,[3,3,3,3],0))
            n=257;base=fill(-10f0,n);base[favorite+1]=2f0
            input=Mantle.Buffer(device,base);logits=Mantle.Buffer(device,zeros(Float32,n))
            sampler=GPUSampling.State(device,n;topk=1,seed=UInt32(17))
            budget=ReasoningBudget.State(device,n,3;limit);plan=nothing
            try
                graph=Mantle.Graph(device)
                Mantle.dispatch!(graph,BonsaiRunner.copy_kernel!,(logits,input,Int32(n)),n;group=256,name="fixture.logits")
                ReasoningBudget.declare!(graph,logits,budget,sampler);GPUSampling.declare!(graph,logits,sampler)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                GPUSampling.activate!(sampler,false);Mantle.run!(plan)
                @test only(Array(sampler.counter))==0
                @test only(Array(budget.force))==0
                @test only(Array(budget.forced_count))==0
                GPUSampling.activate!(sampler,true)
                for (step,token) in enumerate(expected)
                    before=Mantle.submissions(device);Mantle.run!(plan)
                    @test Mantle.submissions(device)-before==1
                    @test only(Array(sampler.selected))==token
                    @test only(Array(sampler.counter))==step
                    @test only(Array(sampler.status))==0
                end
                @test only(Array(budget.forced_count))==forced
                @test only(Array(budget.done))==(limit<100 || favorite==3 ? 1 : 0)
            finally
                plan===nothing || Mantle.free!(plan)
                close(budget);close(sampler);Mantle.free!(input);Mantle.free!(logits)
            end
        end
        # Disabled control must preserve the SAME sampled process, including
        # RNG draws, repeated-token penalties and seen-token history.
        n=257;values=Float32[sin(i*.37)+0.01cos(i*.23) for i in 1:n]
        input=Mantle.Buffer(device,values);a=Mantle.Buffer(device,zeros(Float32,n));b=Mantle.Buffer(device,zeros(Float32,n))
        left=GPUSampling.State(device,n;seed=UInt32(123),presence=0.5f0)
        right=GPUSampling.State(device,n;seed=UInt32(123),presence=0.5f0)
        budget=ReasoningBudget.State(device,n,3;limit=typemax(UInt32));plans=Any[]
        try
            for (logits,sampler,controlled) in ((a,left,false),(b,right,true))
                graph=Mantle.Graph(device)
                Mantle.dispatch!(graph,BonsaiRunner.copy_kernel!,(logits,input,Int32(n)),n;group=256,name="fixture.logits")
                controlled && ReasoningBudget.declare!(graph,logits,budget,sampler)
                GPUSampling.declare!(graph,logits,sampler)
                push!(plans,Mantle.record!(Mantle.Plan(graph;budget=Inf)))
            end
            for step in 1:8
                Mantle.run!(plans[1]);before=Mantle.submissions(device);Mantle.run!(plans[2])
                @test Mantle.submissions(device)-before==1
                @test Array(a)==Array(b)
                @test Array(left.selected)==Array(right.selected)
                @test Array(left.counter)==Array(right.counter)==UInt32[step]
                @test Array(left.seen)==Array(right.seen)
                @test Array(left.status)==Array(right.status)==Int32[0]
            end
            @test only(Array(budget.forced_count))==0
        finally
            foreach(Mantle.free!,reverse(plans));close(budget);close(left);close(right)
            foreach(Mantle.free!,(input,a,b))
        end
        @test_throws ArgumentError ReasoningBudget.State(device,257,257)
        @test_throws ArgumentError ReasoningBudget.State(device,257,3;limit=-1)
        @test_throws ArgumentError ReasoningBudget.State(device,257,true)
    end
end
