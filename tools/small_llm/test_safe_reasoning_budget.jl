isdefined(@__MODULE__,:test_reasoning_budget) || include("test_reasoning_budget.jl")
isdefined(@__MODULE__,:SafeReasoningBudget) || include("safe_reasoning_budget.jl")

function test_safe_reasoning_budget(device)
    @testset "Validated reasoning budget, shared kernels" begin
        test_reasoning_budget(device)
        n=257
        cases=[(;values=zeros(Float32,n),expected=0),
            (;values=vcat(zeros(Float32,n-1),NaN32),expected=1),
            (;values=vcat(zeros(Float32,n-1),Inf32),expected=1),
            (;values=fill(-Inf32,n),expected=1),
            (;values=vcat(zeros(Float32,n-1),-Inf32),expected=0)]
        for (trial,case) in enumerate(cases)
            input=Mantle.Buffer(device,case.values);logits=Mantle.Buffer(device,zeros(Float32,n))
            sampler=GPUSampling.State(device,n;topk=1);budget=SafeReasoningBudget.State(device,n,3;limit=0);plan=nothing
            try
                graph=Mantle.Graph(device)
                Mantle.dispatch!(graph,BonsaiRunner.copy_kernel!,(logits,input,Int32(n)),n;group=256,name="fixture.raw_logits")
                SafeReasoningBudget.declare!(graph,logits,budget,sampler);GPUSampling.declare!(graph,logits,sampler)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                before=Mantle.submissions(device);Mantle.run!(plan)
                @test Mantle.submissions(device)-before==1
                @test only(Array(sampler.status))==case.expected
                @test only(Array(budget.forced_count))==1
                # A later healthy query must NOT clear a prior invalid-query flag.
                input[:]=zeros(Float32,n);Mantle.run!(plan)
                @test only(Array(sampler.status))==case.expected
            finally
                plan===nothing || Mantle.free!(plan)
                close(budget);close(sampler);Mantle.free!(input);Mantle.free!(logits)
            end
        end
    end
end
