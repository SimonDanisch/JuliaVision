include("memory_status_history.jl")
using Test

function test_memory_status_history(device)
    history=Mantle.Buffer(device,Int32[0]);current=Mantle.Buffer(device,Int32[0]);plan=nothing
    try
        @testset "Sticky memory validity, shared kernel" begin
            graph=Mantle.Graph(device);declare_memory_status_history!(graph,history,current)
            plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
            for (code,expected) in ((0,0),(2,2),(0,2),(1,2))
                current[:]=Int32[code];before=Mantle.submissions(device);Mantle.run!(plan)
                @test Mantle.submissions(device)-before==1
                @test only(Array(history))==expected
                @test only(Array(current))==code
            end
            @test_throws ErrorException declare_memory_status_history!(graph,history,history)
        end
    finally
        plan===nothing || Mantle.free!(plan)
        Mantle.free!(history);Mantle.free!(current)
    end
end
