include("compact_bayes_tasks.jl")
include("compact_bayes_admission.jl")
using Test

function test_compact_bayes_interface()
    @testset "Public compact interface, no posterior repair" begin
        for task in bayes_extraction_tasks()
            messages=compact_bayes_messages(task)
            @test length(messages)==2
            @test last(first(messages))=="Follow the requested output interface exactly. No fences, commentary or reasoning text."
            @test endswith(last(last(messages)),"\nProblem:\n"*task.request)
        end
        text="{\"nodes\":[{\"name\":\"a\",\"parents\":[],\"table\":[0.2]}],\"query\":\"a\",\"observed\":{}}"
        @test admit_compact_bayes(text).accepted
        @test admit_compact_bayes("```json\n"*text*"\n```").accepted
        @test !admit_compact_bayes(text*" prose").accepted
        @test !admit_compact_bayes("explanation\n"*text).accepted
        @test !admit_compact_bayes(text*text).accepted
        @test !admit_compact_bayes(replace(text,"0.2"=>"0.2+0.1")).accepted
        @test !admit_compact_bayes(replace(text,"0.2"=>"true")).accepted
        @test !admit_compact_bayes(replace(text,"0.2"=>"1.2")).accepted
        @test admit_compact_bayes(replace(text,"0.2"=>"0.3")).packed.factors[1].values[2]==0.3
    end
end
