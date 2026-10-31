include("rpn_stencil_study.jl")
using Test
function test_rpn_stencil_study()
    @testset "Matched RPN stencil study public jobs, no model/device" begin
        jobs=rpn_stencil_jobs()
        @test length(jobs)==length(unique(j.id for j in jobs))==16
        for (i,task) in enumerate(rpn_stencil_tasks())
            pair=filter(j->j.task==task.id,jobs)
            @test length(pair)==2
            @test [j.enabled for j in pair]==(isodd(i) ? [false,true] : [true,false])
            @test pair[1].messages==pair[2].messages
            @test all(j->j.messages==rpn_stencil_messages(task),pair)
        end
        @test_throws ErrorException run_rpn_stencil_study()
        mktempdir() do directory
            # No quality registration is permitted before the hardware gates.
            if !isfile(joinpath(@__DIR__,"laptop_rpn_gate/identity.json"))
                @test_throws ErrorException freeze_rpn_stencil_study(;directory)
                @test isempty(readdir(directory))
            end
        end
    end
end
