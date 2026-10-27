include("rpn_stencil_contract_v2_study.jl")
using Test
function test_rpn_stencil_contract_v2_study()
    @testset "Posthoc v2 keeps BOTH arms/all tasks, no device/model" begin
        parent=JSON3.read(read(joinpath(@__DIR__,"rpn_stencil_results/conditions.json"),String))
        jobs=rpn_stencil_contract_v2_jobs(parent)
        @test length(jobs)==length(parent.jobs)==16
        @test length(unique(j.id for j in jobs))==16
        for (old,new) in zip(parent.jobs,jobs)
            @test new.id==String(old.id)*"_contract_v2"
            @test new.task==old.task&&new.arm==old.arm&&new.enabled==old.enabled
            @test JSON3.write(new.messages)!=JSON3.write(old.messages)
        end
        for task in rpn_stencil_tasks()
            pair=filter(j->j.task==task.id,jobs)
            @test pair[1].messages==pair[2].messages
        end
    end
end
