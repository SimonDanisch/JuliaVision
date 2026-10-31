include("api_public_smoke.jl")
isdefined(@__MODULE__,:JULIA_ADVANCED_TASKS) || include("julia_advanced_tasks.jl")
using Test

function test_api_public_smoke()
    @testset "Public API execution diagnostics" begin
        for task in filter(task->haskey(API_PUBLIC_FIXTURES,task.id),JULIA_ADVANCED_TASKS)
            # Gold is used only to qualify the infrastructure in its sandbox.
            reference=diagnose_api_public_smoke(task,task.reference)
            @test reference.analysis_completed
            @test reference.payload.completed
            public_only=(;id=task.id,prompt=task.prompt)
            missing_method=diagnose_api_public_smoke(public_only,"solve!(args...)=error(\"PUBLIC_SENTINEL\")")
            @test missing_method.analysis_completed
            @test !missing_method.payload.completed
            @test occursin("PUBLIC_SENTINEL",missing_method.payload.error)
            # Deliberately poison private fields; diagnostic must not read them.
            poisoned=(;id=task.id,prompt=task.prompt,tests="PRIVATE_ASSERTION_SENTINEL",reference="PRIVATE_GOLD_SENTINEL",setup="error(\"PRIVATE_SETUP_SENTINEL\")")
            result=diagnose_api_public_smoke(poisoned,"solve!(args...)=error(\"PUBLIC_SENTINEL\")")
            @test result.analysis_completed
            @test !occursin("PRIVATE_",JSON3.write(result))
        end
    end
end
