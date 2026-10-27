include("kernel_public_smoke.jl")
using Test

function test_kernel_public_smoke()
    rows=Any[]
    @testset "Public kernel diagnostics do not consult private truth" begin
        for task in vulkan_kernel_tasks()
            public=(;id=task.id,prompt=task.prompt,tests=error,reference=error,setup=error)
            result=diagnose_kernel_public_smoke(public,task.reference)
            push!(rows,result)
            @test result.analysis_completed
            @test result.payload!==nothing && result.payload.completed
        end
        task=first(vulkan_kernel_tasks());public=(;id=task.id,prompt=task.prompt)
        invalid=diagnose_kernel_public_smoke(public,"function kernel!(out);nothing;end")
        @test invalid.analysis_completed
        @test !invalid.payload.completed
        wrong=replace(task.reference,"muladd(alpha,x[i],beta)"=>"(muladd(alpha,x[i],beta)+1f0)")
        numerical=diagnose_kernel_public_smoke(public,wrong)
        @test numerical.analysis_completed
        @test numerical.payload.completed # Numeric truth must NOT become a repair trigger.
        malformed=diagnose_kernel_public_smoke(public,"function kernel!(")
        @test malformed.analysis_completed
        @test !malformed.payload.completed
    end
    report=(;complete=true,passed=14,rows,
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("kernel_public_smoke.jl","vulkan_kernel_tasks.jl","julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"kernel_public_smoke_validation.json"),"w");report
end
