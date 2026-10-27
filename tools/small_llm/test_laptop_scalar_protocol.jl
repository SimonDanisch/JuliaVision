include("laptop_scalar_kernel_study.jl")
using Test

function test_laptop_scalar_protocol()
    checks=@testset "Scalar-interface study CPU registration, no generations" begin
        mktempdir() do root
            directory=joinpath(root,"study");p=freeze_laptop_scalar_kernel_study(;directory)
            @test length(p.jobs)==4
            @test all(j->j.arm=="scalar_source",p.jobs)
            @test p.decoding.max_tokens==2048 && p.thinking_budget==1024
            @test occursin("POSTHOC",p.scope)
            @test all(j->occursin("ONE Julia scalar",String(last(last(j.messages)))),p.jobs)
            @test_throws ErrorException freeze_laptop_scalar_kernel_study(;directory)
            report=score_laptop_scalar_kernel_study(;directory)
            @test !report.complete && length(report.rows)==4 && all(r->r.status=="pending",report.rows)
            @test_throws ArgumentError run_laptop_shared_source(;directory,seconds=0)
        end
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="CPU registration/integrity only, not model-quality evidence",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("test_laptop_scalar_protocol.jl","laptop_scalar_kernel_study.jl","scalar_kernel_interface.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"laptop_scalar_protocol_validation.json"),"w");packet
end
