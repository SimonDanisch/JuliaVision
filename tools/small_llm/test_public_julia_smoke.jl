include("public_julia_smoke.jl")
include("humaneval_subset.jl")
using Test
@testset "Public smoke boundary" begin
    prompt="Implement the documented function.\n\n    f(x::Int)::Int\n\njulia> f(1)\n1"
    @test public_smoke_expressions(prompt)==["f(1)"]
    for expression in ("run(`id`)","f(read(\"/etc/passwd\",String))","f(1); error(\"x\")","f(begin 1 end)","eval(1)")
        @test_throws ArgumentError public_smoke_expressions("Implement the documented function.\n\n    f(x::Int)::Int\n\njulia> "*expression)
    end
    task=(;id="smoke_boundary",prompt)
    ok=diagnose_public_smoke(task,"f(x::Int)=x+1")
    @test ok.analysis_completed
    @test only(ok.payload).completed
    @test only(ok.payload).value=="2" # observations, not correctness assertions
    bad=diagnose_public_smoke(task,"f(x::Int)=error(\"public failure\")")
    @test bad.analysis_completed
    @test !only(bad.payload).completed
    @test occursin("public failure",only(bad.payload).error)
    loading=diagnose_public_smoke(task,"using DeliberatelyUnavailableSmokePackage")
    @test !loading.analysis_completed
    @test loading.payload===nothing
    @test occursin("DeliberatelyUnavailableSmokePackage",loading.loading_or_harness_error)
end
@testset "Frozen public examples qualify independently of hidden tests" begin
    for task in load_frozen_humaneval_tasks()
        result=diagnose_public_smoke(task,task.reference)
        @test result.analysis_completed
        @test all(row->row.completed,result.payload)
    end
end
