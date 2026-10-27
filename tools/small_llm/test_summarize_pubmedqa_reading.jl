include("summarize_pubmedqa_reading.jl")
using Test
function test_pubmedqa_paired_changes()
    rows=[(;task="a",arm="x",status="failed",prediction="no"),
        (;task="a",arm="y",status="passed",prediction="yes"),
        (;task="b",arm="x",status="passed",prediction="no"),
        (;task="b",arm="y",status="failed",prediction="maybe"),
        (;task="c",arm="x",status="failed",prediction="yes"),
        (;task="c",arm="y",status="failed",prediction="yes")]
    @testset "Paired changes do not hide regressions or repeated tasks" begin
        r=pubmedqa_paired_changes(rows,"x","y")
        @test r.total==3
        @test r.repairs==r.regressions==1
        @test r.prediction_changes==2
        @test length(r.pairs)==3
        @test_throws ArgumentError pubmedqa_paired_changes(rows[2:end],"x","y")
    end
end
