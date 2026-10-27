include("summarize_compact_bayes_demo.jl")
using Test
function test_summarize_compact_bayes_demo()
    rows=[(;task="t$i",arm="compact_network",status="failed") for i in 1:8]
    @testset "Paired demonstration analysis mocks only" begin
        p=(;complete=true,rows)
        @test compact_bayes_demo_statistics(p,p).tasks==["t$i" for i in 1:8]
        @test_throws ArgumentError compact_bayes_demo_statistics((;complete=false,rows),p)
        @test_throws ArgumentError compact_bayes_demo_statistics((;complete=true,rows=rows[1:7]),p)
        duplicate=copy(rows);duplicate[8]=rows[1]
        @test_throws ArgumentError compact_bayes_demo_statistics((;complete=true,rows=duplicate),p)
        invalid=copy(rows);invalid[1]=(;task="t1",arm="direct",status="passed")
        @test_throws ArgumentError compact_bayes_demo_statistics((;complete=true,rows=invalid),p)
        mismatch=copy(rows);mismatch[8]=(;task="other",arm="compact_network",status="failed")
        @test_throws ArgumentError compact_bayes_demo_statistics((;complete=true,rows=mismatch),p)
    end
end
