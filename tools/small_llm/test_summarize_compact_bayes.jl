include("summarize_compact_bayes.jl")
using Test
function test_summarize_compact_bayes()
    @testset "Compact/full paired arithmetic only" begin
        baseline=(;complete=true,rows=[(;task="t$i",arm,status="failed") for i in 1:8 for arm in ("direct_probability","bayes_network")])
        primary=(;complete=true,rows=[(;task="t$i",arm="compact_network",status=i<=5 ? "passed" : "failed") for i in 1:8])
        stats=compact_bayes_summary_statistics(primary,baseline).relative_to_full_schema
        @test stats.repairs==5&&stats.regressions==0&&stats.unchanged_incorrect==3
        @test stats.before.successes==0&&stats.after.successes==5
        @test_throws ArgumentError compact_bayes_summary_statistics((;primary...,complete=false),baseline)
        @test_throws ArgumentError compact_bayes_summary_statistics((;primary...,rows=primary.rows[1:7]),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,task="other")
        @test_throws ArgumentError compact_bayes_summary_statistics((;primary...,rows=changed),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,task="t2")
        @test_throws ArgumentError compact_bayes_summary_statistics((;primary...,rows=changed),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,status="pending")
        @test_throws ArgumentError compact_bayes_summary_statistics((;primary...,rows=changed),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,arm="bayes_network")
        @test_throws ArgumentError compact_bayes_summary_statistics((;primary...,rows=changed),baseline)
    end
end
