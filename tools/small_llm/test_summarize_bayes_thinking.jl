include("summarize_bayes_thinking.jl")
using Test
function test_summarize_bayes_thinking()
    @testset "Paired thinking arithmetic, no synthetic quality claims" begin
        baseline=(;complete=true,rows=[(;task="t$i",arm,status="failed") for i in 1:8 for arm in ("direct_probability","bayes_network")])
        primary=(;complete=true,rows=[(;r...,status=parse(Int,r.task[2:end])<=(r.arm=="direct_probability" ? 6 : 5) ? "passed" : "failed") for r in baseline.rows])
        result=bayes_thinking_summary_statistics(primary,baseline)
        @test result.comparisons[1].relative_to_nonthinking.repairs==6
        @test result.comparisons[2].relative_to_nonthinking.repairs==5
        @test result.comparisons[1].relative_to_nonthinking.regressions==0
        @test result.thinking_network_vs_direct.statistics.regressions==1
        @test_throws ArgumentError bayes_thinking_summary_statistics((;primary...,complete=false),baseline)
        @test_throws ArgumentError bayes_thinking_summary_statistics((;primary...,rows=primary.rows[1:end-1]),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,task="different")
        @test_throws ArgumentError bayes_thinking_summary_statistics((;primary...,rows=changed),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,task=changed[3].task)
        @test_throws ArgumentError bayes_thinking_summary_statistics((;primary...,rows=changed),baseline)
        changed=copy(primary.rows);changed[1]=(;changed[1]...,status="pending")
        @test_throws ArgumentError bayes_thinking_summary_statistics((;primary...,rows=changed),baseline)
    end
end
