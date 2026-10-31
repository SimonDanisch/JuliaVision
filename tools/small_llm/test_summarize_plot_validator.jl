include("summarize_plot_validator_study.jl")
using Test
function test_summarize_plot_validator()
    @testset "Posthoc paired repair arithmetic retains baseline failures" begin
        baseline=(;complete=true,rows=[(;task="task_$i",arm="plot_spec",status=i<=3 ? "passed" : "failed") for i in 1:8])
        rows=[(;task="task_$i",arm,status=i<=(arm=="self_review" ? 4 : 5) ? "passed" : "failed",retained_original=!(4<=i<=6))
            for i in 1:8 for arm in ("self_review","validator_feedback")]
        result=plot_validator_summary_statistics((;complete=true,rows),baseline)
        @test result.baseline_correct==3&&result.total==8
        @test result.comparisons[1].relative_to_original.repairs==1
        @test result.comparisons[2].relative_to_original.repairs==2
        @test result.feedback_vs_review.statistics.repairs==1
        @test result.feedback_vs_review.statistics.regressions==0
        @test_throws ArgumentError plot_validator_summary_statistics((;complete=false,rows),baseline)
        @test_throws ArgumentError plot_validator_summary_statistics((;complete=true,rows=rows[1:15]),baseline)
        changed=copy(rows);changed[end]=(;changed[end]...,status="passed")
        @test_throws ArgumentError plot_validator_summary_statistics((;complete=true,rows=changed),baseline)
    end
end
