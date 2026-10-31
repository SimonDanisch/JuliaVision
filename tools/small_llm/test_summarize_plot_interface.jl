include("summarize_plot_interface.jl")
using Test
function test_summarize_plot_interface()
    @testset "Paired Makie interpretation never equates rendering with aesthetics" begin
        rows=[(;task="task_$i",arm,status=(i<= (arm=="julia_source" ? 2 : 3)) ? "passed" : "failed",
            admitted=true,specification_correct=i<= (arm=="julia_source" ? 2 : 3),admission_reason="accepted",truncated=false)
            for i in 1:8 for arm in ("julia_source","plot_spec")]
        primary=(;complete=true,rows)
        result=plot_interface_interpretation(primary)
        @test result.arms[1].specification_correct==2
        @test result.arms[2].specification_correct==3
        @test result.comparison.statistics.repairs==1
        @test result.comparison.statistics.regressions==0
        @test result.render.status=="not_attempted"
        @test !result.render.aesthetic_preference_tested
        render=(;complete=false,rows=[(;status="interrupted_first_render")])
        @test plot_interface_interpretation(primary,render).render.status=="incomplete"
        @test_throws ArgumentError plot_interface_interpretation((;complete=false,rows))
        @test_throws ArgumentError plot_interface_interpretation((;complete=true,rows=rows[1:15]))
    end
end
