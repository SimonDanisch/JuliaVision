include("paired_workflow_uncertainty.jl")
using Test
function test_paired_workflow_uncertainty()
    @testset "Paired descriptive uncertainty" begin
        empty=paired_workflow_uncertainty(Bool[],Bool[])
        @test empty.before.estimate===nothing
        @test empty.exact_two_sided_discordance_p==1
        repair=paired_workflow_uncertainty(falses(10),trues(10))
        @test repair.repairs==10&&repair.regressions==0
        @test repair.exact_two_sided_discordance_p==2/1024
        @test repair.before.lower==0
        @test repair.after.upper==1
        reverse=paired_workflow_uncertainty(trues(10),falses(10))
        @test reverse.exact_two_sided_discordance_p==repair.exact_two_sided_discordance_p
        tie=paired_workflow_uncertainty([true,false],[false,true])
        @test tie.exact_two_sided_discordance_p==1
        @test paired_workflow_uncertainty(trues(8),trues(8)).discordant==0
        @test workflow_wilson(4,8).estimate==0.5
        @test isapprox(workflow_wilson(4,8).lower,0.2152160622;atol=1e-9)
        @test_throws ArgumentError workflow_wilson(2,1)
        @test_throws ArgumentError workflow_wilson(0,10001)
        @test_throws ArgumentError paired_workflow_uncertainty([true],[true,false])
        @test_throws ArgumentError paired_workflow_uncertainty([1],[0])
        rows=[(;task="a",arm="before",status="failed"),(;task="a",arm="after",status="passed")]
        @test paired_summary_uncertainty((;complete=true,rows),"before","after").statistics.repairs==1
        @test_throws ArgumentError paired_summary_uncertainty((;complete=false,rows),"before","after")
        @test_throws ArgumentError paired_summary_uncertainty((;rows),"before","absent")
        @test_throws ArgumentError paired_summary_uncertainty((;rows=vcat(rows,rows[1:1])),"before","after")
        changed=[rows[1],(;task="b",arm="after",status="passed")]
        @test_throws ArgumentError paired_summary_uncertainty((;rows=changed),"before","after")
        @test_throws ArgumentError paired_summary_uncertainty((;rows=[rows[1],(;task="a",arm="after",status="pending")]),"before","after")
    end
end
