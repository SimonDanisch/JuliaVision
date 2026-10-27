include("summarize_rpn_pairs.jl")
using Test
function test_summarize_rpn_pairs()
    @testset "Paired RPN analysis keeps pending/losing controls and syntax separate" begin
        jobs=[(;id="t$(i)_"*arm,task="t$i",arm,enabled=arm=="enabled") for i in 1:5 for arm in ("enabled","disabled")]
        statuses=("passed","failed","failed","passed","passed","passed","failed","failed","pending","passed")
        rows=[(;id=j.id,status,syntax_accepted=status=="passed") for (j,status) in zip(jobs,statuses)]
        p=summarize_rpn_pairs(jobs,rows)
        @test !p.complete
        @test p.paired_counts==Dict("enabled_only"=>1,"disabled_only"=>1,"both_pass"=>1,"neither_pass"=>1,"pending"=>1)
        @test p.by_arm["enabled"].conditions==5&&p.by_arm["disabled"].conditions==5
        @test p.by_arm["enabled"].mathematical_passes==2
        @test p.by_arm["disabled"].mathematical_passes==3
        @test_throws ErrorException summarize_rpn_pairs(jobs,reverse(rows))
        @test_throws ErrorException summarize_rpn_pairs(vcat(jobs,[first(jobs)]),vcat(rows,[first(rows)]))
        altered=copy(rows);altered[1]=(;altered[1]...,syntax_accepted=false)
        @test_throws ErrorException summarize_rpn_pairs(jobs,altered)
        altered=copy(rows);altered[1]=(;altered[1]...,status="unknown")
        @test_throws ErrorException summarize_rpn_pairs(jobs,altered)
        @test_throws ErrorException summarize_rpn_pairs(jobs[1:3],rows[1:3])
    end
end
