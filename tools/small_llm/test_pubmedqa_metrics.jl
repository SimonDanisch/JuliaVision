include("pubmedqa_metrics.jl")
using Test
function test_pubmedqa_metrics()
    gold=Dict("a"=>"yes","b"=>"no","c"=>"maybe")
    rows=[(;task,arm,prediction=arm=="perfect" ? label : "invalid") for (task,label) in gold for arm in ("perfect","invalid")]
    @testset "Strict biomedical classification metrics are not safety claims" begin
        result=PubMedQAMetrics.summarize(rows,gold;arms=("perfect","invalid"))
        @test result[1].accuracy==result[1].macro_f1==1
        @test result[2].accuracy==result[2].macro_f1==0
        @test result[2].invalid==3
        @test all(c->c.fn==1&&c.tp==0,result[2].per_class)
        @test result[1].confusion["yes"]["yes"]==1
        @test_throws ArgumentError PubMedQAMetrics.summarize(rows[1:end-1],gold;arms=("perfect","invalid"))
        repeated=copy(rows);repeated[1]=repeated[2]
        @test_throws ArgumentError PubMedQAMetrics.summarize(repeated,gold;arms=("perfect","invalid"))
        bad=copy(rows);bad[1]=(;bad[1]...,prediction="YES")
        @test_throws ArgumentError PubMedQAMetrics.summarize(bad,gold;arms=("perfect","invalid"))
        wrong=copy(gold);wrong["a"]="uncertain"
        @test_throws ArgumentError PubMedQAMetrics.summarize(rows,wrong;arms=("perfect","invalid"))
        one=[(;task="a",arm="x",prediction="yes")]
        @test only(PubMedQAMetrics.summarize(one,Dict("a"=>"yes");arms=("x",))).macro_f1==1/3
        @test_throws ArgumentError PubMedQAMetrics.summarize([],Dict())
        @test_throws ArgumentError PubMedQAMetrics.summarize(one,Dict("a"=>"yes");arms=())
        @test_throws ArgumentError PubMedQAMetrics.summarize(one,Dict("a"=>"yes");arms=("x","x"))
    end
end
