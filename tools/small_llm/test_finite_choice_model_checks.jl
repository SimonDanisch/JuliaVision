include("finite_choice_model_checks.jl")
using Test
function test_finite_choice_model_checks()
    @testset "Full-model gate checker rejects invalid observations" begin
        base=(;tokens=Int32[1,2,3,4,5,6,7,8],seen=Int32[1,2],raw_logits=Float32[1,2],status=0,draws=9,submissions=8,node=1,done=0,rejected=0)
        constrained=(;base...,tokens=Int32[97,98,511,511,511,511,511,511],node=4,done=1)
        paths=[Int32[97,511],Int32[97,98,511]]
        check(a,b,c)=finite_choice_model_checks(a,b,c,paths,511)
        @test check(base,base,constrained).passed
        @test !check(base,(;base...,raw_logits=Float32[1,2.01]),constrained).disabled_identity
        @test !check(base,(;base...,tokens=reverse(base.tokens)),constrained).passed
        @test !check(base,base,(;constrained...,tokens=fill(Int32(97),8))).finite_support
        @test !check(base,base,(;constrained...,tokens=Int32[99,511,511,511,511,511,511,511])).finite_support
        @test !check(base,base,(;constrained...,tokens=Int32[97,98,511,99,511,511,511,511])).finite_support
        @test !check(base,base,(;constrained...,status=1)).passed
        @test !check(base,base,(;constrained...,draws=8)).accounting
        @test !check(base,base,(;constrained...,submissions=9)).accounting
        @test !check(base,base,(;constrained...,raw_logits=Float32[NaN,1])).healthy
        @test !check(base,base,(;constrained...,raw_logits=Float32[Inf,1])).healthy
        @test !check(base,base,(;constrained...,done=0)).finite_support
    end
end
