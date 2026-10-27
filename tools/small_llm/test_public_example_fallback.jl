include("public_example_fallback.jl")
using Test
function test_public_example_fallback()
    F=PublicExampleFallback
    bank=RPNExampleSemantics.Bank([(;input=Float32[-1,0,2],output=Float32[-1,1,5])])
    correct="x0\ntwo\nmul\none\nadd\nfinish\n";wrong="one\nfinish\n"
    @testset "Uniform public-only selection never hides raw infrastructure failures" begin
        @test F.matches(correct,bank)
        @test !F.matches(wrong,bank)
        @test !F.matches("x0\nadd\nfinish",bank)
        @test !F.matches("error(\"arbitrary code\")",bank)
        result=F.choose(correct,bank,(;found=true,program=wrong))
        @test result.selected=="neural_baseline" && result.program==correct
        result=F.choose(wrong,bank,(;found=true,program=correct))
        @test result.selected=="symbolic_public_examples" && result.program==correct
        @test F.choose(wrong,bank,(;found=false,program="")).selected=="none"
        @test F.choose(wrong,bank,(;found=true,program=wrong)).selected=="none"
        @test F.choose(correct,bank,(;found=true,program=correct);infrastructure_ok=false).reason=="infrastructure_failure"
        @test_throws ArgumentError F.choose(wrong,bank,(;found=true,program=correct);infrastructure_ok=1)
        @test_throws ArgumentError F.choose(wrong,bank,(;found=1,program=correct))
    end
end
