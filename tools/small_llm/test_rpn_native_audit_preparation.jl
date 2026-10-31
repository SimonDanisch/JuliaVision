include("prepare_rpn_native_audit.jl")
using Test
function test_rpn_native_audit_preparation()
    @testset "PRIVATE native cases derived independently; no inference or hardware" begin
        fixtures=private_rpn_stencil_fixtures()
        for fixture in fixtures
            cases=private_rpn_native_cases(fixture)
            @test length(cases)==21
            @test Set(c.n for c in cases)==Set((0,1,2,3,7,31,65))
            @test all(c->length(c.initial)==length(c.expected)==c.n+7,cases)
            @test all(c->c.initial[c.n+1:end]==c.expected[c.n+1:end],cases)
            @test all(c->all(isfinite,c.expected),cases)
        end
        projection=private_rpn_native_cases(fixtures[5])
        @test all(c->c.expected[1:c.n]==clamp.(c.x[1:c.n],0f0,1f0),projection)
        # It is forbidden to create a full hardware packet during generation.
        summary=JSON3.read(read(joinpath(@__DIR__,"rpn_stencil_results/summary.json"),String))
        if !summary.complete
            @test_throws ErrorException prepare_rpn_native_audit()
            @test !isfile(joinpath(@__DIR__,"rpn_stencil_results/native_audit_input_PRIVATE.json"))
        end
    end
end
