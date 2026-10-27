include("boundary_example_synthesis.jl")
include("public_example_fallback.jl")
using Test
function test_stencil_boundary_contract()
    C=StencilBoundaryContract
    @testset "Boundary contract independent of task ID and hidden answers" begin
        @test C.recognize("Use the previous clamped neighbor.")==:clamped
        @test C.recognize("Square previous and next periodic neighbors.")==:periodic
        for text in ("Clamp the current value to zero through one.","Not the next periodic neighbor.",
                "Previous periodic neighbor and next clamped neighbor.","Unspecified boundary.")
            @test C.recognize(text)==:unknown
        end
        @test C.accepts("x0\nfinish",:clamped)
        @test C.accepts("x_prev_clamped\nx0\nsub\nzero\nmax\nfinish",:clamped)
        @test !C.accepts("x_prev_periodic\nx0\nsub\nzero\nmax\nfinish",:clamped)
        @test !C.accepts("x_next_clamped\nsquare\nfinish",:periodic)
        @test !C.accepts("arbitrary Julia",:unknown)
        @test_throws ArgumentError C.accepts("x0\nfinish",:invalid)
        # Deliberately ambiguous public cases: both boundary modes fit them.
        bank=RPNExampleSemantics.Bank([(;input=Float32[2,-1,0.5,-2],output=Float32[0,3,0,2.5]),
            (;input=Float32[-0.75],output=Float32[0])])
        wrong="zero\nx_prev_periodic\nx0\nsub\nmax\nfinish"
        @test PublicExampleFallback.matches(wrong,bank)
        @test !C.accepts(wrong,:clamped)
        for mode in (:clamped,:periodic)
            proposal=BoundaryExampleSynthesis.solve(bank,mode)
            @test proposal.found
            @test C.accepts(proposal.program,mode)
            @test PublicExampleFallback.matches(proposal.program,bank)
        end
        @test_throws ArgumentError BoundaryExampleSynthesis.solve(bank,:clamped;seconds=0)
        @test_throws ArgumentError BoundaryExampleSynthesis.solve(bank,:invalid)
    end
end
