include("summarize_alias_transfer3.jl")
using Test
function test_alias3_exception_kind()
    @testset "Exception categories do not invent algorithmic causes" begin
        @test alias3_exception_kind("ERROR: LoadError: UndefVarError: x not defined")=="UndefVarError"
        @test alias3_exception_kind("ERROR: MethodError: no method matching f()")=="MethodError"
        @test alias3_exception_kind("ERROR: ParseError: invalid syntax")=="ParseError"
        @test alias3_exception_kind("AssertionError in hidden harness")=="test_failure_without_classified_julia_exception"
        @test alias3_exception_kind("")=="test_failure_without_classified_julia_exception"
    end
end
