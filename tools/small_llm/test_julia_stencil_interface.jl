include("julia_stencil_interface.jl")
using Test
function test_julia_stencil_interface()
    J=JuliaStencilInterface
    wrap(expr)="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\nout[i]="*expr*"\nend\nreturn nothing\nend"
    @testset "Julia stencil source is parsed, never executed" begin
        for expr in ("x[i]","abs(x[i])","x[i]^2","min(x[max(1,i-1)],x[min(n,i+1)])",
                "0.5f0*(x[mod1(i-1,n)]+x[mod1(i+1,n)])","clamp(x[i],0f0,1f0)","x[i]*x[mod1(i+1,n)]")
            result=J.lower(wrap(expr));@test result.accepted
            @test RPNPointwiseInterface.parse_program(result.program).accepted
            @test occursin("KI.get_global_id",result.code)
        end
        result=J.lower("function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\na=x[i]\nb=a*a\nout[i]=max(0f0,1f0-b)\nend\nend")
        @test result.accepted
        @test !("a" in split(result.program,'\n')) && !("b" in split(result.program,'\n'))
        @test !J.lower(wrap(repeat("-(",100)*"x[i]"*repeat(")",100))).accepted
        for expr in ("x[i+1]","x[n*10]","out[i]","x[1]","x[i]+1.0","x[i]+1",
                "sin(x[i])","eval(x[i])","error(\"oops\")","Base.max(x[i],0f0)",
                "clamp(x[i],1f0,0f0)","x[i]/0f0","unknown","Float32(3)")
            @test !J.lower(wrap(expr)).accepted
        end
        for source in ("error(\"not executed\")",wrap("x[i]")*"\nerror(\"not executed\")",
                replace(wrap("x[i]"),"for i"=>"@inbounds for i"),
                replace(wrap("x[i]"),"out[i]="=>"x[i]="),
                replace(wrap("x[i]"),"i=1:n"=>"i=0:n"),
                replace(wrap("x[i]"),"n::Int32"=>"n::Int"))
            @test !J.lower(source).accepted
        end
    end
end
