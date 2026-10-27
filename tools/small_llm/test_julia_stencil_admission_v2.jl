include("julia_stencil_admission_v2.jl")
using Test
function test_julia_stencil_admission_v2()
    wrap(body)="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\n"*body*"\nend\nend"
    @testset "Helper shadowing fails closed without changing frozen V1" begin
        bad=wrap("mod1=x[i]\nout[i]=x[mod1(i+1,n)]")
        @test JuliaStencilInterface.lower(bad).accepted # reproduced V1 gap
        @test !JuliaStencilAdmissionV2.lower(bad).accepted
        @test occursin("mod1",JuliaStencilAdmissionV2.lower(bad).reason)
        for expr in ("x[i]","x[mod1(i-1,n)]","x[mod1(i+1,n)]","abs(x[i])","max(0f0,x[i])")
            source=wrap("a="*expr*"\nout[i]=a")
            a=JuliaStencilInterface.lower(source);b=JuliaStencilAdmissionV2.lower(source)
            @test a.accepted&&b.accepted
            @test a.program==b.program && a.code==b.code
        end
        @test !JuliaStencilAdmissionV2.lower("error(\"never executed\")").accepted
    end
end
