include("julia_stencil_admission_v3.jl")
using Test
function test_julia_stencil_admission_v3()
    wrap(body)="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\n"*body*"\nend"
    loop="for i=1:n\nout[i]=x[i]\nend"
    @testset "Source control flow is not erased by canonical lowering" begin
        for body in ("return nothing\n"*loop,loop*"\nreturn nothing\nreturn nothing")
            source=wrap(body)
            @test JuliaStencilAdmissionV2.lower(source).accepted # reproduce gap
            @test !JuliaStencilAdmissionV3.lower(source).accepted
        end
        for name in ("_","__","___")
            source=wrap("for i=1:n\n"*name*"=x[i]\nout[i]="*name*"\nend")
            @test !JuliaStencilAdmissionV3.lower(source).accepted
        end
        for body in (loop,loop*"\nreturn nothing","nothing\n"*loop*"\nnothing\nreturn nothing")
            source=wrap(body)
            a=JuliaStencilAdmissionV2.lower(source);b=JuliaStencilAdmissionV3.lower(source)
            @test a.accepted&&b.accepted
            @test a.program==b.program&&a.code==b.code
        end
        @test !JuliaStencilAdmissionV3.lower(wrap("for i=1:n\nmod1=x[i]\nout[i]=x[mod1(i+1,n)]\nend")).accepted
    end
end
