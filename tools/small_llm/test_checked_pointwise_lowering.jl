include("checked_pointwise_lowering.jl")
using Test
@testset "Checked pointwise admission" begin
    code="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32);for i in 1:n;out[i]=x[i]*2f0;end;end"
    original=PointwiseKernelLowering.lower(code);checked=CheckedPointwiseLowering.lower(code)
    @test checked.accepted
    @test checked.code==original.code
    @test eltype(1:Int32(3))===Int
    @test eltype(1:Int(3))===Int
    for limit in ("n/2","n*1f0","Float32(n)")
        source=replace(code,"1:n"=>"1:($limit)")
        @test PointwiseKernelLowering.lower(source).accepted
        @test !CheckedPointwiseLowering.lower(source).accepted
    end
    @test !CheckedPointwiseLowering.lower(replace(code,"out"=>"KI")).accepted
    @test !CheckedPointwiseLowering.lower(replace(code,"out"=>"Base")).accepted
    @test !CheckedPointwiseLowering.lower(replace(code,"out"=>"nothing")).accepted
    @test CheckedPointwiseLowering.lower(replace(code,"1:n"=>"1:div(n,2)")).accepted
    @test !CheckedPointwiseLowering.lower(replace(code,"for i in 1:n"=>"for i in 1:n", "out[i]=x[i]*2f0"=>"out[i]=out[i]+x[i]")).accepted
end
