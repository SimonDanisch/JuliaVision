include("typed_pointwise_kernel_lowering.jl")
using Test

function test_typed_pointwise_kernel_lowering()
    good="""
    using KernelInterface
    kernel!(out::Vector{Float32},x::Vector{Float32},indices::Vector{Int32},n::Int32,m::Int32)=begin
        for i=1:n
            idx=clamp(indices[i],1,m)
            out[i]=x[idx]
        end
    end
    """
    @testset "Typed pointwise admission, no correctness oracle" begin
        result=CheckedTypedPointwiseLowering.lower(good)
        @test result.accepted
        @test occursin("AbstractVector{Int32}",result.code)
        @test occursin("get_global_id",result.code)
        @test result.certificate.stores_per_invocation==1
        @test CheckedTypedPointwiseLowering.lower(replace(good,"out[i]=x[idx]"=>"out[i]=-x[idx]")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"indices::Vector{Int32}"=>"indices::Vector{Int64}")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"indices::Vector{Int32}"=>"indices::Vector{Bool}")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"out::Vector{Float32}"=>"out::Vector{Int32}")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"n::Int32"=>"n::Float32")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"m::Int32"=>"clamp::Int32")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"out[i]=x[idx]"=>"out[i]=out[i]+x[idx]")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"out[i]=x[idx]"=>"out[idx]=x[idx]")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"out[i]=x[idx]"=>"out[i]=unknown(x[idx])")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"for i=1:n"=>"for i=2:n")).accepted
        @test !CheckedTypedPointwiseLowering.lower(replace(good,"out[i]=x[idx]"=>"@inbounds out[i]=x[idx]")).accepted
    end
    good
end
