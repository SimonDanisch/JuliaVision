include("pointwise_kernel_lowering.jl")
using Test
@testset "Restricted pointwise lowering, not a task oracle" begin
    good="function kernel!(out::Vector{Float32}, x::Vector{Float32}, n::Int32);for i in 1:n;out[i]=x[i]*2f0;end;end"
    lowered=PointwiseKernelLowering.lower(good)
    @test lowered.accepted
    @test occursin("AbstractVector{Float32}",lowered.code)
    @test occursin("get_global_id",lowered.code)
    @test lowered.certificate.stores_per_invocation==1
    @test !PointwiseKernelLowering.lower(replace(good,"out[i]=x[i]*2f0"=>"out[1]=x[i]*2f0")).accepted
    @test !PointwiseKernelLowering.lower(replace(good,"out[i]=x[i]*2f0"=>"out[i]=out[i]+x[i]")).accepted
    @test !PointwiseKernelLowering.lower(replace(good,"out[i]=x[i]*2f0"=>"out[i]=unknown(x[i])")).accepted
    @test !PointwiseKernelLowering.lower(replace(good,"out[i]=x[i]*2f0"=>"s=s+x[i];out[i]=s")).accepted
    @test !PointwiseKernelLowering.lower(replace(good,"out[i]=x[i]*2f0"=>"j=KI.get_global_id().x;out[i]=x[i]")).accepted
    @test !PointwiseKernelLowering.lower(replace(good,"for i in 1:n"=>"for i in 2:n")).accepted
    @test !PointwiseKernelLowering.lower(replace(good,"Float32"=>"Float64")).accepted
end
