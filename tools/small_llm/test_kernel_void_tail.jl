include("kernel_void_tail.jl")
using Test
@testset "Kernel void-tail normalization" begin
    source="function kernel!(out,x,n::Int32);i=KI.get_global_id().x;if i<=n;out[i]=x[i]*2f0;end;end"
    result=KernelVoidTail.normalize(source)
    @test result.accepted
    @test occursin("out[i] = x[i] * 2.0f0",result.code)
    @test occursin("nothing",result.code)
    @test !KernelVoidTail.normalize("").accepted
    @test !KernelVoidTail.normalize("function solve(x);x;end").accepted
    @test !KernelVoidTail.normalize(replace(source,"out,x"=>"out,nothing")).accepted
    @test !KernelVoidTail.normalize(replace(source,"i=KI"=>"nothing=1;i=KI")).accepted
    # Explicit valued returns are NOT silently rewritten or certified void.
    valued=KernelVoidTail.normalize("kernel!(out)=return out")
    @test valued.accepted
    @test occursin("return out",valued.code)
end
