include("kernel_bounds_advisory.jl")
using Test
@testset "Global-store bounds advisories, not proofs" begin
    check=KernelBoundsAdvisory.direct_global_stores
    prefix="i=Int32(KI.get_global_id().x);"
    @test length(check(prefix*"out[i]=x[i]"))==1
    @test isempty(check(prefix*"if i<=n;out[i]=x[i];end"))
    @test isempty(check(prefix*"i<=n && (@inbounds out[i]=x[i])"))
    @test isempty(check(prefix*"if n>=i;out[i]=x[i];end"))
    @test isempty(check(prefix*"if 1<=i<=n;out[i]=x[i];end"))
    @test isempty(check(prefix*"if i<=rows*cols;out[i]=x[i];end"))
    @test length(check(prefix*"if i<=n || flag;out[i]=x[i];end"))==1
    @test isempty(check(prefix*"if i<=n && flag;out[i]=x[i];end"))
    @test length(check(prefix*"for j in 1:n;out[i]=x[j];end"))==1
    @test isempty(check("for i in 1:n;out[i]=x[i];end"))
    @test length(check(prefix*"if flag;out[i]=x[i];else;out[i]=0;end"))==2
    @test KernelBoundsAdvisory.global_index_calls(prefix)==1
    @test KernelBoundsAdvisory.global_index_calls("for i in 1:n;out[i]=x[i];end")==0
    @test KernelBoundsAdvisory.global_index_calls("# KI.get_global_id()\n\"KI.get_global_id()\"")==0
end
