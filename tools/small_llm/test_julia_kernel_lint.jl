include("julia_kernel_lint.jl")
using Test
@testset "Advisory atomic extrema AST check" begin
    for code in ("Atomix.@atomic a[1] = max(a[1],x)","Atomix.@atomic :monotonic a[1] = min(x,a[1])",
                 "@atomic a[i] = Base.max(a[i],x)")
        rows=JuliaKernelLint.atomic_extrema_assignments(code)
        @test length(rows)==1
        @test only(rows).severity=="advisory"
    end
    for code in ("Atomix.@atomic max(a[1],x)","Atomix.@atomic :monotonic min(a[1],x)",
                 "Atomix.@atomic a[1] += x","Atomix.@atomic a[1] = x",
                 "Atomix.@atomic a[1] = max(b[1],x)","a[1]=max(a[1],x)")
        @test isempty(JuliaKernelLint.atomic_extrema_assignments(code))
    end
    @test isempty(JuliaKernelLint.atomic_extrema_assignments("error(\"not executed\")"))
    @test_throws ArgumentError JuliaKernelLint.atomic_extrema_assignments("x"^65537)
end
