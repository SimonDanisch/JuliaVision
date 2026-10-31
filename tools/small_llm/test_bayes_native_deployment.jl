include("deploy_bayes_native_audit.jl")
using Test
function test_bayes_native_deployment()
    @testset "Native deployment exposes source ONLY" begin
        updates=bayes_native_source_updates()
        @test length(updates)==8
        @test Set(keys(updates))==Set(BAYES_NATIVE_PUBLIC_FILES)
        @test all(f->basename(f)==f&&endswith(f,".jl")||f==keys(updates))
        @test !any(f->endswith(f,".json")||occursin("private_packet",f),keys(updates))
        @test all(bytes->!isempty(bytes),values(updates))
    end
end
