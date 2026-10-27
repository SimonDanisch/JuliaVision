include("stencil_boundary_max_proof.jl")
using Test
function test_stencil_boundary_max_proof()
    P=StencilBoundaryMaxProof
    @testset "Exact max coverage proof is not a task-specific repair" begin
        redundant="x0\nx_prev_periodic\nx_next_periodic\nmax\nmax\nx_prev_clamped\nx_next_clamped\nmax\nmax\nfinish"
        @test !StencilBoundaryContract.accepts(redundant,:periodic)
        proof=P.certify(redundant,:periodic)
        @test proof.accepted&&length(proof.rules)==2
        @test !P.certify("x0\nx_prev_clamped\nmax\nfinish",:periodic).accepted
        @test !P.certify("x0\nx_next_clamped\nmax\nfinish",:periodic).accepted
        @test !P.certify("x0\nx_prev_periodic\nmax\nx_prev_clamped\nmin\nfinish",:periodic).accepted
        @test !P.certify(redundant,:clamped).accepted
        @test P.certify("x0\nx_prev_periodic\nmax\nfinish",:periodic).accepted
        @test !P.certify("malformed",:periodic).accepted
        @test_throws ArgumentError P.certify(redundant,:invalid)
        for n in 1:8,i in 1:n
            @test max(1,i-1) in (i,mod1(i-1,n))
            @test min(n,i+1) in (i,mod1(i+1,n))
        end
    end
end
