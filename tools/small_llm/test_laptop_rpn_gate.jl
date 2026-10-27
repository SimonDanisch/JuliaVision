include("laptop_rpn_gate.jl")
using Test
function test_laptop_rpn_gate()
    @testset "CPU-only RPN gate registration, private proofs not deployed" begin
        mktempdir() do directory
            p=freeze_laptop_rpn_gate(;directory)
            @test p.phases==["kernels","identity"]
            @test p.decoding.max_tokens==32
            @test length(p.cpu_proof_commitments)==5
            @test length(p.action_token_ids)==18
            @test length(unique(Tuple.(p.action_token_ids)))==18
            @test !haskey(p.source_sha256,"qualify_rpn_pointwise_semantics.jl")
            @test !haskey(p.source_sha256,"rpn_pointwise_semantics_validation.json")
            @test haskey(p.source_sha256,"sampled_rpn_program.jl")
            @test haskey(p.source_sha256,"sampled_julia.jl")
            @test readdir(directory)==["conditions.json"]
            @test_throws ErrorException freeze_laptop_rpn_gate(;directory)
        end
        @test_throws ArgumentError qualify_laptop_rpn_gate(:unknown)
        refused=try
            qualify_laptop_rpn_gate(:kernels)
            "unexpected success"
        catch err
            sprint(showerror,err)
        end
        @test occursin("Laptop snapshot only",refused)
    end
end
