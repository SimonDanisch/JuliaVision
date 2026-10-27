include("audit_plot_validator_render_reuse.jl")
using Test
function test_plot_validator_render_reuse()
    @testset "Exact-original render reuse is not new rendering" begin
        row=(;admitted=true,retained_original=true,selected_generation_id="original",generation_sha256="hash")
        prior=(;id="original",generation_sha256="hash",rendered=true,within_canvas=true)
        @test plot_validator_render_reuse_decision(row,prior)=="reused_original_unclipped"
        @test plot_validator_render_reuse_decision((;row...,admitted=false),nothing)=="unsupported_output"
        @test plot_validator_render_reuse_decision((;row...,retained_original=false),nothing)=="pending_new_plan_render"
        @test plot_validator_render_reuse_decision(row,(;prior...,within_canvas=false))=="reused_original_clipped"
        @test_throws ErrorException plot_validator_render_reuse_decision(row,nothing)
        @test_throws ErrorException plot_validator_render_reuse_decision(row,(;prior...,generation_sha256="wrong"))
    end
end
