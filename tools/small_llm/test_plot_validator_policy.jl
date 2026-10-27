include("plot_validator_policy.jl")
include("plot_interface_references.jl")
using Test
function test_plot_validator_policy()
    @testset "Public admission policy never consults hidden correctness" begin
        good=(;answer=JSON3.write(first(plot_interface_references())),truncated=false)
        @test !plot_validator_trigger(good).triggered
        wrong=JSON3.read(good.answer,Dict{String,Any});wrong["axes"][1]["title"]="Semantically WRONG"
        @test !plot_validator_trigger((;answer=JSON3.write(wrong),truncated=false)).triggered
        @test plot_validator_trigger((;good...,truncated=true)).triggered
        @test plot_validator_trigger((;answer="{}",truncated=false)).triggered
        @test_throws ArgumentError plot_validator_trigger((;inference_error="synthetic"))
        original=JSON3.read(JSON3.write(["system"=>"Public schema","user"=>"Public request"]))
        bad=(;answer="{}",truncated=false)
        control=plot_validator_revision_messages(original,bad,"self_review")
        tool=plot_validator_revision_messages(original,bad,"validator_feedback")
        @test length(control)==length(tool)==4
        @test control[1:3]==tool[1:3]
        @test last(control[3])=="{}"
        @test occursin("validator observation",last(tool[4]))
        @test !occursin("Unknown or missing fields",last(control[4]))
        @test_throws ArgumentError plot_validator_revision_messages(original,good,"self_review")
        @test_throws ArgumentError plot_validator_revision_messages(original,bad,"hidden_feedback")
        @test_throws ArgumentError plot_validator_revision_messages(JSON3.read("[]"),bad,"self_review")
    end
end
