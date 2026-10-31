include("laptop_controller_controls.jl")
using Test

@testset "same-host control registration (CPU only)" begin
    mktempdir() do tmp
        directory=joinpath(tmp,"controls")
        p=freeze_laptop_controller_controls(;directory)
        @test length(p.jobs)==6
        @test length(p.pilot_output_sha256)==6
        @test p.thinking_limit==Int(typemax(UInt32))
        @test p.decoding.max_tokens==2048
        @test p.thinking_budget==1024 # preserves pilot metadata; only thinking_limit is used
        @test occursin("Posthoc",p.scope)
        @test_throws ErrorException freeze_laptop_controller_controls(;directory)
        s=laptop_control_state(directory)
        @test length(s.packet.jobs)==6
        @test_throws ArgumentError run_laptop_controller_controls(;directory,seconds=0)
        @test_throws ArgumentError run_laptop_controller_controls(;directory,seconds=0)
        # Invalid source receipt must fail before any device/tokenizer setup.
        packet=JSON3.read(read(s.path,String),Dict{String,Any})
        packet["source_sha256"]["laptop_controller_controls.jl"]="invalid"
        open(io->JSON3.pretty(io,packet),s.path,"w")
        @test_throws ErrorException laptop_control_state(directory)
    end
end
