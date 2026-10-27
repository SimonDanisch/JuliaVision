include("laptop_sampling_full_model.jl")
using Test

function test_laptop_sampling_registration()
    tests=@testset "Laptop sampled checkpoint registration, CPU only" begin
        mktempdir() do directory
            p=freeze_laptop_sampling_gate(;directory)
            @test p.capacity==256
            @test p.prepare_decode==4 && p.continuation_decode==8
            @test !isempty(p.ids)
            @test laptop_sampling_gate_state(directory).packet.portable_manifest_sha256==p.portable_manifest_sha256
            @test_throws ErrorException freeze_laptop_sampling_gate(;directory)
            @test_throws ErrorException qualify_laptop_sampling_gate(:invalid;directory)
            @test_throws ErrorException qualify_laptop_sampling_gate(:prepare;seconds=0,directory)
            path=joinpath(directory,"conditions.json")
            bad=JSON3.read(read(path,String),Dict{String,Any})
            bad["source_sha256"]["sampling_checkpoint.jl"]=repeat("0",64)
            open(io->JSON3.pretty(io,bad),path,"w")
            @test_throws ErrorException laptop_sampling_gate_state(directory)
        end
    end
    c=Test.get_test_counts(tests)
    report=(;complete=true,scope="CPU registration/integrity only; NOT full-model GPU continuation qualification",
        passed=c.passes+c.cumulative_passes,source_sha256=Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in
            ("laptop_sampling_full_model.jl","test_laptop_sampling_registration.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"laptop_sampling_registration_validation.json"),"w");report
end
