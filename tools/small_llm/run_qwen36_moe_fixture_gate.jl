include("device_finish.jl")
include("test_qwen36_moe_model.jl")
include("test_fused_residual_norm.jl")
using JSON3,SHA,Test

function run_qwen36_moe_fixture_gate(expected_sources;output=joinpath(@__DIR__,"qwen36_moe_fixture_gate/result.json"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    for (file,hash) in pairs(expected_sources)
        path=String(file)=="BonsaiRunner/src/kernels.jl" ? joinpath(dirname(dirname(@__DIR__)),String(file)) : joinpath(@__DIR__,String(file))
        bytes2hex(sha256(read(path)))==hash||error("Fixture gate source drift: $file")
    end
    ispath(output)&&error("Preserve first fixture attempt");mkpath(dirname(output))
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"source_sha256"=>expected_sources,
        "scope"=>"Tiny4-layer full MoE plus shared fused-normalization hardware qualification; NOT35B checkpoint or task-quality/throughput proof")
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon");rows=Ref{Any}(nothing)
            try
                tests=@testset "Full tiny MoE hardware fixture and residual regression" begin
                    test_fused_residual_norm(device)
                    rows[]=test_qwen36_moe_model(device,joinpath(@__DIR__,"qwen36_moe_tiny_reference"))
                end
                counts=Test.get_test_counts(tests)
                report["checks"]=counts.passes+counts.cumulative_passes
                report["rows"]=rows[];report["complete"]=true
            finally
                finish_device!(device);report["normal_final_waitidle"]=true;save()
            end
            nothing
        end
    end
end
