include("run_compact_bayes_demo_native_audit.jl")
using Test
function test_compact_bayes_demo_native_audit(device,bytes,hash)
    @testset "All admitted generated networks, including wrong ones" begin
        p=validate_compact_demo_native_payload(bytes,hash)
        @test length(p.rows)==4&&length(p.generation_pids)==8
        @test_throws ErrorException validate_compact_demo_native_payload(bytes,"wrong")
        mutate(f)=begin
            x=JSON3.read(String(copy(bytes)),Dict{String,Any});f(x);b=Vector{UInt8}(codeunits(JSON3.write(x)))
            validate_compact_demo_native_payload(b,bytes2hex(sha256(b)))
        end
        @test_throws ErrorException mutate(x->(x["rows"]=[]))
        @test_throws ErrorException mutate(x->(x["generation_pids"][2]=x["generation_pids"][1]))
        @test_throws ErrorException mutate(x->(x["rows"][1]["network_sha256"]="wrong"))
        @test_throws ErrorException mutate(x->(x["rows"][1]["expected"]["defined"][1]=2))
        @test_throws ErrorException mutate(x->(x["rows"][1]["expected"]["prior"]=[]))
        @test_throws ErrorException mutate(x->(x["source_sha256"]["portable_evidence.jl"]="wrong"))
        for row in p.rows
            @test audit_bayes_native_case(device,String(row.network),row.expected).passed
        end
    end
end
