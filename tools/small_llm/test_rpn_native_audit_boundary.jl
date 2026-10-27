include("run_rpn_native_audit.jl")
using Test
function test_rpn_native_audit_boundary()
    @testset "Native private audit input boundary, no Device or model" begin
        rows=[(;id="test_"*string(i)) for i in 1:16]
        sources=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in RPN_NATIVE_AUDIT_SOURCES)
        packet=(;rows,controls=fill((;id="control"),4),source_sha256=sources)
        bytes=JSON3.write(packet);hash=bytes2hex(sha256(bytes))
        @test length(validate_rpn_native_audit_payload(bytes,hash).rows)==16
        @test_throws ErrorException validate_rpn_native_audit_payload(bytes,"wrong")
        for change in ((;rows=rows[1:15]),(;rows=fill(first(rows),16)),(;controls=Any[]),
                (;source_sha256=Dict("../escape"=>"wrong")),(;source_sha256=merge(sources,Dict("rpn_kernel_device_audit.jl"=>"wrong"))),
                (;source_sha256=Dict{String,String}()))
            changed=JSON3.write((;packet...,change...))
            @test_throws ErrorException validate_rpn_native_audit_payload(changed,bytes2hex(sha256(changed)))
        end
        @test_throws ErrorException run_rpn_native_audit(bytes;expected_sha256=hash)
    end
end
