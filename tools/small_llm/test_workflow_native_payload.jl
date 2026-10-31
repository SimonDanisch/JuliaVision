include("run_workflow_native_audit.jl")
using Test
function test_workflow_native_payload()
    sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in WORKFLOW_NATIVE_SOURCES)
    base=(;rows=[(;id="row"*string(i)) for i in 1:24],controls=[(;id="control"*string(i)) for i in 1:4],source_sha256=sources)
    encode(p)=Vector{UInt8}(codeunits(JSON3.write(p)))
    validate(p)=let b=encode(p);validate_workflow_native_payload(b,bytes2hex(sha256(b))) end
    @testset "Bounded variable-size grading packet, no model/device setup" begin
        @test length(validate(base).rows)==24
        @test length(validate((;base...,rows=[(;id="single")])).rows)==1
        @test_throws ErrorException validate((;base...,rows=Any[]))
        @test_throws ErrorException validate((;base...,rows=[(;id=string(i)) for i in 1:65]))
        @test_throws ErrorException validate((;base...,rows=[(;id="duplicate"),(;id="duplicate")]))
        @test_throws ErrorException validate((;base...,controls=base.controls[1:3]))
        @test_throws ErrorException validate_workflow_native_payload(encode(base),"wrong")
        changed=copy(sources);changed["device_finish.jl"]="wrong"
        @test_throws ErrorException validate((;base...,source_sha256=changed))
        delete!(changed,"device_finish.jl");changed["../device_finish.jl"]="wrong"
        @test_throws ErrorException validate((;base...,source_sha256=changed))
    end
end
