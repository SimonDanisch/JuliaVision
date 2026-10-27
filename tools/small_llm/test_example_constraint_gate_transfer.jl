include("example_constraint_gate_transfer.jl")
using Test
function test_example_constraint_gate_transfer()
    @testset "Public synthetic gate has no benchmark/reference payload" begin
        packet=example_constraint_gate_payload()
        @test length(packet.updates)==18
        @test Set(keys(packet.updates))==Set(EXAMPLE_GATE_SOURCES)
        @test all(name->basename(name)==name&&(endswith(name,".jl")||endswith(name,".py")),keys(packet.updates))
        @test all(name->!occursin("PRIVATE",name)&&!occursin("qualify_rpn_stencil",name),keys(packet.updates))
        p=JSON3.read(packet.conditions)
        @test p.expected_checks==1627
        @test_throws ErrorException run_laptop_example_constraint_gate() # ROOT guard before Device
        @test all(bytes2hex(sha256(data))==p.sources[file] for (file,data) in packet.updates)
    end
end
