include("laptop_rpn_gate_transfer.jl")
using Test
function test_laptop_rpn_gate_transfer()
    @testset "RPN gate public source packet, no private proof transfer" begin
        mktempdir() do directory
            file=joinpath(directory,"production.jl");write(file,"# public")
            path=joinpath(directory,"conditions.json")
            save(name,hash)=open(io->JSON3.write(io,(;source_sha256=Dict(name=>hash))),path,"w")
            save("production.jl",bytes2hex(sha256(read(file))))
            p=rpn_gate_source_payload(path;source_root=directory)
            @test Set(keys(p.updates))==Set(["production.jl"])
            @test !haskey(p.updates,"bigcodebench_x_doc_results/conditions.json")
            @test !haskey(p.updates,"laptop_rpn_gate/conditions.json")
            @test p.updates["production.jl"]==read(file)
            write(file,"# changed")
            @test_throws ErrorException rpn_gate_source_payload(path;source_root=directory)
            for forbidden in ("rpn_pointwise_semantics_validation.json","PRIVATE_selected_reference_tests.jsonl","../escape.jl","summary.json")
                save(forbidden,"not-a-hash")
                @test_throws ErrorException rpn_gate_source_payload(path;source_root=directory)
            end
        end
    end
end
