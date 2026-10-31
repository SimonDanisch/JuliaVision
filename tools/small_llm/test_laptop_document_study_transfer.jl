include("laptop_document_study_transfer.jl")
using Test
function test_laptop_document_study_transfer()
    @testset "Public-only document deployment boundaries" begin
        mktempdir() do directory
            write(joinpath(directory,"module.jl"),"# public source")
            data=read(joinpath(directory,"module.jl"))
            p=(;source_sha256=Dict("module.jl"=>bytes2hex(sha256(data))))
            path=joinpath(directory,"conditions.json");open(io->JSON3.write(io,p),path,"w")
            packet=document_study_payload(path;source_root=directory)
            @test Set(keys(packet.updates))==Set(("module.jl","bigcodebench_x_doc_results/conditions.json"))
            @test validate_document_study_payload(packet)===nothing
            mock=(f,packet)->Dict(n=>bytes2hex(sha256(d)) for (n,d) in packet.updates)
            @test deploy_document_study(mock,packet).complete
            @test_throws ErrorException deploy_document_study((f,p)->Dict{String,String}(),packet)
            @test_throws ErrorException validate_document_study_payload((;root="/tmp/other",updates=packet.updates))
            for bad in ("../escape.jl","PRIVATE_selected_reference_tests.jsonl","summary.json","candidate_score.json","nested/file.jl")
                @test_throws ErrorException validate_document_study_payload((;root=packet.root,updates=Dict(bad=>UInt8[])))
            end
            write(joinpath(directory,"module.jl"),"# changed")
            @test_throws ErrorException document_study_payload(path;source_root=directory)
        end
    end
end
