include("rpn_public_example_transfer.jl")
using Test
function test_rpn_public_example_transfer()
    @testset "Numeric-mask deployment excludes PRIVATE formulas and scorers" begin
        mktempdir() do root
            parentdir=mkpath(joinpath(root,"laptop_example_identity_gate"))
            parent=joinpath(parentdir,"conditions.json");write(parent,JSON3.write((;sources=Dict("production.jl"=>"unused"))))
            names=("production.jl","rpn_public_example_tasks.jl","rpn_public_example_study.jl")
            sources=Dict(name=>begin
                path=joinpath(root,name);write(path,"# public "*name);bytes2hex(sha256(read(path)))
            end for name in names)
            path=joinpath(root,"conditions.json");parenthash=bytes2hex(sha256(read(parent)))
            save(s,h=parenthash)=write(path,JSON3.write((;source_sha256=s,identity_conditions_sha256=h)))
            save(sources);packet=public_example_source_payload(path;source_root=root)
            @test Set(keys(packet.updates))==Set(names)
            for forbidden in ("qualify_rpn_public_example_tasks.jl","score_rpn_public_example_study.jl","PRIVATE_cases.json","../escape.jl")
                save(merge(sources,Dict(forbidden=>"unused")))
                @test_throws ErrorException public_example_source_payload(path;source_root=root)
            end
            save(sources,"wrong")
            @test_throws ErrorException public_example_source_payload(path;source_root=root)
            save(sources);write(joinpath(root,"production.jl"),"# drift")
            @test_throws ErrorException public_example_source_payload(path;source_root=root)
        end
    end
end
