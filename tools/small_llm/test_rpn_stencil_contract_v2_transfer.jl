include("rpn_stencil_contract_v2_transfer.jl")
using Test
function test_rpn_stencil_contract_v2_transfer()
    @testset "Posthoc v2 source set excludes private grading/feedback" begin
        mktempdir() do root
            parentdir=mkpath(joinpath(root,"rpn_stencil_results"));parent=joinpath(parentdir,"conditions.json")
            write(parent,JSON3.write((;source_sha256=Dict("production.jl"=>"unused"))))
            sources=Dict(name=>begin
                path=joinpath(root,name);write(path,"# public "*name);bytes2hex(sha256(read(path)))
            end for name in ("production.jl","rpn_stencil_contract_v2.jl","rpn_stencil_contract_v2_study.jl"))
            path=joinpath(root,"conditions.json");parenthash=bytes2hex(sha256(read(parent)))
            save(s,h=parenthash)=write(path,JSON3.write((;source_sha256=s,format_parent_conditions_sha256=h)))
            save(sources);p=rpn_contract_v2_source_payload(path;source_root=root)
            @test Set(keys(p.updates))==Set(keys(sources))
            for forbidden in ("qualify_rpn_stencil_tasks.jl","score_rpn_stencil_study.jl","native_audit_input_PRIVATE.json","../escape.jl")
                save(merge(sources,Dict(forbidden=>"unused")))
                @test_throws ErrorException rpn_contract_v2_source_payload(path;source_root=root)
            end
            save(sources,"wrong")
            @test_throws ErrorException rpn_contract_v2_source_payload(path;source_root=root)
            save(sources);write(joinpath(root,"production.jl"),"# changed")
            @test_throws ErrorException rpn_contract_v2_source_payload(path;source_root=root)
        end
    end
end
