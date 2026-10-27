include("rpn_stencil_transfer.jl")
using Test
function test_rpn_stencil_transfer()
    @testset "RPN stencil positive source set excludes private tests/scorer" begin
        mktempdir() do root
            gate=mkpath(joinpath(root,"laptop_rpn_gate"))
            write(joinpath(gate,"conditions.json"),JSON3.write((;source_sha256=Dict("production.jl"=>"unused"))))
            sources=Dict(name=>begin
                path=joinpath(root,name);write(path,"# public "*name);bytes2hex(sha256(read(path)))
            end for name in ("production.jl","rpn_stencil_tasks.jl","rpn_stencil_study.jl"))
            path=joinpath(root,"conditions.json")
            save(s)=write(path,JSON3.write((;source_sha256=s)))
            save(sources)
            p=rpn_stencil_source_payload(path;source_root=root)
            @test Set(keys(p.updates))==Set(keys(sources))
            @test !haskey(p.updates,"qualify_rpn_stencil_tasks.jl")
            @test !haskey(p.updates,"score_rpn_stencil_study.jl")
            for private in ("qualify_rpn_stencil_tasks.jl","score_rpn_stencil_study.jl","../escape.jl","rpn_stencil_reference_validation.json")
                save(merge(sources,Dict(private=>"unused")))
                @test_throws ErrorException rpn_stencil_source_payload(path;source_root=root)
            end
            save(Dict(k=>v for (k,v) in sources if k!="production.jl"))
            @test_throws ErrorException rpn_stencil_source_payload(path;source_root=root)
            save(sources);write(joinpath(root,"production.jl"),"# drift")
            @test_throws ErrorException rpn_stencil_source_payload(path;source_root=root)
        end
    end
end
