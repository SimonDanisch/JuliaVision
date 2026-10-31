include("collect_public_example_outputs.jl")
using Test
function test_collect_public_example_outputs()
    @testset "Dependency-free read-only collection boundary" begin
        @test public_example_output_names([(;id="task_enabled")])==["task_enabled"]
        @test_throws ArgumentError public_example_output_names(Any[])
        @test_throws ArgumentError public_example_output_names([(;id="a"),(;id="a")])
        for name in ("","..",".","../escape","x/y","x\\y","bad\0name")
            @test_throws ArgumentError public_example_output_names([(;id=name)])
        end
        mktempdir() do root
            names=["a","b"]
            files=Dict("a.json"=>Vector{UInt8}(codeunits("output")),"a_attempt.json"=>Vector{UInt8}(codeunits("attempt")))
            @test save_immutable_example_outputs(root,files,names)==1
            @test save_immutable_example_outputs(root,files,names)==1
            @test read(joinpath(root,"a.json"),String)=="output"
            @test_throws ErrorException save_immutable_example_outputs(root,Dict("a.json"=>UInt8[]),names)
            @test_throws ErrorException save_immutable_example_outputs(root,Dict("../escape.json"=>UInt8[]),names)
            bad=merge(files,Dict("a.json"=>UInt8[],"b.json"=>UInt8[],"b_attempt.json"=>UInt8[]))
            @test_throws ErrorException save_immutable_example_outputs(root,bad,names)
            @test !isfile(joinpath(root,"b.json"))
            @test read(joinpath(root,"a.json"),String)=="output"
        end
    end
end
