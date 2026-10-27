include("collect_compact_bayes_demo_resume.jl")
using Test
function test_verified_demo_artifacts()
    @testset "Verified artifact copies are immutable and nonempty" begin
        mktempdir() do directory
            bytes=Vector{UInt8}(codeunits("{\"complete\":true}"));hash=bytes2hex(sha256(bytes));item=(;bytes,sha256=hash)
            @test save_verified_demo_artifacts(directory,Dict("a.json"=>item),["a.json"])==1
            @test read(joinpath(directory,"a.json"))==bytes
            @test save_verified_demo_artifacts(directory,Dict("a.json"=>item),["a.json"])==1
            empty=(;bytes=UInt8[],sha256=bytes2hex(sha256(UInt8[])))
            @test_throws ErrorException save_verified_demo_artifacts(directory,Dict("b.json"=>empty),["b.json"])
            @test !ispath(joinpath(directory,"b.json"))
            @test_throws ErrorException save_verified_demo_artifacts(directory,Dict("b.json"=>(;bytes,sha256="wrong")),["b.json"])
            @test_throws ErrorException save_verified_demo_artifacts(directory,Dict("b.json"=>item),["a.json"])
            @test_throws ErrorException save_verified_demo_artifacts(directory,Dict("../b.json"=>item),["../b.json"])
            other=(;bytes=UInt8[1],sha256=bytes2hex(sha256(UInt8[1])))
            @test_throws ErrorException save_verified_demo_artifacts(directory,Dict("a.json"=>other),["a.json"])
            @test read(joinpath(directory,"a.json"))==bytes
        end
    end
end
