include("bigcodebench_x_revision_policy.jl")
using Test
function test_bcbx_revision_policy()
    tests=@testset "Public-static versus self-review policy, no hidden evaluation" begin
        mktempdir() do directory
            path=joinpath(directory,"policy.json");p=freeze_bcbx_revision_policy(;path)
            @test length(p.order)==16
            @test length(unique(j.id for j in p.order))==16
            @test count(j->j.mode=="public_static",p.order)==8
            @test p.decoding.max_tokens==4096&&p.thinking_budget==2048
            @test_throws ErrorException freeze_bcbx_revision_policy(;path)
        end
        m=JSON3.read("[{\"system\":\"Julia programmer\"},{\"user\":\"Read stdin and print result\"}]")
        b=Dict("Makie"=>["Figure"]);code="using CSV;Makie.backend()"
        static=bcbx_public_revision_messages(m,code,"public_static",b)
        review=bcbx_public_revision_messages(m,code,"self_review",b)
        @test static[1:3]==review[1:3]
        @test last(static[1:3])==("assistant"=>code)
        @test occursin("CSV",last(static).second)
        @test occursin("Makie.backend",last(static).second)
        @test !occursin("Makie.backend",last(review).second)
        @test_throws ErrorException bcbx_public_revision_messages(m,code,"hidden_tests",b)
    end
    c=Test.get_test_counts(tests);report=(;complete=true,passed=c.passes+c.cumulative_passes,
        scope="CPU message/policy qualification only; no model revision, hidden evaluation or quality gain",
        source_sha256=Dict(f=>bcbx_policy_digest(joinpath(@__DIR__,f)) for f in ("bigcodebench_x_revision_policy.jl","test_bigcodebench_x_revision_policy.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"bigcodebench_x_revision_policy_validation.json"),"w");report
end
