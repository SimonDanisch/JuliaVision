include("lint_revision_study.jl")
using Test
function test_lint_revision_study()
    @testset "Separate same-cohort revision registration, no baseline overwrite" begin
        mktempdir() do directory
            p=freeze_lint_revision(;directory)
            @test p.ready_for_generation&&length(p.jobs)==16
            @test p.condition_legend.direct=="self_review"
            @test p.condition_legend.public_docs=="static_inventory_and_expanded_public_docs"
            @test all(j->j.eligible,p.jobs)
            @test all(j->length(j.messages)==4,p.jobs)
            @test all(j->j.condition_name in ("self_review","lint_and_expanded_docs"),p.jobs)
            @test length(unique(j.original_id for j in p.jobs))==8
            @test all(p.jobs[i].original_generation_sha256==p.jobs[i+1].original_generation_sha256 for i in 1:2:16)
            @test_throws ErrorException freeze_lint_revision(;directory)
        end
    end
end
