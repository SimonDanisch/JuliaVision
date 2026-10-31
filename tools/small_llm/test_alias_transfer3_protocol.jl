include("alias_transfer3_protocol.jl")
using Test
function test_alias_transfer3_protocol()
    @testset "Third external cohort: public-only query and paired retention" begin
        index=AliasDocumentRetrieval.Index([(;qualified="Example.mean",text="mean computes an average",binding_id=1),
            (;qualified="Other.mean",text="mean computes an average",binding_id=1)])
        prompts=[(;task_id="External/"*string(i),prompt="compute mean from stdin") for i in 1:8]
        counter=m->length(m[2].second)
        jobs=alias_transfer3_jobs(prompts,index,counter;system="Julia contract")
        @test length(jobs)==16
        @test length(unique(j.id for j in jobs))==16
        @test [j.arm for j in jobs[1:4]]==["direct","public_docs","public_docs","direct"]
        @test all(j->j.eligible,jobs)
        for i in 1:2:16
            a,b=jobs[i:i+1]
            @test a.task==b.task&&a.eligible==b.eligible
            @test a.messages[1]==b.messages[1]
            @test startswith(a.messages[2].second,prompts[(i+1)÷2].prompt)
            @test startswith(b.messages[2].second,prompts[(i+1)÷2].prompt)
        end
        @test all(j->j.arm=="direct" ? isempty(j.retrieved_names) : j.retrieved_names==["Example.mean"],jobs)
        limited=alias_transfer3_jobs(prompts,index,m->occursin("Example.mean",m[2].second) ? 5000 : 10;system="contract")
        @test all(j->!j.eligible&&j.skip_reason=="paired_context_capacity_failure",limited)
        @test_throws ArgumentError alias_transfer3_jobs(prompts[1:7],index,counter;system="contract")
        @test_throws ArgumentError alias_transfer3_jobs(fill(first(prompts),8),index,counter;system="contract")
        private_rows=[(;p...,reference="forbidden") for p in prompts]
        @test_throws ArgumentError alias_transfer3_jobs(private_rows,index,counter;system="contract")
        @test_throws ArgumentError alias_transfer3_jobs(prompts,index,m->0;system="contract")
    end
end
