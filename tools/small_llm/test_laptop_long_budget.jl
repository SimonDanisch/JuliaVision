include("laptop_long_budget.jl")
using Test
function test_laptop_long_budget()
    tests=@testset "PRIVATE long-budget protocol/scoring, not model outputs" begin
        mktempdir() do directory
            p=freeze_laptop_long_budget(;directory)
            @test length(p.jobs)==2
            @test p.decoding.max_tokens==6144
            @test_throws ErrorException freeze_laptop_long_budget(;directory)
            @test !score_laptop_long_budget(;directory).complete
            s=laptop_long_budget_state(directory);task=first(transfer_budget_state().state.tasks)
            for job in p.jobs
                path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
                open(io->JSON3.pretty(io,(;private=true)),attempt,"w")
                row=(;id=job.id,task=job.task,arm=job.arm,messages=job.messages,
                    conditions_sha256=long_budget_digest(s.path),attempt_sha256=long_budget_digest(attempt),
                    budget_tokens=6144,sampling=p.decoding,decode_steps_executed=4,decode_submissions=4,
                    code=task.reference,generated_tokens=4,truncated=false,finished_thinking=true)
                open(io->JSON3.pretty(io,row),path,"w")
            end
            summary=score_laptop_long_budget(;directory)
            @test summary.complete
            @test all(r->r.status=="passed",summary.rows)
            @test score_laptop_long_budget(;directory).rows==summary.rows
            path=joinpath(directory,String(first(p.jobs).id)*".json")
            bad=JSON3.read(read(path,String),Dict{String,Any});bad["decode_submissions"]=5
            open(io->JSON3.pretty(io,bad),path,"w")
            @test_throws ErrorException score_laptop_long_budget(;directory)
        end
    end
    c=Test.get_test_counts(tests)
    report=(;complete=true,passed=c.passes+c.cumulative_passes,
        scope="Trusted reference/scoring qualification only, NOT model intelligence",
        source_sha256=Dict(f=>long_budget_digest(joinpath(@__DIR__,f)) for f in ("laptop_long_budget.jl","test_laptop_long_budget.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"laptop_long_budget_validation.json"),"w");report
end
