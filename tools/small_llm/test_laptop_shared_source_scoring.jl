include("score_laptop_shared_source.jl")
using Test

function test_laptop_shared_source_scoring()
    checks=@testset "PRIVATE trusted-reference scorer fixtures, NOT model evidence" begin
        mktempdir() do root
            directory=joinpath(root,"PRIVATE_REFERENCE_FIXTURE")
            p=freeze_laptop_shared_source(;directory)
            path=joinpath(directory,"conditions.json");digest(p)=bytes2hex(open(sha256,p))
            for job in p.jobs
                task=only(filter(t->t.id==job.task,shared_source_kernel_tasks()))
                lowered=CheckedTypedPointwiseLowering.lower(task.reference_cpu)
                @test lowered.accepted
                code=job.arm=="cpu_source" ? task.reference_cpu : lowered.code
                output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
                open(io->JSON3.pretty(io,(;conditions_sha256=digest(path))),attempt,"w")
                candidate=(;attempt_sha256=digest(attempt),conditions_sha256=digest(path),messages=job.messages,
                    thinking_budget=p.thinking_budget,budget_tokens=p.decoding.max_tokens,decode_steps_executed=1,decode_submissions=1,
                    sampling=p.decoding,code,truncated=false,forced_closures=0)
                open(io->JSON3.pretty(io,candidate),output,"w")
            end
            report=score_laptop_shared_source(;directory)
            @test report.complete
            @test length(report.rows)==8
            @test all(r->r.status=="passed",report.rows)
            @test count(r->r.cpu_passed===true,report.rows)==4
            @test count(r->r.cpu_passed===nothing,report.rows)==4
            @test score_laptop_shared_source(;directory).rows==report.rows
            firstoutput=joinpath(directory,String(first(p.jobs).id)*".json")
            candidate=JSON3.read(read(firstoutput,String),Dict{String,Any})
            candidate["decode_submissions"]=2
            open(io->JSON3.pretty(io,candidate),firstoutput,"w")
            @test_throws ErrorException score_laptop_shared_source(;directory)
        end
    end
    n=Test.get_test_counts(checks)
    report=(;complete=true,scope="PRIVATE reference-only CPU/software-Vulkan scorer qualification; no neural generations or model-quality evidence",
        passed=n.passes+n.cumulative_passes,source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("test_laptop_shared_source_scoring.jl","score_laptop_shared_source.jl","laptop_shared_source.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"laptop_shared_source_scorer_validation.json"),"w");report
end
