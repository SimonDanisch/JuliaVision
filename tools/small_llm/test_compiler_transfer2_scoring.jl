include("compiler_transfer2_benchmark.jl")
using Test

function test_compiler_transfer2_scoring()
    checks=@testset "PRIVATE transfer2 end-to-end scoring fixtures, NOT model outputs" begin
        mktempdir() do root
            directory=joinpath(root,"PRIVATE_FIXTURE");p=freeze_compiler_transfer2_benchmark(;directory)
            path=joinpath(directory,"conditions.json");digest(p)=bytes2hex(open(sha256,p))
            for job in p.jobs
                task=only(filter(t->t.id==job.task,compiler_transfer2_tasks()))
                reference=CanonicalScalarInterface.lower(task.reference_value,task.interface_id)
                @test reference.accepted
                code=job.arm=="scalar_program" ? task.reference_value : reference.code
                output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
                open(io->JSON3.pretty(io,(;conditions_sha256=digest(path))),attempt,"w")
                row=(;id=job.id,task=job.task,arm=job.arm,messages=job.messages,code,
                    conditions_sha256=digest(path),attempt_sha256=digest(attempt),sampling=p.decoding,
                    budget_tokens=p.decoding.max_tokens,thinking_budget=p.thinking_budget,
                    decode_steps_executed=1,decode_submissions=1,truncated=false)
                open(io->JSON3.pretty(io,row),output,"w")
            end
            r=score_compiler_transfer2(;directory)
            @test r.complete
            @test length(r.rows)==8
            @test all(row->row.status=="passed",r.rows)
            @test score_compiler_transfer2(;directory).rows==r.rows
            output=joinpath(directory,String(first(p.jobs).id)*".json")
            row=JSON3.read(read(output,String),Dict{String,Any});row["decode_submissions"]=2
            open(io->JSON3.pretty(io,row),output,"w")
            @test_throws ErrorException score_compiler_transfer2(;directory)
        end
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="PRIVATE trusted-reference grader fixtures ONLY, not neural quality",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("test_compiler_transfer2_scoring.jl","compiler_transfer2_benchmark.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"compiler_transfer2_scorer_validation.json"),"w");packet
end
