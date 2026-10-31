include("bigcodebench_x_revisions.jl")
using Test
function test_bcbx_revisions()
    tests=@testset "PRIVATE derived16 revision protocol and skip fixtures" begin
        mktempdir() do root
            baseline=joinpath(root,"baseline");mkpath(baseline)
            cp(joinpath(@__DIR__,"bigcodebench_x_results/conditions.json"),joinpath(baseline,"conditions.json"))
            parent=JSON3.read(read(joinpath(baseline,"conditions.json"),String))
            for job in parent.jobs
                path=joinpath(baseline,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
                open(io->JSON3.pretty(io,(;private_fixture=true)),attempt,"w")
                row=(;id=job.id,task=job.task,arm=job.arm,messages=job.messages,code="println(17)",
                    attempt_sha256=bcbx_digest(attempt),conditions_sha256=bcbx_digest(joinpath(baseline,"conditions.json")),
                    decode_submissions=4,decode_steps_executed=4,sampling=(;parent.decoding...,draws=5),
                    budget_tokens=6144,thinking_budget=4096)
                open(io->JSON3.pretty(io,row),path,"w")
            end
            directory=joinpath(root,"revisions");p=freeze_bcbx_revisions(;directory,baseline_directory=baseline)
            @test length(p.jobs)==16
            @test all(j->j.eligible,p.jobs)
            @test length(p.baseline_generation_sha256)==8
            @test count(j->j.arm=="public_static",p.jobs)==8
            @test_throws ErrorException freeze_bcbx_revisions(;directory,baseline_directory=baseline)
            run(`gen/bcb-x-tools/bin/python $(joinpath(@__DIR__,"test_score_bigcodebench_x_revisions.py")) $directory $baseline $BCBX_SNAPSHOT`)
            # Both arms retain a synthetic upstream inference error. No device
            # is created while recording these explicit non-generation skips.
            lastpath=joinpath(baseline,String(last(parent.jobs).id)*".json")
            bad=JSON3.read(read(lastpath,String),Dict{String,Any});bad["inference_error"]="PRIVATE upstream fixture"
            open(io->JSON3.pretty(io,bad),lastpath,"w")
            skips=joinpath(root,"skips");sp=freeze_bcbx_revisions(;directory=skips,baseline_directory=baseline)
            @test count(j->!j.eligible,sp.jobs)==2
            for job in sp.jobs
                job.eligible||continue
                open(io->JSON3.pretty(io,(;private_placeholder=true)),joinpath(skips,String(job.id)*".json"),"w")
            end
            @test run_bcbx_revisions(;directory=skips).not_generated
            @test run_bcbx_revisions(;directory=skips).not_generated
            @test run_bcbx_revisions(;directory=skips).complete
        end
    end
    c=Test.get_test_counts(tests);report=(;complete=true,passed=c.passes+c.cumulative_passes,
        scope="PRIVATE all16 derived protocol, mocked scorer and no-device skips only; not model-quality evidence",
        source_sha256=Dict(f=>bcbx_digest(joinpath(@__DIR__,f)) for f in ("bigcodebench_x_revisions.jl","test_bigcodebench_x_revisions.jl","score_bigcodebench_x_revisions.py")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"bigcodebench_x_revisions_validation.json"),"w");report
end
