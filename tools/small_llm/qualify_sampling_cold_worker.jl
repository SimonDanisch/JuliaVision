isdefined(@__MODULE__,:checkpoint_fixture) || include("test_sampling_checkpoint_device.jl")

function qualify_sampling_cold_worker(phase::Symbol)
    phase in (:prepare,:restore) || error("unknown qualification phase")
    root=joinpath(@__DIR__,"sampling_cold_worker")
    receiptpath=joinpath(root,"expected.json")
    output=joinpath(@__DIR__,"sampling_cold_worker_validation.json")
    isfile(output) && error("preserve previous qualification")
    sources=Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in
        ("sampling_checkpoint.jl","test_sampling_checkpoint_device.jl","qualify_sampling_cold_worker.jl"))
    device=Mantle.Device("llvmpipe");fixture=nothing
    try
        Q=SmallLLMChecks.Qwen35Experiment
        if phase==:prepare
            ispath(root) && error("preserve previous/partial fixture")
            mkpath(root);fixture=checkpoint_fixture(device)
            Q.step!(fixture.session,2);Q.decode!(fixture.session,3)
            save_sampling_checkpoint(fixture.session,fixture.sampler,joinpath(root,"checkpoint");identity="cold_worker_fixture_v1")
            Q.decode!(fixture.session,8);Mantle.waitidle(device)
            expected=Any[]
            for (i,(name,resource)) in enumerate(checkpoint_resources(fixture.session,fixture.sampler))
                file="expected_"*lpad(string(i),3,'0')*".bin";path=joinpath(root,file)
                open(io->write(io,checkpoint_array(resource)),path,"w")
                push!(expected,(;name,file,sha256=checkpoint_digest(path)))
            end
            receipt=(;producer_pid=getpid(),position=fixture.session.position,resources=expected,
                source_sha256=sources,checkpoint_sha256=checkpoint_digest(joinpath(root,"checkpoint","manifest.json")))
            open(io->JSON3.pretty(io,receipt),receiptpath,"w")
            (;phase,producer_pid=getpid(),requires_fresh_worker=true)
        else
            p=JSON3.read(read(receiptpath,String))
            p.producer_pid!=getpid() || error("must use a different fresh worker process")
            checkpoint_plain(p.source_sha256)==sources || error("fixture source drift")
            p.checkpoint_sha256==checkpoint_digest(joinpath(root,"checkpoint","manifest.json")) || error("checkpoint manifest drift")
            fixture=checkpoint_fixture(device)
            restore_sampling_checkpoint!(fixture.session,fixture.sampler,joinpath(root,"checkpoint");identity="cold_worker_fixture_v1")
            tests=@testset "Fresh-process sampled-state restore, software only" begin
                @test fixture.session.position==4
                before=Mantle.submissions(device);Q.decode!(fixture.session,8)
                @test Mantle.submissions(device)-before==8
                Mantle.waitidle(device)
                @test fixture.session.position==p.position==12
                named=checkpoint_resources(fixture.session,fixture.sampler)
                @test length(named)==length(p.resources)
                for (i,((name,resource),row)) in enumerate(zip(named,p.resources))
                    @test row.name==name
                    row.file=="expected_"*lpad(string(i),3,'0')*".bin" || error("unexpected fixture filename")
                    path=joinpath(root,String(row.file));islink(path) && error("fixture symlink")
                    actual=checkpoint_array(resource)
                    sizeof(actual)==filesize(path) && row.sha256==checkpoint_digest(path) || error("expected bytes drift")
                    expected=similar(actual);read!(path,expected)
                    @test isequal(actual,expected)
                end
            end
            Mantle.waitidle(device);counts=Test.get_test_counts(tests)
            report=(;complete=true,passed=counts.passes+counts.cumulative_passes,
                scope="fresh-PROCESS cold resume on synthetic shared-kernel sampler/cache fixture, not full Qwen qualification",
                producer_pid=p.producer_pid,consumer_pid=getpid(),source_sha256=sources,
                expected_receipt_sha256=checkpoint_digest(receiptpath),normal_final_waitidle=true)
            open(io->JSON3.pretty(io,report),output,"w");report
        end
    finally
        fixture===nothing || (close(fixture.session);close(fixture.sampler))
    end
end
