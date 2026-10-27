# Separate portable-host qualification; preserves the original Bosgame registration.
include("laptop_shared_source.jl")
isdefined(@__MODULE__,:CheckpointSamplingHook) || include("sampling_checkpoint.jl")
using Test

function freeze_laptop_sampling_gate(;directory=joinpath(@__DIR__,"laptop_sampling_full_model_gate"))
    ispath(directory) && !isempty(readdir(directory)) && error("preserve registration")
    gatepath=joinpath(@__DIR__,"laptop_portable_identity_gate.json")
    gate=JSON3.read(read(gatepath,String))
    gate.complete && gate.normal_final_waitidle && gate.full_model_passed==10 || error("portable gate incomplete")
    reference=joinpath(@__DIR__,"conversation_native_reference.json")
    ids=Int.(only(filter(r->r.id=="baseline_short",JSON3.read(read(reference,String)).rows)).julia_ids)
    migration=JSON3.read(read(joinpath(@__DIR__,"laptop_pending_results/conditions.json"),String))
    sources=merge(Dict(String(k)=>String(v) for (k,v) in pairs(migration.source_sha256)),
        Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in
            ("laptop_sampling_full_model.jl",SAMPLING_CHECKPOINT_SOURCES...)))
    qualifications=Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in
        ("sampling_checkpoint_software_validation.json","sampling_cold_worker_validation.json"))
    for file in keys(qualifications)
        JSON3.read(read(joinpath(@__DIR__,file),String)).complete || error("software qualification incomplete")
    end
    p=(;scope="Laptop exact portable4B sampled-state cold-worker checkpoint gate; NOT task-quality evidence. Original Bosgame gate remains unchanged.",
        capacity=256,ids,prepare_decode=4,continuation_decode=8,
        portable_manifest_sha256=gate.portable_manifest_sha256,gate_sha256=checkpoint_digest(gatepath),
        reference_sha256=checkpoint_digest(reference),qualifications,source_sha256=sources)
    mkpath(directory);open(io->JSON3.pretty(io,p),joinpath(directory,"conditions.json"),"w");p
end

function laptop_sampling_gate_state(directory)
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        checkpoint_digest(joinpath(@__DIR__,String(file)))==hash || error("gate source drift: $file")
    end
    for (file,hash) in pairs(p.qualifications)
        checkpoint_digest(joinpath(@__DIR__,String(file)))==hash || error("software qualification drift")
    end
    checkpoint_digest(joinpath(@__DIR__,"conversation_native_reference.json"))==p.reference_sha256 || error("prompt drift")
    checkpoint_digest(joinpath(@__DIR__,"laptop_portable_identity_gate.json"))==p.gate_sha256 || error("portable gate drift")
    (;path,packet=p)
end

function qualify_laptop_sampling_gate(phase::Symbol;seconds=210,directory=joinpath(@__DIR__,"laptop_sampling_full_model_gate"))
    phase in (:prepare,:restore) && seconds>0 || error("phase and time budget required")
    s=laptop_sampling_gate_state(directory);p=s.packet
    output=joinpath(directory,String(phase)*".json");ispath(output) && error("preserve prior attempt")
    portable="gen/qwen35-4b-portable"
    checkpoint_digest(joinpath(portable,"portable.json"))==p.portable_manifest_sha256 || error("model identity drift")
    identity=String(p.portable_manifest_sha256);Q=SmallLLMChecks.Qwen35Experiment
    let
        started=time();device=Mantle.Device("radeon");model=nothing;sampler=nothing;session=nothing
        report=Dict{String,Any}("complete"=>false,"phase"=>String(phase),"pid"=>getpid(),
            "conditions_sha256"=>checkpoint_digest(s.path),"runtime_provenance"=>inference_runtime_provenance())
        save()=open(io->JSON3.pretty(io,report),output,"w")
        try
            save();model=load_portable_qwen35(Q,portable,device)
            time()<started+seconds-75 || return save()
            sampler=GPUSampling.State(device,Int(model.config.vocab_size);seed=UInt32(20261024),topk=20,temperature=0.6f0,top_p=0.95f0)
            session=Q.session(model;capacity=Int(p.capacity),kv_type=Float32,attention=:split,logit_hook=CheckpointSamplingHook(sampler))
            if phase==:prepare
                ids=Int.(p.ids);GPUSampling.activate!(sampler,false)
                Q.prefill!(session,ids[1:end-1];chunk=8);GPUSampling.activate!(sampler,true);Q.step!(session,last(ids))
                Q.decode!(session,Int(p.prepare_decode))
                save_sampling_checkpoint(session,sampler,joinpath(directory,"checkpoint");identity)
            else
                prepared=JSON3.read(read(joinpath(directory,"prepare.json"),String))
                prepared.complete && prepared.normal_final_waitidle && prepared.pid!=getpid() || error("fresh-worker normal prepare required")
                prepared.conditions_sha256==checkpoint_digest(s.path) || error("prepare drift")
                restore_sampling_checkpoint!(session,sampler,joinpath(directory,"checkpoint");identity)
            end
            before=Mantle.submissions(device);Q.decode!(session,Int(p.continuation_decode));Mantle.waitidle(device)
            @test Mantle.submissions(device)-before==p.continuation_decode
            resources=checkpoint_resources(session,sampler)
            if phase==:prepare
                expected=Any[]
                for (i,(name,resource)) in enumerate(resources)
                    file="expected_"*lpad(string(i),3,'0')*".bin";path=joinpath(directory,file)
                    ispath(path) && error("preserve partial expected state")
                    open(io->write(io,checkpoint_array(resource)),path,"w")
                    push!(expected,(;name,file,sha256=checkpoint_digest(path)))
                end
                report["expected"]=expected
            else
                prepared=JSON3.read(read(joinpath(directory,"prepare.json"),String))
                tests=@testset "Laptop real4B cold-worker sampled continuation" begin
                    @test length(prepared.expected)==length(resources)
                    @test session.position==prepared.final_position
                    for (i,((name,resource),row)) in enumerate(zip(resources,prepared.expected))
                        @test name==row.name
                        row.file=="expected_"*lpad(string(i),3,'0')*".bin" || error("expected filename")
                        path=joinpath(directory,String(row.file));islink(path) && error("expected symlink")
                        actual=checkpoint_array(resource)
                        sizeof(actual)==filesize(path) && row.sha256==checkpoint_digest(path) || error("expected integrity")
                        expected=similar(actual);read!(path,expected);@test isequal(actual,expected)
                    end
                end
                counts=Test.get_test_counts(tests);report["passed"]=counts.passes+counts.cumulative_passes
                report["prepare_sha256"]=checkpoint_digest(joinpath(directory,"prepare.json"))
            end
            report["final_position"]=session.position;report["sampler_counter"]=Int(only(Array(sampler.counter)));save()
        finally
            try
                session===nothing || close(session)
                sampler===nothing || close(sampler)
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
        report["normal_final_waitidle"]=true;report["complete"]=true;report["elapsed_seconds"]=time()-started;save();report
    end
end
