isdefined(@__MODULE__,:CheckpointSamplingHook) || include("sampling_checkpoint.jl")
isdefined(@__MODULE__,:qualified_qwen_cache) || include("run_humaneval_subset.jl")
using Test

function freeze_sampling_full_model_gate()
    root=joinpath(@__DIR__,"sampling_full_model_gate");mkpath(root)
    path=joinpath(root,"conditions.json");isfile(path) && error("gate already registered")
    qualifications=("sampling_checkpoint_software_validation.json","sampling_cold_worker_validation.json")
    for file in qualifications
        p=JSON3.read(read(joinpath(@__DIR__,file),String));p.complete || error("incomplete software gate")
        for (source,hash) in pairs(p.source_sha256)
            checkpoint_digest(joinpath(@__DIR__,String(source)))==hash || error("software qualification drift")
        end
    end
    base=qualified_qwen_cache()
    candidates=filter(d->all(isfile(joinpath(base,d,f)) for f in ("index.json","source.json","weights.bin")),readdir(base))
    length(candidates)==1 || error("cache identity ambiguous")
    files=vcat([joinpath(base,only(candidates),f) for f in ("index.json","source.json","weights.bin")],
        [joinpath("gen/qwen35-4b",f) for f in ("config.json","tokenizer.json")])
    identities=sort([abspath(f)=>checkpoint_digest(f) for f in files];by=first)
    reference=joinpath(@__DIR__,"conversation_native_reference.json")
    ids=Int.(only(filter(r->r.id=="baseline_short",JSON3.read(read(reference,String)).rows)).julia_ids)
    p=(;scope="real4B sampled-state cold-worker checkpoint qualification, NOT task-quality evidence",capacity=256,
        model_dir="gen/qwen35-4b",cachebase=base,identity=bytes2hex(sha256(JSON3.write(identities))),identity_files=identities,
        ids,prepare_decode=4,continuation_decode=8,reference_sha256=checkpoint_digest(reference),
        qualifications=Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in qualifications),
        source_sha256=Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in
            ("qualify_sampling_full_model.jl",SAMPLING_CHECKPOINT_SOURCES...)))
    open(io->JSON3.pretty(io,p),path,"w");p
end

function sampling_full_gate_state()
    root=joinpath(@__DIR__,"sampling_full_model_gate");path=joinpath(root,"conditions.json")
    p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        checkpoint_digest(joinpath(@__DIR__,String(file)))==hash || error("full gate source drift")
    end
    for pair in p.identity_files
        # JSON3 encodes a Pair as a one-field object.
        file=String(first(keys(pair)));hash=String(first(values(pair)))
        checkpoint_digest(file)==hash || error("model/cache/tokenizer drift")
    end
    for (file,hash) in pairs(p.qualifications)
        checkpoint_digest(joinpath(@__DIR__,String(file)))==hash || error("software receipt drift")
    end
    p.reference_sha256==checkpoint_digest(joinpath(@__DIR__,"conversation_native_reference.json")) || error("prompt receipt drift")
    (;root,path,packet=p)
end

function qualify_sampling_full_model(phase::Symbol;seconds=210)
    phase in (:prepare,:restore) && seconds>0 || error("phase and time budget required")
    s=sampling_full_gate_state();p=s.packet
    output=joinpath(s.root,String(phase)*".json");isfile(output) && error("preserve prior gate attempt")
    Q=SmallLLMChecks.Qwen35Experiment;cache=qualified_qwen_cache()
    let
        started=time();device=Mantle.Device("radeon");model=nothing;sampler=nothing;session=nothing
        report=Dict{String,Any}("complete"=>false,"phase"=>String(phase),"producer_or_consumer_pid"=>getpid(),
            "conditions_sha256"=>checkpoint_digest(s.path),"runtime_provenance"=>inference_runtime_provenance())
        save()=open(io->JSON3.pretty(io,report),output,"w")
        try
            save();model=Q.loadmodel(String(p.model_dir);quantize=:q4k,device,quantcache=cache)
            time()<started+seconds-75 || return save()
            sampler=GPUSampling.State(device,Int(model.config.vocab_size);seed=UInt32(20261024),topk=20,temperature=0.6f0,top_p=0.95f0)
            session=Q.session(model;capacity=Int(p.capacity),kv_type=Float32,attention=:split,logit_hook=CheckpointSamplingHook(sampler))
            if phase==:prepare
                ids=Int.(p.ids);GPUSampling.activate!(sampler,false)
                Q.prefill!(session,ids[1:end-1];chunk=8);GPUSampling.activate!(sampler,true);Q.step!(session,last(ids))
                Q.decode!(session,Int(p.prepare_decode))
                save_sampling_checkpoint(session,sampler,joinpath(s.root,"checkpoint");identity=String(p.identity))
            else
                prepared=JSON3.read(read(joinpath(s.root,"prepare.json"),String))
                prepared.complete && prepared.normal_final_waitidle && prepared.producer_or_consumer_pid!=getpid() || error("fresh-worker normal prepare required")
                prepared.conditions_sha256==checkpoint_digest(s.path) || error("prepare conditions drift")
                restore_sampling_checkpoint!(session,sampler,joinpath(s.root,"checkpoint");identity=String(p.identity))
            end
            before=Mantle.submissions(device);Q.decode!(session,Int(p.continuation_decode));Mantle.waitidle(device)
            @test Mantle.submissions(device)-before==p.continuation_decode
            resources=checkpoint_resources(session,sampler);expected=Any[]
            if phase==:prepare
                for (i,(name,resource)) in enumerate(resources)
                    file="expected_"*lpad(string(i),3,'0')*".bin";path=joinpath(s.root,file)
                    isfile(path) && error("preserve partial expected state")
                    open(io->write(io,checkpoint_array(resource)),path,"w")
                    push!(expected,(;name,file,sha256=checkpoint_digest(path)))
                end
                report["expected"]=expected
            else
                prepared=JSON3.read(read(joinpath(s.root,"prepare.json"),String))
                tests=@testset "Real4B cold-worker sampled continuation" begin
                    @test length(prepared.expected)==length(resources)
                    @test session.position==prepared.final_position
                    for (i,((name,resource),row)) in enumerate(zip(resources,prepared.expected))
                        @test name==row.name
                        row.file=="expected_"*lpad(string(i),3,'0')*".bin" || error("expected filename")
                        path=joinpath(s.root,String(row.file));islink(path) && error("expected symlink")
                        actual=checkpoint_array(resource)
                        sizeof(actual)==filesize(path) && row.sha256==checkpoint_digest(path) || error("expected byte integrity")
                        expected=similar(actual);read!(path,expected);@test isequal(actual,expected)
                    end
                end
                counts=Test.get_test_counts(tests);report["passed"]=counts.passes+counts.cumulative_passes
                report["prepare_sha256"]=checkpoint_digest(joinpath(s.root,"prepare.json"))
            end
            report["final_position"]=session.position
            report["sampler_counter"]=Int(only(Array(sampler.counter)))
            save()
        finally
            try
                session===nothing || close(session)
                sampler===nothing || close(sampler)
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
        report["complete"]=true;report["normal_final_waitidle"]=true;report["elapsed_seconds"]=time()-started;save();report
    end
end
