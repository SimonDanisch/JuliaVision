# Load after qwen36_gguf.jl and device_finish.jl. All comparisons execute GPU graphs.
using JSON3,SHA

function benchmark_qwen36_optim(checkpoint,config,output;tokens=64,repeats=2,variants=nothing,external_binary=nothing,isolate_external=false)
    Base.active_project()=="/home/sim/Programmieren/VulkanDev/Project.toml"||error("Root project required")
    ispath(output)&&error("Do not overwrite an ablation receipt")
    G=Qwen36GGUF;R=G.R
    filesize(checkpoint)==G.CHECKPOINT_BYTES&&first(split(read(`sha256sum $checkpoint`,String)))==G.CHECKPOINT_SHA256||error("Checkpoint hash")
    prompt=Int[27175,667,17839,958,3145,63368,5638,13]
    cases=[("baseline",()),("reuse_q8",(:reuse_q8,)),
        ("narrow_experts",(:narrow_experts,)),("small_dense",(:small_dense,)),
        ("fuse_shared",(:fuse_shared,)),
        ("all",(:reuse_q8,:narrow_experts,:small_dense,:fuse_shared)),("baseline_after",())]
    variants===nothing||(cases=variants)
    report=Dict{String,Any}("complete"=>false,"checkpoint_sha256"=>G.CHECKPOINT_SHA256,
        "scope"=>"Same full-model GPU path, fixed prompt, Float32 activation scales and KV; short-context performance, not task quality",
        "project"=>Base.active_project(),"prompt"=>prompt,"tokens"=>tokens,"repeats"=>repeats,
        "isolate_external_model_residency"=>isolate_external,
        "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("qwen36_moe_model.jl","qwen36_optim.jl","qwen36_gguf_kernels.jl","benchmark_qwen36_optim.jl")),
        "cases"=>Any[])
    mantle_src=dirname(pathof(Mantle))
    report["mantle_source_sha256"]=Dict(relpath(joinpath(root,f),mantle_src)=>bytes2hex(sha256(read(joinpath(root,f))))
        for (root,_,files) in walkdir(mantle_src) for f in files if endswith(f,".jl"))
    save()=open(io->JSON3.pretty(io,report),output,"w")
    baseline_logits=nothing;baseline_predictions=nothing;baseline_final_logits=nothing
    let
        device=Mantle.Device("radeon");model=nothing;s=nothing;start=time()
        try
            report["pid"]=getpid();report["device"]=string(device);save()
            model=G.loadmodel(checkpoint,config,device;verify_hash=false)
            R.prepare_shared_gateup!(model)
            # Compile the host feedback driver outside measured intervals,
            # then discard its state. Every sample starts at the same prompt.
            s=R.session(model;capacity=256,head_layout=:tiled,kv_type=Float32)
            for token in prompt;R.step!(s,token);end
            R.decode!(s,1);Mantle.waitidle(device);close(s);s=nothing
            for (name,flags) in cases
                time()-start<210||break
                row=Dict{String,Any}("name"=>name,"flags"=>flags,"samples"=>Any[])
                push!(report["cases"],row);save()
                if flags===nothing
                    tokens==64&&external_binary!==nothing||error("External comparison requires64 tokens and binary")
                    if isolate_external
                        model===nothing||close(model);model=nothing
                        Mantle.waitidle(device)
                        row["pool_trim"]=Mantle.trim_gpu_pool!(device)
                        row["mantle_bytes_before_external"]=Mantle.gpu_live_bytes(device)
                        row["mantle_bytes_before_external"]<512*1024^2||error("Mantle model residency was not released")
                        save()
                    end
                    binary=abspath(external_binary);report["external_binary_sha256"]=bytes2hex(open(sha256,binary))
                    for trial in 1:repeats
                        time()-start<210||break
                        directory=abspath(output*".external.$name.$trial");log=directory*".log"
                        (ispath(directory)||ispath(log))&&error("Preserve external comparison artifacts")
                        command=`$binary $(abspath(checkpoint)) $directory $("Radeon 8060S")`
                        open(log,"w") do io
                            run(pipeline(`timeout --signal=TERM --kill-after=5s 60s $command`;stdout=io,stderr=io))
                        end
                        ref=JSON3.read(read(joinpath(directory,"reference.json"),String))
                        ref.complete&&ref.device=="Vulkan Radeon"&&Int.(ref.tokens)==prompt||error("External conditions")
                        sample=(;trial,tokens_per_second=Float64(ref.decode_tokens_per_second),
                            elapsed_seconds=Float64(ref.decode_seconds),backend="llama.cpp Vulkan",directory,
                            same_generated_sequence=Int.(ref.generated_tokens)==baseline_predictions)
                        push!(row["samples"],sample);save();println(name," ",sample);flush(stdout)
                    end
                    continue
                end
                if model===nothing
                    reload=time();model=G.loadmodel(checkpoint,config,device;verify_hash=false)
                    R.prepare_shared_gateup!(model);row["reload_seconds"]=time()-reload
                end
                for trial in 1:repeats
                    time()-start<225||break
                    build=time();s=R.session(model;capacity=256,head_layout=:tiled,kv_type=Float32,optimizations=flags)
                    row["build_seconds"]=time()-build;row["passes"]=length(s.plan.passes)
                    row["transient_peak_bytes"]=s.plan.peak;row["transient_naive_bytes"]=s.plan.naive
                    row["barrier_passes"]=count(p->p.barrier!==nothing,s.plan.passes)
                    row["alias_handover_transitions"]=sum(count(t->t.resource==0,p.pre) for p in s.plan.passes)
                    for token in prompt;R.step!(s,token);end
                    Mantle.waitidle(device);logits=Array(s.logits)
                    baseline_logits===nothing&&(baseline_logits=copy(logits))
                    delta=logits-baseline_logits
                    before=Mantle.submissions(device);t0=time_ns()
                    R.decode!(s,tokens);queued=(time_ns()-t0)/1e9;Mantle.waitidle(device)
                    elapsed=(time_ns()-t0)/1e9;submissions=Mantle.submissions(device)-before
                    predictions=Int.(Array(s.predictions)[length(prompt)+1:length(prompt)+tokens])
                    baseline_predictions===nothing&&(baseline_predictions=copy(predictions))
                    final_logits=Array(s.logits)
                    baseline_final_logits===nothing&&(baseline_final_logits=copy(final_logits))
                    sample=(;trial,elapsed_seconds=elapsed,queue_seconds=queued,tokens_per_second=tokens/elapsed,submissions,
                        prompt_logits_finite=all(isfinite,logits),max_prompt_logit_change=maximum(abs,delta),
                        prompt_logit_rmse=sqrt(sum(abs2,delta)/length(delta)),
                        prompt_logits_bitidentical=reinterpret(UInt32,logits)==reinterpret(UInt32,baseline_logits),
                        same_prompt_argmax=argmax(logits)==argmax(baseline_logits),
                        same_generated_sequence=predictions==baseline_predictions,predictions,
                        final_logits_finite=all(isfinite,final_logits),
                        max_final_logit_change=maximum(abs,final_logits-baseline_final_logits),
                        final_logits_bitidentical=reinterpret(UInt32,final_logits)==reinterpret(UInt32,baseline_final_logits))
                    push!(row["samples"],sample);save();println(name," ",sample);flush(stdout)
                    submissions==tokens&&sample.prompt_logits_finite&&sample.final_logits_finite||error("Submission/finite check")
                    close(s);s=nothing
                end
            end
            report["complete"]=length(report["cases"])==length(cases)&&all(r->length(r["samples"])==repeats,report["cases"])
            report["all_greedy_sequences_unchanged"]=all(r->all(s->s.same_generated_sequence,r["samples"]),report["cases"])
            report["full_model_equivalence_proven"]=false
        catch err
            report["error"]=sprint(showerror,err);rethrow()
        finally
            s===nothing||close(s);model===nothing||close(model)
            report["normal_final_waitidle"]=false
            try
                finish_device!(device);report["normal_final_waitidle"]=true
                isolate_external&&(report["final_pool_trim"]=Mantle.trim_gpu_pool!(device))
            finally
                save()
            end
        end
    end
    report
end
