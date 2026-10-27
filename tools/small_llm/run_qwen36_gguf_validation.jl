# Include after qwen36_moe_model.jl, qwen36_gguf.jl and device_finish.jl.
using JSON3,SHA,LinearAlgebra,Test

function run_qwen36_gguf_validation(checkpoint,config,reference_dir,output;benchmark_tokens=64)
    Base.active_project()=="/home/sim/Programmieren/VulkanDev/Project.toml"||error("Root project required")
    ispath(output)&&error("Preserve first full-checkpoint validation attempt")
    G=Qwen36GGUF;R=G.R
    filesize(checkpoint)==G.CHECKPOINT_BYTES&&first(split(read(`sha256sum $checkpoint`,String)))==G.CHECKPOINT_SHA256||error("Unverified packed checkpoint")
    reference=JSON3.read(read(joinpath(reference_dir,"reference.json"),String))
    provenance=JSON3.read(read(joinpath(reference_dir,"provenance.json"),String))
    reference.complete&&reference.device=="Vulkan Radeon"&&provenance.checkpoint_sha256==G.CHECKPOINT_SHA256||error("Wrong/incomplete Vulkan baseline")
    benchmark_tokens==reference.decode_tokens||error("Matched comparison requires identical decode length")
    reference_path=joinpath(reference_dir,"logits.f32")
    bytes2hex(open(sha256,reference_path))==provenance.logits_sha256||error("Reference data corrupt")
    expected=reshape(reinterpret(Float32,read(reference_path)),Int(reference.vocab),:)
    size(expected,2)==length(reference.tokens)||error("Reference shape")
    report=Dict{String,Any}("complete"=>false,"host"=>gethostname(),"project"=>Base.active_project(),
        "checkpoint_sha256"=>G.CHECKPOINT_SHA256,"reference_provenance"=>provenance,
        "scope"=>"Full quantized35B GPU numerical comparison and short-context decode; NOT equivalence proof, intelligence benchmark or4K throughput",
        "activation_scale_precision"=>"Mantle Float32; llama.cpp Vulkan Float16 (different Q8 arithmetic)",
        "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("qwen36_moe_model.jl","qwen36_gguf.jl","qwen36_gguf_kernels.jl","run_qwen36_gguf_validation.jl")))
    save()=begin mkpath(dirname(output));open(io->JSON3.pretty(io,report),output,"w");end
    let
        let
            device=Mantle.Device("radeon");model=nothing;session=nothing;started=time()
            try
                report["device"]=string(device);report["pid"]=getpid();save()
                load_started=time();model=G.loadmodel(checkpoint,config,device;verify_hash=false)
                report["load_seconds"]=time()-load_started;save()
                build_started=time()
                session=R.session(model;capacity=256,head_layout=:tiled,kv_type=Float32)
                report["session_build_seconds"]=time()-build_started
                report["model_and_session_load_seconds"]=time()-load_started
                rows=Any[];report["rows"]=rows
                for (i,token) in enumerate(reference.tokens)
                    time()-started<220||error("Work deadline before reference step")
                    before=Mantle.submissions(device);R.step!(session,Int(token));Mantle.waitidle(device)
                    graph_submissions=Mantle.submissions(device)-before
                    logits=Array(session.logits);want=expected[:,i]
                    row=(;position=i,token=Int(token),submissions=graph_submissions,
                        finite=all(isfinite,logits),max_logit_error=maximum(abs,logits-want),
                        rmse=sqrt(sum(abs2,logits-want)/length(logits)),
                        cosine=dot(logits,want)/(norm(logits)*norm(want)),
                        next_token=argmax(logits)-1,reference_next_token=argmax(want)-1)
                    push!(rows,row);save();println(row);flush(stdout)
                    row.submissions==1&&row.finite||error("GPU graph submission/finite-output check failed")
                end
                report["reference_max_abs_0p1_pass"]=all(row->row.max_logit_error<0.1,rows)
                report["reference_argmax_agreement"]=count(row->row.next_token==row.reference_next_token,rows)
                report["full_model_equivalence_verified"]=false;save()
                # Keep numerical discrepancies visible, not a timing acceptance
                # claim. Known-answer kernel checks are a separate qualification.
                # The
                # context is the actual short prompt, not a fictitious4K prefill.
                time()-started<210||error("No conservative margin for speed sample")
                before=Mantle.submissions(device);t0=time_ns()
                R.decode!(session,benchmark_tokens);Mantle.waitidle(device)
                elapsed=(time_ns()-t0)/1e9
                @assert Mantle.submissions(device)-before==benchmark_tokens
                @assert all(isfinite,Array(session.logits))
                report["benchmark"]=(;tokens=benchmark_tokens,elapsed_seconds=elapsed,
                    tokens_per_second=benchmark_tokens/elapsed,prompt_tokens=length(reference.tokens),
                    kv_type="Float32",capacity=256,submissions=benchmark_tokens,
                    token_feedback="device-resident; no per-token readback or wait")
                # Also measure the synchronized host-greedy mode used by the
                # external harness. Reset state so prompt/context are identical.
                close(session);session=R.session(model;capacity=256,head_layout=:tiled,kv_type=Float32)
                prefill_start=time_ns()
                for token in reference.tokens;R.step!(session,Int(token));end
                Mantle.waitidle(device)
                report["warm_sequential_prefill"]=(;tokens=length(reference.tokens),elapsed_seconds=(time_ns()-prefill_start)/1e9,
                    external_elapsed_seconds=reference.prefill_seconds,mode="one-token graph replay; final sync")
                next_token=Int(only(Array(session.best)));generated=Int[];t0=time_ns()
                for _ in 1:benchmark_tokens
                    R.step!(session,next_token)
                    logits=Array(session.logits)
                    next_token=argmax(logits)-1;push!(generated,next_token)
                end
                Mantle.waitidle(device);elapsed=(time_ns()-t0)/1e9
                report["synchronized_greedy_benchmark"]=(;tokens=benchmark_tokens,elapsed_seconds=elapsed,
                    tokens_per_second=benchmark_tokens/elapsed,kv_type="Float32",capacity=256,
                    token_feedback="host logits readback and argmax included",generated_tokens=generated,
                    external_tokens_per_second=reference.decode_tokens_per_second,
                    same_generated_sequence=generated==Int.(reference.generated_tokens))
                report["complete"]=true
            catch err
                report["error"]=sprint(showerror,err);rethrow()
            finally
                session===nothing||close(session);model===nothing||close(model)
                report["normal_final_waitidle"]=false
                try
                    finish_device!(device);report["normal_final_waitidle"]=true
                finally
                    save()
                end
            end
        end
    end
    report
end
