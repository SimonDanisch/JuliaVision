isdefined(@__MODULE__,:SmallLLMChecks) || include("run_humaneval_subset.jl")
isdefined(@__MODULE__,:test_knn_memory) || include("test_knn_memory.jl")
isdefined(@__MODULE__,:test_logit_features) || include("test_logit_features.jl")
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")
include("argument_extent_audit.jl")
using LinearAlgebra, Test, JSON3, SHA

function verify_post_extent_engine(;seconds=240,model_dir="gen/qwen35-4b",
        output=joinpath(@__DIR__,"post_extent_engine_validation.json"))
    seconds>0 || throw(ArgumentError("time budget"))
    isfile(output) && error("preserve previous validation attempt")
    Q=SmallLLMChecks.Qwen35Experiment
    reference_path=joinpath(@__DIR__,"conversation_native_reference.json")
    metadata=JSON3.read(read(reference_path,String))
    case=only(filter(row->row.id=="baseline_short",metadata.rows))
    tensors=Q.DK.readsafetensors(replace(reference_path,".json"=>".safetensors");mmap=true)
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing
        evidence=Dict{String,Any}("complete"=>false,"scope"=>"post-argument-extent-fix infrastructure/native parity, NOT intelligence",
            "runtime_provenance"=>inference_runtime_provenance(),"phase"=>"fresh Device setup succeeded")
        evidence["experiment_source_sha256"]=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("knn_memory.jl","logit_features.jl","qwen35.jl","prefill_qwen35.jl","verify_post_extent_engine.jl"))
        save()=begin
            evidence["elapsed_seconds"]=time()-started
            open(io->JSON3.pretty(io,evidence),output,"w")
            evidence
        end
        try
            save()
            for (label,check) in (("memory",test_knn_memory),("features",test_logit_features))
                time()<deadline-140 || return save()
                counts=Test.get_test_counts(check(device))
                evidence[label]=(;passed=counts.passes+counts.cumulative_passes,
                    failed=counts.fails+counts.cumulative_fails,errors=counts.errors+counts.cumulative_errors)
                evidence["phase"]=label*" shared kernels passed";save()
            end
            time()<deadline-110 || return save()
            model=Q.loadmodel(model_dir;quantize=:none,device)
            evidence["phase"]="FP32 native model loaded";save()
            ids=Int.(case.julia_ids);session=Q.session(model;capacity=256,kv_type=Float32,attention=:split)
            try
                extent=audit_plan_argument_extents(session.plan)
                @test extent.all_fit
                evidence["decode_argument_extents"]=extent;save()
                for first in 1:8:length(ids)
                    time()<deadline-25 || error("native parity cleanup margin")
                    Q.prefill!(session,ids[first:min(first+7,length(ids))];chunk=8)
                    Mantle.waitidle(device)
                end
                got=Array(session.logits);want=vec(tensors[String(case.id)])
                relative_error=norm(Float64.(got).-want)/norm(Float64.(want))
                maxabs=maximum(abs,Float64.(got).-want)
                before=Mantle.submissions(device);Q.decode!(session,length(case.reference_greedy_ids))
                submissions=Mantle.submissions(device)-before
                tokens=Int.(Array(session.predictions)[length(ids):length(ids)+length(case.reference_greedy_ids)-1])
                @test all(isfinite,got)
                @test relative_error<2e-4
                @test maxabs<0.02
                @test tokens==Int.(case.reference_greedy_ids)
                @test submissions==length(tokens)
                evidence["native"]=(;input_tokens=length(ids),relative_rms=relative_error,maxabs,tokens,
                    native_tokens=Int.(case.reference_greedy_ids),decode_submissions=submissions,
                    reference_sha256=bytes2hex(sha256(read(reference_path))))
            finally
                close(session)
            end
            evidence["phase"]="all requested checks passed";evidence["complete"]=true;save()
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end
