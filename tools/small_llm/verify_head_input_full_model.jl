isdefined(@__MODULE__,:HeadInputTap) || include("head_input_tap.jl")
isdefined(@__MODULE__,:qualified_qwen_cache) || include("run_humaneval_subset.jl")
using Test, LinearAlgebra, JSON3, SHA

function verify_head_input_full_model(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    output=joinpath(@__DIR__,"head_input_full_model_validation.json")
    isfile(output) && error("preserve previous gate attempt")
    reference=JSON3.read(read(joinpath(@__DIR__,"conversation_native_reference.json"),String))
    ids=Int.(only(filter(row->row.id=="baseline_short",reference.rows)).julia_ids)
    Q=SmallLLMChecks.Qwen35Experiment;cache=qualified_qwen_cache()
    let
        started=time();device=Mantle.Device("radeon");model=nothing;query=nothing
        report=Dict{String,Any}("complete"=>false,"scope"=>"read-only full-model native head-input capture; NOT intelligence",
            "phase"=>"fresh device setup","runtime_provenance"=>inference_runtime_provenance(),
            "source_sha256"=>Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
                ("verify_head_input_full_model.jl","head_input_tap.jl","qwen35.jl","prefill_qwen35.jl")))
        save()=begin
            report["elapsed_seconds"]=time()-started
            open(io->JSON3.pretty(io,report),output,"w");report
        end
        function observe()
            session=Q.session(model;capacity=256,kv_type=Float32,attention=:split)
            try
                Q.prefill!(session,ids;chunk=8);Mantle.waitidle(device)
                pre=Array(session.logits)
                before=Mantle.submissions(device);Q.decode!(session,8);Mantle.waitidle(device)
                submissions=Mantle.submissions(device)-before
                tokens=Int.(Array(session.predictions)[length(ids):length(ids)+7])
                (;pre,post=Array(session.logits),tokens,submissions)
            finally
                close(session)
            end
        end
        try
            save();model=Q.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            report["phase"]="model loaded";save()
            time()<started+seconds-70 || return save()
            original=model.weights["lm_head"]
            direct=observe();query=Mantle.Buffer(device,zeros(Float32,Int(model.config.hidden_size)))
            tapped=with_head_input_tap(observe,model,query)
            hidden=Array(query)
            @test direct.pre==tapped.pre
            @test direct.post==tapped.post
            @test direct.tokens==tapped.tokens
            @test direct.submissions==tapped.submissions==8
            @test model.weights["lm_head"]===original
            @test length(hidden)==Int(model.config.hidden_size)
            @test all(isfinite,hidden)
            @test norm(hidden)>0
            report["checks"]=(;passed=8,exact_prefill_logits=true,exact_decode_logits=true,exact_tokens=true,
                decode_submissions=tapped.submissions,native_width=length(hidden),head_input_norm=norm(hidden),tokens=tapped.tokens)
            report["phase"]="capture invariants passed";save()
        finally
            try
                query===nothing || Mantle.free!(query)
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
        report["normal_final_waitidle"]=true;report["complete"]=true;save()
    end
end
