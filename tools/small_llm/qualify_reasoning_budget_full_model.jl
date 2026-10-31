isdefined(@__MODULE__,:sampled_budget_julia_answer) || include("sampled_budget_julia.jl")
isdefined(@__MODULE__,:qualified_qwen_cache) || include("run_humaneval_subset.jl")
isdefined(@__MODULE__,:test_reasoning_budget) || include("test_reasoning_budget.jl")

function qualify_reasoning_budget_full_model(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    output=joinpath(@__DIR__,"reasoning_budget_full_model_validation.json")
    isfile(output) && error("preserve prior gate attempt")
    qualification=JSON3.read(read(joinpath(@__DIR__,"reasoning_budget_software_validation.json"),String))
    qualification.complete || error("software gate required")
    for (file,hash) in pairs(qualification.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("software qualification drift")
    end
    Q=SmallLLMChecks.Qwen35Experiment;T=SmallLLMChecks.QwenText;cache=qualified_qwen_cache()
    tokenizer=T.Tokenizer("gen/qwen35-4b");marker=T.encode(tokenizer,"</think>")
    length(marker)==1 && T.decode(tokenizer,marker;skipspecial=false)=="</think>" || error("single native token required")
    messages=["system"=>"You are a Julia programmer. Return only executable code.",
        "user"=>"Write a Julia function f(x) returning x+1. Return only Julia code."]
    ids=T.chat_ids(tokenizer,messages;thinking=true)
    let
        started=time();device=Mantle.Device("radeon");model=nothing
        report=Dict{String,Any}("complete"=>false,"scope"=>"real4B controller identity/activation gate, NOT task-quality evidence",
            "runtime_provenance"=>inference_runtime_provenance(),"messages"=>messages,
            "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                ("qualify_reasoning_budget_full_model.jl","reasoning_budget.jl","sampled_budget_julia.jl",
                    "test_reasoning_budget.jl","gpu_sampling.jl","qwen35.jl","prefill_qwen35.jl")))
        save()=open(io->JSON3.pretty(io,report),output,"w")
        function observe(control)
            sampler=GPUSampling.State(device,Int(model.config.vocab_size);seed=UInt32(20261024),topk=20)
            budget=nothing;session=nothing
            try
                control===:baseline || (budget=ReasoningBudget.State(device,Int(model.config.vocab_size),only(marker);
                    limit=control===:disabled ? typemax(UInt32) : UInt32(0)))
                GPUSampling.activate!(sampler,false)
                hook=(graph,logits)->begin
                    budget===nothing || ReasoningBudget.declare!(graph,logits,budget,sampler)
                    GPUSampling.declare!(graph,logits,sampler)
                end
                session=Q.session(model;capacity=256,kv_type=Float32,attention=:split,logit_hook=hook)
                Q.prefill!(session,ids[1:end-1];chunk=8);GPUSampling.activate!(sampler,true);Q.step!(session,last(ids))
                before=Mantle.submissions(device);Q.decode!(session,8);Mantle.waitidle(device)
                decode_submissions=Mantle.submissions(device)-before
                (;tokens=Int.(Array(session.predictions)[length(ids):length(ids)+7]),logits=Array(session.logits),
                    seen=Array(sampler.seen),draws=Int(only(Array(sampler.counter))),status=only(Array(sampler.status)),
                    submissions=decode_submissions,
                    forced=budget===nothing ? 0 : Int(only(Array(budget.forced_count))))
            finally
                session===nothing || close(session)
                budget===nothing || close(budget)
                close(sampler)
            end
        end
        try
            save();counts=Test.get_test_counts(test_reasoning_budget(device))
            report["hardware_fixture_passed"]=counts.passes+counts.cumulative_passes;save()
            time()<started+seconds-140 || return save()
            model=Q.loadmodel("gen/qwen35-4b";quantize=:q4k,device,quantcache=cache)
            time()<started+seconds-90 || return save()
            direct=observe(:baseline);report["phase"]="baseline observed";save()
            time()<started+seconds-60 || return save()
            disabled=observe(:disabled);report["phase"]="disabled observed";save()
            time()<started+seconds-35 || return save()
            forced=observe(:force)
            tests=@testset "Real4B reasoning-budget identity and activation" begin
                @test direct.tokens==disabled.tokens
                @test direct.logits==disabled.logits
                @test direct.seen==disabled.seen
                @test direct.draws==disabled.draws==forced.draws==9
                @test direct.status==disabled.status==forced.status==0
                @test direct.submissions==disabled.submissions==forced.submissions==8
                @test direct.forced==disabled.forced==0
                @test forced.forced==1
                @test first(forced.tokens)==only(marker)
            end
            c=Test.get_test_counts(tests);report["full_model_passed"]=c.passes+c.cumulative_passes
            report["closing_token"]=only(marker);report["direct_tokens"]=direct.tokens;report["forced_tokens"]=forced.tokens;save()
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
        report["complete"]=true;report["normal_final_waitidle"]=true;report["elapsed_seconds"]=time()-started;save();report
    end
end
