include("laptop_pending_study.jl")
include("finite_choice_constraint_gpu.jl")
include("finite_choice_model_checks.jl")
import BonsaiRunner

function run_finite_choice_model_gate(;directory=joinpath(@__DIR__,"finite_choice_model_gate"))
    normpath(pwd())=="/tmp/small-llm-bosgame-20261025"&&Sys.total_memory()<=40*1024^3||error("Laptop snapshot only")
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    digest(path)=bytes2hex(sha256(read(path)))
    p.ready&&p.steps==8&&p.seed==20261025||error("Frozen gate policy")
    for (f,h) in pairs(p.source_sha256)
        digest(joinpath(@__DIR__,String(f)))==h||error("Gate source drift")
    end
    digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256||error("Model drift")
    output=joinpath(directory,"result.json");ispath(output)&&error("Preserve first gate attempt")
    Q=SmallLLMChecks.Qwen35Experiment;T=SmallLLMChecks.QwenText
    tokenizer=T.Tokenizer("gen/qwen35-4b-portable")
    grammar=FiniteChoiceConstraint.Grammar(s->T.encode(tokenizer,s),ids->T.decode(tokenizer,ids;skipspecial=false),String.(p.choices),Int(p.vocabulary),tokenizer.bpe.eos)
    all(length(s)<=8 for s in grammar.sequences)||error("Support cannot complete within gate budget")
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in p.messages]
    ids=T.chat_ids(tokenizer,messages;thinking=false);length(ids)+10<=256||error("Gate capacity")
    report=Dict{String,Any}("complete"=>false,"pid"=>getpid(),"conditions_sha256"=>digest(path),
        "scope"=>p.scope,"source_sha256"=>p.source_sha256,"runtime_provenance"=>inference_runtime_provenance())
    save()=open(io->JSON3.pretty(io,report),output,"w")
    let
        let
            save();device=Mantle.Device("radeon");model=nothing
            function observe(control)
                sampler=GPUSampling.State(device,Int(model.config.vocab_size);seed=UInt32(p.seed),topk=20,temperature=.6f0,top_p=.95f0,presence=0f0,fallback=tokenizer.bpe.eos)
                constraint=nothing;session=nothing;raw=nothing
                try
                    control==:baseline||(constraint=FiniteChoiceConstraintGPU.State(device,grammar;enabled=control==:constrained))
                    raw=Mantle.Buffer(device,zeros(Float32,Int(model.config.vocab_size)))
                    GPUSampling.activate!(sampler,false)
                    hook=(graph,logits)->begin
                        Mantle.dispatch!(graph,BonsaiRunner.copy_kernel!,(raw,logits,Int32(model.config.vocab_size)),Int(model.config.vocab_size);group=256,name="choice_gate.raw_logits")
                        constraint===nothing ? GPUSampling.declare!(graph,logits,sampler) : FiniteChoiceConstraintGPU.declare!(graph,logits,constraint,sampler)
                    end
                    session=Q.session(model;capacity=256,kv_type=Float32,attention=:split,logit_hook=hook)
                    Q.prefill!(session,ids[1:end-1];chunk=8);GPUSampling.activate!(sampler,true);Q.step!(session,last(ids))
                    before=Mantle.submissions(device);Q.decode!(session,8);Mantle.waitidle(device)
                    submissions=Mantle.submissions(device)-before
                    (;tokens=Int32.(Array(session.predictions)[length(ids):length(ids)+7]),raw_logits=Array(raw),seen=Array(sampler.seen),
                        draws=Int(only(Array(sampler.counter))),status=Int(only(Array(sampler.status))),submissions,
                        node=constraint===nothing ? 1 : Int(only(Array(constraint.node))),
                        done=constraint===nothing ? 0 : Int(only(Array(constraint.done))),
                        rejected=constraint===nothing ? 0 : Int(only(Array(constraint.rejected))))
                finally
                    session===nothing||close(session)
                    constraint===nothing||close(constraint)
                    raw===nothing||Mantle.free!(raw)
                    close(sampler)
                end
            end
            try
                model=load_portable_qwen35(Q,"gen/qwen35-4b-portable",device)
                model.config.vocab_size==p.vocabulary||error("Model vocabulary")
                observations=Any[]
                for control in (:baseline,:disabled,:constrained)
                    push!(observations,observe(control));report["phase"]=String(control);save()
                end
                result=finite_choice_model_checks(observations...,grammar.sequences,grammar.eos)
                report["result"]=result;report["tokens"]=Dict(String(k)=>o.tokens for (k,o) in zip((:baseline,:disabled,:constrained),observations))
                report["raw_logit_sha256"]=Dict(String(k)=>bytes2hex(sha256(reinterpret(UInt8,o.raw_logits))) for (k,o) in zip((:baseline,:disabled,:constrained),observations));save()
                result.passed||error("Full-model finite-choice integration failed")
            finally
                try model===nothing||close(model)
                finally finish_device!(device) end
            end
            report["complete"]=true;report["normal_final_waitidle"]=true;save();report
        end
    end
end
