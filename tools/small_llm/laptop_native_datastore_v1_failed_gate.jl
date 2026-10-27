include("laptop_pending_study.jl")
isdefined(@__MODULE__,:HeadInputTap) || include("head_input_tap.jl")
isdefined(@__MODULE__,:LogitFeatures) || include("logit_features.jl")
using LinearAlgebra

function freeze_laptop_native_datastore(;directory=joinpath(@__DIR__,"laptop_native_datastore"))
    oldpath=joinpath(@__DIR__,"native_hidden_datastore/registration.json")
    old=JSON3.read(read(oldpath,String))
    config=JSON3.read(read(joinpath(String(old.model_dir),"config.json"),String)).text_config
    Int(config.hidden_size)==old.width || error("registered hidden width drift")
    gatepath=joinpath(@__DIR__,"laptop_portable_identity_gate.json");gate=JSON3.read(read(gatepath,String))
    gate.complete && gate.normal_final_waitidle && gate.full_model_passed==10 || error("portable gate missing")
    for row in old.rows
        bytes2hex(sha256(String(row.source)))==row.source_sha256 || error("published source text drift")
        # Only source-token predictions are captured, not generated solutions.
        2<=length(row.token_ids)<=769 || error("source token count")
    end
    ispath(directory)&&!isempty(readdir(directory))&&error("refuse overwrite")
    packet=(;scope="Laptop native final-RMSNorm hidden-key capture of ONLY the previously frozen published source corpus; infrastructure, NOT task-quality evidence",
        prompt=old.prompt,rows=old.rows,width=old.width,vocabulary=Int(config.vocab_size),thinking=false,
        alignment="after source token i predicts token i+1; no cross-document/EOS labels",
        original_registration_sha256=bytes2hex(open(sha256,oldpath)),gate_sha256=bytes2hex(open(sha256,gatepath)),
        portable_manifest_sha256=gate.portable_manifest_sha256,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("laptop_native_datastore.jl","laptop_pending_study.jl","portable_qwen35.jl","head_input_tap.jl",
                "logit_features.jl","qwen35.jl","prefill_qwen35.jl","qwen_weight_cache.jl","device_finish.jl","runtime_provenance.jl")))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"registration.json"),"w");packet
end

function laptop_native_datastore_state(directory)
    path=joinpath(directory,"registration.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash || error("capture source drift")
    end
    bytes2hex(open(sha256,joinpath(@__DIR__,"native_hidden_datastore/registration.json")))==p.original_registration_sha256 || error("original registration drift")
    bytes2hex(open(sha256,joinpath(@__DIR__,"laptop_portable_identity_gate.json")))==p.gate_sha256 || error("portable gate drift")
    (;path,packet=p)
end

function capture_laptop_native_datastore(;seconds=210,directory=joinpath(@__DIR__,"laptop_native_datastore"))
    seconds>0 || throw(ArgumentError("time budget"))
    state=laptop_native_datastore_state(directory);p=state.packet
    bytes2hex(open(sha256,"gen/qwen35-4b-portable/portable.json"))==p.portable_manifest_sha256 || error("tensor drift")
    digest(path)=bytes2hex(open(sha256,path))
    Q=SmallLLMChecks.Qwen35Experiment;T=SmallLLMChecks.QwenText
    tokenizer=T.Tokenizer("gen/qwen35-4b-portable")
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in p.prompt]
    prompt=T.chat_ids(tokenizer,messages;thinking=false)
    pending=filter(r->!isfile(joinpath(directory,String(r.id)*".json")),p.rows)
    isempty(pending)&&return (;complete=true)
    any(isfile(joinpath(directory,String(r.id)*"_attempt.json")) for r in pending)&&error("unfinished capture attempt; preserve it")
    let
        started=time();device=Mantle.Device("radeon");model=nothing;query=nothing;completed=String[]
        try
            model=load_portable_qwen35(Q,"gen/qwen35-4b-portable",device)
            Int(model.config.hidden_size)==p.width||error("hidden width drift")
            query=Mantle.Buffer(device,zeros(Float32,Int(p.width)))
            gatepath=joinpath(directory,"head_gate.json")
            if !isfile(gatepath)
                reference=JSON3.read(read(joinpath(@__DIR__,"conversation_native_reference.json"),String))
                ids=Int.(only(filter(r->r.id=="baseline_short",reference.rows)).julia_ids)
                function observe()
                    session=Q.session(model;capacity=256,kv_type=Float32,attention=:split)
                    try
                        Q.prefill!(session,ids;chunk=8);Mantle.waitidle(device);pre=Array(session.logits)
                        before=Mantle.submissions(device);Q.decode!(session,8);Mantle.waitidle(device)
                        (;pre,post=Array(session.logits),tokens=Int.(Array(session.predictions)[length(ids):length(ids)+7]),submissions=Mantle.submissions(device)-before)
                    finally close(session) end
                end
                open(io->JSON3.pretty(io,(;complete=false,registration_sha256=digest(state.path))),gatepath,"w")
                original=model.weights["lm_head"];direct=observe();tapped=with_head_input_tap(observe,model,query);hidden=Array(query)
                direct.pre==tapped.pre && direct.post==tapped.post && direct.tokens==tapped.tokens && direct.submissions==tapped.submissions==8 || error("head tap perturbs model")
                model.weights["lm_head"]===original && length(hidden)==p.width && all(isfinite,hidden) && norm(hidden)>0 || error("head tap invariant")
                gate=(;complete=true,checks=8,registration_sha256=digest(state.path),runtime_provenance=inference_runtime_provenance())
                open(io->JSON3.pretty(io,gate),gatepath,"w")
            end
            gate=JSON3.read(read(gatepath,String))
            gate.complete && gate.registration_sha256==digest(state.path) || error("head gate incomplete")
            for row in pending
                time()<started+seconds-65 || return (;complete=false,completed)
                id=String(row.id);output=joinpath(directory,id*".json");attempt=joinpath(directory,id*"_attempt.json")
                isfile(attempt)&&error("refuse capture retry")
                ids=Int.(row.token_ids);records=length(ids)-1
                open(io->JSON3.pretty(io,(;registration_sha256=digest(state.path),started_unix_seconds=time())),attempt,"w")
                capture=LogitFeatures.Capture(device,Int(p.width),records)
                try
                    keys=with_head_input_tap(model,query) do
                        session=Q.session(model;capacity=nextpow(2,length(prompt)+length(ids)+32),
                            logit_hook=(g,logits)->LogitFeatures.declare_capture!(g,query,capture))
                        try
                            Q.prefill!(session,prompt;chunk=8);Mantle.waitidle(device)
                            LogitFeatures.reset!(capture);LogitFeatures.activate!(capture)
                            for i=1:records
                                time()<started+seconds-20||error("capture cleanup margin")
                                Q.step!(session,ids[i]);i%8==0&&Mantle.waitidle(device)
                            end
                            Mantle.waitidle(device)
                            only(Array(capture.counter))==records&&only(Array(capture.status))==0||error("capture count/overflow")
                            Array(capture.history)
                        finally close(session) end
                    end
                    all(isfinite,keys)&&all(j->sum(abs2,view(keys,:,j))>0,axes(keys,2))||error("invalid native keys")
                    binary=id*".f32";binarypath=joinpath(directory,binary);isfile(binarypath)&&error("partial binary exists")
                    open(io->write(io,keys),binarypath,"w")
                    receipt=(;id,width=Int(p.width),records,labels=ids[2:end],keys_file=binary,keys_sha256=digest(binarypath),
                        endian_bom=Int(Base.ENDIAN_BOM),registration_sha256=digest(state.path),attempt_sha256=digest(attempt),runtime_provenance=inference_runtime_provenance())
                    open(io->JSON3.pretty(io,receipt),output,"w");push!(completed,id)
                finally close(capture) end
            end
            (;complete=true,completed)
        finally
            try
                query===nothing||Mantle.free!(query);model===nothing||close(model)
            finally finish_device!(device) end
        end
    end
end
