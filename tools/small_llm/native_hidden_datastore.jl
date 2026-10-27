isdefined(@__MODULE__,:verify_projection_checkpoint) || include("projection_datastore.jl")
isdefined(@__MODULE__,:with_head_input_tap) || include("head_input_tap.jl")

const NATIVE_DATASTORE_SOURCES=("native_hidden_datastore.jl","head_input_tap.jl",
    "logit_features.jl","qwen35.jl","prefill_qwen35.jl","qwen_weight_cache.jl")

function require_native_head_gate()
    path=joinpath(@__DIR__,"head_input_full_model_validation.json")
    gate=JSON3.read(read(path,String))
    gate.complete && gate.normal_final_waitidle && gate.checks.passed==8 || error("incomplete head gate")
    for (file,hash) in pairs(gate.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("head gate source drift")
    end
    for (package,descriptor) in inference_runtime_provenance()
        descriptor.source_sha256==Dict(String(k)=>String(v) for (k,v) in pairs(gate.runtime_provenance[Symbol(package)].source_sha256)) || error("head gate runtime drift")
    end
    path
end

function freeze_native_hidden_datastore()
    gate=require_native_head_gate()
    sourcepath=joinpath(@__DIR__,"projection_datastore","registration.json")
    source=JSON3.read(read(sourcepath,String));verify_projection_checkpoint(source)
    directory=joinpath(@__DIR__,"native_hidden_datastore");mkpath(directory)
    path=joinpath(directory,"registration.json");isfile(path) && error("already frozen")
    config=JSON3.read(read(joinpath(String(source.model_dir),"config.json"),String))
    packet=(;scope="source-only native final-RMSNorm hidden keys, not projected logits; no quality claim",
        sources="reuse EXACT published source/token/prompt corpus from the negative projection pilot, never generated answers or private tests",
        alignment="state after source token i predicts source token i+1; no EOS/end-of-source cross-document labels",
        representation="unmodified Float32 final RMSNorm input to lm_head; raw squared-L2 keys and queries, no learned projection",
        model_dir=source.model_dir,checkpoint_sha256=source.checkpoint_sha256,config_sha256=source.config_sha256,
        prompt=source.prompt,thinking=false,rows=source.rows,width=Int(config.text_config.hidden_size),
        prior_registration_sha256=bytes2hex(sha256(read(sourcepath))),head_gate_sha256=bytes2hex(sha256(read(gate))),
        source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in NATIVE_DATASTORE_SOURCES))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function native_hidden_datastore_state()
    gate=require_native_head_gate()
    directory=joinpath(@__DIR__,"native_hidden_datastore");path=joinpath(directory,"registration.json")
    packet=JSON3.read(read(path,String));verify_projection_checkpoint(packet)
    packet.head_gate_sha256==bytes2hex(sha256(read(gate))) || error("gate receipt drift")
    for (file,hash) in pairs(packet.source_sha256)
        bytes2hex(sha256(read(joinpath(@__DIR__,String(file)))))==hash || error("native capture source drift")
    end
    sourcepath=joinpath(@__DIR__,"projection_datastore","registration.json")
    packet.prior_registration_sha256==bytes2hex(sha256(read(sourcepath))) || error("published source registration drift")
    (;directory,path,packet)
end

function build_native_hidden_datastore(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    state=native_hidden_datastore_state();p=state.packet
    cache=qualified_qwen_cache();Q=SmallLLMChecks.Qwen35Experiment;T=SmallLLMChecks.QwenText
    tokenizer=T.Tokenizer(String(p.model_dir))
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in p.prompt]
    prompt=T.chat_ids(tokenizer,messages;thinking=false)
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=Q.loadmodel(String(p.model_dir);quantize=:q4k,device,quantcache=cache)
            Int(model.config.hidden_size)==p.width || error("native width drift")
            for row in p.rows
                output=joinpath(state.directory,String(row.id)*".json");isfile(output) && continue
                time()<started+seconds-80 || return (;complete=false,completed)
                ids=Int.(row.token_ids);records=length(ids)-1;width=Int(p.width)
                capture=LogitFeatures.Capture(device,width,records)
                query=Mantle.Buffer(device,zeros(Float32,width))
                try
                    keys=with_head_input_tap(model,query) do
                        session=Q.session(model;capacity=nextpow(2,length(prompt)+length(ids)+32),
                            logit_hook=(graph,logits)->LogitFeatures.declare_capture!(graph,query,capture))
                        try
                            Q.prefill!(session,prompt;chunk=8);Mantle.waitidle(device)
                            LogitFeatures.reset!(capture);LogitFeatures.activate!(capture)
                            for i in 1:records
                                time()<started+seconds-25 || error("capture cleanup margin")
                                Q.step!(session,ids[i]);i%8==0 && Mantle.waitidle(device)
                            end
                            Mantle.waitidle(device)
                            only(Array(capture.counter))==records && only(Array(capture.status))==0 || error("capture count/overflow")
                            Array(capture.history)
                        finally
                            close(session)
                        end
                    end
                    all(isfinite,keys) && all(j->sum(abs2,view(keys,:,j))>0,axes(keys,2)) || error("invalid hidden keys")
                    binary=String(row.id)*".f32";binarypath=joinpath(state.directory,binary)
                    isfile(binarypath) && error("preserve partial binary")
                    open(io->write(io,keys),binarypath,"w")
                    receipt=(;id=row.id,width,records,labels=ids[2:end],keys_file=binary,
                        keys_sha256=bytes2hex(sha256(read(binarypath))),endian_bom=Int(Base.ENDIAN_BOM),
                        registration_sha256=bytes2hex(sha256(read(state.path))),runtime_provenance=inference_runtime_provenance())
                    open(io->JSON3.pretty(io,receipt),output,"w");push!(completed,String(row.id))
                finally
                    close(capture);Mantle.free!(query)
                end
            end
            (;complete=true,completed)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end
