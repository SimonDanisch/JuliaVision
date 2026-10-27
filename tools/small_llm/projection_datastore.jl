isdefined(@__MODULE__,:run_humaneval_baselines) || include("run_humaneval_subset.jl")
isdefined(@__MODULE__,:LogitFeatures) || include("logit_features.jl")
isdefined(@__MODULE__,:KNNMemory) || include("knn_memory.jl")

function projection_checkpoint_identity(directory)
    index=JSON3.read(read(joinpath(directory,"model.safetensors.index.json"),String))
    files=sort!(unique!(vcat(["config.json","tokenizer.json","model.safetensors.index.json"],String.(collect(values(index.weight_map))))))
    all(file->basename(file)==file,files) || error("checkpoint path escape")
    Dict(file=>bytes2hex(open(sha256,joinpath(directory,file))) for file in files)
end

function verify_projection_checkpoint(packet)
    current=projection_checkpoint_identity(String(packet.model_dir))
    expected=Dict(String(k)=>String(v) for (k,v) in pairs(packet.checkpoint_sha256))
    current==expected || error("datastore checkpoint/tokenizer drift")
    nothing
end

function complete_source_prefix(text;expressions=2)
    1<=expressions<=4 || throw(ArgumentError("source expression bound"))
    position=firstindex(text);pieces=String[]
    for _ in 1:expressions
        expression,next=Meta.parse(text,position;greedy=true,raise=false)
        expression===nothing && break
        expression isa Expr && expression.head in (:error,:incomplete) && break
        push!(pieces,String(text[position:prevind(text,next)]));position=next
        position>lastindex(text) && break
    end
    length(pieces)==expressions || throw(ArgumentError("not enough complete published source expressions"))
    join(pieces,"\n")
end

function freeze_projection_datastore(;model_dir="gen/qwen35-4b")
    directory=joinpath(@__DIR__,"projection_datastore");mkpath(directory)
    path=joinpath(directory,"registration.json");isfile(path) && error("datastore registration already frozen")
    snapshot_path=joinpath(@__DIR__,"installed_method_cards.json")
    snapshot=JSON3.read(read(snapshot_path,String))
    tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
    vocabulary=Int(JSON3.read(read(joinpath(model_dir,"config.json"),String)).text_config.vocab_size)
    feature_ids=Int32[mod(17+1009i,vocabulary) for i in 0:511]
    length(unique(feature_ids))==512 || error("feature projection IDs collided")
    rows=map((("makie_update",only(snapshot.makie_update)),("kernel_call",only(snapshot.kernel_call)))) do (id,card)
        source=complete_source_prefix(String(card.text))
        ids=SmallLLMChecks.QwenText.encode(tokenizer,source)
        9<=length(ids)<=768 || error("bounded source token count")
        (;id,source,token_ids=ids,source_sha256=bytes2hex(sha256(source)),
            installed_file=String(card.file),installed_file_sha256=String(card.source_sha256))
    end
    prompt=["system"=>"You are a Julia programmer. Return only executable Julia code.",
        "user"=>"Continue the Julia source code. Code only."]
    report=(;scope="small published-API output-projection datastore pilot, not native-hidden-state kNN-LM or trained latent fusion",
        sources="ONLY bounded installed method definitions; no scoring assertions, authored gold solutions or generated candidates",
        model_dir,checkpoint_sha256=projection_checkpoint_identity(model_dir),
        config_sha256=bytes2hex(sha256(read(joinpath(model_dir,"config.json")))),
        snapshot_sha256=bytes2hex(sha256(read(snapshot_path))),feature_ids,
        projection="fixed native logit subset, centered and RMS-normalized using the SAME shared kernel during capture and lookup",
        prompt,thinking=false,rows,source_sha256=bytes2hex(sha256(read(@__FILE__))),
        feature_source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"logit_features.jl")))))
    open(io->JSON3.pretty(io,report),path,"w")
    report
end

function require_post_extent_gate()
    path=joinpath(@__DIR__,"post_extent_engine_validation.json")
    isfile(path) || error("post-extent Radeon/native gate required first")
    gate=JSON3.read(read(path,String));gate.complete || error("incomplete post-extent gate")
    current=inference_runtime_provenance()
    for (package,descriptor) in current
        descriptor.source_sha256==Dict(String(k)=>String(v) for (k,v) in pairs(gate.runtime_provenance[Symbol(package)].source_sha256)) ||
            error("runtime changed since post-extent gate: $package")
    end
    for file in ("knn_memory.jl","logit_features.jl","qwen35.jl","prefill_qwen35.jl")
        gate.experiment_source_sha256[Symbol(file)]==bytes2hex(sha256(read(joinpath(@__DIR__,file)))) || error("memory/model source changed since gate")
    end
    gate
end

function build_projection_datastore(;seconds=240)
    seconds>0 || throw(ArgumentError("time budget"))
    require_post_extent_gate()
    directory=joinpath(@__DIR__,"projection_datastore");registration=joinpath(directory,"registration.json")
    packet=JSON3.read(read(registration,String))
    verify_projection_checkpoint(packet)
    packet.source_sha256==bytes2hex(sha256(read(@__FILE__))) || error("datastore builder drift")
    packet.config_sha256==bytes2hex(sha256(read(joinpath(String(packet.model_dir),"config.json")))) || error("model config drift")
    packet.feature_source_sha256==bytes2hex(sha256(read(joinpath(@__DIR__,"logit_features.jl")))) || error("feature source drift")
    cache=qualified_qwen_cache();Q=SmallLLMChecks.Qwen35Experiment;T=SmallLLMChecks.QwenText
    tokenizer=T.Tokenizer(String(packet.model_dir));messages=[String(first(keys(m)))=>String(first(values(m))) for m in packet.prompt]
    prompt=T.chat_ids(tokenizer,messages;thinking=false)
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=Q.loadmodel(String(packet.model_dir);quantize=:q4k,device,quantcache=cache)
            for row in packet.rows
                path=joinpath(directory,String(row.id)*".json");isfile(path) && continue
                time()<deadline-90 || return (;complete=false,completed)
                ids=Int.(row.token_ids);records=length(ids)-1;width=length(packet.feature_ids)
                features=LogitFeatures.Features(device,Int(model.config.vocab_size),Int32.(packet.feature_ids))
                capture=LogitFeatures.Capture(device,width,records)
                query=Mantle.Buffer(device,zeros(Float32,width));session=nothing
                try
                    hook=(graph,logits)->begin
                        LogitFeatures.declare!(graph,query,logits,features)
                        LogitFeatures.declare_capture!(graph,query,capture)
                    end
                    session=Q.session(model;capacity=nextpow(2,length(prompt)+length(ids)+32),logit_hook=hook)
                    Q.prefill!(session,prompt;chunk=8);Mantle.waitidle(device)
                    LogitFeatures.reset!(capture);LogitFeatures.activate!(capture)
                    for i in 1:records
                        time()<deadline-25 || error("bounded datastore capture margin")
                        Q.step!(session,ids[i])
                        # Offline diagnostic ingestion, not generation token feedback.
                        i%8==0 && Mantle.waitidle(device)
                    end
                    Mantle.waitidle(device)
                    only(Array(capture.counter))==records && only(Array(capture.status))==0 || error("capture count/overflow")
                    keys=Array(capture.history);all(isfinite,keys) || error("invalid source feature vectors")
                    binary=String(row.id)*".f32"
                    isfile(joinpath(directory,binary)) && error("unowned/partial datastore binary exists")
                    open(io->write(io,keys),joinpath(directory,binary),"w")
                    result=(;id=row.id,records,width,keys_file=binary,keys_sha256=bytes2hex(sha256(read(joinpath(directory,binary)))),
                        endian_bom=Int(Base.ENDIAN_BOM),labels=ids[2:end],
                        registration_sha256=bytes2hex(sha256(read(registration))),
                        runtime_provenance=inference_runtime_provenance(),elapsed_seconds=time()-started)
                    open(io->JSON3.pretty(io,result),path,"w");push!(completed,String(row.id))
                    println(row.id," model-specific datastore records saved: ",records);flush(stdout)
                finally
                    session===nothing || close(session)
                    close(features);close(capture);Mantle.free!(query)
                end
            end
            (;complete=true,completed,elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end

"""Read bounded, hashed raw arrays only; never deserialize executable Julia objects."""
function load_projection_datastore(;directory=joinpath(@__DIR__,"projection_datastore"))
    registration=joinpath(directory,"registration.json")
    packet=JSON3.read(read(registration,String))
    verify_projection_checkpoint(packet)
    registration_sha=bytes2hex(sha256(read(registration)))
    width=length(packet.feature_ids)
    1<=width<=65536 || error("invalid feature width")
    vocabulary=Int(JSON3.read(read(joinpath(String(packet.model_dir),"config.json"),String)).text_config.vocab_size)
    packet.config_sha256==bytes2hex(sha256(read(joinpath(String(packet.model_dir),"config.json")))) || error("model config drift")
    matrices=Matrix{Float32}[];labels=Int32[]
    seen=Set{String}()
    for source in packet.rows
        id=String(source.id)
        occursin(r"^[a-z_]+$",id) && !(id in seen) || error("invalid/duplicate datastore ID")
        push!(seen,id)
        row=JSON3.read(read(joinpath(directory,id*".json"),String))
        row.registration_sha256==registration_sha || error("datastore registration mismatch")
        row.id==id && row.width==width && 1<=row.records<=768 || error("datastore dimensions")
        row.endian_bom==Int(Base.ENDIAN_BOM) || error("datastore endianness")
        String(row.keys_file)==id*".f32" || error("unexpected datastore filename")
        path=joinpath(directory,String(row.keys_file))
        islink(path) && error("datastore symlink rejected")
        filesize(path)==4*width*Int(row.records) || error("datastore byte count")
        bytes2hex(sha256(read(path)))==row.keys_sha256 || error("datastore checksum")
        row.labels==source.token_ids[2:end] || error("source-label alignment")
        length(row.labels)==row.records && all(x->x isa Integer && !(x isa Bool) && 0<=x<vocabulary,row.labels) || error("datastore token labels")
        data=Matrix{Float32}(undef,width,Int(row.records));read!(path,data)
        all(isfinite,data) || error("nonfinite datastore keys")
        push!(matrices,data);append!(labels,Int32.(row.labels))
    end
    isempty(matrices) && error("empty datastore")
    (;keys=hcat(matrices...),labels,feature_ids=Int32.(packet.feature_ids),vocabulary,
        registration_sha256=registration_sha)
end
