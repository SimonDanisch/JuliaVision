"""Native MoE checkpoint packing/loading, separate from frozen dense experiments.

All quantization uses the existing DNNKernels GPU quantizers. Cache files are
per tensor, committed data-first/receipt-last, and source/content hashed. No
constructor here deletes a cache on a policy mismatch.
"""
module Qwen36NativeCache
using JSON3,SHA
import Mantle,DNNKernels
const DK=DNNKernels
const POLICY="qwen36-native-v1:Q4_K-experts+dense;F32-router+embedding;Q6_K-independent-head"
const TYPES=Dict("Float32"=>Float32,"Float16"=>Float16,"UVec4"=>DK.UVec4)

function native_name(key)
    key=="lm_head.weight"&&return "lm_head"
    for prefix in ("model.language_model.","model.")
        startswith(key,prefix)||continue
        name=chopprefix(key,prefix)
        (name in ("embed_tokens.weight","norm.weight")||startswith(name,"layers."))&&return name
    end
    nothing
end

function expected_names(c)
    names=Set(["embed_tokens.weight","norm.weight","lm_head"])
    for il in 0:Int(c.num_hidden_layers)-1
        p="layers.$il."
        union!(names,p.*["input_layernorm.weight","post_attention_layernorm.weight"])
        union!(names,p.*"mlp.".*["gate.weight","experts.gate_up_proj","experts.down_proj",
            "shared_expert.gate_proj.weight","shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight","shared_expert_gate.weight"])
        if c.layer_types[il+1]=="linear_attention"
            union!(names,p.*"linear_attn.".*["A_log","dt_bias","conv1d.weight","norm.weight",
                "in_proj_qkv.weight","in_proj_z.weight","in_proj_a.weight","in_proj_b.weight","out_proj.weight"])
        else
            union!(names,p.*"self_attn.".*["q_proj.weight","k_proj.weight","v_proj.weight",
                "o_proj.weight","q_norm.weight","k_norm.weight"])
        end
    end
    names
end

function policy(name,shape)
    if occursin(".experts.",name)
        length(shape)==3&&shape[1]%256==0||throw(DimensionMismatch("Expert shape"))
        return "experts_q4k"
    end
    name=="lm_head"&&return "q6k"
    name=="embed_tokens.weight"&&return "raw"
    endswith(name,"mlp.gate.weight")&&return "dense"
    length(shape)==2&&return shape[1]%256==0&&shape[2]>=128 ? "q4k" : "dense"
    "raw"
end

function floating(raw,dtype)
    dtype in ("F32","F16","BF16")||throw(ArgumentError("Unsupported model dtype: $dtype"))
    dtype=="BF16" ? reinterpret.(Float32,UInt32.(raw).<<16) : Float32.(raw)
end

function transform!(x,name)
    if endswith(name,"input_layernorm.weight")||endswith(name,"post_attention_layernorm.weight")||
        name=="norm.weight"||endswith(name,"q_norm.weight")||endswith(name,"k_norm.weight")
        x.+=1f0
    elseif endswith(name,"A_log")
        x.=.-exp.(x)
    end
    x
end

function header(path)
    open(path) do io
        length=read(io,UInt64)
        length<=64*1024^2||error("Safetensors header too large")
        JSON3.read(read(io,Int(length)))
    end
end

function commit_json(path,value)
    ispath(path)&&error("Preserve existing receipt: $path")
    temp=path*".partial"
    ispath(temp)&&error("Incomplete prior receipt: $temp")
    open(io->JSON3.pretty(io,value),temp,"w")
    mv(temp,path)
end

function open_cache(checkpoint,directory,device)
    config=read(joinpath(checkpoint,"config.json"))
    descriptor=(;version=1,policy=POLICY,config_sha256=bytes2hex(sha256(config)),
        source_manifest_sha256=bytes2hex(sha256(read(joinpath(checkpoint,
            isfile(joinpath(checkpoint,"publisher-manifest.json")) ? "publisher-manifest.json" : "model.safetensors")))),
        quantizer_sha256=[bytes2hex(sha256(read(joinpath(dirname(pathof(DK)),f)))) for f in ("q4k.jl","q6k.jl")],
        packer_sha256=bytes2hex(sha256(read(@__FILE__))),
        device=string(Mantle.devicename(device)),subgroup=Mantle.caps(device).subgroup)
    canonical=JSON3.write(descriptor)
    fingerprint=bytes2hex(sha256(canonical))
    marker=joinpath(directory,"source.json")
    if isfile(marker)
        JSON3.read(read(marker,String)).fingerprint==fingerprint||error("Cache provenance mismatch; no files overwritten")
    else
        isdir(directory)&&!isempty(readdir(directory))&&error("Unowned nonempty cache")
        mkpath(directory);commit_json(marker,(;fingerprint,descriptor))
    end
    (;directory=abspath(directory),fingerprint,config=JSON3.read(config).text_config)
end

stem(name)=bytes2hex(sha256(name))

function validate_entry(cache,name;source_sha256=nothing)
    path=joinpath(cache.directory,stem(name)*".json")
    isfile(path)||return nothing
    entry=JSON3.read(read(path,String))
    entry.name==name&&entry.fingerprint==cache.fingerprint||error("Wrong tensor receipt")
    source_sha256===nothing||entry.source_sha256==source_sha256||error("Tensor source mismatch")
    bin=joinpath(cache.directory,stem(name)*".bin")
    filesize(bin)==entry.bytes&&bytes2hex(open(sha256,bin))==entry.sha256||error("Tensor data corrupt: $name")
    entry
end

function store_entry(cache,name,kind,shape,arrays,source_sha256)
    stempath=joinpath(cache.directory,stem(name));target=stempath*".bin";temp=target*".partial"
    ispath(target)&&error("Uncommitted/existing tensor data; refusing overwrite: $name")
    ispath(temp)&&error("Incomplete prior tensor: $name")
    specs=Any[]
    open(temp,"w") do io
        for array in arrays
            x=Array(array);T=eltype(x)
            type=T==DK.UVec4 ? "UVec4" : string(T)
            haskey(TYPES,type)||error("Unsupported cache array type")
            push!(specs,(;type,dims=collect(size(x)),offset=position(io),bytes=sizeof(x)))
            write(io,x)
        end
    end
    entry=(;name,kind,shape=collect(shape),arrays=specs,source_sha256,
        fingerprint=cache.fingerprint,bytes=filesize(temp),sha256=bytes2hex(open(sha256,temp)))
    mv(temp,target);commit_json(stempath*".json",entry)
    entry
end

function pack_tensor!(R,cache,name,raw,dtype,source_sha256,device)
    previous=validate_entry(cache,name;source_sha256)
    previous===nothing||return previous
    shape=size(raw);kind=policy(name,shape)
    x=floating(raw,dtype);owned=Any[]
    try
        if kind in ("q4k","q6k","experts_q4k")
            k=shape[1];m=prod(shape[2:end])
            # Flatten expert rows in native order; no cyclic head/expert remix.
            dense=Mantle.Buffer(device,permutedims(reshape(x,k,m)));push!(owned,dense)
            packed=kind=="q6k" ? DK.quantizeq6k(device,dense) : DK.quantizeq4k(device,dense)
            arrays=DK.kqarrays(packed);append!(owned,arrays)
            Mantle.waitidle(device)
            store_entry(cache,name,kind,shape,arrays,source_sha256)
        else
            transform!(x,name)
            store_entry(cache,name,kind,shape,(x,),source_sha256)
        end
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end

function pack_shard!(R,cache,checkpoint,file,device;expected_sha256)
    basename(file)==file||error("Shard must be a basename")
    path=joinpath(checkpoint,file)
    source_sha256=bytes2hex(open(sha256,path))
    source_sha256==expected_sha256||error("Shard hash mismatch")
    specs=header(path);tensors=DK.readsafetensors(path;mmap=true);count=0
    for key in sort!(collect(keys(tensors)))
        name=native_name(key);name===nothing&&continue
        name in expected_names(cache.config)||error("Unexpected text tensor: $name")
        pack_tensor!(R,cache,name,tensors[key],String(specs[Symbol(key)].dtype),source_sha256,device)
        count+=1
        println("Cached ",name);flush(stdout)
    end
    count
end

function finalize!(cache)
    names=sort!(collect(expected_names(cache.config)))
    entries=Dict(name=>validate_entry(cache,name) for name in names)
    all(x->x!==nothing,values(entries))||error("Checkpoint cache incomplete")
    path=joinpath(cache.directory,"complete.json")
    if isfile(path)
        old=JSON3.read(read(path,String));old.fingerprint==cache.fingerprint||error("Completion drift")
    else
        commit_json(path,(;fingerprint=cache.fingerprint,names))
    end
    length(names)
end

function loadmodel(R,cache,device)
    R.validate_configuration(cache.config)
    complete=JSON3.read(read(joinpath(cache.directory,"complete.json"),String))
    complete.fingerprint==cache.fingerprint&&Set(String.(complete.names))==expected_names(cache.config)||error("Incomplete/wrong cache")
    model=R.Model(device,cache.config,Dict{String,Any}(),Any[],:q4k)
    try
        for name in String.(complete.names)
            entry=validate_entry(cache,name);shape=Int.(entry.shape);arrays=Any[]
            open(joinpath(cache.directory,stem(name)*".bin")) do io
                for spec in entry.arrays
                    T=TYPES[String(spec.type)];dims=Tuple(Int.(spec.dims))
                    prod(dims)*sizeof(T)==spec.bytes||error("Cache extent mismatch")
                    seek(io,Int(spec.offset));x=Array{T}(undef,dims);read!(io,x)
                    buffer=Mantle.Buffer(device,x);push!(model.owned,buffer);push!(arrays,buffer)
                end
            end
            if entry.kind in ("q4k","experts_q4k","q6k")
                m=prod(shape[2:end]);k=shape[1]
                packed=entry.kind=="q6k" ? DK.Q6KMatrix(arrays...,m,k) : DK.Q4KMatrix(only(arrays),m,k)
                model.weights[name]=entry.kind=="experts_q4k" ? R.M.Q4Experts(packed,shape[2],shape[3]) : packed
            elseif entry.kind=="dense"
                model.weights[name]=R.Q.DenseWeight(only(arrays),shape[2],shape[1])
            elseif entry.kind=="raw"
                model.weights[name]=only(arrays)
            else
                error("Unknown tensor cache kind")
            end
        end
        model
    catch
        close(model);rethrow()
    end
end
end
