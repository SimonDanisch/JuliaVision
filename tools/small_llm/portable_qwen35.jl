# Portable loading of the ALREADY QUALIFIED quantized tensors. No quantizer,
# inference kernel, token rule or existing checkpoint/cache file is changed.
using JSON3,SHA
import DNNKernels,Mantle

function export_portable_qwen35(checkpoint,cache_directory,destination)
    ispath(destination) && error("refuse to overwrite portable export")
    source=JSON3.read(read(joinpath(cache_directory,"source.json"),String))
    index=JSON3.read(read(joinpath(cache_directory,"index.json"),String))
    index.version==DNNKernels.QUANTCACHE_VERSION && index.scheme=="qwen35_native_v1" || error("cache format")
    for (file,hash) in pairs(source.descriptor.sources)
        basename(String(file))==String(file) || error("checkpoint path")
        bytes2hex(open(sha256,joinpath(checkpoint,String(file))))==hash || error("checkpoint source changed")
    end
    mkpath(destination);mkpath(joinpath(destination,"cache"))
    for file in ("index.json","weights.bin","source.json")
        run(`cp --reflink=auto -a $(joinpath(cache_directory,file)) $(joinpath(destination,"cache"))`)
    end
    for file in ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE")
        cp(joinpath(checkpoint,file),joinpath(destination,file))
    end
    records=Any[];seen=Set{String}();native=JSON3.read(read(joinpath(checkpoint,"model.safetensors.index.json"),String))
    payload=joinpath(destination,"residual.f32")
    open(payload,"w") do io
        for file in sort!(unique(String.(collect(values(native.weight_map)))))
            tensors=DNNKernels.readsafetensors(joinpath(checkpoint,file);mmap=true)
            for key in sort!(collect(keys(tensors)))
                startswith(key,"model.language_model.") || continue
                name=replace(key,"model.language_model."=>"";count=1);raw=tensors[key]
                name in seen && error("duplicate weight");push!(seen,name)
                if ndims(raw)==2 && name!="embed_tokens.weight"
                    k,m=size(raw)
                    if m>=128 && k%256==0
                        entry=index.entries[Symbol(name)]
                        entry.kind=="q4k" && Tuple(Int.(entry.shape))==(m,k) || error("missing qualified cached weight")
                        push!(records,(;name,kind="cache",shape=[m,k]));continue
                    end
                end
                # Exactly the existing native loader's conversion policy.
                x=eltype(raw)===UInt16 ? reinterpret.(Float32,UInt32.(raw).<<16) : Float32.(raw)
                kind="buffer";m=0;k=0
                if ndims(x)==2 && name!="embed_tokens.weight"
                    k,m=size(x);kind="dense"
                elseif endswith(name,"input_layernorm.weight") || endswith(name,"post_attention_layernorm.weight") ||
                        name=="norm.weight" || endswith(name,"q_norm.weight") || endswith(name,"k_norm.weight")
                    x .+= 1f0
                elseif endswith(name,"A_log")
                    x=.-exp.(x)
                end
                offset=position(io);write(io,x)
                push!(records,(;name,kind,dims=collect(size(x)),offset,bytes=sizeof(x),m,k,
                    sha256=bytes2hex(sha256(reinterpret(UInt8,vec(x))))))
            end
        end
    end
    head=index.entries.lm_head;head.kind=="q6k" || error("qualified tied head missing")
    push!(records,(;name="lm_head",kind="cache",shape=Int.(head.shape)))
    Set(String(r.name) for r in records if r.kind=="cache")==Set(String.(keys(index.entries))) || error("cache coverage mismatch")
    files=Dict(file=>bytes2hex(open(sha256,joinpath(destination,file))) for file in
        ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE","residual.f32",
            "cache/index.json","cache/weights.bin","cache/source.json"))
    report=(;version=1,scope="exact qualified Q4/Q6 arrays plus native-loader Float32 residuals; export is NOT hardware qualification",
        endian_bom=Int(Base.ENDIAN_BOM),source_fingerprint=source.fingerprint,
        source_descriptor=source.descriptor,records,files,
        loader_sha256=bytes2hex(open(sha256,@__FILE__)),
        native_loader_sha256=bytes2hex(open(sha256,joinpath(@__DIR__,"qwen35.jl"))))
    open(io->JSON3.pretty(io,report),joinpath(destination,"portable.json"),"w");report
end

function validate_portable_qwen35(directory)
    p=JSON3.read(read(joinpath(directory,"portable.json"),String))
    p.version==1 && p.endian_bom==Int(Base.ENDIAN_BOM) || error("portable format/endian mismatch")
    p.loader_sha256==bytes2hex(open(sha256,@__FILE__)) || error("portable loader drift")
    p.native_loader_sha256==bytes2hex(open(sha256,joinpath(@__DIR__,"qwen35.jl"))) || error("native loader drift")
    length(p.records)<=2048 || error("weight count")
    required=Set(("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE","residual.f32",
        "cache/index.json","cache/weights.bin","cache/source.json"))
    Set(String.(keys(p.files)))==required || error("incomplete portable file manifest")
    names=String[];cursor=0
    for (file,hash) in pairs(p.files)
        String(file) in ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","LICENSE","residual.f32",
            "cache/index.json","cache/weights.bin","cache/source.json") || error("unexpected file")
        path=joinpath(directory,String(file));islink(path) && error("portable symlink")
        bytes2hex(open(sha256,path))==hash || error("portable checksum: $file")
    end
    for record in p.records
        push!(names,String(record.name))
        if record.kind!="cache"
            record.kind in ("buffer","dense") || error("weight kind")
            all(d->d isa Integer && !(d isa Bool) && 0<d<=1000000,record.dims) || error("weight dimensions")
            bytes=4*prod(Int128.(record.dims))
            0<bytes<=4*Int128(1024)^3 && bytes==record.bytes && record.offset==cursor || error("residual extent")
            record.kind=="dense" && (length(record.dims)!=2 || Tuple(Int.(record.dims))!=(record.k,record.m)) && error("dense shape")
            cursor+=Int(bytes)
        end
    end
    length(unique(names))==length(names) && cursor==filesize(joinpath(directory,"residual.f32")) || error("weight coverage")
    index=JSON3.read(read(joinpath(directory,"cache/index.json"),String))
    index.version==DNNKernels.QUANTCACHE_VERSION && index.scheme=="qwen35_native_v1" || error("cache format; no constructor invoked")
    Set(String(r.name) for r in p.records if r.kind=="cache")==Set(String.(keys(index.entries))) || error("cache coverage")
    p
end

function load_portable_qwen35(Q,directory,device)
    p=validate_portable_qwen35(directory)
    occursin("8060S",string(Mantle.devicename(device))) && Mantle.caps(device).subgroup==p.source_descriptor.subgroup ||
        error("first portable gate is restricted to matching8060S/subgroup; other hardware not qualified")
    for file in ("q4k.jl","q6k.jl")
        bytes2hex(open(sha256,joinpath(dirname(pathof(DNNKernels)),file)))==p.source_descriptor.kernels[Symbol(file)] || error("quantized kernel drift")
    end
    c=JSON3.read(read(joinpath(directory,"config.json"),String)).text_config
    model=Q.Model(device,c,Dict{String,Any}(),Any[],:q4k)
    cache=DNNKernels.QuantCache(joinpath(directory,"cache"),:qwen35_native_v1)
    try
        open(joinpath(directory,"residual.f32"),"r") do io
            for record in p.records
                name=String(record.name)
                if record.kind=="cache"
                    model.weights[name]=Q.QwenWeightCache.load_owned!(device,cache,name,model.owned)
                else
                    seek(io,Int(record.offset));x=Array{Float32}(undef,Tuple(Int.(record.dims)));read!(io,x)
                    bytes2hex(sha256(reinterpret(UInt8,vec(x))))==record.sha256 || error("residual tensor checksum")
                    b=Mantle.Buffer(device,x);push!(model.owned,b)
                    model.weights[name]=record.kind=="dense" ? Q.DenseWeight(b,Int(record.m),Int(record.k)) : b
                end
            end
        end
        model
    catch
        close(model);rethrow()
    finally
        DNNKernels.closecache!(cache)
    end
end
