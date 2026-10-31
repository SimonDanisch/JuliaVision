isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")
using JSON3, SHA

struct CheckpointSamplingHook{S}
    state::S
end
(hook::CheckpointSamplingHook)(graph,logits)=GPUSampling.declare!(graph,logits,hook.state)

const SAMPLING_CHECKPOINT_SOURCES=("sampling_checkpoint.jl","qwen35.jl","prefill_qwen35.jl","gpu_sampling.jl")
checkpoint_digest(path)=bytes2hex(open(sha256,path))
checkpoint_plain(x)=JSON3.read(JSON3.write(x),Dict{String,Any})

function checkpoint_resources(session,sampler)
    session.logit_hook isa CheckpointSamplingHook && session.logit_hook.state===sampler ||
        error("only the explicit sampler-only hook is checkpointable; memory/other hooks need their own state protocol")
    named=Pair{String,Any}["session.token"=>session.token,"session.position"=>session.posref,
        "session.logits"=>session.logits,"session.best"=>session.best,"session.predictions"=>session.predictions]
    for layer in sort(collect(keys(session.recurrent)))
        history,state=session.recurrent[layer]
        push!(named,"recurrent.$layer.history"=>history,"recurrent.$layer.state"=>state)
    end
    for layer in sort(collect(keys(session.kv)))
        key,value=session.kv[layer];push!(named,"kv.$layer.key"=>key,"kv.$layer.value"=>value)
    end
    Set(objectid(last(p)) for p in named)==Set(objectid.(session.owned)) || error("unaccounted session-owned state")
    length(named)==length(session.owned) || error("aliased session state")
    for field in (:active,:counter,:seen,:selected,:status)
        push!(named,"sampler.$field"=>getproperty(sampler,field))
    end
    all(p->last(p).dev===session.model.device,named) || error("mixed checkpoint devices")
    named
end

checkpoint_shape(resource)=resource isa Mantle.GPURef ? (1,) : size(Mantle.storage(resource))
checkpoint_array(resource)=resource isa Mantle.GPURef ? Array(Mantle.storage(resource)) : Array(resource)
checkpoint_sampling_options(s)=(;s.vocabulary,s.topk,s.temperature,s.top_p,s.presence,seed=Int(s.seed),s.fallback)
checkpoint_sources()=Dict(f=>checkpoint_digest(joinpath(@__DIR__,f)) for f in SAMPLING_CHECKPOINT_SOURCES)

function checkpoint_scalar_invariants(position,vocabulary,scalars)
    scalars["session.position"]==position || error("saved GPU/host position mismatch")
    token=scalars["session.token"]
    0<=token<vocabulary && token==scalars["session.best"]==scalars["sampler.selected"] || error("saved token feedback mismatch")
    scalars["sampler.active"]==1 && scalars["sampler.status"]==0 || error("checkpoint must be in valid active sampling mode")
    0<scalars["sampler.counter"]<=position || error("saved RNG counter bounds")
    nothing
end

"""Cold-resume state, not executable plans or model weights.

Call only between completed decode batches, with exclusive ownership of the
session and coordinated device. `identity` MUST name the caller's independently
verified checkpoint/tokenizer/cache identity, not merely the model dimensions.
Version1 stores whole KV allocations (including unused capacity); it is not a
compact or production checkpoint format. No Julia Serialization is loaded.
"""
function save_sampling_checkpoint(session,sampler,directory;identity::AbstractString)
    isempty(identity) && error("verified model/tokenizer identity required")
    ispath(directory) && error("preserve existing/partial checkpoint")
    named=checkpoint_resources(session,sampler)
    0<session.position<=session.capacity || error("checkpoint requires ingested state")
    Mantle.waitidle(session.model.device)
    gpu_position=only(checkpoint_array(session.posref))
    gpu_position==session.position || error("host/device position disagreement")
    scalar_names=Set(("session.position","session.token","session.best","sampler.selected","sampler.active","sampler.status","sampler.counter"))
    checkpoint_scalar_invariants(session.position,sampler.vocabulary,
        Dict(name=>only(checkpoint_array(resource)) for (name,resource) in named if name in scalar_names))
    mkpath(directory);rows=Any[]
    for (i,(name,resource)) in enumerate(named)
        resource isa Mantle.GPURef && (@atomic resource.dirty) && error("pending host ref store is not checkpointed")
        data=checkpoint_array(resource)
        eltype(data) in (Float32,Float16,Int32,UInt32) || error("unsupported raw checkpoint dtype")
        filename=lpad(string(i),3,'0')*".bin";path=joinpath(directory,filename)
        open(io->write(io,data),path,"w")
        push!(rows,(;name,file=filename,type=string(eltype(data)),shape=collect(size(data)),
            bytes=filesize(path),sha256=checkpoint_digest(path)))
    end
    report=(;complete=true,version=1,identity=String(identity),capacity=session.capacity,position=session.position,
        attention=String(session.attention),config=checkpoint_plain(session.model.config),
        sampler=checkpoint_sampling_options(sampler),endian_bom=Int(Base.ENDIAN_BOM),
        resources=rows,source_sha256=checkpoint_sources(),runtime_provenance=inference_runtime_provenance())
    # The manifest is committed LAST; an interrupted directory is never accepted.
    open(io->JSON3.pretty(io,report),joinpath(directory,"manifest.json"),"w");report
end

function restore_sampling_checkpoint!(session,sampler,directory;identity::AbstractString)
    named=checkpoint_resources(session,sampler)
    session.position==0 || error("restore only into a freshly constructed, unused session")
    manifest=joinpath(directory,"manifest.json")
    islink(manifest) && error("manifest symlink")
    filesize(manifest)<=1048576 || error("oversized manifest")
    p=JSON3.read(read(manifest,String))
    p.complete && p.version==1 && p.identity==identity || error("checkpoint version/identity")
    p.endian_bom==Int(Base.ENDIAN_BOM) && p.capacity==session.capacity && 0<p.position<=session.capacity || error("checkpoint capacity/position/endian")
    p.attention==String(session.attention) && checkpoint_plain(p.config)==checkpoint_plain(session.model.config) || error("model configuration mismatch")
    checkpoint_plain(p.sampler)==checkpoint_plain(checkpoint_sampling_options(sampler)) || error("sampler parameters mismatch")
    checkpoint_plain(p.source_sha256)==checkpoint_sources() || error("checkpoint source drift")
    checkpoint_plain(p.runtime_provenance)==checkpoint_plain(inference_runtime_provenance()) || error("checkpoint runtime drift")
    length(p.resources)==length(named) || error("resource count mismatch")
    # Validate ALL metadata/checksums before writing any target resource.
    scalars=Dict{String,Any}()
    scalar_names=Set(("session.position","session.token","session.best","sampler.selected","sampler.active","sampler.status","sampler.counter"))
    for (i,((name,resource),row)) in enumerate(zip(named,p.resources))
        row.name==name && row.file==lpad(string(i),3,'0')*".bin" || error("resource identity/order")
        row.type==string(eltype(resource)) && Tuple(row.shape)==checkpoint_shape(resource) || error("resource dtype/shape")
        path=joinpath(directory,String(row.file));islink(path) && error("raw resource symlink")
        expected=sizeof(eltype(resource))*prod(checkpoint_shape(resource))
        row.bytes==expected==filesize(path) && row.sha256==checkpoint_digest(path) || error("resource size/checksum")
        if name in scalar_names
            value=Vector{eltype(resource)}(undef,1);read!(path,value);scalars[name]=only(value)
        end
    end
    checkpoint_scalar_invariants(Int(p.position),sampler.vocabulary,scalars)
    for ((_,resource),row) in zip(named,p.resources)
        data=Array{eltype(resource)}(undef,checkpoint_shape(resource))
        read!(joinpath(directory,String(row.file)),data)
        if resource isa Mantle.GPURef
            # Queue the ACTUAL device value, not the stale host pending payload.
            resource[]=only(data)
        else
            copyto!(resource,data)
        end
    end
    session.position=Int(p.position)
    session
end
