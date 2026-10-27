module QwenWeightCache
using SHA, JSON3
import DNNKernels, Mantle, KernelAbstractions
const DK=DNNKernels

function open_cache(checkpoint,base,device)
    index=JSON3.read(read(joinpath(checkpoint,"model.safetensors.index.json"),String))
    files=sort!(unique(vcat(["config.json","model.safetensors.index.json"],String.(collect(values(index.weight_map))))))
    all(f->basename(f)==f,files) || throw(ArgumentError("checkpoint paths must be local basenames"))
    sources=Dict(f=>open(sha256,joinpath(checkpoint,f))|>bytes2hex for f in files)
    kernels=Dict(f=>bytes2hex(sha256(read(joinpath(dirname(pathof(DK)),f)))) for f in ("q4k.jl","q6k.jl"))
    # A stable, explicitly versioned native HF transpose/quantization policy.
    # Kernel source and actual checkpoint bytes, not merely shapes or mtimes.
    descriptor=(;version=1,policy="HF(k,m)->DK(m,k);Q4_K eligible dense;Q6_K tied head",
        cache_format=DK.QUANTCACHE_VERSION,sources,kernels,
        device=string(Mantle.devicename(device)),subgroup=Mantle.caps(device).subgroup)
    canonical=JSON3.write((;descriptor.version,descriptor.policy,descriptor.cache_format,
        sources=sort!(collect(sources);by=first),kernels=sort!(collect(kernels);by=first),descriptor.device,descriptor.subgroup))
    fingerprint=bytes2hex(sha256(canonical));directory=joinpath(abspath(base),fingerprint)
    marker=joinpath(directory,"source.json")
    if isdir(directory)&&!isfile(marker)&&!isempty(readdir(directory))
        throw(ArgumentError("refusing nonempty unowned cache directory"))
    end
    if isfile(marker)
        JSON3.read(read(marker,String)).fingerprint==fingerprint || throw(ArgumentError("cache source marker mismatch"))
    end
    cacheindex=joinpath(directory,"index.json")
    if isfile(cacheindex)
        saved=JSON3.read(read(cacheindex,String))
        saved.version==DK.QUANTCACHE_VERSION && saved.scheme=="qwen35_native_v1" ||
            throw(ArgumentError("cache format mismatch; refusing automatic overwrite"))
    end
    mkpath(directory)
    isfile(marker) || open(io->JSON3.pretty(io,(;fingerprint,descriptor)),marker,"w")
    (;cache=DK.QuantCache(directory,:qwen35_native_v1),directory,fingerprint)
end

"""Adopt existing cache arrays into this runner's explicit Buffer ownership.

The existing cache returns backend arrays, whereas Qwen sessions own Buffers.
Use a conservative exact-byte round trip; no quantization is repeated and no
new storage format or separate CPU quantizer is introduced.
"""
function load_owned!(device,cache,key,owned)
    weight=DK.cacheload(device,cache,key)
    original=DK.cachearrays(weight)
    try
        arrays=map(original) do array
            buffer=Mantle.Buffer(device,Array(array));push!(owned,buffer);buffer
        end
        DK.fromcache(Val(Symbol(DK.cachekind(weight))),arrays,size(weight))
    finally
        foreach(KernelAbstractions.unsafe_free!,original)
    end
end
end
