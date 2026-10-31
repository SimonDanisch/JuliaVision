# Decode attention at long positions, pass by pass: the split-K attention and the
# combine, each recorded alone and timed against the bytes it has to read.
isdefined(@__MODULE__,:Qwen36MoEExperiment) || include("qwen36_moe_model.jl")
using Statistics,Random,Printf

"""Median seconds of `reps` runs of a plan holding just `declare(g)`."""
function timepass(device,declare;reps=20,warm=3)
    g=Mantle.Graph(device);declare(g)
    plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
    try
        for _ in 1:warm;Mantle.run!(plan);Mantle.waitidle(device);end
        median([(t=time_ns();Mantle.run!(plan);Mantle.waitidle(device);(time_ns()-t)/1e9) for _ in 1:reps])
    finally
        Mantle.free!(plan)
    end
end

function long_attention_passes(device;position=65535,kvtype=Float32,nkv=2,nqh=16,mode=:grouped)
    Q=Qwen36MoEExperiment.Q;BR=Q.BR
    capacity=position+1;chunks=cld(capacity,64);rng=MersenneTwister(1)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        kc=up(kvtype.(randn(rng,Float32,256,nkv,capacity)./16));vc=up(kvtype.(randn(rng,Float32,256,nkv,capacity)))
        q=up(randn(rng,Float32,256nqh));gate=up(ones(Float32,256nqh));out=up(zeros(Float32,256nqh))
        pm=up(zeros(Float32,nqh*chunks));ps=up(zeros(Float32,nqh*chunks));pa=up(zeros(Float32,nqh*chunks*256))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        attention=timepass(device,g->Q.declare_attention!(g,mode,pm,ps,pa,q,kc,vc,pos,nkv,nqh,chunks;name="attention"))
        combine=timepass(device,g->Q.dispatch(g,BR.attention_combine_kernel!,(out,pm,ps,pa,gate,pos,Int32(chunks)),nqh*256;name="combine"))
        kvbytes=2*nkv*256*sizeof(kvtype)*capacity
        partial=nqh*chunks*(256+2)*4
        (;position,kvtype,attention_ms=1000attention,combine_ms=1000combine,kv_mib=kvbytes/2^20,
         attention_gbs=(kvbytes+partial)/attention/1e9,combine_gbs=partial/combine/1e9)
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end
