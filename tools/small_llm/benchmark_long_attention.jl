# Decode attention at long positions, pass by pass: the split-K attention and the
# combine, each recorded alone and timed against the bytes it has to read.
isdefined(@__MODULE__,:Qwen36MoEExperiment) || include("qwen36_moe_model.jl")
using Statistics,Random,Printf

"""Minimum seconds over `reps` runs of a plan holding just `declare(g)`. The
minimum, not the median: the GPU is shared with other work (a render farm),
which stalls single runs by hundreds of milliseconds, and the minimum is the
kernel's time when it has the device."""
function timepass(device,declare;reps=40,warm=3)
    g=Mantle.Graph(device);declare(g)
    plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
    try
        for _ in 1:warm;Mantle.run!(plan);Mantle.waitidle(device);end
        minimum([(t=time_ns();Mantle.run!(plan);Mantle.waitidle(device);(time_ns()-t)/1e9) for _ in 1:reps])
    finally
        Mantle.free!(plan)
    end
end

function long_attention_passes(device;position=65535,kvtype=Float32,nkv=2,nqh=16,mode=:grouped,slots=nothing)
    Q=Qwen36MoEExperiment.Q;BR=Q.BR
    capacity=position+1;chunks=slots===nothing ? Q.attention_slots(mode,capacity) : slots;rng=MersenneTwister(1)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        kc=up(kvtype.(randn(rng,Float32,256,nkv,capacity)./16));vc=up(kvtype.(randn(rng,Float32,256,nkv,capacity)))
        q=up(randn(rng,Float32,256nqh));gate=up(ones(Float32,256nqh));out=up(zeros(Float32,256nqh))
        pm=up(zeros(Float32,nqh*chunks));ps=up(zeros(Float32,nqh*chunks));pa=up(zeros(Float32,nqh*chunks*256))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        attention=timepass(device,g->Q.declare_attention!(g,mode,pm,ps,pa,q,kc,vc,pos,nkv,nqh,chunks;name="attention"))
        combine=timepass(device,g->Q.declare_attention_combine!(g,mode,out,pm,ps,pa,gate,pos,chunks,nqh;name="combine"))
        kvbytes=2*nkv*256*sizeof(kvtype)*capacity
        partial=nqh*chunks*(256+2)*4
        (;position,kvtype,mode,slots=chunks,attention_ms=1000attention,combine_ms=1000combine,kv_mib=kvbytes/2^20,
         attention_gbs=(kvbytes+partial)/attention/1e9,combine_gbs=partial/combine/1e9)
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end

"""The chunk-wide prefill attention for `n` tokens starting at `position`, against
the bytes of the cache it reads once."""
function long_prefill_attention(device;position=65536,n=256,kvtype=Float16,nkv=2,nqh=16,np=2)
    Q=Qwen36MoEExperiment.Q;capacity=position+n;rng=MersenneTwister(1)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        cache()=kvtype===Int8 ? Q.Q8KV(up(rand(rng,UInt32,64,nkv,capacity)),up(fill(1f-2,8,nkv,capacity))) :
            up(kvtype.(randn(rng,Float32,256,nkv,capacity)./16))
        kc,vc=cache(),cache()
        q=up(randn(rng,Float32,256nqh,n));gate=up(ones(Float32,256nqh,n));out=up(zeros(Float32,256nqh,n))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        t=timepass(device,g->Q.declare_prefill_attention!(g,out,q,gate,kc,vc,pos,nkv,nqh,n;np,name="prefill_attention");reps=10)
        rowbytes=kvtype===Int8 ? 288 : 256sizeof(kvtype)
        kvbytes=2*nkv*rowbytes*(position+n)
        flops=4.0*nqh*256*n*(position+n/2)
        (;position,n,kvtype,np,ms=1000t,per_token_us=1e6t/n,kv_gbs=kvbytes/t/1e9,tflops=flops/t/1e12)
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end

"""Minimum milliseconds of each named variant of a pass, interleaved: `rounds`
rounds, every variant once a round in a rotated order, so a stall from other work
on the GPU costs every variant alike. `variants` maps a name to `declare(g)`."""
function interleaved_ms(device,variants;rounds=30)
    names=collect(keys(variants))
    plans=Dict(k=>begin g=Mantle.Graph(device);variants[k](g);Mantle.record!(Mantle.Plan(g;budget=Inf)) end for k in names)
    best=Dict(k=>Inf for k in names)
    try
        for k in names;Mantle.run!(plans[k]);Mantle.waitidle(device);end
        for r in 1:rounds,k in circshift(names,r)
            t=time_ns();Mantle.run!(plans[k]);Mantle.waitidle(device)
            best[k]=min(best[k],(time_ns()-t)/1e6)
        end
    finally
        foreach(Mantle.free!,values(plans))
    end
    best
end

"""The chunk attention's variants at one depth, interleaved: `(kvtype, np, qs)`."""
function prefill_attention_variants(device;position=96000,n=512,configs=((Int8,2,2),(Int8,2,4),(Float16,2,2),(Float16,2,4)),nkv=2,nqh=16,rounds=20,clock=interleaved_ms)
    Q=Qwen36MoEExperiment.Q;capacity=position+n;rng=MersenneTwister(1)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        caches=Dict{Any,Any}()
        for T in unique(first.(configs))
            caches[T]=T===Int8 ? ntuple(_->Q.Q8KV(up(rand(rng,UInt32,64,nkv,capacity)),up(fill(1f-2,8,nkv,capacity))),2) :
                ntuple(_->up(T.(randn(rng,Float32,256,nkv,capacity)./16)),2)
        end
        q=up(randn(rng,Float32,256nqh,n));gate=up(ones(Float32,256nqh,n));out=up(zeros(Float32,256nqh,n))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        variants=Dict(c=>(g->Q.declare_prefill_attention!(g,out,q,gate,caches[c[1]]...,pos,nkv,nqh,n;np=c[2],qs=c[3],vt=length(c)>3&&c[4],name="pa")) for c in configs)
        ms=clock(device,variants;rounds)
        flops=4.0*nqh*256*n*(position+n/2)
        Dict(k=>(ms=v,tflops=flops/(v/1e3)/1e12) for (k,v) in ms)
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end

"""Like `interleaved_ms`, but each variant's GPU timestamps around its passes
(median per pass, summed), which leave out the time a submission waits for the
GPU while other work holds it."""
function interleaved_gpu_ms(device,variants;rounds=10)
    names=collect(keys(variants))
    plans=Dict(k=>begin g=Mantle.Graph(device);variants[k](g);Mantle.record!(Mantle.Plan(g;budget=Inf,profile=true)) end for k in names)
    try
        for r in 1:rounds,k in circshift(names,r)
            Mantle.run!(plans[k]);Mantle.waitidle(device)
        end
        Dict(k=>sum(t.gpu_ms for t in Mantle.timings(plans[k]) if t.kind===:compute) for k in names)
    finally
        foreach(Mantle.free!,values(plans))
    end
end

"""Flash-decoding over Float16, q8_0 and q4_0 caches at one depth, by GPU
timestamps, interleaved: milliseconds and the cache bytes a millisecond reads."""
function decode_attention_formats(device;position=131071,nkv=2,nqh=16,rounds=10,formats=(Float16,Int8,Qwen36MoEExperiment.Q.Q4_0))
    Q=Qwen36MoEExperiment.Q;capacity=position+1;rng=MersenneTwister(1);slots=Q.FLASH_SPLITS
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        # Values for float caches and scales, random words for quantized ones.
        alloc(T,dims)=up(T<:AbstractFloat ? T.(randn(rng,Float32,dims)./16) : rand(rng,T,dims))
        caches=Dict(T=>(Q.kvcache(alloc,T,nkv,capacity),Q.kvcache(alloc,T,nkv,capacity)) for T in formats)
        q=up(randn(rng,Float32,256nqh));gate=up(ones(Float32,256nqh));out=up(zeros(Float32,256nqh))
        pm=up(zeros(Float32,nqh*slots));ps=up(zeros(Float32,nqh*slots));pa=up(zeros(Float32,nqh*slots*256))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        variants=Dict(T=>(g->(Q.declare_attention!(g,:flash,pm,ps,pa,q,caches[T]...,pos,nkv,nqh,slots;name="attention");
            Q.declare_attention_combine!(g,:flash,out,pm,ps,pa,gate,pos,slots,nqh;name="combine"))) for T in formats)
        ms=interleaved_gpu_ms(device,variants;rounds)
        rowbytes(T)=T===Int8 ? 288 : T===Q.Q4_0 ? 160 : 256sizeof(T)
        Dict(T=>(ms=ms[T],gbs=2*nkv*rowbytes(T)*capacity/(ms[T]/1e3)/1e9) for T in formats)
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end

"""The grouped expert products of one layer for a chunk of `n` tokens with
routing drawn like the model's (top-k of random logits), by GPU timestamps,
interleaved: columns a pass (`nc`) and rows a group (`rows`) per variant."""
function grouped_expert_variants(model;layer=0,n=512,configs=((8,2),(16,2),(16,1)),rounds=6)
    M=Qwen36MoEExperiment;c=model.config;dev=model.device;rng=MersenneTwister(2)
    topk=Int(c.num_experts_per_tok);ne=Int(c.num_experts);w=Int(c.hidden_size);f=Int(c.moe_intermediate_size)
    p="layers.$layer.mlp.";W=model.weights[p*"experts.gate_up_proj"];D=model.weights[p*"experts.down_proj"]
    owned=Any[];up(x)=(b=Mantle.Buffer(dev,x);push!(owned,b);b)
    try
        sel=reduce(hcat,[Int32.(partialsortperm(randn(rng,ne),1:topk;rev=true).-1) for _ in 1:n])
        selected=up(sel);status=up(zeros(Int32,n))
        x=up(randn(rng,Float32,w,n));a=up(randn(rng,Float32,f,topk,n))
        gu=up(zeros(Float32,2f,topk,n));routed=up(zeros(Float32,w,topk,n))
        variants=Dict((nc,rows)=>(g->begin
            order,offsets=M.declare_expert_buckets!(g,selected,status,ne;name="buckets")
            M.grouped_routed!(g,gu,W,x,order,offsets,selected,status;name="gate_up",nc,rows)
            M.grouped_routed!(g,routed,D,a,order,offsets,selected,status;per_expert=true,name="down",nc,rows)
        end) for (nc,rows) in configs)
        names=collect(keys(variants))
        plans=Dict(k=>begin g=Mantle.Graph(dev);variants[k](g);Mantle.record!(Mantle.Plan(g;budget=Inf,profile=true)) end for k in names)
        try
            for r in 1:rounds,k in circshift(names,r);Mantle.run!(plans[k]);Mantle.waitidle(dev);end
            Dict(k=>Dict(t.name=>t.gpu_ms for t in Mantle.timings(plans[k]) if t.kind===:compute) for k in names)
        finally
            foreach(Mantle.free!,values(plans))
        end
    finally
        foreach(Mantle.free!,reverse(owned))
    end
end
