# Flash-decoding attention against a Float64 reference and the grouped kernel:
# single positions, the edges of a split step (31, 32) and long caches, for
# Float32 and Float16 caches. Same assertion on every backend.
isdefined(@__MODULE__,:Qwen36MoEExperiment) || include("qwen36_moe_model.jl")
using Test,Random

function attention_reference(q,kc,vc,gate,position,nkv,nqh)
    R=nqh÷nkv;out=zeros(Float64,256nqh)
    for h in 0:nqh-1
        kh=h÷R;qh=Float64.(q[h*256+1:h*256+256])
        s=[sum(qh.*Float64.(kc[:,kh+1,p+1]))/16 for p in 0:position]
        w=exp.(s.-maximum(s));w./=sum(w)
        o=sum(w[p+1].*Float64.(vc[:,kh+1,p+1]) for p in 0:position)
        out[h*256+1:h*256+256].=o.*Float64.(gate[h*256+1:h*256+256])
    end
    out
end

function attention_device(device,mode,q,kc,vc,gate,position,nkv,nqh)
    Q=Qwen36MoEExperiment.Q;capacity=size(kc,3);slots=Q.attention_slots(mode,capacity)
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b);plan=nothing
    try
        bq,bk,bv,bg=up(q),up(kc),up(vc),up(gate);out=up(zeros(Float32,256nqh))
        pm=up(zeros(Float32,nqh*slots));ps=up(zeros(Float32,nqh*slots));pa=up(zeros(Float32,nqh*slots*256))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        g=Mantle.Graph(device)
        Q.declare_attention!(g,mode,pm,ps,pa,bq,bk,bv,pos,nkv,nqh,slots;name="attention")
        Q.declare_attention_combine!(g,mode,out,pm,ps,pa,bg,pos,slots,nqh;name="combine")
        plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
        Float64.(Array(out))
    finally
        plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
    end
end

function test_flash_decode(device;nkv=2,nqh=16)
    rng=MersenneTwister(7)
    @testset "flash-decoding attention" begin
        for T in (Float32,Float16),(capacity,positions) in ((128,(0,31,32,100)),(4096,(4095,)),(65536,(65535,)))
            kc=T.(randn(rng,Float32,256,nkv,capacity));vc=T.(randn(rng,Float32,256,nkv,capacity))
            q=randn(rng,Float32,256nqh);gate=rand(rng,Float32,256nqh)
            for position in positions
                want=attention_reference(q,kc,vc,gate,position,nkv,nqh)
                flash=attention_device(device,:flash,q,kc,vc,gate,position,nkv,nqh)
                grouped=attention_device(device,:grouped,q,kc,vc,gate,position,nkv,nqh)
                @test maximum(abs.(flash.-want))<=2f-5*max(1,maximum(abs.(want)))
                @test maximum(abs.(flash.-grouped))<=2f-5*max(1,maximum(abs.(want)))
            end
        end
    end
end

# q8_0 as the device stores it: four int8 per UInt32 word, one scale per 32.
function q8rows(x::AbstractArray{Float32,3})
    nd,nk,np=size(x);words=zeros(UInt32,64,nk,np);scales=zeros(Float32,8,nk,np);deq=similar(x)
    for p in 1:np,h in 1:nk,b in 1:8
        blk=x[32(b-1)+1:32b,h,p];s=maximum(abs.(blk))/127f0;scales[b,h,p]=s
        q=s==0f0 ? zeros(Int32,32) : Int32.(clamp.(round.(blk./s),-127f0,127f0))
        deq[32(b-1)+1:32b,h,p]=Float32.(q).*s
        for w in 1:8
            words[8(b-1)+w,h,p]=reduce(|,[(reinterpret(UInt32,q[4(w-1)+i])&0xff)<<(8(i-1)) for i in 1:4])
        end
    end
    words,scales,deq
end

# q4_0 as the device stores it: llama.cpp's quantize_row_q4_0_ref per block of
# 32, eight nibbles per UInt32 word, the scale in Float32.
function q4rows(x::AbstractArray{Float32,3})
    nd,nk,np=size(x);words=zeros(UInt32,32,nk,np);scales=zeros(Float32,8,nk,np);deq=similar(x)
    for p in 1:np,h in 1:nk,b in 1:8
        blk=x[32(b-1)+1:32b,h,p]
        m=0f0
        for v in blk;abs(v)>abs(m) && (m=v);end
        d=m/-8f0;id=d==0f0 ? 0f0 : 1f0/d
        q=[min(Int32(15),unsafe_trunc(Int32,v*id+8.5f0)) for v in blk]
        scales[b,h,p]=d;deq[32(b-1)+1:32b,h,p]=Float32.(q.-Int32(8)).*d
        for w in 1:4
            words[4(b-1)+w,h,p]=reduce(|,[UInt32(q[8(w-1)+i])<<(4(i-1)) for i in 1:8])
        end
    end
    words,scales,deq
end

# The stored integers of a quantized cache's words: q8_0 as signed bytes, q4_0
# as the nibbles 0..15.
storedints(::typeof(q8rows),words)=Int.(reinterpret(Int8,words))
storedints(::typeof(q4rows),words)=[Int((w>>(4(i-1)))&0xf) for i in 1:8,w in words]

function attention_device_quant(device,T,q,words,kscales,vwords,vscales,gate,position,nkv,nqh)
    Q=Qwen36MoEExperiment.Q;slots=Q.attention_slots(:flash,size(words,3))
    owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b);plan=nothing
    try
        wrap=T===Int8 ? Q.Q8KV : Q.Q4KV
        kc=wrap(up(words),up(kscales));vc=wrap(up(vwords),up(vscales))
        bq,bg=up(q),up(gate);out=up(zeros(Float32,256nqh))
        pm=up(zeros(Float32,nqh*slots));ps=up(zeros(Float32,nqh*slots));pa=up(zeros(Float32,nqh*slots*256))
        pos=Mantle.GPURef(device,Int32(position));push!(owned,pos)
        g=Mantle.Graph(device)
        Q.declare_attention!(g,:flash,pm,ps,pa,bq,kc,vc,pos,nkv,nqh,slots;name="attention")
        Q.declare_attention_combine!(g,:flash,out,pm,ps,pa,bg,pos,slots,nqh;name="combine")
        plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
        Float64.(Array(out))
    finally
        plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
    end
end

function test_flash_decode_quant(device;nkv=2,nqh=16)
    Q=Qwen36MoEExperiment.Q;rng=MersenneTwister(8)
    for (T,rows,name) in ((Int8,q8rows,"q8_0"),(Q.Q4_0,q4rows,"q4_0"))
        @testset "$name KV cache" begin
            # The writer: within half a step of the floating writer's values, a
            # whole step where q4_0 clamps the largest value of the other sign.
            capacity=8;theta=1f7;eps=1f-6
            norm=1f0.+0.1f0.*randn(rng,Float32,256)
            owned=Any[];up(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b);plan=nothing
            try
                kf=up(zeros(Float32,256,nkv,capacity));vf=up(zeros(Float32,256,nkv,capacity))
                alloc(E,dims)=up(zeros(E,dims))
                kq=Q.kvcache(alloc,T,nkv,capacity);vq=Q.kvcache(alloc,T,nkv,capacity)
                kx=up(randn(rng,Float32,256nkv));vx=up(randn(rng,Float32,256nkv));nb=up(norm)
                pos=Mantle.GPURef(device,Int32(0));push!(owned,pos)
                g=Mantle.Graph(device)
                Q.declare_prepare_kv!(g,kf,vf,kx,vx,nb,pos,nkv,theta,eps;name="float")
                Q.declare_prepare_kv!(g,kq,vq,kx,vx,nb,pos,nkv,theta,eps;name="quant")
                plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
                for p in (0,3,7)
                    pos[]=Int32(p);Mantle.run!(plan)
                end
                Mantle.waitidle(device)
                for (fl,qc) in ((kf,kq),(vf,vq))
                    ref=Array(fl);_,hs,_=rows(ref)
                    ints=reshape(storedints(rows,Array(qc.q)),256,nkv,capacity);sc=Array(qc.scales)
                    deq=T===Int8 ? Float32.(ints).*repeat(sc,inner=(32,1,1)) : Float32.(ints.-8).*repeat(sc,inner=(32,1,1))
                    step=abs.(repeat(sc,inner=(32,1,1)))
                    bound=T===Int8 ? 0.5f0.*step : ifelse.(ints.==15,1f0,0.5f0).*step
                    for p in (1,4,8)
                        @test all(abs.(deq[:,:,p].-ref[:,:,p]).<=bound[:,:,p].+1f-6)
                        @test maximum(abs.(ref[:,:,p]))>0
                        # The device's scales are the host quantizer's of the same values.
                        @test maximum(abs.(sc[:,:,p].-hs[:,:,p]))<=1f-6*maximum(abs.(hs[:,:,p]))
                    end
                end
            finally
                plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,reverse(owned))
            end
            # The reader: flash attention over the cache is attention over its values.
            for (capacity,positions) in ((128,(0,31,100)),(4096,(4095,)))
                k=randn(rng,Float32,256,nkv,capacity);v=randn(rng,Float32,256,nkv,capacity)
                kw,ks,kd=rows(k);vw,vs,vd=rows(v)
                q=randn(rng,Float32,256nqh);gate=rand(rng,Float32,256nqh)
                for position in positions
                    want=attention_reference(q,kd,vd,gate,position,nkv,nqh)
                    got=attention_device_quant(device,T,q,kw,ks,vw,vs,gate,position,nkv,nqh)
                    @test maximum(abs.(got.-want))<=2f-5*max(1,maximum(abs.(want)))
                end
            end
        end
    end
end
