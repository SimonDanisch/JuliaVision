# Included in Qwen35Experiment after grouped_attention_qwen35.jl. Flash-decoding
# for one query token: a fixed number of splits per KV head, each covering an
# equal share of the positions up to the current one, so the work stays spread
# over the device at 4k and at 256k alike, and the combine reads FLASH_SPLITS
# partials per head instead of one per 64 positions.
#
# A wave takes SG/R positions at a time for all R query heads of its KV head:
# every lane forms partial dot products over its 256/SG dimensions, and one
# transposed butterfly over the subgroup reduces all SG of them at once, about
# one shuffle per dot product, leaving score `lane` in lane `lane`. The grouped
# kernel reduced each head's dot product on its own (six shuffles each) and ran
# its softmax across the workgroup with barriers; at 64k positions it read the
# cache at half the rate a copy reaches. Each wave keeps an online softmax, and
# the waves of a workgroup merge only once, at the end.
const FLASH_SPLITS=128
const FLASH_STEP=32          # span granularity: a multiple of a workgroup's step (16 positions for R=8)

"""Positions per split for `n` positions: an equal share, rounded up to a step."""
@inline flash_span(n::Int32,splits::Int32)=cld(cld(n,splits),Int32(FLASH_STEP))*Int32(FLASH_STEP)

# Lane `lane` ends with the subgroup-wide sum of value `lane+1`: a butterfly
# that halves the values at every step, written out as straight-line code. A
# recursive version shrank the tuple at every level, which inference widens to
# `Any` and the device compiler cannot take.
@generated function transposed_sum(vals::NTuple{N,Float32},lane::Int32) where N
    ex=Expr(:block);cur=[Symbol(:v0_,i) for i in 1:N]
    for i in 1:N;push!(ex.args,:($(cur[i])=vals[$i]));end
    n=N;step=0
    while n>1
        h=n÷2;step+=1
        up=Symbol(:up,step);partner=Symbol(:partner,step)
        push!(ex.args,:($up=(lane&Int32($h))!=Int32(0)))
        push!(ex.args,:($partner=(lane⊻Int32($h))+Int32(1)))
        nxt=[Symbol(:v,step,:_,i) for i in 1:h]
        for i in 1:h
            push!(ex.args,:($(nxt[i])=ifelse($up,$(cur[i+h]),$(cur[i]))+KI.shfl(ifelse($up,$(cur[i]),$(cur[i+h])),$partner)))
        end
        cur=nxt;n=h
    end
    push!(ex.args,cur[1])
    ex
end

# Max and sum over the lanes that hold the same head within a block of N
# values (lanes R apart, below N).
@inline function headmax(x::Float32,lane::Int32,::Val{R},::Val{N}) where {R,N}
    s=Int32(R)
    while s<Int32(N)
        x=max(x,KI.shfl(x,(lane⊻s)+Int32(1)));s<<=1
    end
    x
end
@inline function headsum(x::Float32,lane::Int32,::Val{R},::Val{N}) where {R,N}
    s=Int32(R)
    while s<Int32(N)
        x+=KI.shfl(x,(lane⊻s)+Int32(1));s<<=1
    end
    x
end

"""An 8-bit KV cache (q8_0): int8 values packed four to a UInt32 word, with
one Float32 scale per 32 values of a (position, KV head) row. Packing into
words keeps every access 32-bit, as BonsaiRunner's int8 cache does, so no
driver needs 8-bit storage. A row is 288 bytes against 512 in Float16."""
struct Q8KV{Q,S}
    q::Q          # UInt32 (64, nk, capacity)
    scales::S     # Float32 (8, nk, capacity)
end

"""A 4-bit KV cache (q4_0): eight values to a UInt32 word, value `i` of a word
at bits `4(i-1)`, with one Float32 scale per 32 values of a (position, KV head)
row. The quantization is llama.cpp's `quantize_row_q4_0_ref`: the block's value
of largest magnitude `xm` (the first of equals) gives `d = xm / -8`, a value is
stored as `min(15, trunc(x/d + 8.5))` and read as `(q - 8) d`. The scale stays
Float32 where llama.cpp rounds it to Float16. A row is 160 bytes."""
struct Q4KV{Q,S}
    q::Q          # UInt32 (32, nk, capacity)
    scales::S     # Float32 (8, nk, capacity)
end

"""The `kv_type` of a q4_0 cache, which has no 4-bit element type to name it."""
abstract type Q4_0 end

const QuantizedKV=Union{Q8KV,Q4KV}

# What a kernel reads a cache row as, passed to it as an instance.
struct FloatRows end
struct Q8Rows end
struct Q4Rows end
rowformat(c)=FloatRows()
rowformat(::Q8KV)=Q8Rows()
rowformat(::Q4KV)=Q4Rows()

"""A K or V cache of element type `T` for `nk` heads; `alloc(T, dims)` allocates."""
kvcache(alloc,::Type{T},nk,capacity) where T<:AbstractFloat=alloc(T,(256,nk,capacity))
kvcache(alloc,::Type{Int8},nk,capacity)=Q8KV(alloc(UInt32,(64,nk,capacity)),alloc(Float32,(8,nk,capacity)))
kvcache(alloc,::Type{Q4_0},nk,capacity)=Q4KV(alloc(UInt32,(32,nk,capacity)),alloc(Float32,(8,nk,capacity)))

# Sizes are type parameters of these helpers, so every tuple length and index
# is a compile-time constant: a closure that captured them as values made the
# kernel's `ntuple` calls dynamic, which no device compiler accepts.
@inline loadrow(c,row::Int32,::Val{E}) where E=ntuple(e->Float32(@inbounds c[row+Int32(e)]),Val(E))
# A lane's E values of row (p, kh): read as stored, or decoded from words and
# the row block's scale (E values never cross a 32-value block).
@inline cacherow(c,sc,p::Int32,nk::Int32,kh::Int32,d0::Int32,ve::Val{E},::FloatRows) where E=
    loadrow(c,(p*nk+kh)*Int32(256)+d0,ve)
@inline function cacherow(c,sc,p::Int32,nk::Int32,kh::Int32,d0::Int32,::Val{E},::Q8Rows) where E
    scale=@inbounds sc[(p*nk+kh)*Int32(8)+d0÷Int32(32)+Int32(1)]
    w0=(p*nk+kh)*Int32(64)+d0÷Int32(4)
    words=ntuple(i->@inbounds(c[w0+Int32(i)]),Val(E÷4))
    ntuple(e->scale*Float32(reinterpret(Int32,words[(e-1)÷4+1]<<UInt32(24-8*((e-1)%4)))>>Int32(24)),Val(E))
end
@inline function cacherow(c,sc,p::Int32,nk::Int32,kh::Int32,d0::Int32,::Val{E},::Q4Rows) where E
    row=p*nk+kh
    scale=@inbounds sc[row*Int32(8)+d0÷Int32(32)+Int32(1)]
    ntuple(Val(E)) do e
        i=d0+Int32(e-1)
        w=@inbounds c[row*Int32(32)+i÷Int32(8)+Int32(1)]
        scale*Float32(Int32((w>>(UInt32(4)*UInt32(i%Int32(8))))&0x0000000f)-Int32(8))
    end
end
@inline function loadrows(c,sc,base::Int32,last::Int32,nk::Int32,kh::Int32,d0::Int32,::Val{P},ve::Val{E},fmt) where {P,E}
    ntuple(pi->cacherow(c,sc,min(base+Int32(pi-1),last-Int32(1)),nk,kh,d0,ve,fmt),Val(P))
end

# A quantized row as the cache holds it (`fetchrow`: words and scale, two or
# three registers) and its values (`decoderow`), so the loop can have the next
# step's rows in flight while it computes this step's. At 128k positions (GPU
# time, interleaved) that took q4_0 from 0.83 to 0.74 ms and q8_0 from 0.88 to
# 0.87; both stay short of their bytes' time, bound by the arithmetic a
# position takes. Float rows are not fetched ahead: they are E registers each,
# and that kernel already reads at the bus's rate (1.3 ms, 204 GB/s).
pipelined(::FloatRows)=false
pipelined(::Union{Q8Rows,Q4Rows})=true
@inline function fetchrow(c,sc,p::Int32,nk::Int32,kh::Int32,d0::Int32,::Val{E},::Q8Rows) where E
    w0=(p*nk+kh)*Int32(64)+d0÷Int32(4)
    (ntuple(i->@inbounds(c[w0+Int32(i)]),Val(E÷4)),@inbounds(sc[(p*nk+kh)*Int32(8)+d0÷Int32(32)+Int32(1)]))
end
@inline decoderow(r,::Val{E},::Q8Rows) where E=
    ntuple(e->r[2]*Float32(reinterpret(Int32,r[1][(e-1)÷4+1]<<UInt32(24-8*((e-1)%4)))>>Int32(24)),Val(E))
# E values from d0 lie in cld(E, 8) words: d0 is a multiple of E, so for E = 4
# they are one half of a word and for E >= 8 whole words.
@inline function fetchrow(c,sc,p::Int32,nk::Int32,kh::Int32,d0::Int32,::Val{E},::Q4Rows) where E
    row=p*nk+kh
    (ntuple(i->@inbounds(c[row*Int32(32)+d0÷Int32(8)+Int32(i)]),Val(cld(E,8))),
     @inbounds(sc[row*Int32(8)+d0÷Int32(32)+Int32(1)]),d0%Int32(8))
end
@inline decoderow(r,::Val{E},::Q4Rows) where E=ntuple(Val(E)) do e
    i=r[3]+Int32(e-1)
    r[2]*Float32(Int32((r[1][i÷Int32(8)+Int32(1)]>>(UInt32(4)*UInt32(i%Int32(8))))&0x0000000f)-Int32(8))
end
@inline fetchrows(c,sc,base::Int32,last::Int32,nk::Int32,kh::Int32,d0::Int32,::Val{P},ve,fmt) where P=
    ntuple(pi->fetchrow(c,sc,min(base+Int32(pi-1),last-Int32(1)),nk,kh,d0,ve,fmt),Val(P))
@inline decoderows(raw,ve,fmt)=map(r->decoderow(r,ve,fmt),raw)
@inline function partdot(qr,ks,v::Int,::Val{R},::Val{E}) where {R,E}
    pi=(v-1)÷R+1;h=(v-1)%R+1
    sum(ntuple(e->qr[h][e]*ks[pi][e],Val(E)))
end
# Lengths are types; element indices are plain integers, which homogeneous
# tuples index stably and LLVM folds once the loops are unrolled. Each weight
# is fetched by shuffle when its V row is used: holding all of them took 64
# registers, and the kernel ran at the 256-register ceiling with 4 subgroups
# per SIMD, too few to hide the cache's latency.
@inline function accrow(acc,alphas,p::Float32,vs,j::Int,::Val{R},vp::Val{P},::Val{E}) where {R,P,E}
    ws=ntuple(pi->KI.shfl(p,Int32((pi-1)*R+j)),vp)
    ntuple(e->acc[j][e]*alphas[j]+sum(ntuple(pi->ws[pi]*vs[pi][e],vp)),Val(E))
end
# A closure that captured `acc`, which the loop reassigns, would box it; these
# helpers take it as an argument instead.
@inline nextacc(acc,alphas,p,vs,vr::Val{R},vp,ve) where R=ntuple(j->accrow(acc,alphas,p,vs,j,vr,vp,ve),vr)
# Wave `wave` adds its rescaled values to its KV head's accumulator at `off`;
# the maxima it rescales by are those of the waves u0:u0+nwh-1 of that head.
@inline function mergewave!(wacc,wstat,acc,wave::Int,d0::Int32,vr::Val{R},ve,u0::Int,nwh::Int,off::Int32) where R
    ntuple(vr) do j
        m=-Inf32
        for u in u0:u0+nwh-1;m=max(m,@inbounds wstat[u*R+j]);end
        f=m==-Inf32 ? 0f0 : exp(@inbounds(wstat[wave*R+j])-m)
        addwave!(wacc,acc,f,j,d0+off,ve)
    end
    nothing
end
@inline function addwave!(wacc,acc,f,j::Int,d0::Int32,::Val{E}) where E
    ntuple(Val(E)) do e
        @inbounds wacc[Int32((j-1)*256)+d0+Int32(e)]+=acc[j][e]*f
        nothing
    end
    nothing
end

# A workgroup serves G KV heads over its span, NW/G waves each. With G=nk it
# reads every position's whole cache row (all heads, side by side in memory)
# at once; one head per workgroup read half of each row, and the other half
# came from another workgroup at another time.
function attention_flash!(pm,ps,pa,q,kc,vc,ksc,vsc,position,nk::Int32,splits::Int32,vr::Val{R},vsg::Val{SG},::Val{G},kf,vf) where {R,SG,G}
    # A wave takes P positions for all R heads: N=P*R values, half a subgroup.
    ve=Val(256÷SG);vp=Val(SG÷R÷2);vn=Val(SG÷2);NW=256÷SG;P=SG÷R÷2;N=SG÷2;WH=NW÷G
    wacc=KI.localmemory(Float32,Val((256R*G,)),Val(1))
    wstat=KI.localmemory(Float32,Val((2*(256÷SG)*R,)),Val(2))
    t=Int32(KI.get_local_id().x)-Int32(1)
    wg=Int32(KI.get_group_id().x)-Int32(1)
    kgroup=wg÷splits;sp=wg%splits
    npos=Int32(position[1])+Int32(1)
    span=flash_span(npos,splits)
    first=sp*span;last=min(npos,first+span)
    first>=npos && return nothing          # uniform across the workgroup
    lane=t%Int32(SG);w=t÷Int32(SG);d0=lane*Int32(256÷SG)
    g=w÷Int32(WH);wi=w%Int32(WH);kh=kgroup*Int32(G)+g
    qr=ntuple(j->loadrow(q,(kh*Int32(R)+Int32(j-1))*Int32(256)+d0,ve),vr)
    mh=-Inf32;lh=0f0
    acc=ntuple(_->ntuple(_->0f0,ve),vr)
    base=first+wi*Int32(P)
    PIPE=pipelined(kf)&&pipelined(vf)
    if PIPE
        rk=fetchrows(kc,ksc,base,last,nk,kh,d0,vp,ve,kf)
        rv=fetchrows(vc,vsc,base,last,nk,kh,d0,vp,ve,vf)
    end
    while base<last
        nextbase=base+Int32(WH*P)
        # One K row per position, reused by all R heads; value index pi*R+h.
        if PIPE
            rk2=fetchrows(kc,ksc,nextbase,last,nk,kh,d0,vp,ve,kf)
            rv2=fetchrows(vc,vsc,nextbase,last,nk,kh,d0,vp,ve,vf)
        end
        # One assignment: a closure below captures `ks`, and a variable
        # assigned twice is boxed.
        ks=PIPE ? decoderows(rk,ve,kf) : loadrows(kc,ksc,base,last,nk,kh,d0,vp,ve,kf)
        parts=ntuple(v->partdot(qr,ks,v,vr,ve),vn)
        # N values over SG lanes: the butterfly leaves value lane%N in each
        # lane, summed over the lanes below N; one more step adds the upper half.
        s=transposed_sum(parts,lane)
        s=(s+KI.shfl(s,(lane⊻Int32(N))+Int32(1)))*0.0625f0
        v=lane%Int32(N)
        valid=base+v÷Int32(R)<last
        s=valid ? s : -Inf32
        mnew=max(mh,headmax(s,lane,vr,vn))
        alpha=mnew==-Inf32 ? 1f0 : exp(mh-mnew)
        p=valid ? exp(s-mnew) : 0f0
        lh=lh*alpha+headsum(p,lane,vr,vn);mh=mnew
        # Every lane needs every head's rescale; weights come by shuffle as used.
        alphas=ntuple(j->KI.shfl(alpha,Int32(j)),vr)
        vs=PIPE ? decoderows(rv,ve,vf) : loadrows(vc,vsc,base,last,nk,kh,d0,vp,ve,vf)
        acc=nextacc(acc,alphas,p,vs,vr,vp,ve)
        if PIPE
            rk=rk2;rv=rv2
        end
        base=nextbase
    end
    # Merge each head's waves: their maxima, then their rescaled sums and values.
    if lane<Int32(R)
        @inbounds begin wstat[w*Int32(R)+lane+Int32(1)]=mh;wstat[Int32(NW*R)+w*Int32(R)+lane+Int32(1)]=lh;end
    end
    i=t
    while i<Int32(256R*G)
        @inbounds wacc[i+Int32(1)]=0f0;i+=Int32(256)
    end
    KI.barrier()
    for step in 0:WH-1
        wi==Int32(step) && mergewave!(wacc,wstat,acc,Int(w),d0,vr,ve,Int(g)*WH,WH,g*Int32(256R))
        KI.barrier()
    end
    for gg in 0:G-1,j in 1:R
        m=-Inf32
        for u in gg*WH:gg*WH+WH-1;m=max(m,@inbounds wstat[u*R+j]);end
        l=0f0
        for u in gg*WH:gg*WH+WH-1
            mu=@inbounds wstat[u*R+j]
            l+=mu==-Inf32 ? 0f0 : exp(mu-m)*@inbounds(wstat[NW*R+u*R+j])
        end
        slot=((kgroup*Int32(G)+Int32(gg))*Int32(R)+Int32(j-1))*splits+sp
        @inbounds pa[slot*Int32(256)+t+Int32(1)]=wacc[Int32((gg*R+j-1)*256)+t+Int32(1)]
        t==0 && (@inbounds begin pm[slot+Int32(1)]=m;ps[slot+Int32(1)]=l;end)
    end
    nothing
end

# The prepare step of a quantized cache: the floating kernel's K normalisation
# and RoPE, then every block of 32 quantized (`quantize`) and packed into words
# (`packword`) as the cache's format has it.
prepare_kv_quant!(kq,ks,vq,vs,k,v,norm,position,nk::Int32,theta::Float32,eps::Float32,fmt)=
    prepare_kv_quant_at!(kq,ks,vq,vs,k,v,norm,Int32(position[1]),nk,theta,eps,Int32(KI.get_group_id().x)-Int32(1),Int32(0),fmt)
# Every column of a prompt chunk: one workgroup a (KV head, token).
function prepare_kv_quant_chunk!(kq,ks,vq,vs,k,v,norm,positions,nk::Int32,theta::Float32,eps::Float32,fmt)
    t=Int32(KI.get_group_id().x)-Int32(1);tok=t÷nk
    prepare_kv_quant_at!(kq,ks,vq,vs,k,v,norm,Int32(@inbounds positions[tok+Int32(1)]),nk,theta,eps,t%nk,tok*nk*Int32(256),fmt)
end

# The larger in magnitude of two values, the first of equals (`a` comes first).
@inline bigger(a::Float32,b::Float32)=abs(b)>abs(a) ? b : a
# A block's scale and a value's stored integer, from the block's largest value.
@inline function quantize(::Q8Rows,m::Float32,x::Float32)
    scale=abs(m)/127f0
    scale,(scale==0f0 ? 0f0 : clamp(round(x/scale),-127f0,127f0))
end
@inline function quantize(::Q4Rows,m::Float32,x::Float32)
    d=m/-8f0
    id=d==0f0 ? 0f0 : 1f0/d
    d,Float32(min(Int32(15),unsafe_trunc(Int32,x*id+8.5f0)))
end
qwords(::Q8Rows)=64
qwords(::Q4Rows)=32
# Word `w` of a row from its stored integers at `o + 1 ...` in `sh`.
@inline packword(::Q8Rows,sh,o::Int32,w::Int32)=
    reduce(|,ntuple(i->(reinterpret(UInt32,Int32(@inbounds sh[o+Int32(4)*w+Int32(i)]))&UInt32(0xff))<<UInt32(8*(i-1)),Val(4)))
@inline packword(::Q4Rows,sh,o::Int32,w::Int32)=
    reduce(|,ntuple(i->(UInt32(Int32(@inbounds sh[o+Int32(8)*w+Int32(i)]))&UInt32(0xf))<<UInt32(4*(i-1)),Val(8)))

# Head `h` of the K and V vectors at `src` (0-based) into the rows of `pos`.
@inline function prepare_kv_quant_at!(kq,ks,vq,vs,k,v,norm,pos::Int32,nk::Int32,theta::Float32,eps::Float32,h::Int32,src::Int32,fmt)
    sh=KI.localmemory(Float32,Val((512,)),Val(1))
    am=KI.localmemory(Float32,Val((512,)),Val(2))
    d=Int32(KI.get_local_id().x)-Int32(1)
    base=src+h*Int32(256)
    @inbounds begin
        value=Float32(k[base+d+Int32(1)])
        sh[d+Int32(1)]=value;sh[Int32(256)+d+Int32(1)]=value*value
    end
    KI.barrier()
    stride=Int32(128)
    while stride>0
        d<stride && (@inbounds sh[Int32(256)+d+Int32(1)]+=sh[Int32(256)+d+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    scale=inv(sqrt(@inbounds(sh[257])/256f0+eps))
    @inbounds sh[d+Int32(1)]*=scale*norm[d+Int32(1)]
    KI.barrier()
    if d<32
        angle=Float32(pos)*exp(-log(theta)*Float32(2d)/64f0)
        co,si=cos(angle),sin(angle)
        @inbounds begin
            a,b=sh[d+Int32(1)],sh[d+Int32(33)]
            sh[d+Int32(1)]=a*co-b*si;sh[d+Int32(33)]=a*si+b*co
        end
    end
    KI.barrier()
    kx=@inbounds sh[d+Int32(1)];vx=Float32(@inbounds v[base+d+Int32(1)])
    # Each block's value of largest magnitude, the lower index winning a tie:
    # q4_0 takes its sign, q8_0 only its size.
    @inbounds begin am[d+Int32(1)]=kx;am[Int32(256)+d+Int32(1)]=vx;end
    KI.barrier()
    stride=Int32(16)
    while stride>0
        if d%Int32(32)<stride
            @inbounds begin
                am[d+Int32(1)]=bigger(am[d+Int32(1)],am[d+stride+Int32(1)])
                am[Int32(256)+d+Int32(1)]=bigger(am[Int32(256)+d+Int32(1)],am[Int32(256)+d+stride+Int32(1)])
            end
        end
        KI.barrier();stride>>=1
    end
    blk=d÷Int32(32)
    kscale,qk=quantize(fmt,@inbounds(am[blk*Int32(32)+Int32(1)]),kx)
    vscale,qv=quantize(fmt,@inbounds(am[Int32(256)+blk*Int32(32)+Int32(1)]),vx)
    KI.barrier()
    @inbounds begin sh[d+Int32(1)]=qk;sh[Int32(256)+d+Int32(1)]=qv;end
    KI.barrier()
    row=pos*nk+h;W=Int32(qwords(fmt))
    if d<W
        @inbounds begin
            kq[row*W+d+Int32(1)]=packword(fmt,sh,Int32(0),d)
            vq[row*W+d+Int32(1)]=packword(fmt,sh,Int32(256),d)
        end
    end
    if d%Int32(32)==Int32(0)
        @inbounds begin ks[row*Int32(8)+blk+Int32(1)]=kscale;vs[row*Int32(8)+blk+Int32(1)]=vscale;end
    end
    nothing
end

function attention_flash_combine!(out,pm,ps,pa,gate,position,splits::Int32)
    d=Int32(KI.get_local_id().x)-Int32(1)
    h=Int32(KI.get_group_id().x)-Int32(1)
    npos=Int32(position[1])+Int32(1)
    used=cld(npos,flash_span(npos,splits))
    base=h*splits
    m=-Inf32
    for c in Int32(0):used-Int32(1)
        m=max(m,@inbounds pm[base+c+Int32(1)])
    end
    l=0f0;o=0f0
    for c in Int32(0):used-Int32(1)
        f=exp(@inbounds(pm[base+c+Int32(1)])-m)
        l=muladd(f,@inbounds(ps[base+c+Int32(1)]),l)
        o=muladd(f,@inbounds(pa[(base+c)*Int32(256)+d+Int32(1)]),o)
    end
    i=h*Int32(256)+d+Int32(1)
    @inbounds out[i]=eltype(out)((o/l)*Float32(gate[i]))
    nothing
end

for (kernel,writes) in ((attention_flash!,(1,2,3)),(attention_flash_combine!,(1,)),(prepare_kv_quant!,(1,2,3,4)),(prepare_kv_quant_chunk!,(1,2,3,4)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end
