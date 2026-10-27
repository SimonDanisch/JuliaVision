# Included in Qwen35Experiment. The same kernel runs on CPU Vulkan and Radeon.
# One workgroup handles all queries sharing a KV head for a 64-token chunk.
@inline grouped_value_acc(acc,w,p,value,::Val{R}) where R =
    ntuple(j->muladd((@inbounds w[Int32(64*(j-1))+p+Int32(1)]),value,acc[j]),Val(R))

function attention_grouped!(pm,ps,pa,q,kc,vc,position,nk::Int32,chunks::Int32,
                            ::Val{R},::Val{SG}) where {R,SG}
    w=KI.localmemory(Float32,Val((128R,)),Val(1))
    stat=KI.localmemory(Float32,Val((2R,)),Val(2))
    t=Int32(KI.get_local_id().x)-Int32(1)
    wg=Int32(KI.get_group_id().x)-Int32(1)
    kh=wg÷chunks;chunk=wg%chunks;p0=chunk*Int32(64)
    p0>Int32(position[1]) && return nothing
    n=min(Int32(64),Int32(position[1])-p0+Int32(1))
    lane=t%Int32(SG);subgroup=t÷Int32(SG)
    qr=ntuple(Val(R)) do j
        ntuple(Val(256÷SG)) do u
            @inbounds Float32(q[(kh*Int32(R)+Int32(j-1))*Int32(256)+lane+Int32(SG*(u-1))+Int32(1)])
        end
    end
    p=subgroup
    while p<64
        # Clamp inactive positions to a valid row; mask their scores below.
        base=((p0+min(p,n-Int32(1)))*nk+kh)*Int32(256)
        kr=ntuple(u->Float32(@inbounds kc[base+lane+Int32(SG*(u-1))+Int32(1)]),Val(256÷SG))
        scores=ntuple(Val(R)) do j
            dot=0f0
            for u in 1:256÷SG
                dot=muladd(qr[j][u],kr[u],dot)
            end
            KI.sub_group_reduce_add(dot)
        end
        if lane==0
            for j in 1:R
                @inbounds w[Int32(64*(j-1))+p+Int32(1)]=p<n ? scores[j]*0.0625f0 : -Inf32
            end
        end
        p+=Int32(256÷SG)
    end
    KI.barrier()
    # Four 64-work-item head segments fit in a workgroup. Tile the softmax
    # when eight queries share a KV head, keeping K/V reuse across all eight.
    for firsthead in Int32(0):Int32(4):Int32(R-1)
        u=t+firsthead*Int32(64)
        active=u<Int32(64R)
        j=u÷Int32(64);r=u%Int32(64)
        active && (@inbounds w[Int32(64R)+u+Int32(1)]=w[u+Int32(1)])
        KI.barrier()
        stride=Int32(32)
        while stride>0
            if active && r<stride
                at=Int32(64R)+u+Int32(1)
                @inbounds w[at]=max(w[at],w[at+stride])
            end
            KI.barrier();stride>>=1
        end
        active && r==0 && (@inbounds stat[j+Int32(1)]=w[Int32(64R)+u+Int32(1)])
        KI.barrier()
        if active
            m=@inbounds stat[j+Int32(1)]
            e=exp((@inbounds w[u+Int32(1)])-m)
            @inbounds begin w[u+Int32(1)]=e;w[Int32(64R)+u+Int32(1)]=e;end
        end
        KI.barrier()
        stride=Int32(32)
        while stride>0
            if active && r<stride
                at=Int32(64R)+u+Int32(1)
                @inbounds w[at]+=w[at+stride]
            end
            KI.barrier();stride>>=1
        end
        active && r==0 && (@inbounds stat[Int32(R)+j+Int32(1)]=w[Int32(64R)+u+Int32(1)])
        KI.barrier()
    end
    acc=ntuple(_->0f0,Val(R))
    for p in Int32(0):n-Int32(1)
        # Load a value once for all R query heads, not once per head.
        value=Float32(@inbounds vc[((p0+p)*nk+kh)*Int32(256)+t+Int32(1)])
        acc=grouped_value_acc(acc,w,p,value,Val(R))
    end
    for j in 1:R
        slot=(kh*Int32(R)+Int32(j-1))*chunks+chunk
        @inbounds pa[slot*Int32(256)+t+Int32(1)]=acc[j]
        if t==0
            @inbounds begin pm[slot+Int32(1)]=stat[j];ps[slot+Int32(1)]=stat[R+j];end
        end
    end
    nothing
end

# The flash kernel's cache arguments: values, then scales (the values again,
# unused, for a floating cache).
flashargs(kc,vc)=(kc,vc,kc,vc)
flashargs(kc::QuantizedKV,vc::QuantizedKV)=(kc.q,vc.q,kc.scales,vc.scales)

function declare_attention!(g,mode,pm,ps,pa,q,kc,vc,pos,nkv,nqh,chunks;name)
    sg=Mantle.caps(g.dev).subgroup
    kc isa QuantizedKV && mode!==:flash && throw(ArgumentError("a quantized KV cache needs flash attention"))
    if mode===:grouped
        ratio=nqh÷nkv
        ratio in (1,2,4,8) || throw(ArgumentError("unsupported grouped-query ratio"))
        dispatch(g,attention_grouped!,
            (pm,ps,pa,q,kc,vc,pos,Int32(nkv),Int32(chunks),Val(ratio),Val(sg)),
            nkv*chunks*256;name)
    elseif mode===:flash
        ratio=nqh÷nkv
        ratio in (1,2,4,8) && 256%sg==0 && sg%(2ratio)==0 || throw(ArgumentError("unsupported flash-decoding shape"))
        # All KV heads of a span in one workgroup when its waves divide among them.
        heads=(256÷sg)%nkv==0 ? nkv : 1
        dispatch(g,attention_flash!,
            (pm,ps,pa,q,flashargs(kc,vc)...,pos,Int32(nkv),Int32(chunks),Val(ratio),Val(sg),Val(heads),rowformat(kc),rowformat(vc)),
            (nkv÷heads)*chunks*256;group=256,name)
    elseif mode===:split
        dispatch(g,attention_partial!,
            (pm,ps,pa,q,kc,vc,pos,Int32(nkv),Int32(nqh÷nkv),Int32(chunks),Val(sg)),
            nqh*chunks*256;name)
    else
        throw(ArgumentError("unknown attention implementation"))
    end
end

"""How many partial results per query head an attention mode writes: one per 64
positions of the cache, or a fixed number of flash-decoding splits."""
attention_slots(mode::Symbol,capacity::Integer)=mode===:flash ? FLASH_SPLITS : cld(capacity,64)

"""Merge a mode's partial results into the gated attention output."""
function declare_attention_combine!(g,mode,out,pm,ps,pa,gate,pos,slots,nqh;name)
    kernel=mode===:flash ? attention_flash_combine! : BR.attention_combine_kernel!
    dispatch(g,kernel,(out,pm,ps,pa,gate,pos,Int32(slots)),nqh*256;name)
end

"""Write one position's K (normalised, rotated) and V into the cache."""
declare_prepare_kv!(g,kc,vc,k,v,norm,pos,nkv,theta,eps;name)=
    dispatch(g,prepare_kv!,(kc,vc,k,v,norm,pos,Int32(nkv),theta,eps),nkv*256;name)
declare_prepare_kv!(g,kc::QuantizedKV,vc::QuantizedKV,k,v,norm,pos,nkv,theta,eps;name)=
    dispatch(g,prepare_kv_quant!,(kc.q,kc.scales,vc.q,vc.scales,k,v,norm,pos,Int32(nkv),theta,eps,rowformat(kc)),nkv*256;group=256,name)
