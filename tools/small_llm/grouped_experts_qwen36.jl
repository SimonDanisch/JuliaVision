# Included in Qwen36MoEExperiment. The routed experts of a wide prompt chunk,
# grouped by expert. The decode kernels run one (token, expert) pair a subgroup,
# so a chunk of 256 tokens reads 2048 experts' weights a layer where 256 exist;
# that was 69% of the chunk. Here the slots of each expert are listed first, and
# a workgroup takes one expert's rows and goes through its slots eight at a time,
# reading the weights once per eight. The arithmetic per (row, slot) is the
# decode kernels', in the same order.

const BUCKET_TILE=2048           # selections staged at a time

"""
    bucket_experts!(order, offsets, selected, status, ne, slots, Val(TOPK))

The slots `token*TOPK + k` that chose each expert, in slot order: expert `e`'s
are `order[offsets[e+1]+1 : offsets[e+2]]`. One thread an expert, each counting
its own over the selections staged in shared memory 2048 at a time: no atomics,
so the order and every result after it are the same on every run. Slots of a
token the router refused (status not 0) are in no list; the combine writes that
token as NaN without reading them.
"""
function bucket_experts!(order,offsets,selected,status,ne::Int32,slots::Int32,::Val{TOPK}) where TOPK
    counts=KI.localmemory(Int32,Val((257,)),Val(1))
    tile=KI.localmemory(Int32,Val((BUCKET_TILE,)),Val(2))
    e=Int32(KI.get_local_id().x)-Int32(1)
    c=Int32(0)
    t0=Int32(0)
    while t0<slots
        stagechoices!(tile,selected,status,t0,slots,e,Val(TOPK))
        KI.barrier()
        if e<ne
            for i in Int32(1):min(Int32(BUCKET_TILE),slots-t0)
                c+=@inbounds(tile[i])==e ? Int32(1) : Int32(0)
            end
        end
        KI.barrier()
        t0+=Int32(BUCKET_TILE)
    end
    @inbounds counts[e+Int32(1)]=c
    KI.barrier()
    if e==Int32(0)
        acc=Int32(0)
        for i in Int32(1):Int32(256)
            @inbounds begin v=counts[i];counts[i]=acc;acc+=v;end
        end
        @inbounds counts[257]=acc
    end
    KI.barrier()
    at=@inbounds counts[e+Int32(1)]
    e<ne && (@inbounds offsets[e+Int32(1)]=at)
    e==Int32(0) && (@inbounds offsets[ne+Int32(1)]=counts[257])
    t0=Int32(0)
    while t0<slots
        stagechoices!(tile,selected,status,t0,slots,e,Val(TOPK))
        KI.barrier()
        if e<ne
            for i in Int32(1):min(Int32(BUCKET_TILE),slots-t0)
                if @inbounds(tile[i])==e
                    @inbounds order[at+Int32(1)]=t0+i-Int32(1);at+=Int32(1)
                end
            end
        end
        KI.barrier()
        t0+=Int32(BUCKET_TILE)
    end
    nothing
end
# The expert of each slot t0 .. t0+BUCKET_TILE-1, or -1 for a refused token or
# past the last slot; 256 threads, coalesced.
@inline function stagechoices!(tile,selected,status,t0::Int32,slots::Int32,e::Int32,::Val{TOPK}) where TOPK
    i=e
    while i<Int32(BUCKET_TILE)
        s=t0+i
        v=Int32(-1)
        if s<slots
            @inbounds v=status[s÷Int32(TOPK)+Int32(1)]==Int32(0) ? selected[s+Int32(1)] : Int32(-1)
        end
        @inbounds tile[i+Int32(1)]=v
        i+=Int32(256)
    end
    nothing
end

# A K-quant block of a weight row as the routed kernels read it: the header
# (scales, d, dmin), this lane's quant vector, and the Q5 high bits (the header
# again for Q4, unused), then the two integer dot products with the activations.
@inline function kqblock(::Val{4},Q,row::Int32,nb::Int32,b::Int32,j::Int32,h::Int32)
    vb=(row*nb+b)*Int32(DK.Q4K_VECS)
    hd=@inbounds Q[vb+Int32(1)]
    (hd,@inbounds(Q[vb+Int32(2)+Int32(2)*j+h]),hd)
end
@inline kqdots(::Val{4},qv,hb,xlo,xhi,j::Int32)=(DK.q4k_dot(qv,xlo,DK.q4k_lo),DK.q4k_dot(qv,xhi,DK.q4k_hi))

# The sum over an aligned group of W lanes (8, 16 or 32), in its first lane.
@inline function kqsum(v::Float32,::Val{W}) where W
    W>=32 && (v+=KI.shfl_down(v,16))
    W>=16 && (v+=KI.shfl_down(v,8))
    v+=KI.shfl_down(v,4);v+=KI.shfl_down(v,2);v+=KI.shfl_down(v,1)
    v
end

"""
    grouped_kq_gemv!(out, Q, xq, xds, order, offsets, m, Val(NB), Val(FORMAT), Val(TOPK), Val(PER_EXPERT), Val(NC), Val(R))

`out[:, slot] = W[expert] * x[:, column(slot)]` for every listed slot of every
expert, `W` the experts' K-quant rows (`FORMAT` 4 or 5, `NB` blocks of 256 a
row) stacked expert-major. Eight lanes take a block, so a row has `8NB` lanes up
to 32, and a 32-lane group takes `32 ÷ 8NB` row sets of `R` (≤ 4) rows: at the
experts' 512-wide down projection one row set a group left half the lanes idle.
A workgroup is one expert's `8 * (32 ÷ 8NB) * R` rows; it takes that expert's
slots `NC` (≤ 16) at a time. The column of a slot is the slot itself
(`PER_EXPERT`) or its token.
"""
function grouped_kq_gemv!(out,Q,xq,xds,order,offsets,m::Int32,::Val{NB},
        vf::Val{FORMAT},::Val{TOPK},::Val{PER_EXPERT},::Val{NC},::Val{R}) where {NB,FORMAT,TOPK,PER_EXPERT,NC,R}
    LG=min(32,8NB);SPLIT=32÷LG;nb=Int32(NB)
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(32)
    tiles=cld(m,Int32(8SPLIT*R))
    wg=Int32(KI.get_group_id().x)-Int32(1)
    e=wg÷tiles;tile=wg%tiles
    first=@inbounds offsets[e+Int32(1)]
    count=@inbounds(offsets[e+Int32(2)])-first
    count==Int32(0) && return nothing          # uniform across the workgroup
    l=lane%Int32(LG)
    r0=((tile*Int32(8)+lid÷Int32(32))*Int32(SPLIT)+lane÷Int32(LG))*Int32(R)
    p=l%Int32(8);j=p÷Int32(2);h=p%Int32(2)
    keepmin=h==Int32(0) ? 1f0 : 0f0
    cols=nb*Int32(16)                          # activation vectors, and floats, a column
    c0=Int32(0)
    while c0<count
        # A group past the expert's last slot repeats that slot and stores nothing.
        Base.Cartesian.@nexprs 16 c->if c<=NC
            slot_c=@inbounds order[first+min(c0+Int32(c-1),count-Int32(1))+Int32(1)]
            xo_c=(PER_EXPERT ? slot_c : slot_c÷Int32(TOPK))*cols
            Base.Cartesian.@nexprs 4 r->if r<=R;acc_c_r=0f0;end
        end
        b=l÷Int32(8)
        while b<nb
            Base.Cartesian.@nexprs 4 r->if r<=R
                hd_r,qv_r,hb_r=kqblock(vf,Q,e*m+min(r0+Int32(r-1),m-Int32(1)),nb,b,j,h)
                d_r=DK.q4k_half(hd_r[1].value,0)
                dmin_r=DK.q4k_half(hd_r[1].value,1)
                sl_r,ml_r=DK.q4k_scalemin(hd_r,Int32(2)*j)
                sh_r,mh_r=DK.q4k_scalemin(hd_r,Int32(2)*j+Int32(1))
            end
            xv=b*Int32(16)+Int32(4)*j+h
            xs=(b*Int32(8)+Int32(2)*j)*Int32(2)
            Base.Cartesian.@nexprs 16 c->if c<=NC
                xlo=@inbounds xq[xo_c+xv+Int32(1)]
                xhi=@inbounds xq[xo_c+xv+Int32(3)]
                dlo,slo=@inbounds xds[xo_c+xs+Int32(1)],xds[xo_c+xs+Int32(2)]
                dhi,shi=@inbounds xds[xo_c+xs+Int32(3)],xds[xo_c+xs+Int32(4)]
                Base.Cartesian.@nexprs 4 r->if r<=R
                    il,ih=kqdots(vf,qv_r,hb_r,xlo,xhi,j)
                    acc_c_r+=d_r*(Float32(sl_r)*dlo*Float32(il)+Float32(sh_r)*dhi*Float32(ih))-
                             keepmin*dmin_r*(Float32(ml_r)*slo+Float32(mh_r)*shi)
                end
            end
            b+=Int32(LG÷8)
        end
        Base.Cartesian.@nexprs 16 c->Base.Cartesian.@nexprs 4 r->if c<=NC&&r<=R
            v_c_r=kqsum(acc_c_r,Val(LG))
            row_c_r=r0+Int32(r-1)
            if l==Int32(0)&&row_c_r<m&&c0+Int32(c-1)<count
                @inbounds out[slot_c*m+row_c_r+Int32(1)]=v_c_r
            end
        end
        c0+=Int32(NC)
    end
    nothing
end

for (kernel,writes) in ((bucket_experts!,(1,2)),(grouped_kq_gemv!,(1,)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end

"""The expert lists of a chunk's router selection, `(order, offsets)`."""
function declare_expert_buckets!(g,selected,status,ne;name)
    topk,tokens=size(selected)
    ne<=256 || throw(ArgumentError("expert buckets take at most 256 experts"))
    order=Mantle.Transient.Buffer(g,Int32,(topk*tokens,))
    offsets=Mantle.Transient.Buffer(g,Int32,(ne+1,))
    Q.dispatch(g,bucket_experts!,(order,offsets,selected,status,Int32(ne),Int32(topk*tokens),Val(topk)),256;group=256,name)
    order,offsets
end

kqformat(::M.Q4Experts)=Val(4)
expertshape(W::M.Q4Experts)=size(W)
kqrows(W::M.Q4Experts)=W.packed.q

"""DECLARE a routed projection over expert lists: `out` is `(m, topk, tokens)`."""
function grouped_routed!(g,out,W,x,order,offsets,selected,status;per_expert=false,name,optimizations=(),nc=8,rows=2)
    k,m,ne=expertshape(W);topk,tokens=size(selected)
    size(out)==(m,topk,tokens)||throw(DimensionMismatch("grouped expert output"))
    length(kqrows(W))<=typemax(Int32)&&length(out)<=typemax(Int32)||throw(ArgumentError("grouped expert addressing"))
    columns=per_expert ? topk*tokens : tokens
    qx=x isa Q8Input ? x : prepare_q8!(g,x,k,columns;status,topk,per_expert,name=name*".q8",parallel=:parallel_q8 in optimizations)
    qx.k==k&&qx.columns==columns||throw(DimensionMismatch("grouped expert Q8 extent"))
    nb=k÷256;split=32÷min(32,8nb)
    Q.dispatch(g,grouped_kq_gemv!,
        (out,kqrows(W),qx.q,qx.ds,order,offsets,Int32(m),Val(nb),kqformat(W),Val(topk),Val(per_expert),Val(nc),Val(rows)),
        ne*cld(m,8split*rows)*256;group=256,name)
    nothing
end
