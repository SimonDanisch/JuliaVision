# Included in Qwen35MoE. Reuse DNNKernels' Q4_K format, Q8 activations and
# packed integer-dot arithmetic; add device-side expert indexing.
import DNNKernels
const DK=DNNKernels

struct Q4Experts{A<:DK.Q4KMatrix}
    packed::A
    rows::Int
    experts::Int
    function Q4Experts(packed::A,rows,experts) where {A<:DK.Q4KMatrix}
        rows>0 && experts>0 && size(packed,1)==rows*experts && size(packed,2)%256==0 ||
            throw(DimensionMismatch("packed expert dimensions"))
        new{A}(packed,rows,experts)
    end
end
Base.size(W::Q4Experts)=(size(W.packed,2),W.rows,W.experts)
Base.size(W::Q4Experts,d::Integer)=d<=3 ? size(W)[d] : 1

function checked_q8act!(xq,xds,x,status,k::Int32,n::Int32,topk::Int32,::Val{PER_EXPERT}) where PER_EXPERT
    t=Int32(KI.get_global_id().x)-Int32(1)
    if t<n
        nk=k÷Int32(32);col=t÷nk;token=PER_EXPERT ? col÷topk : col
        if status[token+Int32(1)]==0
            DK.q8act_kernel!(xq,xds,x,k,n)
        else
            # Do not convert a bad token's NaN activation to an integer.
            base=t*Int32(2)
            z=ntuple(_->VecElement(UInt32(0)),Val(4))
            @inbounds begin
                xq[base+Int32(1)]=z;xq[base+Int32(2)]=z
                xds[base+Int32(1)]=0f0;xds[base+Int32(2)]=0f0
            end
        end
    end
    nothing
end

function routed_q4_gemv!(out,Q,xq,xds,selected,status,m::Int32,ne::Int32,topk::Int32,
                         rows::Int32,nb::Int32,::Val{PER_EXPERT},::Val{SG}) where {PER_EXPERT,SG}
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(32)
    row=(Int32(KI.get_group_id().x)-Int32(1))*Int32(8)+lid÷Int32(32)
    safe=min(row,rows-Int32(1));slot=safe÷m;token=slot÷topk
    expert=@inbounds selected[slot+Int32(1)]
    valid=row<rows && status[token+Int32(1)]==0 && 0<=expert<ne
    acc=0f0
    if valid
        physical=expert*m+safe%m
        col=PER_EXPERT ? slot : token
        cols=nb*Int32(16);offset=col*cols
        p=lane%Int32(8);j=p÷Int32(2);h=p%Int32(2)
        b=lane÷Int32(8)
        while b<nb
            vb=(physical*nb+b)*Int32(DK.Q4K_VECS)
            hd=@inbounds Q[vb+Int32(1)]
            qv=@inbounds Q[vb+Int32(2)+Int32(2)*j+h]
            d=DK.q4k_half(hd[1].value,0);dm=DK.q4k_half(hd[1].value,1)
            sl,ml=DK.q4k_scalemin(hd,Int32(2)*j)
            sh,mh=DK.q4k_scalemin(hd,Int32(2)*j+Int32(1))
            xv=offset+b*Int32(16)+Int32(4)*j+h
            xs=offset+(b*Int32(8)+Int32(2)*j)*Int32(2)
            @inbounds begin
                xlo=xq[xv+Int32(1)];xhi=xq[xv+Int32(3)]
                dlo=xds[xs+Int32(1)];slo=xds[xs+Int32(2)]
                dhi=xds[xs+Int32(3)];shi=xds[xs+Int32(4)]
            end
            il=DK.q4k_dot(qv,xlo,DK.q4k_lo);ih=DK.q4k_dot(qv,xhi,DK.q4k_hi)
            acc+=d*(Float32(sl)*dlo*Float32(il)+Float32(sh)*dhi*Float32(ih))-
                (h==0 ? 1f0 : 0f0)*dm*(Float32(ml)*slo+Float32(mh)*shi)
            b+=Int32(4)
        end
    end
    if SG>=32
        acc=DK.q4k_sum32(acc)
    else
        # A software Vulkan subgroup can be 8 lanes. Reduce the same logical
        # 32-lane row through shared memory, without a second CPU algorithm.
        partial=KI.localmemory(Float32,Val((256,)),Val(1))
        @inbounds partial[lid+Int32(1)]=acc
        KI.barrier()
        stride=Int32(16)
        while stride>0
            lane<stride && (@inbounds partial[lid+Int32(1)]+=partial[lid+stride+Int32(1)])
            KI.barrier();stride>>=1
        end
        acc=@inbounds partial[lid+Int32(1)]
    end
    if lane==0 && row<rows
        @inbounds out[row+Int32(1)]=status[token+Int32(1)]==0 && !(0<=expert<ne) ? NaN32 : acc
    end
    nothing
end

function declare_routed_linear!(g,out,W::Q4Experts,x,selected,status;per_expert=false,name="moe.expert_q4")
    k,m,ne=size(W);topk,tokens=size(selected)
    1<=topk<=ne && tokens>0 || throw(ArgumentError("invalid expert selection dimensions"))
    size(out)==(m,topk,tokens) || throw(DimensionMismatch("expert Q4 output"))
    size(x)==(per_expert ? (k,topk,tokens) : (k,tokens)) && length(status)==tokens ||
        throw(DimensionMismatch("expert Q4 input"))
    length(W.packed.q)<=typemax(Int32) && length(out)<=typemax(Int32) ||
        throw(ArgumentError("expert Q4 exceeds 32-bit addressing"))
    columns=per_expert ? topk*tokens : tokens
    xq=Mantle.Transient.Buffer(g,DK.UVec4,(k÷16,columns))
    xds=Mantle.Transient.Buffer(g,Float32,(k÷16,columns))
    kerneldispatch!(g,checked_q8act!,
        (xq,xds,x,status,Int32(k),Int32(columns*(k÷32)),Int32(topk),Val(per_expert)),
        columns*(k÷32);group=256,name=name*".q8")
    kerneldispatch!(g,routed_q4_gemv!,
        (out,W.packed.q,xq,xds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(out)),
         Int32(k÷256),Val(per_expert),Val(Mantle.caps(g.dev).subgroup)),
        cld(length(out),8)*256;group=256,name)
end
