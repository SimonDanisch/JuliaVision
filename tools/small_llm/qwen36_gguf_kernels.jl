# Included inside Qwen36GGUF. Reuse DNNKernels' Q8 activations, scale decoding,
# integer dot products and subgroup reduction for GGUF's Q5_K extension.
struct Q5KMatrix{A}
    q::A
    m::Int
    k::Int
end
struct Q5Experts{A}
    packed::A
    rows::Int
    experts::Int
end
R.supports_q8(::Q5KMatrix)=true
R.supports_q8(::Q5Experts)=true

@inline function q5_dot(q,h,x,shift::Int32,high::Bool)
    dot=Int32(0)
    Base.Cartesian.@nexprs 4 w -> begin
        lo=high ? DK.q4k_hi(q[w].value) : DK.q4k_lo(q[w].value)
        hi=((h[w].value>>shift)&UInt32(0x01010101))<<4
        dot+=KI.dot4x8(lo|hi,x[w].value)
    end
    dot
end

# A Q5_K block for the grouped expert kernel: header, quants, high bits.
@inline function R.kqblock(::Val{5},Q,row::Int32,nb::Int32,b::Int32,j::Int32,h::Int32)
    vb=(row*nb+b)*Int32(11)
    @inbounds (Q[vb+Int32(1)],Q[vb+Int32(4)+Int32(2)*j+h],Q[vb+Int32(2)+h])
end
@inline R.kqdots(::Val{5},qv,hb,xlo,xhi,j::Int32)=
    (q5_dot(qv,hb,xlo,Int32(2)*j,false),q5_dot(qv,hb,xhi,Int32(2)*j+Int32(1),true))
R.kqformat(::Q5Experts)=Val(5)
R.expertshape(W::Q5Experts)=(W.packed.k,W.rows,W.experts)
R.kqrows(W::Q5Experts)=W.packed.q

q5_gemv!(args...)=q5_gemv_rows!(args...,Val(32))
q5_gemv_rows!(args...)=q5_gemv_tuned!(args...,Val(256))
function q5_gemv_tuned!(out,Q,xq,xds,selected,status,m::Int32,ne::Int32,topk::Int32,
                  rows::Int32,nb::Int32,::Val{PER_EXPERT},::Val{ROUTED},::Val{SG},::Val{LANES},::Val{WG}) where {PER_EXPERT,ROUTED,SG,LANES,WG}
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(LANES)
    row=(Int32(KI.get_group_id().x)-Int32(1))*Int32(WG÷LANES)+lid÷Int32(LANES)
    safe=min(row,rows-Int32(1));slot=ROUTED ? safe÷m : Int32(0)
    token=ROUTED ? slot÷topk : Int32(0)
    expert=ROUTED ? (@inbounds selected[slot+Int32(1)]) : Int32(0)
    okay=ROUTED ? (@inbounds status[token+Int32(1)]==0) : true
    valid=row<rows&&okay&&0<=expert<ne
    acc=0f0
    if valid
        physical=ROUTED ? expert*m+safe%m : safe
        col=PER_EXPERT ? slot : token
        offset=col*nb*Int32(16)
        p=lane%Int32(8);j=p÷Int32(2);h=p%Int32(2);b=lane÷Int32(8)
        while b<nb
            vb=(physical*nb+b)*Int32(11)
            @inbounds begin
                hd=Q[vb+Int32(1)]
                highbits=Q[vb+Int32(2)+h]
                qv=Q[vb+Int32(4)+Int32(2)*j+h]
                xv=offset+b*Int32(16)+Int32(4)*j+h
                xs=offset+(b*Int32(8)+Int32(2)*j)*Int32(2)
                xlo=xq[xv+Int32(1)];xhi=xq[xv+Int32(3)]
                dlo=xds[xs+Int32(1)];slo=xds[xs+Int32(2)]
                dhi=xds[xs+Int32(3)];shi=xds[xs+Int32(4)]
            end
            d=DK.q4k_half(hd[1].value,0);dm=DK.q4k_half(hd[1].value,1)
            sl,ml=DK.q4k_scalemin(hd,Int32(2)*j)
            sh,mh=DK.q4k_scalemin(hd,Int32(2)*j+Int32(1))
            il=q5_dot(qv,highbits,xlo,Int32(2)*j,false)
            ih=q5_dot(qv,highbits,xhi,Int32(2)*j+Int32(1),true)
            acc+=d*(Float32(sl)*dlo*Float32(il)+Float32(sh)*dhi*Float32(ih))-
                (h==0 ? 1f0 : 0f0)*dm*(Float32(ml)*slo+Float32(mh)*shi)
            b+=Int32(LANES÷8)
        end
    end
    if SG>=LANES
        acc=R.sum_row(acc,Val(LANES))
    else
        partial=KI.localmemory(Float32,Val((WG,)),Val(1))
        @inbounds partial[lid+Int32(1)]=acc
        KI.barrier();stride=Int32(LANES÷2)
        while stride>0
            lane<stride&&(@inbounds partial[lid+Int32(1)]+=partial[lid+stride+Int32(1)])
            KI.barrier();stride>>=1
        end
        acc=@inbounds partial[lid+Int32(1)]
    end
    if lane==0&&row<rows
        @inbounds out[row+Int32(1)]=okay&&!(0<=expert<ne) ? NaN32 : acc
    end
    nothing
end

function R.linear_opt!(g,out,A::Q5KMatrix,x::R.Q8Input;name,optimizations=())
    m,k=A.m,A.k
    k==x.k&&x.columns==1||throw(DimensionMismatch("Q5 shared input"))
    Q.dispatch(g,q5_gemv!,
        (out,A.q,x.q,x.ds,out,out,Int32(m),Int32(1),Int32(1),Int32(m),Int32(k÷256),Val(false),Val(false),Val(Mantle.caps(g.dev).subgroup)),
        cld(m,8)*256;name)
end
function R.linear_opt!(g,out,A::Q5KMatrix,x;name,optimizations=())
    :parallel_q8 in optimizations||return Q.linear!(g,out,A,x;name)
    qx=R.prepare_q8!(g,x,A.k,1;name=name*".q8",parallel=true)
    R.linear_opt!(g,out,A,qx;name,optimizations)
end

function R.routed_opt!(g,out,W::Q5Experts,x,selected,status;per_expert=false,name,optimizations=())
    if !(x isa R.Q8Input)&&!(:narrow_experts in optimizations)
        return M.declare_routed_linear!(g,out,W,x,selected,status;per_expert,name)
    end
    k,m,ne=W.packed.k,W.rows,W.experts;topk,tokens=size(selected)
    1<=topk<=ne&&size(out)==(m,topk,tokens)&&length(status)==tokens||throw(DimensionMismatch("Routed Q5 outputs/status"))
    length(W.packed.q)<=typemax(Int32)&&length(out)<=typemax(Int32)||throw(ArgumentError("Routed Q5 addressing"))
    columns=per_expert ? topk*tokens : tokens
    qx=x isa R.Q8Input ? x : R.prepare_q8!(g,x,k,columns;status,topk,per_expert,name=name*".q8",parallel=:parallel_q8 in optimizations)
    qx.k==k&&qx.columns==columns||throw(DimensionMismatch("Q5 shared routed input"))
    lanes=:narrow_experts in optimizations ? min(32,k÷32) : 32
    wg=:down_wg64 in optimizations&&per_expert&&k==512 ? 64 : 256
    Q.dispatch(g,q5_gemv_tuned!,
        (out,W.packed.q,qx.q,qx.ds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(out)),Int32(k÷256),Val(per_expert),Val(true),Val(Mantle.caps(g.dev).subgroup),Val(lanes),Val(wg)),
        cld(length(out),wg÷lanes)*wg;group=wg,name)
end

function Q.linear!(g,out,A::Q5KMatrix,x;name)
    m,k=A.m,A.k
    xq=Mantle.Transient.Buffer(g,DK.UVec4,(k÷16,1))
    xds=Mantle.Transient.Buffer(g,Float32,(k÷16,1))
    Q.dispatch(g,DK.q8act_kernel!,(xq,xds,x,Int32(k),Int32(k÷32)),k÷32;name=name*".q8")
    Q.dispatch(g,q5_gemv!,
        (out,A.q,xq,xds,out,out,Int32(m),Int32(1),Int32(1),Int32(m),Int32(k÷256),Val(false),Val(false),Val(Mantle.caps(g.dev).subgroup)),
        cld(m,8)*256;name)
end

function fill_int32!(a,value::Int32,n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds a[i]=value)
    nothing
end

# Every column a slot of one expert: the list `0:n-1` and its offsets `[0, n]`.
function single_expert_list!(order,offsets,n::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    i<n && (@inbounds order[i+Int32(1)]=i)
    i==Int32(0) && (@inbounds begin offsets[1]=Int32(0);offsets[2]=n;end)
    nothing
end
DK.kernelresults(::typeof(single_expert_list!))=(1,2)

# A Q5_K product over `n` columns. Up to the bit-exact width, the routed kernel
# with the one expert chosen for every column, so each column reads its own Q8
# activation and the row arithmetic is the single-column kernel's. Wider, the
# grouped expert kernel over that one expert, which reads the weight once per
# eight columns instead of once per column.
function Q.linear_batch!(g,out,A::Q5KMatrix,x,n;name)
    m,k=A.m,A.k
    xq=Mantle.Transient.Buffer(g,DK.UVec4,(k÷16,n))
    xds=Mantle.Transient.Buffer(g,Float32,(k÷16,n))
    Q.dispatch(g,DK.q8act_kernel!,(xq,xds,x,Int32(k),Int32(n*(k÷32))),n*(k÷32);name=name*".q8")
    if Q.exactchunk(n)
        first=Mantle.Transient.Buffer(g,Int32,(n,))
        Q.dispatch(g,fill_int32!,(first,Int32(0),Int32(n)),n;name=name*".expert")
        Q.dispatch(g,q5_gemv!,
            (out,A.q,xq,xds,first,first,Int32(m),Int32(1),Int32(1),Int32(m*n),Int32(k÷256),Val(false),Val(true),Val(Mantle.caps(g.dev).subgroup)),
            cld(m*n,8)*256;name)
    else
        order=Mantle.Transient.Buffer(g,Int32,(n,));offsets=Mantle.Transient.Buffer(g,Int32,(2,))
        Q.dispatch(g,single_expert_list!,(order,offsets,Int32(n)),n;name=name*".columns")
        nb=k÷256;split=32÷min(32,8nb)
        Q.dispatch(g,R.grouped_kq_gemv!,
            (out,A.q,xq,xds,order,offsets,Int32(m),Val(nb),Val(5),Val(1),Val(true),Val(8),Val(2)),
            cld(m,16split)*256;group=256,name)
    end
end

function M.declare_routed_linear!(g,out,W::Q5Experts,x,selected,status;per_expert=false,name="moe.expert_q5")
    k,m,ne=W.packed.k,W.rows,W.experts;topk,tokens=size(selected)
    1<=topk<=ne&&size(out)==(m,topk,tokens)||throw(DimensionMismatch("Expert Q5 output"))
    size(x)==(per_expert ? (k,topk,tokens) : (k,tokens))&&length(status)==tokens||throw(DimensionMismatch("Expert Q5 input"))
    columns=per_expert ? topk*tokens : tokens
    xq=Mantle.Transient.Buffer(g,DK.UVec4,(k÷16,columns))
    xds=Mantle.Transient.Buffer(g,Float32,(k÷16,columns))
    Q.dispatch(g,M.checked_q8act!,
        (xq,xds,x,status,Int32(k),Int32(columns*(k÷32)),Int32(topk),Val(per_expert)),columns*(k÷32);name=name*".q8")
    Q.dispatch(g,q5_gemv!,
        (out,W.packed.q,xq,xds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(out)),Int32(k÷256),Val(per_expert),Val(true),Val(Mantle.caps(g.dev).subgroup)),
        cld(length(out),8)*256;name)
end

@inline function q4_embedding_value(Q,row::Int32,i::Int32,width::Int32)
    b=i÷Int32(256);v=i%Int32(256)
    vb=(row*(width÷Int32(256))+b)*Int32(DK.Q4K_VECS)
    @inbounds hd=Q[vb+Int32(1)]
    j=v÷Int32(32);byte=Int32(32)*(j÷Int32(2))+v%Int32(32)
    @inbounds qv=Q[vb+Int32(2)+byte÷Int32(16)]
    wi=byte%Int32(16)÷Int32(4)
    word=ifelse(wi==0,qv[1].value,ifelse(wi==1,qv[2].value,ifelse(wi==2,qv[3].value,qv[4].value)))
    q=(word>>(Int32(8)*(byte%Int32(4))+Int32(4)*(j%Int32(2))))&UInt32(15)
    sc,mi=DK.q4k_scalemin(hd,j)
    DK.q4k_half(hd[1].value,0)*Float32(sc)*Float32(q)-DK.q4k_half(hd[1].value,1)*Float32(mi)
end
function embedding_q4!(out,Q,token,width::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    i<width&&(@inbounds out[i+Int32(1)]=q4_embedding_value(Q,Int32(@inbounds token[1]),i,width))
    nothing
end
function embedding_q4_batch!(out,Q,tokens,width::Int32,total::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<total
        row=Int32(@inbounds tokens[i÷width+Int32(1)])
        @inbounds out[i+Int32(1)]=q4_embedding_value(Q,row,i%width,width)
    end
    nothing
end

function R.declare_embedding!(g,out,weight::DK.Q4KMatrix,token,width)
    Q.dispatch(g,embedding_q4!,(out,weight.q,token,Int32(width)),width;name="embedding_q4")
end
function R.declare_embedding_batch!(g,out,weight::DK.Q4KMatrix,tokens,width,n)
    Q.dispatch(g,embedding_q4_batch!,(out,weight.q,tokens,Int32(width),Int32(width*n)),width*n;name="prompt.embedding_q4")
end

for (kernel,writes) in ((q5_gemv!,(1,)),(q5_gemv_tuned!,(1,)),(embedding_q4!,(1,)),(embedding_q4_batch!,(1,)),(fill_int32!,(1,)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end
