# Included in Qwen36MoEExperiment. Explicit producer sharing is local to one
# input value/version; never cache by a mutable Buffer's identity across layers.
const KI=Q.KI
struct Q8Input{X,A,B}
    source::X
    q::A
    ds::B
    k::Int
    columns::Int
end
source(x)=x
source(x::Q8Input)=x.source
supports_q8(x)=false
supports_q8(x::DK.KQuantMatrix)=true
supports_q8(x::M.Q4Experts)=true

# A register-resident value adapter used only inside a kernel. Reuse the exact
# Q8 producer below without storing and reloading its activation input.
struct LaneValue
    value::Float32
end
@inline Base.getindex(x::LaneValue, ::Integer) = x.value

function swiglu_q8!(act,q,ds,gate,up,status,k::Int32,columns::Int32,topk::Int32,
                    ::Val{PACKED},::Val{CHECK},::Val{PER},::Val{SG}) where {PACKED,CHECK,PER,SG}
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<k*columns
        col=i÷k;token=PER ? col÷topk : col
        okay=CHECK ? (@inbounds status[token+Int32(1)]==0) : true
        value=0f0
        if okay
            if PACKED
                base=col*k*Int32(2)+i%k+Int32(1)
                @inbounds a=Float32(gate[base])
                @inbounds b=Float32(gate[base+k])
                e=exp(-abs(a));sig=a>=0f0 ? inv(1f0+e) : e/(1f0+e)
                value=(a*sig)*b
            else
                @inbounds a=Float32(gate[i+Int32(1)])
                @inbounds b=Float32(up[i+Int32(1)])
                value=a/(1f0+exp(-a))*b
            end
        end
        @inbounds act[i+Int32(1)]=value
        parallel_q8!(q,ds,LaneValue(value),status,k,columns,topk,Val(CHECK),Val(PER),Val(SG))
    end
    nothing
end

function prepare_swiglu_q8!(g,act,gate,up,k,columns;
                            packed=false,status=nothing,topk=1,per_expert=false,name)
    k>0&&k%256==0&&columns>0&&length(act)==k*columns||throw(DimensionMismatch("SwiGLU Q8 extent"))
    length(gate)==(packed ? 2k*columns : k*columns)&&
        (packed||length(up)==length(act))||throw(DimensionMismatch("SwiGLU input extent"))
    sg=Mantle.caps(g.dev).subgroup
    sg>=32&&sg%32==0||throw(ArgumentError("SwiGLU Q8 requires whole 32-lane subgroups"))
    k*columns<=typemax(Int32)||throw(ArgumentError("SwiGLU Q8 addressing"))
    if status!==nothing
        topk>0&&(!per_expert||columns%topk==0)&&length(status)==(per_expert ? columns÷topk : columns)||throw(DimensionMismatch("SwiGLU status extent"))
    end
    q=Mantle.Transient.Buffer(g,DK.UVec4,(k÷16,columns))
    ds=Mantle.Transient.Buffer(g,Float32,(k÷16,columns))
    Q.dispatch(g,swiglu_q8!,
        (act,q,ds,gate,up,status===nothing ? ds : status,Int32(k),Int32(columns),Int32(topk),
         Val(packed),Val(status!==nothing),Val(per_expert),Val(sg)),k*columns;name)
    Q8Input(act,q,ds,k,columns)
end

function parallel_q8!(q,ds,x,status,k::Int32,columns::Int32,topk::Int32,::Val{CHECK},::Val{PER},::Val{SG}) where {CHECK,PER,SG}
    i=Int32(KI.get_global_id().x)-Int32(1);lane=i&Int32(31)
    if i<k*columns
        block=i÷Int32(32);col=i÷k;token=PER ? col÷topk : col
        okay=CHECK ? (@inbounds status[token+Int32(1)]==0) : true
        value=okay ? Float32(@inbounds x[i+Int32(1)]) : 0f0
        base=(i%Int32(SG))&Int32(-32);amax=abs(value);shift=Int32(16)
        while shift>0
            amax=max(amax,KI.shfl(amax,base+(lane⊻shift)+Int32(1)));shift>>=1
        end
        d=amax/127f0;inverse=d==0f0 ? 0f0 : 1f0/d
        quant=DK.q4k_round(value*inverse);sum=quant;shift=Int32(16)
        while shift>0
            sum+=KI.shfl(sum,base+(lane⊻shift)+Int32(1));shift>>=1
        end
        quad=base+(lane&Int32(-4))+Int32(1)
        a=KI.shfl(quant,quad);b=KI.shfl(quant,quad+Int32(1));c=KI.shfl(quant,quad+Int32(2));e=KI.shfl(quant,quad+Int32(3))
        word=DK.q4k_bytes(a%UInt8,b%UInt8,c%UInt8,e%UInt8)
        half=base+(lane&Int32(-16))+Int32(1)
        w0=KI.shfl(word,half);w1=KI.shfl(word,half+Int32(4));w2=KI.shfl(word,half+Int32(8));w3=KI.shfl(word,half+Int32(12))
        if lane%Int32(16)==0
            @inbounds q[block*Int32(2)+lane÷Int32(16)+Int32(1)]=(VecElement(w0),VecElement(w1),VecElement(w2),VecElement(w3))
        end
        if lane==0
            @inbounds begin ds[block*Int32(2)+Int32(1)]=d;ds[block*Int32(2)+Int32(2)]=d*Float32(sum);end
        end
    end
    nothing
end

function prepare_q8!(g,x,k,columns;status=nothing,topk=1,per_expert=false,name,parallel=false)
    k>0&&k%256==0&&columns>0&&length(x)==k*columns||throw(DimensionMismatch("Q8 input extent"))
    k*columns<=typemax(Int32)||throw(ArgumentError("Q8 input addressing"))
    if status!==nothing
        topk>0&&(!per_expert||columns%topk==0)&&length(status)==(per_expert ? columns÷topk : columns)||throw(DimensionMismatch("Q8 status extent"))
    end
    q=Mantle.Transient.Buffer(g,DK.UVec4,(k÷16,columns))
    ds=Mantle.Transient.Buffer(g,Float32,(k÷16,columns))
    n=columns*(k÷32)
    if parallel&&Mantle.caps(g.dev).subgroup>=32
        Q.dispatch(g,parallel_q8!,(q,ds,x,status===nothing ? ds : status,Int32(k),Int32(columns),Int32(topk),Val(status!==nothing),Val(per_expert),Val(Mantle.caps(g.dev).subgroup)),k*columns;name)
    elseif status===nothing
        Q.dispatch(g,DK.q8act_kernel!,(q,ds,x,Int32(k),Int32(n)),n;name)
    else
        Q.dispatch(g,M.checked_q8act!,
            (q,ds,x,status,Int32(k),Int32(n),Int32(topk),Val(per_expert)),n;name)
    end
    Q8Input(x,q,ds,k,columns)
end

function maybe_prepare_q8!(g,x,weights;optimizations=(),status=nothing,name)
    :reuse_q8 in optimizations&&any(supports_q8,weights)||return x
    prepare_q8!(g,x,length(x),1;status,name,parallel=:parallel_q8 in optimizations)
end

# Unroll loads while preserving the original64-lane FMA/reduction order. Router
# rounding can change later activation quantization, so algebraic equivalence is
# not sufficient here: full-model bit identity is checked separately.
dense_small!(args...)=dense_small_tuned!(args...,Val(256))
dense_small_tuned!(args...)=dense_small_unrolled!(args...,Val(4))
function dense_small_unrolled!(out,W,x,k::Int32,m::Int32,::Val{LANES},::Val{SG},::Val{WG},::Val{UNROLL}) where {LANES,SG,WG,UNROLL}
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(LANES)
    row=(Int32(KI.get_group_id().x)-Int32(1))*Int32(WG÷LANES)+lid÷Int32(LANES)
    a0=0f0
    if row<m
        i=lane;base=row*k
        if UNROLL==32&&k==Int32(2048)
            Base.Cartesian.@nexprs 32 j -> a0=muladd((@inbounds W[base+i+Int32((j-1)*LANES+1)]),(@inbounds x[i+Int32((j-1)*LANES+1)]),a0)
            i+=Int32(2048)
        end
        while i+Int32((UNROLL-1)*LANES)<k
            if UNROLL==32
                Base.Cartesian.@nexprs 32 j -> a0=muladd((@inbounds W[base+i+Int32((j-1)*LANES+1)]),(@inbounds x[i+Int32((j-1)*LANES+1)]),a0)
            elseif UNROLL==16
                Base.Cartesian.@nexprs 16 j -> a0=muladd((@inbounds W[base+i+Int32((j-1)*LANES+1)]),(@inbounds x[i+Int32((j-1)*LANES+1)]),a0)
            else
                Base.Cartesian.@nexprs 4 j -> a0=muladd((@inbounds W[base+i+Int32((j-1)*LANES+1)]),(@inbounds x[i+Int32((j-1)*LANES+1)]),a0)
            end
            i+=Int32(UNROLL*LANES)
        end
        while i<k
            @inbounds a0=muladd(W[base+i+Int32(1)],x[i+Int32(1)],a0)
            i+=Int32(LANES)
        end
    end
    acc=KI.sub_group_reduce_add(a0)
    if SG==LANES
        lane==0&&row<m&&(@inbounds out[row+Int32(1)]=acc)
    else
        partial=KI.localmemory(Float32,Val((WG÷SG,)),Val(1))
        lid%Int32(SG)==0&&(@inbounds partial[lid÷Int32(SG)+Int32(1)]=acc)
        KI.barrier()
        if lane==0&&row<m
            total=0f0;base=(lid÷Int32(LANES))*Int32(LANES÷SG)
            for j in Int32(1):Int32(LANES÷SG);total+=@inbounds partial[base+j];end
            @inbounds out[row+Int32(1)]=total
        end
    end
    nothing
end

function linear_opt!(g,out,A,x;name,optimizations=())
    Q.linear!(g,out,A,source(x);name)
end

# A 64-lane logical row on a 32-lane device: each physical lane keeps the
# original low/high accumulators separate, then reproduces the same two
# subgroup sums and ordered addition. No cross-subgroup shared-memory barrier.
function dense_small_pair32!(out,W,x,k::Int32,m::Int32,::Val{WG}) where WG
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(32)
    row=(Int32(KI.get_group_id().x)-Int32(1))*Int32(WG÷32)+lid÷Int32(32)
    a=0f0;b=0f0
    if row<m
        base=row*k;i=lane
        if k==Int32(2048)
            Base.Cartesian.@nexprs 32 j -> begin
                a=muladd((@inbounds W[base+i+Int32((j-1)*64+1)]),(@inbounds x[i+Int32((j-1)*64+1)]),a)
                b=muladd((@inbounds W[base+i+Int32((j-1)*64+33)]),(@inbounds x[i+Int32((j-1)*64+33)]),b)
            end
        else
            while i<k
                a=muladd((@inbounds W[base+i+Int32(1)]),(@inbounds x[i+Int32(1)]),a)
                if i+Int32(32)<k
                    b=muladd((@inbounds W[base+i+Int32(33)]),(@inbounds x[i+Int32(33)]),b)
                end
                i+=Int32(64)
            end
        end
    end
    lo=KI.sub_group_reduce_add(a);hi=KI.sub_group_reduce_add(b)
    if lane==0&&row<m
        total=0f0;total+=lo;total+=hi
        @inbounds out[row+Int32(1)]=total
    end
    nothing
end
function linear_opt!(g,out,A::Q.DenseWeight,x;name,optimizations=())
    if :small_dense in optimizations&&A.m<=260&&A.k%4==0&&eltype(A.data)==Float32
        lanes=64;sg=Mantle.caps(g.dev).subgroup
        if :capability_dense in optimizations&&sg==32
            wg=min(128,Mantle.caps(g.dev).workgrouplimit)
            wg>=32&&wg%32==0||throw(ArgumentError("Dense subgroup workgroup extent"))
            Q.dispatch(g,dense_small_pair32!,(out,A.data,source(x),Int32(A.k),Int32(A.m),Val(wg)),
                cld(A.m,wg÷32)*wg;group=wg,name)
            return nothing
        end
        wg=:router_wg64 in optimizations&&A.m>1 ? 64 : 256
        unroll=:dense_unroll32 in optimizations ? 32 : :dense_unroll16 in optimizations ? 16 : 4
        Q.dispatch(g,dense_small_unrolled!,(out,A.data,source(x),Int32(A.k),Int32(A.m),Val(lanes),Val(sg),Val(wg),Val(unroll)),
            cld(A.m,wg÷lanes)*wg;group=wg,name)
    else
        Q.linear!(g,out,A,source(x);name)
    end
end

"""Concatenate router and shared scalar rows without changing row arithmetic."""
function prepare_router_gate!(model)
    for il in 0:Int(model.config.num_hidden_layers)-1
        p="layers.$il.mlp.";key=p*"router_and_shared_gate"
        haskey(model.weights,key)&&continue
        router=model.weights[p*"gate.weight"];scalar=model.weights[p*"shared_expert_gate.weight"]
        if router isa Q.DenseWeight&&scalar isa Q.DenseWeight&&router.k==scalar.k&&scalar.m==1
            b=Mantle.Buffer(model.device,hcat(Array(router.data),Array(scalar.data)))
            push!(model.owned,b);model.weights[key]=Q.DenseWeight(b,router.m+1,router.k)
        end
    end
    model
end

# Keep the original256-thread tree order, replacing only its within-subgroup
# shared-memory/barrier stages with shuffles. Optional residual addition occurs
# before the same FP32 FMA as a separate add followed by RMS normalization.
boundary_norm!(out,x,branch,weight,n,eps,add,sg)=boundary_norm_impl!(out,x,branch,weight,out,out,n,eps,add,sg,Val(0),Val(false))
boundary_norm_q8!(out,x,branch,weight,q,ds,n,eps,add,sg,width)=boundary_norm_impl!(out,x,branch,weight,q,ds,n,eps,add,sg,width,Val(true))
function boundary_norm_impl!(out,x,branch,weight,q,ds,n::Int32,eps::Float32,::Val{ADD},::Val{SG},::Val{WIDTH},::Val{Q8}) where {ADD,SG,WIDTH,Q8}
    sh=KI.localmemory(Float32,Val((256,)),Val(1));t=Int32(KI.get_local_id().x)-Int32(1)
    total=0f0;i=t
    while i<n
        @inbounds v=x[i+Int32(1)]
        if ADD
            @inbounds begin v+=branch[i+Int32(1)];x[i+Int32(1)]=v;end
        end
        total=muladd(v,v,total);i+=Int32(256)
    end
    @inbounds sh[t+Int32(1)]=total
    KI.barrier();stride=Int32(128)
    while stride>=Int32(SG)
        t<stride&&(@inbounds sh[t+Int32(1)]+=sh[t+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    if t<Int32(SG)
        @inbounds value=sh[t+Int32(1)]
        while stride>0
            value+=KI.shfl_down(value,stride);stride>>=1
        end
        t==0&&(@inbounds sh[Int32(1)]=value)
    end
    KI.barrier();scale=inv(sqrt((@inbounds sh[Int32(1)])/Float32(n)+eps));i=t
    if Q8
        normalized=KI.localmemory(Float32,Val((WIDTH,)),Val(2))
        while i<n
            @inbounds begin
                value=x[i+Int32(1)]*scale*weight[i+Int32(1)]
                out[i+Int32(1)]=value;normalized[i+Int32(1)]=value
            end
            i+=Int32(256)
        end
        KI.barrier()
        DK.q8act_kernel!(q,ds,normalized,n,n÷Int32(32))
    else
        while i<n
            @inbounds out[i+Int32(1)]=x[i+Int32(1)]*scale*weight[i+Int32(1)]
            i+=Int32(256)
        end
    end
    nothing
end
function linear_opt!(g,out,A::DK.KQuantMatrix,x::Q8Input;name,optimizations=())
    m,k=size(A);k==x.k&&x.columns==1||throw(DimensionMismatch("Shared Q8 extent"))
    r=DK.q4krows(1)
    Q.dispatch(g,DK.kqgemv_kernels(A)[(1,r)],
        (out,DK.kqarrays(A)...,x.q,x.ds,out,identity,Int32(m),Int32(k÷256),Val(false)),
        cld(cld(m,r),DK.Q4K_WG÷32)*DK.Q4K_WG;group=DK.Q4K_WG,name)
end
function linear_opt!(g,out,A::DK.KQuantMatrix,x;name,optimizations=())
    :parallel_q8 in optimizations||return Q.linear!(g,out,A,x;name)
    qx=prepare_q8!(g,x,size(A,2),1;name=name*".q8",parallel=true)
    linear_opt!(g,out,A,qx;name,optimizations)
end

@inline function sum_row(acc,::Val{LANES}) where LANES
    LANES>=32&&(acc+=KI.shfl_down(acc,16))
    LANES>=16&&(acc+=KI.shfl_down(acc,8))
    acc+=KI.shfl_down(acc,4);acc+=KI.shfl_down(acc,2);acc+=KI.shfl_down(acc,1)
    acc
end

# Same packed Q4 arithmetic, but use8/16/32 lanes for256/512/>=1024 inputs.
function routed_q4_narrow!(args...)
    routed_q4_tuned!(args...,Val(256))
end
function routed_q4_tuned!(out,W,xq,ds,selected,status,m::Int32,ne::Int32,topk::Int32,
                           rows::Int32,nb::Int32,::Val{PER},::Val{LANES},::Val{SG},::Val{WG}) where {PER,LANES,SG,WG}
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(LANES)
    row=(Int32(KI.get_group_id().x)-Int32(1))*Int32(WG÷LANES)+lid÷Int32(LANES)
    safe=min(row,rows-Int32(1));slot=safe÷m;token=slot÷topk
    expert=@inbounds selected[slot+Int32(1)];okay=@inbounds status[token+Int32(1)]==0
    acc=0f0
    if row<rows&&okay&&0<=expert<ne
        physical=expert*m+safe%m;col=PER ? slot : token;offset=col*nb*Int32(16)
        p=lane%Int32(8);j=p÷Int32(2);h=p%Int32(2);b=lane÷Int32(8)
        while b<nb
            base=(physical*nb+b)*Int32(DK.Q4K_VECS)
            @inbounds hd=W[base+Int32(1)]
            @inbounds qv=W[base+Int32(2)+Int32(2)*j+h]
            d=DK.q4k_half(hd[1].value,0);dm=DK.q4k_half(hd[1].value,1)
            sl,ml=DK.q4k_scalemin(hd,Int32(2)*j);sh,mh=DK.q4k_scalemin(hd,Int32(2)*j+Int32(1))
            xv=offset+b*Int32(16)+Int32(4)*j+h;xs=offset+(b*Int32(8)+Int32(2)*j)*Int32(2)
            @inbounds begin
                lo=xq[xv+Int32(1)];hi=xq[xv+Int32(3)]
                dl=ds[xs+Int32(1)];smlo=ds[xs+Int32(2)];dh=ds[xs+Int32(3)];smhi=ds[xs+Int32(4)]
            end
            il=DK.q4k_dot(qv,lo,DK.q4k_lo);ih=DK.q4k_dot(qv,hi,DK.q4k_hi)
            acc+=d*(Float32(sl)*dl*Float32(il)+Float32(sh)*dh*Float32(ih))-
                (h==0 ? 1f0 : 0f0)*dm*(Float32(ml)*smlo+Float32(mh)*smhi)
            b+=Int32(LANES÷8)
        end
    end
    if SG>=LANES
        acc=sum_row(acc,Val(LANES))
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
    lane==0&&row<rows&&(@inbounds out[row+Int32(1)]=okay&&!(0<=expert<ne) ? NaN32 : acc)
    nothing
end

routed_opt!(g,out,W,x,selected,status;name,per_expert=false,optimizations=())=
    M.declare_routed_linear!(g,out,W,source(x),selected,status;name,per_expert)
function routed_opt!(g,out,W::M.Q4Experts,x,selected,status;name,per_expert=false,optimizations=())
    if !(x isa Q8Input)&&!(:narrow_experts in optimizations)
        return M.declare_routed_linear!(g,out,W,x,selected,status;name,per_expert)
    end
    k,m,ne=size(W);topk,tokens=size(selected);columns=per_expert ? topk*tokens : tokens
    1<=topk<=ne&&size(out)==(m,topk,tokens)&&length(status)==tokens||throw(DimensionMismatch("Routed Q4 outputs/status"))
    length(W.packed.q)<=typemax(Int32)&&length(out)<=typemax(Int32)||throw(ArgumentError("Routed Q4 addressing"))
    qx=x isa Q8Input ? x : prepare_q8!(g,x,k,columns;status,topk,per_expert,name=name*".q8",parallel=:parallel_q8 in optimizations)
    qx.k==k&&qx.columns==columns||throw(DimensionMismatch("Routed shared Q8 extent"))
    lanes=:narrow_experts in optimizations ? min(32,k÷32) : 32
    wg=:down_wg64 in optimizations&&per_expert&&k==512 ? 64 : 256
    if :paired_expert_rows in optimizations&&!per_expert&&k>=1024&&m%2==0&&Mantle.caps(g.dev).subgroup>=32
        caps=Mantle.caps(g.dev);wg=min(4caps.subgroup,256,caps.workgrouplimit)
        wg%32==0||throw(ArgumentError("Paired expert workgroup extent"))
        Q.dispatch(g,routed_q4_pair_general!,
            (out,W.packed.q,qx.q,qx.ds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(out)),Int32(k÷256),Val(wg)),
            cld(length(out),2*(wg÷32))*wg;group=wg,name)
        return nothing
    end
    if :down_rows2 in optimizations&&per_expert&&k==512&&m%2==0&&Mantle.caps(g.dev).subgroup>=16
        args=(out,W.packed.q,qx.q,qx.ds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(out)),Val(Mantle.caps(g.dev).subgroup))
        Q.dispatch(g,routed_q4_rows2!,args,cld(length(out),8)*64;group=64,name)
        return nothing
    end
    kernel=:narrow_experts in optimizations ? routed_q4_narrow! : M.routed_q4_gemv!
    args=(out,W.packed.q,qx.q,qx.ds,selected,status,Int32(m),Int32(ne),Int32(topk),Int32(length(out)),Int32(k÷256),Val(per_expert))
    if :narrow_experts in optimizations
        Q.dispatch(g,routed_q4_tuned!,(args...,Val(lanes),Val(Mantle.caps(g.dev).subgroup),Val(wg)),cld(length(out),wg÷lanes)*wg;group=wg,name)
    else
        Q.dispatch(g,kernel,(args...,Val(Mantle.caps(g.dev).subgroup)),cld(length(out),8)*256;name)
    end
end

@inline function q4_row_term(W,row,b,j,h,lo,hi,dl,slo,dh,shi)
    q4_row_term(W,row,b,j,h,lo,hi,dl,slo,dh,shi,Int32(2))
end
@inline function q4_row_term(W,row,b,j,h,lo,hi,dl,slo,dh,shi,nb::Int32)
    base=(row*nb+b)*Int32(DK.Q4K_VECS)
    @inbounds begin hd=W[base+Int32(1)];qv=W[base+Int32(2)+Int32(2)*j+h];end
    d=DK.q4k_half(hd[1].value,0);dm=DK.q4k_half(hd[1].value,1)
    sl,ml=DK.q4k_scalemin(hd,Int32(2)*j);sh,mh=DK.q4k_scalemin(hd,Int32(2)*j+Int32(1))
    il=DK.q4k_dot(qv,lo,DK.q4k_lo);ih=DK.q4k_dot(qv,hi,DK.q4k_hi)
    d*(Float32(sl)*dl*Float32(il)+Float32(sh)*dh*Float32(ih))-
        (h==0 ? 1f0 : 0f0)*dm*(Float32(ml)*slo+Float32(mh)*shi)
end

function routed_q4_pair_general!(out,W,xq,ds,selected,status,m::Int32,ne::Int32,topk::Int32,
                                  rows::Int32,nb::Int32,::Val{WG}) where WG
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(32)
    row=((Int32(KI.get_group_id().x)-Int32(1))*Int32(WG÷32)+lid÷Int32(32))*Int32(2)
    safe=min(row,rows-Int32(2));slot=safe÷m;token=slot÷topk
    @inbounds begin expert=selected[slot+Int32(1)];okay=status[token+Int32(1)]==0;end
    a=0f0;c=0f0
    if row<rows&&okay&&0<=expert<ne
        physical=expert*m+safe%m;p=lane%Int32(8);j=p÷Int32(2);h=p%Int32(2);b=lane÷Int32(8)
        offset=token*nb*Int32(16)
        while b<nb
            xv=offset+b*Int32(16)+Int32(4)*j+h;xs=offset+(b*Int32(8)+Int32(2)*j)*Int32(2)
            @inbounds begin lo=xq[xv+Int32(1)];hi=xq[xv+Int32(3)];dl=ds[xs+Int32(1)];slo=ds[xs+Int32(2)];dh=ds[xs+Int32(3)];shi=ds[xs+Int32(4)];end
            a+=q4_row_term(W,physical,b,j,h,lo,hi,dl,slo,dh,shi,nb)
            c+=q4_row_term(W,physical+Int32(1),b,j,h,lo,hi,dl,slo,dh,shi,nb)
            b+=Int32(4)
        end
    end
    a=sum_row(a,Val(32));c=sum_row(c,Val(32))
    if lane==0&&row<rows
        bad=okay&&!(0<=expert<ne)
        @inbounds begin out[row+Int32(1)]=bad ? NaN32 : a;out[row+Int32(2)]=bad ? NaN32 : c;end
    end
    nothing
end

# Two output rows share one activation load. Each row keeps the exact original
#16-lane arithmetic; even row counts keep a pair inside one selected expert.
function routed_q4_rows2!(out,W,xq,ds,selected,status,m::Int32,ne::Int32,topk::Int32,rows::Int32,::Val{SG}) where SG
    lid=Int32(KI.get_local_id().x)-Int32(1);lane=lid%Int32(16)
    row=((Int32(KI.get_group_id().x)-Int32(1))*Int32(4)+lid÷Int32(16))*Int32(2)
    safe=min(row,rows-Int32(2));slot=safe÷m;token=slot÷topk
    @inbounds begin expert=selected[slot+Int32(1)];okay=status[token+Int32(1)]==0;end
    a=0f0;c=0f0
    if row<rows&&okay&&0<=expert<ne
        physical=expert*m+safe%m;p=lane%Int32(8);j=p÷Int32(2);h=p%Int32(2);b=lane÷Int32(8)
        offset=slot*Int32(32);xv=offset+b*Int32(16)+Int32(4)*j+h;xs=offset+(b*Int32(8)+Int32(2)*j)*Int32(2)
        @inbounds begin lo=xq[xv+Int32(1)];hi=xq[xv+Int32(3)];dl=ds[xs+Int32(1)];slo=ds[xs+Int32(2)];dh=ds[xs+Int32(3)];shi=ds[xs+Int32(4)];end
        a+=q4_row_term(W,physical,b,j,h,lo,hi,dl,slo,dh,shi)
        c+=q4_row_term(W,physical+Int32(1),b,j,h,lo,hi,dl,slo,dh,shi)
    end
    a=sum_row(a,Val(16));c=sum_row(c,Val(16))
    if lane==0&&row<rows
        bad=okay&&!(0<=expert<ne)
        @inbounds begin out[row+Int32(1)]=bad ? NaN32 : a;out[row+Int32(2)]=bad ? NaN32 : c;end
    end
    nothing
end

"""Optional load-time row concatenation; no requantization or precision change."""
function prepare_shared_gateup!(model)
    for il in 0:Int(model.config.num_hidden_layers)-1
        p="layers.$il.mlp.shared_expert.";key=p*"gate_up_proj"
        haskey(model.weights,key)&&continue
        gate=model.weights[p*"gate_proj.weight"];up=model.weights[p*"up_proj.weight"]
        if gate isa DK.Q4KMatrix&&up isa DK.Q4KMatrix&&size(gate)==size(up)
            m,k=size(gate);data=vcat(vec(Array(gate.q)),vec(Array(up.q)))
            buffer=Mantle.Buffer(model.device,data);push!(model.owned,buffer)
            model.weights[key]=DK.Q4KMatrix(buffer,2m,k)
        elseif gate isa Q.DenseWeight&&up isa Q.DenseWeight&&gate.m==up.m&&gate.k==up.k
            buffer=Mantle.Buffer(model.device,hcat(Array(gate.data),Array(up.data)));push!(model.owned,buffer)
            model.weights[key]=Q.DenseWeight(buffer,2*gate.m,gate.k)
        end
    end
    model
end

for (kernel,writes) in ((swiglu_q8!,(1,2,3)),(parallel_q8!,(1,2)),(dense_small_unrolled!,(1,)),
                        (dense_small_pair32!,(1,)),(routed_q4_tuned!,(1,)),(routed_q4_pair_general!,(1,)),
                        (routed_q4_rows2!,(1,)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end
# The fused residual also writes the residual stream back.
DK.kernelresults(::typeof(boundary_norm!),args::Tuple)=args[7]===Val(true) ? (1,2) : (1,)
DK.kernelresults(::typeof(boundary_norm_q8!),args::Tuple)=args[9]===Val(true) ? (1,2,5,6) : (1,5,6)
