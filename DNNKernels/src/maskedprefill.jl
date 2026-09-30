# A dense two-product alternative for long queries. Each head uses the staged
# GEMM path, which the batched register-only GEMM cannot select.
#
# `nq >= 2048` is a judgement, not a divisibility bound, and it is the one line
# here worth revisiting. A grouped-query export folds the group into the query
# extent, so `nq = group * prompt` and the 32B model's 128-token bucket arrives
# as `nq = 1024`. This kernel is faster than flash there too -- 0.122 against
# 0.300 ms a layer recorded in isolation -- but attention is only 2.4% of that
# bucket (`tools/attribute_prefill_ops.jl`: its 64 matmuls are 82%), so the
# whole-graph gain is ~11 ms of 631, under the spread between two recordings of
# the same graph. What is not under the noise is fidelity: against the unfused
# graph the caches land at 0.0209 and 0.0282 relative through this path and
# 0.0170 and 0.0249 through flash. Below `nq = 2048` that trade is not worth
# taking; at and above it the same fidelity buys 3.9% of the 256-token bucket
# and 4.6% of the 512-token one, so it is.
function maskedprefill_applicable(caps::M.DeviceCaps,q,k,v,mask)
    caps.coopmat && caps.coopmatsubgroup == 32 && caps.tile == 16 || return false
    caps.workgrouplimit >= 256 && caps.sharedbudget >= 21504 || return false
    all(a->eltype(a)===Float16,(q,k,v,mask)) && ndims(mask) == 4 || return false
    e,nq,h,b=size(q); nk=size(k,2)
    e==128 && b==1 && nq>=2048 && nq%128==0 && 128<=nk<=1024 && nk%128==0 || return false
    size(mask)==(nk,nq,1,1) || return false
    max(e*nq*h*b,nk*nq*h*b) <= typemax(Int32)
end

function masked_prefill_softmax!(p, s, mask,
                                                  scale, ::Val{NK}, ::Val{NQ}) where {NK,NQ}
    lane = Int32(KI.get_local_id().x)-Int32(1)
    row = Int32(KI.get_group_id().x)-Int32(1)
    vals = StaticArrays.MArray{Tuple{cld(NK,32)}, Float32}(undef)
    mx = -Inf32
    @inbounds for j in 0:cld(NK,32)-1
        key = lane+Int32(32j)
        x = key < NK ? Float32(Float16(Float16(s[1+key+row*Int32(NK)]*scale)+
                        mask[1+key+(row%Int32(NQ))*Int32(NK)])) : -Inf32
        vals[j+1] = x
        mx = max(mx,x)
    end
    Base.Cartesian.@nexprs 5 d -> begin
        delta = Int32(1 << (d-1))
        mx = max(mx,Lava.subgroup_shuffle(mx,UInt32(lane ⊻ Int32(delta))))
    end
    total = 0f0
    @inbounds for j in 1:cld(NK,32)
        x = exp(vals[j]-mx)
        vals[j] = x
        total += x
    end
    Base.Cartesian.@nexprs 5 d -> begin
        delta = Int32(1 << (d-1))
        total += Lava.subgroup_shuffle(total,UInt32(lane ⊻ Int32(delta)))
    end
    @inbounds for j in 0:cld(NK,32)-1
        key = lane+Int32(32j)
        if key < NK
            p[1+key+row*Int32(NK)] = Float16(vals[j+1]/total)
        end
    end
    return nothing
end

function masked_batch_gemm!(C,A,B,
        ::Val{M},::Val{N},::Val{K},as,bs) where {M,N,K}
    sa = KI.localmemory(Mantle.GemmV2, Val((72*32,)), Val(1))
    sb = KI.localmemory(Mantle.GemmV2, Val((24*128,)), Val(2))
    tid=Int32(KI.get_local_id().x)-Int32(1)
    block=Int32(KI.get_group_id().x)-Int32(1)
    batch=block÷Int32((M÷128)*(N÷128))
    tile=block%Int32((M÷128)*(N÷128))
    tm=(tile%Int32(M÷128))*Int32(128)
    tn=(tile÷Int32(M÷128))*Int32(128)
    sg=tid÷Int32(32); sm=(sg%Int32(2))*Int32(4); sn=(sg÷Int32(2))*Int32(2)
    Base.Cartesian.@nexprs 2 j -> Base.Cartesian.@nexprs 4 i ->
        c_i_j=zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
    for kb in Int32(0):Int32(K÷32-1)
        k0=kb*Int32(32)
        @inbounds for r in Int32(0):Int32(7)
            idx=tid+r*Int32(256)
            # Int32 indices (`splitidx` returns `Int`): an Int64 one loses
            # the pair's second load on Lava, as in `q8gemm.jl`.
            p,kk=map(Int32, Mantle.splitidx(idx,Val(64)))
            ai=Int32(1)+tm+Int32(2)*p+(k0+kk)*Int32(M)+batch*as
            sa[Int32(1)+p+kk*Int32(72)]=(VecElement(A[ai]),VecElement(A[ai+Int32(1)]))
            p,j=map(Int32, Mantle.splitidx(idx,Val(16)))
            bi=Int32(1)+k0+Int32(2)*p+(tn+j)*Int32(K)+batch*bs
            sb[Int32(1)+p+j*Int32(24)]=(VecElement(B[bi]),VecElement(B[bi+Int32(1)]))
        end
        KI.barrier()
        Base.Cartesian.@nexprs 2 u -> begin
            kt=Int32((u-1)*16)
            Base.Cartesian.@nexprs 4 i -> begin
                a= Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(
                    sa,Int32(1)+(sm+Int32(i-1))*Int32(8)+kt*Int32(72),Int32(72))
                Base.Cartesian.@nexprs 2 j -> begin
                    b= Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(
                        sb,Int32(1)+kt÷Int32(2)+(sn+Int32(j-1))*Int32(16*24),Int32(24))
                    c_i_j=muladd(a,b,c_i_j)
                end
            end
        end
        KI.barrier()
    end
    Base.Cartesian.@nexprs 2 j -> Base.Cartesian.@nexprs 4 i ->
        Mantle.accstore!(C,Int32(1)+tm+(sm+Int32(i-1))*Int32(16)+
            (tn+(sn+Int32(j-1))*Int32(16))*Int32(M)+batch*Int32(M*N),
            Int32(M),c_i_j,identity)
    return nothing
end

"""
    maskedprefill_dispatch!(emitctx, op, out, q, k, v, mask, scale) -> out

Declare the staged masked attention: `k` transposed, the score product, the
masked softmax and the value product, four passes over dense operands. The
eager form this replaces took the value operand through its strided root; the
declared operands are dense, so its batch stride is `e * nk`.
"""
function maskedprefill_dispatch!(emitctx,op,out,q,k,v,mask,scale)
    e,nq,h,b=size(q); nk=size(k,2)
    kt=scratch(emitctx,Float16,nk,e,h,b)
    s=scratch(emitctx,Float16,nk,nq,h,b)
    p=scratch(emitctx,Float16,nk,nq,h,b)
    transposeLE_dispatch!(emitctx.g,kt,k;name="$(op.id).kt")
    M.dispatch!(emitctx.g,masked_batch_gemm!,
        (s,kt,q,Val(nk),Val(nq),Val(e),Int32(nk*e),Int32(e*nq)),
        (nk÷128)*(nq÷128)*h*b*256;group=256,name="$(op.id).scores")
    M.dispatch!(emitctx.g,masked_prefill_softmax!,
        (p,s,mask,Float32(scale),Val(nk),Val(nq)),
        nq*h*b*32;group=32,name="$(op.id).softmax")
    M.dispatch!(emitctx.g,masked_batch_gemm!,
        (out,v,p,Val(e),Val(nq),Val(nk),Int32(e*nk),Int32(nk*nq)),
        (e÷128)*(nq÷128)*h*b*256;group=256,name="$(op.id).apply")
    out
end
