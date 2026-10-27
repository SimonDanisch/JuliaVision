# Included in Qwen35Experiment after flash_decode_qwen35.jl. Causal attention
# for a whole prompt chunk on cooperative matrices: every key block is read from
# the cache once per chunk rather than once per token, which at 128k positions
# was two thirds of a chunk's time.
#
# DNNKernels' `attn_flash_rows!` structure at head width 256. A 16-row tile is
# two tokens times the eight query heads of one KV head, so a K/V block staged
# in shared memory serves every head of the group. Two subgroups share a tile
# and split the head width: each holds half of Q (8 A fragments) and half of O
# (8 accumulators), which is what `attn_flash_rows!` holds at width 128. Their
# half scores meet in shared memory, each subgroup runs the softmax for eight of
# the sixteen rows, and both read the whole P tile back.
#
# Staging converts whatever the cache holds (Float32, Float16, q8_0, q4_0) into
# Float16, the only operand type the matrix units take, so Q is packed to
# Float16 too. That costs bit identity with decode, which keeps Float32 scores.

const PREFILL_H4=NTuple{4,VecElement{Float16}}
const PREFILL_BC=32          # keys per block: two score tiles
const PREFILL_KS4=66         # H4 per staged key row: 256 halves and 8 of padding
const PREFILL_VT4=10         # H4 per row of transposed V: 32 keys and 8 of padding

# Q for the prefill attention: Float16, rows ordered (KV head, token, query head
# of its group), so the sixteen rows of a tile are one matrix of stride 256.
function pack_prefill_q!(qh,q,n::Int32,::Val{NQ},::Val{R}) where {NQ,R}
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<n*Int32(256NQ)
        e=i%Int32(256);h=(i÷Int32(256))%Int32(NQ);t=i÷Int32(256NQ)
        kh=h÷Int32(R);hh=h%Int32(R)
        @inbounds qh[((kh*n+t)*Int32(R)+hh)*Int32(256)+e+Int32(1)]=Float16(q[i+Int32(1)])
    end
    nothing
end

# Sixteen values of cache row (p, kh) from e16, as four H4.
@inline function stageunit(c::AbstractArray{Float16},sc,p::Int32,nk::Int32,kh::Int32,e16::Int32,::FloatRows)
    ptr=reinterpret(Ptr{PREFILL_H4},pointer(c)+2*Int((p*nk+kh)*Int32(256)+e16))
    (Core.Intrinsics.pointerref(ptr,1,16),Core.Intrinsics.pointerref(ptr,2,8),
     Core.Intrinsics.pointerref(ptr,3,16),Core.Intrinsics.pointerref(ptr,4,8))
end
@inline tohalf4(x::NTuple{4,VecElement{Float32}})=
    (VecElement(Float16(x[1].value)),VecElement(Float16(x[2].value)),VecElement(Float16(x[3].value)),VecElement(Float16(x[4].value)))
@inline function stageunit(c::AbstractArray{Float32},sc,p::Int32,nk::Int32,kh::Int32,e16::Int32,::FloatRows)
    c4=reinterpret(NTuple{4,VecElement{Float32}},c)
    b=((p*nk+kh)*Int32(256)+e16)÷Int32(4)
    @inbounds (tohalf4(c4[b+Int32(1)]),tohalf4(c4[b+Int32(2)]),tohalf4(c4[b+Int32(3)]),tohalf4(c4[b+Int32(4)]))
end
# Four int8 of a word as halves, times the block's scale. `Float16(s * x)` would
# narrow a float product, which Lava spells as integer rounding (a driver may fuse
# the product into the conversion), twenty operations a value: the q8_0 cache
# staged at a fifth of the Float16 one's speed. Here no value is narrowed. A byte
# with its sign bit flipped is x + 128, and as the low mantissa of the half
# 0x6400 (1024) it is the half 1152 + x, so subtracting 1152 gives x exactly; the
# one rounding is the product with the scale, in half precision. The scale's own
# narrowing is once a sixteen.
@inline q8half(w::UInt32,i::Int,hs::Float16)=
    VecElement((reinterpret(Float16,UInt16(0x6400|(((w>>UInt32(8i))&0x000000ff)⊻0x00000080)))-Float16(1152))*hs)
@inline q8half4(w::UInt32,hs::Float16)=(q8half(w,0,hs),q8half(w,1,hs),q8half(w,2,hs),q8half(w,3,hs))
@inline function stageunit(c,sc,p::Int32,nk::Int32,kh::Int32,e16::Int32,::Q8Rows)
    c4=reinterpret(NTuple{4,VecElement{UInt32}},c)
    row=p*nk+kh
    w=@inbounds c4[row*Int32(16)+e16÷Int32(16)+Int32(1)]
    hs=Float16(@inbounds sc[row*Int32(8)+e16÷Int32(32)+Int32(1)])
    (q8half4(w[1].value,hs),q8half4(w[2].value,hs),q8half4(w[3].value,hs),q8half4(w[4].value,hs))
end
# The same for four-bit values: 0x6400 with the stored nibble q as its low
# mantissa is the half 1024 + q, so subtracting 1032 gives q - 8 exactly.
@inline q4half(w::UInt32,i::Int,hs::Float16)=
    VecElement((reinterpret(Float16,UInt16(0x6400|((w>>UInt32(4i))&0x0000000f)))-Float16(1032))*hs)
@inline q4half4(w::UInt32,o::Int,hs::Float16)=(q4half(w,o,hs),q4half(w,o+1,hs),q4half(w,o+2,hs),q4half(w,o+3,hs))
@inline function stageunit(c,sc,p::Int32,nk::Int32,kh::Int32,e16::Int32,::Q4Rows)
    c2=reinterpret(NTuple{2,VecElement{UInt32}},c)
    row=p*nk+kh
    w=@inbounds c2[row*Int32(16)+e16÷Int32(16)+Int32(1)]
    hs=Float16(@inbounds sc[row*Int32(8)+e16÷Int32(32)+Int32(1)])
    (q4half4(w[1].value,0,hs),q4half4(w[1].value,4,hs),q4half4(w[2].value,0,hs),q4half4(w[2].value,4,hs))
end
@inline function storeunit!(kv,at::Int32,u)
    @inbounds begin kv[at]=u[1];kv[at+Int32(1)]=u[2];kv[at+Int32(2)]=u[3];kv[at+Int32(3)]=u[4];end
    nothing
end

"""
    attention_prefill!(out, qh, gate, kc, vc, ksc, vsc, start, n, nk, splits, pm, ps, po, Val(R), Val(NP), Val(QS), kf, vf, Val(PARTIAL), Val(VT))

`out[:, h, t] = gate[:, h, t] .* softmax(K[:, 0:start+t]ᵀ q[:, h, t] / 16) V` for
the `n` tokens of a chunk that starts at position `start[1]`, whose keys and
values are already in the cache. `qh` is `pack_prefill_q!`'s. One workgroup is
`NP` tiles of one KV head, `QS` subgroups a tile, over one of `splits` equal
spans of the keys. With `PARTIAL` a workgroup leaves its rows' maximum (base 2),
sum and unnormalised values in `pm`, `ps` and `po` for
`attention_prefill_combine!`; without it `splits` is 1 and it writes `out`.
A short chunk deep in a context is a handful of tiles: four tokens are two
workgroups for the whole device, which the key split spreads out again.

The `QS` subgroups of a tile split the head width: each holds `256/QS` columns of
Q and of O. At `QS = 2` that is 64 registers of Q fragments and 64 of O, and with
the scores, the next block's K or V in flight and P the kernel needs 277-342
registers: RADV spills 21-86 of them. `QS = 4` halves both and spills nothing
(192 registers at `NP = 4`).

What decides the speed is the rows a workgroup has, because every workgroup
reads its KV head's whole cache: 32 rows make 64 FLOP a Float16 byte, ~14
TFLOP/s at the bus's 215 GB/s, which is where `NP = 2, QS = 2` ran (12). GPU
time at depth 96k, 512 tokens, interleaved: Float16 67.9 ms at 32 rows, 58.5 at
64 (`NP = 4, QS = 4`) or 128 (`NP = 8, QS = 2`, which spills); q8_0 96.8, 67.1,
64.0. Neither the spills nor the V reads below changed the time at 32 rows.

With `VT` V is staged transposed. A P·V operand is sixteen keys of one column;
from key-major rows that is sixteen 2-byte reads a fragment, 256 a key block
for 32 matrix products (counted in the ISA), and transposed it is four 8-byte
reads. A thread then stages four consecutive keys of one 16-column slice and
writes them as sixteen 4-key groups, which needs 128 threads (`NP * QS == 4`).
It measured the same as row-major V (69.5 against 69.0 ms, Float16, 32 rows).
"""
function attention_prefill!(out,qh,gate,kc,vc,ksc,vsc,start,n::Int32,nk::Int32,splits::Int32,pm,ps,po,
        ::Val{R},::Val{NP},::Val{QS},kf,vf,::Val{PARTIAL},::Val{VT}) where {R,NP,QS,PARTIAL,VT}
    # Sizes from the type parameters, never from a local (see `attn_flash_rows!`).
    kv=KI.localmemory(PREFILL_H4,Val((max(PREFILL_BC*PREFILL_KS4,VT ? 256*PREFILL_VT4 : 0),)),Val(1))   # K, then V
    sw=KI.localmemory(Float32,Val((NP*QS*16*33,)),Val(2))                 # partial scores, (r, c) at r*33 + c
    pw=KI.localmemory(PREFILL_H4,Val((NP*16*8,)),Val(3))                  # P, (r, c) at r*32 + c
    aw=KI.localmemory(Float32,Val((NP*16,)),Val(4))                       # a row's factor
    NT=Int32(32QS*NP)
    ED=256÷QS                    # head columns a subgroup holds
    EQ=Int32(ED÷4)               # ... in H4
    RS=16÷QS                     # softmax rows a subgroup takes
    LR=32÷RS                     # lanes a row
    KL=32÷LR                     # keys a lane
    ks=Int32(PREFILL_KS4)
    tid=Int32(KI.get_local_id().x)-Int32(1)
    wg=Int32(KI.get_group_id().x)-Int32(1)
    rows=Int32(R)*n
    tiles=cld(rows,Int32(16NP))
    kh=wg÷(tiles*splits);tile=(wg%(tiles*splits))÷splits;sp=wg%splits
    w=tid÷Int32(32);lane=tid%Int32(32)
    pr=w÷Int32(QS);part=w%Int32(QS)
    # A row extent the tiles do not divide slides the last tile back; the rows
    # it re-covers are computed twice, identically.
    r0=min((tile*Int32(NP)+pr)*Int32(16),rows-Int32(16))
    pos0=Int32(start[1])
    # The softmax's view: LR lanes a row, KL keys each, RS rows a subgroup.
    srow=part*Int32(RS)+lane÷Int32(LR)
    kpart=lane%Int32(LR)
    lim=pos0+(r0+srow)÷Int32(R)                      # the row's last key
    rmin=min(tile*Int32(16NP),rows-Int32(16))
    rmax=min((tile*Int32(NP)+Int32(NP-1))*Int32(16),rows-Int32(16))+Int32(15)
    kend=pos0+rmax÷Int32(R)+Int32(1)                 # keys the workgroup reads
    kfree=pos0+rmin÷Int32(R)                         # keys every row may see
    # This split's keys: an equal share of a whole number of blocks.
    span=cld(cld(kend,splits),Int32(PREFILL_BC))*Int32(PREFILL_BC)
    kbeg=sp*span;kstop=min(kend,kbeg+span)
    kb0=kbeg÷Int32(PREFILL_BC);nkb=cld(kstop,Int32(PREFILL_BC))
    # Unit `i` of this thread is key `keyb + (i-1)*kstep`, sixteen values from e16.
    keyb=tid÷Int32(16);e16=(tid%Int32(16))*Int32(16);kstep=NT÷Int32(16)
    # Transposed V: keys 4kg .. 4kg+3, sixteen columns from e16v.
    kg=tid%Int32(8);e16v=(tid÷Int32(8))*Int32(16);vt4=Int32(PREFILL_VT4)
    sl2=0.0625f0*1.442695f0

    @inbounds begin
        qg=(kh*rows+r0)*Int32(256)+part*Int32(ED)+Int32(1)
        Base.Cartesian.@nexprs 8 et->if et<=ED÷16
            qa_et=Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(pointer(qh),qg+Int32((et-1)*16),Int32(256),Val(true))
            acco_et=zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
        end
        mrun=-Inf32;lrun=0f0
        Base.Cartesian.@nexprs 4 i->if i<=16÷(QS*NP)
            ku_i=stageunit(kc,ksc,min(kbeg+keyb+Int32(i-1)*kstep,kend-Int32(1)),nk,kh,e16,kf)
        end
        for kb in kb0:nkb-Int32(1)
            k0=kb*Int32(PREFILL_BC)
            # K(kb) from registers into shared; V(kb) starts loading.
            Base.Cartesian.@nexprs 4 i->if i<=16÷(QS*NP)
                sk_i=Int32(1)+(keyb+Int32(i-1)*kstep)*ks+e16÷Int32(4)
                storeunit!(kv,sk_i,ku_i)
                vu_i=VT ? stageunit(vc,vsc,min(k0+Int32(4)*kg+Int32(i-1),kend-Int32(1)),nk,kh,e16v,vf) :
                    stageunit(vc,vsc,min(k0+keyb+Int32(i-1)*kstep,kend-Int32(1)),nk,kh,e16,vf)
            end
            KI.barrier()

            Base.Cartesian.@nexprs 2 ct->begin
                s_ct=zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
                Base.Cartesian.@nexprs 8 et->if et<=ED÷16
                    kbm=Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(kv,
                        Int32(1)+Int32((ct-1)*16)*ks+part*EQ+Int32((et-1)*4),ks,Val(false))
                    s_ct=muladd(qa_et,kbm,s_ct)
                end
                copyto!(sw,Int32(1)+(pr*Int32(QS)+part)*Int32(16*33)+Int32((ct-1)*16),Int32(33),s_ct,Val(true))
            end
            KI.barrier()

            # Every subgroup has its scores, so K's buffer takes V.
            if VT
                Base.Cartesian.@nexprs 16 e->begin
                    kv[Int32(1)+(e16v+Int32(e-1))*vt4+kg]=(VecElement(vu_1[(e-1)÷4+1][(e-1)%4+1].value),
                        VecElement(vu_2[(e-1)÷4+1][(e-1)%4+1].value),VecElement(vu_3[(e-1)÷4+1][(e-1)%4+1].value),
                        VecElement(vu_4[(e-1)÷4+1][(e-1)%4+1].value))
                end
            else
                Base.Cartesian.@nexprs 4 i->if i<=16÷(QS*NP)
                    storeunit!(kv,sk_i,vu_i)
                end
            end
            sb=Int32(1)+pr*Int32(QS*16*33)+srow*Int32(33)+kpart*Int32(KL)
            Base.Cartesian.@nexprs 8 j->if j<=KL
                x_j=sw[sb+Int32(j-1)]
                Base.Cartesian.@nexprs 3 q->if q<QS
                    x_j+=sw[sb+Int32(q*16*33+j-1)]
                end
                x_j*=sl2
            end
            if k0+Int32(PREFILL_BC-1)>kfree
                kfirst=k0+kpart*Int32(KL)
                Base.Cartesian.@nexprs 8 j->if j<=KL
                    x_j=kfirst+Int32(j-1)<=lim ? x_j : -Inf32
                end
            end
            mloc=x_1
            Base.Cartesian.@nexprs 7 j->if j+1<=KL
                mloc=max(mloc,x_{j+1})
            end
            Base.Cartesian.@nexprs 3 d->if (1<<(d-1))<LR
                mloc=max(mloc,KI.shfl(mloc,(lane⊻Int32(1<<(d-1)))+Int32(1)))
            end
            # A block a row cannot see leaves `mrun` as it was and gives it weight
            # 0. Key 0 is every row's, so the first split's `mnew` is finite from
            # its first block on; a later split can see none of a row's keys
            # (the chunk's own, for its first tokens), and stays at -Inf, 0.
            mnew=max(mrun,mloc)
            alpha=mnew==-Inf32 ? 1f0 : exp2(mrun-mnew)
            Base.Cartesian.@nexprs 8 j->if j<=KL
                p_j=mnew==-Inf32 ? 0f0 : exp2(x_j-mnew)
            end
            lsum=p_1
            Base.Cartesian.@nexprs 7 j->if j+1<=KL
                lsum+=p_{j+1}
            end
            Base.Cartesian.@nexprs 3 d->if (1<<(d-1))<LR
                lsum+=KI.shfl(lsum,(lane⊻Int32(1<<(d-1)))+Int32(1))
            end
            lrun=lrun*alpha+lsum;mrun=mnew
            pb=Int32(1)+pr*Int32(16*8)+srow*Int32(8)+kpart*Int32(KL÷4)
            pw[pb]=(VecElement(Float16(p_1)),VecElement(Float16(p_2)),VecElement(Float16(p_3)),VecElement(Float16(p_4)))
            if KL==8
                pw[pb+Int32(1)]=(VecElement(Float16(p_5)),VecElement(Float16(p_6)),VecElement(Float16(p_7)),VecElement(Float16(p_8)))
            end
            kpart==Int32(0) && (aw[Int32(1)+pr*Int32(16)+srow]=alpha)
            KI.barrier()

            # K(kb+1) loads while P·V runs.
            if kb+Int32(1)<nkb
                k1=k0+Int32(PREFILL_BC)
                Base.Cartesian.@nexprs 4 i->if i<=16÷(QS*NP)
                    ku_i=stageunit(kc,ksc,min(k1+keyb+Int32(i-1)*kstep,kend-Int32(1)),nk,kh,e16,kf)
                end
            end
            fac=Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(aw,Int32(1)+pr*Int32(16),Int32(0),Val(false))
            Base.Cartesian.@nexprs 2 kt->pa_kt=Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(pw,
                Int32(1)+pr*Int32(16*8)+Int32((kt-1)*4),Int32(8),Val(true))
            Base.Cartesian.@nexprs 8 t->if t<=ED÷16
                acco_t=Mantle.coopmat_mul(acco_t,fac)
                Base.Cartesian.@nexprs 2 kt->begin
                    vbm=VT ? Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(kv,
                            Int32(1)+(part*Int32(ED)+Int32((t-1)*16))*vt4+Int32((kt-1)*4),vt4,Val(false)) :
                        Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(kv,
                            Int32(1)+Int32((kt-1)*16)*ks+part*EQ+Int32((t-1)*4),ks,Val(true))
                    acco_t=muladd(pa_kt,vbm,acco_t)
                end
            end
            KI.barrier()
        end

        if PARTIAL
            if kpart==Int32(0)
                slot=(kh*rows+r0+srow)*splits+sp
                pm[slot+Int32(1)]=mrun;ps[slot+Int32(1)]=lrun
            end
        else
            kpart==Int32(0) && (aw[Int32(1)+pr*Int32(16)+srow]=1f0/lrun)
        end
        # Which row and column each accumulator component is, measured: the
        # layout is the implementation's.
        for idx in tid:NT:Int32(255)
            sw[Int32(1)+idx]=Float32(idx)
        end
        KI.barrier()
        idxmat=Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(sw,Int32(1),Int32(16),Val(false))
        fac=Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(aw,Int32(1)+pr*Int32(16),Int32(0),Val(false))
        Base.Cartesian.@nexprs 8 i->begin
            oc_i=unsafe_trunc(Int32,Mantle.coopmat_getcomp(idxmat,Int32(i-1)))
            row_i=r0+oc_i%Int32(16)
            # The output column of (row, e): token-major, query heads within;
            # a partial's, (row, split)-major.
            ob_i=PARTIAL ? ((kh*rows+row_i)*splits+sp)*Int32(256)+part*Int32(ED)+oc_i÷Int32(16)+Int32(1) :
                ((row_i÷Int32(R))*nk*Int32(R)+kh*Int32(R)+row_i%Int32(R))*Int32(256)+part*Int32(ED)+oc_i÷Int32(16)+Int32(1)
        end
        Base.Cartesian.@nexprs 8 t->if t<=ED÷16
            PARTIAL || (acco_t=Mantle.coopmat_mul(acco_t,fac))
            Base.Cartesian.@nexprs 8 i->begin
                o_i=ob_i+Int32((t-1)*16)
                if PARTIAL
                    po[o_i]=Mantle.coopmat_getcomp(acco_t,Int32(i-1))
                else
                    out[o_i]=eltype(out)(Mantle.coopmat_getcomp(acco_t,Int32(i-1))*Float32(gate[o_i]))
                end
            end
        end
    end
    nothing
end

# The splits of each (KV head, row): one workgroup a row, a thread a value.
function attention_prefill_combine!(out,pm,ps,po,gate,n::Int32,nk::Int32,splits::Int32,::Val{R}) where R
    d=Int32(KI.get_local_id().x)-Int32(1)
    t=Int32(KI.get_group_id().x)-Int32(1)
    rows=Int32(R)*n;kh=t÷rows;row=t%rows
    base=t*splits
    m=-Inf32
    for s in Int32(0):splits-Int32(1)
        m=max(m,@inbounds pm[base+s+Int32(1)])
    end
    # The first split holds key 0, so `m` is finite.
    l=0f0;o=0f0
    for s in Int32(0):splits-Int32(1)
        ms=@inbounds pm[base+s+Int32(1)]
        if ms!=-Inf32
            f=exp2(ms-m)
            l=muladd(f,@inbounds(ps[base+s+Int32(1)]),l)
            o=muladd(f,@inbounds(po[(base+s)*Int32(256)+d+Int32(1)]),o)
        end
    end
    i=((row÷Int32(R))*nk*Int32(R)+kh*Int32(R)+row%Int32(R))*Int32(256)+d+Int32(1)
    @inbounds out[i]=eltype(out)((o/l)*Float32(gate[i]))
    nothing
end

"""Workgroup tiles per KV head: `NP` 16-row tiles of `R*n` rows."""
prefill_tiles(R,n,np)=cld(R*n,16np)

"""Key splits for a chunk's attention: enough workgroups for two a core, up to
64. The plan serves every position, so the count cannot depend on the depth; at
a shallow one the later splits find no keys and leave at once."""
prefill_splits(caps,workgroups)=clamp(cld(2*max(caps.cores,1),workgroups),1,64)

"""DECLARE the chunk's attention: pack Q, then one causal pass over the cache,
split over the keys (and merged) where the chunk alone cannot fill the device.
`q` and `gate` are the chunk's prepared queries and gates, `(256nqh, n)`."""
function declare_prefill_attention!(g,out,q,gate,kc,vc,start,nkv,nqh,n;np=4,qs=4,vt=false,name,splits=nothing)
    vt && np*qs!=4 && throw(ArgumentError("transposed V is staged by 128 threads: np * qs == 4"))
    R=nqh÷nkv
    16%R==0 && R*n>=16 || throw(ArgumentError("prefill attention needs 16 % (heads per KV head) == 0 and at least 16 rows"))
    qh=Mantle.Transient.Buffer(g,Float16,(256nqh*n,))
    dispatch(g,pack_prefill_q!,(qh,q,Int32(n),Val(nqh),Val(R)),256nqh*n;name=name*".pack_q")
    wgs=nkv*prefill_tiles(R,n,np)
    sp=splits===nothing ? prefill_splits(Mantle.caps(g.dev),wgs) : splits
    if sp==1
        dispatch(g,attention_prefill!,(out,qh,gate,flashargs(kc,vc)...,start,Int32(n),Int32(nkv),Int32(1),out,out,out,
            Val(R),Val(np),Val(qs),rowformat(kc),rowformat(vc),Val(false),Val(vt)),wgs*32qs*np;group=32qs*np,name)
    else
        rows=R*n*nkv
        pm=Mantle.Transient.Buffer(g,Float32,(rows*sp,));ps=Mantle.Transient.Buffer(g,Float32,(rows*sp,))
        po=Mantle.Transient.Buffer(g,Float32,(256*rows*sp,))
        dispatch(g,attention_prefill!,(out,qh,gate,flashargs(kc,vc)...,start,Int32(n),Int32(nkv),Int32(sp),pm,ps,po,
            Val(R),Val(np),Val(qs),rowformat(kc),rowformat(vc),Val(true),Val(vt)),wgs*sp*32qs*np;group=32qs*np,name)
        dispatch(g,attention_prefill_combine!,(out,pm,ps,po,gate,Int32(n),Int32(nkv),Int32(sp),Val(R)),rows*256;group=256,name=name*".combine")
    end
    out
end

# BonsaiRunner's `prepare_q_kernel!` and `prepare_kv!` for every column of a
# chunk, one workgroup a (head, token), each token at its own position. The
# decode kernels stay as they are: they are what bit-exact chunks reproduce.
function prepare_q_chunk!(q,gate,qfull,norm,positions,theta_base::Float32,eps::Float32,nqh::Int32)
    sh=KI.localmemory(Float32,Val((512,)),Val(1))
    d=Int32(KI.get_local_id().x)-Int32(1)
    t=Int32(KI.get_group_id().x)-Int32(1)
    h=t%nqh;tok=t÷nqh
    pos=Int32(@inbounds positions[tok+Int32(1)])
    base=tok*nqh*Int32(512)+h*Int32(512)
    @inbounds begin
        x=Float32(qfull[base+d+Int32(1)])
        sh[d+Int32(1)]=x
        sh[Int32(256)+d+Int32(1)]=x*x
    end
    KI.barrier()
    step=Int32(128)
    while step>Int32(0)
        d<step && (@inbounds sh[Int32(256)+d+Int32(1)]+=sh[Int32(256)+d+step+Int32(1)])
        KI.barrier()
        step>>=1
    end
    scale=inv(sqrt(@inbounds(sh[Int32(257)])/256f0+eps))
    @inbounds sh[d+Int32(1)]*=scale*Float32(norm[d+Int32(1)])
    KI.barrier()
    if d<Int32(32)
        angle=Float32(pos)*exp(-log(theta_base)*Float32(2d)/64f0)
        c,s=cos(angle),sin(angle)
        @inbounds begin
            a=sh[d+Int32(1)];b=sh[d+Int32(32)+Int32(1)]
            sh[d+Int32(1)]=a*c-b*s;sh[d+Int32(32)+Int32(1)]=a*s+b*c
        end
    end
    KI.barrier()
    o=tok*nqh*Int32(256)+h*Int32(256)+d+Int32(1)
    @inbounds begin
        q[o]=eltype(q)(sh[d+Int32(1)])
        g=Float32(qfull[base+Int32(256)+d+Int32(1)])
        gate[o]=eltype(gate)(1f0/(1f0+exp(-g)))
    end
    nothing
end

function prepare_kv_chunk!(kc,vc,k,v,norm,positions,nk::Int32,theta::Float32,eps::Float32)
    sh=KI.localmemory(Float32,Val((512,)),Val(1))
    d=Int32(KI.get_local_id().x)-Int32(1)
    t=Int32(KI.get_group_id().x)-Int32(1)
    h=t%nk;tok=t÷nk
    pos=Int32(@inbounds positions[tok+Int32(1)])
    base=tok*nk*Int32(256)+h*Int32(256)
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
    scale=inv(sqrt(@inbounds(sh[Int32(257)])/256f0+eps))
    @inbounds sh[d+Int32(1)]*=scale*norm[d+Int32(1)]
    KI.barrier()
    if d<Int32(32)
        angle=Float32(pos)*exp(-log(theta)*Float32(2d)/64f0)
        co,si=cos(angle),sin(angle)
        @inbounds begin
            a,b=sh[d+Int32(1)],sh[d+Int32(33)]
            sh[d+Int32(1)]=a*co-b*si;sh[d+Int32(33)]=a*si+b*co
        end
    end
    KI.barrier()
    at=pos*nk*Int32(256)+h*Int32(256)+d+Int32(1)
    @inbounds begin
        kc[at]=eltype(kc)(sh[d+Int32(1)])
        vc[at]=eltype(vc)(v[base+d+Int32(1)])
    end
    nothing
end

"""DECLARE the chunk's prepared queries and gates, and its keys and values
written into the cache, each token at `positions[token]`."""
function declare_prepare_chunk!(g,aq,qgate,qfull,qnorm,kc,vc,k,v,knorm,positions,nqh,nkv,theta,eps,n;name)
    dispatch(g,prepare_q_chunk!,(aq,qgate,qfull,qnorm,positions,theta,eps,Int32(nqh)),nqh*n*256;group=256,name=name*".prepare_q")
    declare_prepare_kv_chunk!(g,kc,vc,k,v,knorm,positions,nkv,theta,eps,n;name=name*".prepare_kv")
end
declare_prepare_kv_chunk!(g,kc,vc,k,v,norm,positions,nkv,theta,eps,n;name)=
    dispatch(g,prepare_kv_chunk!,(kc,vc,k,v,norm,positions,Int32(nkv),theta,eps),nkv*n*256;group=256,name)
declare_prepare_kv_chunk!(g,kc::QuantizedKV,vc::QuantizedKV,k,v,norm,positions,nkv,theta,eps,n;name)=
    dispatch(g,prepare_kv_quant_chunk!,(kc.q,kc.scales,vc.q,vc.scales,k,v,norm,positions,Int32(nkv),theta,eps,rowformat(kc)),nkv*n*256;group=256,name)

for (kernel,writes) in ((pack_prefill_q!,(1,)),(attention_prefill!,(1,12,13,14)),(attention_prefill_combine!,(1,)),(prepare_q_chunk!,(1,2)),(prepare_kv_chunk!,(1,2)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end

"""Whether the chunk-wide attention can run `n` columns: a device with the matrix
kernels and a head grouping a tile divides."""
chunk_attention_runs(dev,n,nkv,nqh)=DK.coopmatkernels(Mantle.caps(dev))&&16%(nqh÷nkv)==0&&(nqh÷nkv)*n>=16

"""Whether a prompt chunk of `n` columns takes the chunk-wide attention: wider
than the products that reproduce decode bit for bit (which already gives up
identity), where it runs."""
chunk_attention(dev,n,nkv,nqh)=!exactchunk(n)&&chunk_attention_runs(dev,n,nkv,nqh)
