"""
Flash attention where every subgroup owns sixteen query rows end to end.

[`attn_flash_cm_spatial4!`](@ref) shares one block of rows between the subgroups
of a workgroup: the score tiles are dealt round the subgroups, one thread per row
runs the softmax through shared memory, and K and V are read from global memory
by every subgroup that needs them. At Qwen-Image 2.1's attention (`E = 128`,
4096 queries, 4118 keys, 32 heads) that is 27 ms where torch's AOTriton kernel
takes 8.5, and the difference is not arithmetic: both issue the same 128 WMMAs
per 64 rows and 32 keys, but the shared-row kernel issues ~3.5x the other
instructions and 36x the memory loads doing it.

This is AOTriton's structure, in portable cooperative matrices:

  * **A subgroup's rows are its own.** Q·Kᵀ, the softmax and P·V for sixteen rows
    never leave the subgroup, so no subgroup idles while another computes scores
    and the softmax is not one thread per row.
  * **Q lives in registers** for the whole key loop, `E/16` A fragments.
  * **K and V are staged once per workgroup** with 16-byte copies and read back
    as fragments by all four subgroups.
  * **The next block's loads are in flight during this one's products**: V(kb)
    loads while Q·Kᵀ runs, K(kb+1) while P·V runs, and each lands in shared
    memory only once the block before it is no longer being read.

What the cooperative-matrix extension does not give, it works around rather than
assumes. The accumulator layout is opaque, so a row's maximum and sum cannot be
taken in registers: the scores go to a subgroup-private slice of shared memory,
two lanes read each row, and `P` comes back as an A fragment. The factor that
rescales `O` is applied as a stride-0 matrix, so no component is ever named.

## Shared memory is the occupancy

RADV runs compute workgroups in CU mode, 64 KB of shared memory each, allocated
in 1 KB steps. Three 128-thread workgroups — six subgroups a SIMD, which is also
what 240 registers allow — need at most 21504 bytes each, and at `E = 128` this
kernel uses exactly that: K then V in one 8.5 KB buffer (V takes K's place once
every subgroup has its scores), the fp32 scores 8.25 KB, `P` 4 KB, the row
factors 256 bytes. Measured at Qwen-Image 2.1's shape:

    layout                                   subgroups/SIMD   ms
    K, V, S, P in separate buffers (31.7 KB)        4         13.3
    P in K's buffer (26.6 KB)                       4         12.7
    this (21.5 KB, loads pipelined)                 6         10.8
    shipped `attn_flash_cm_spatial4!`               8         27.2

**Tried and not kept: staging V transposed.** A V fragment is sixteen keys of
one `e`, and row-major V makes that sixteen 2-byte loads a fragment — 256 a
block, plus ~140 moves the register allocator spends assembling them. Storing
Vᵀ turns each fragment into four 8-byte loads and cut the loop from 906
instructions to 782, and it measured 11.6 ms against 10.8: the allocator spent
the saving on swaps (172 moves and swaps in that loop). Rows of Vᵀ must also be
8-byte aligned — 68-byte rows measured 27 ms, the misaligned wide loads being
that slow.
"""

"""A 16-byte K or V chunk as it is staged: two of these."""
const FLASHROWS_H4 = NTuple{4,VecElement{Float16}}

"""Keys per block. Two score tiles a subgroup, and the staging assumes it."""
const FLASHROWS_BC = 32

"""Subgroups a workgroup. Four: see the table in the file's docstring."""
const FLASHROWS_NW = 4

@inline flashrows_chunk(base::Ptr{Float16}, off::Int) =
    reinterpret(Ptr{FLASHROWS_H4}, base + 2 * off)

"""
    attn_flash_rows!(out, q, k, v, scale, qbase, qsL, qsH, qsB, kbase, …, Val(E), Val(NW), Val(OUTPERM), Lq, Lk)

`out[:, i, h, b] = softmax(scale * K[:, :, h, b]ᵀ q[:, i, h, b]) V` for every
query `i`, with `q`, `k`, `v` read at 1-based `base + e + stride_L*l + stride_H*(h-1) + stride_B*(b-1)`.
With `OUTPERM`, `out` is `(E, H, L, B)` and row `i` of head `h` goes to
`out[:, h, i, b]`: the order a following `(0, 2, 1, 3)` permute asks for, which
then needs no pass of its own (see `sdpaoutputpermute`).
`E` contiguous and 16-byte-aligned key rows are [`flashrows_plan`](@ref)'s to
check.
"""
function attn_flash_rows!(out, q, k, v, scale::Float32,
        qbase::Int32, qsL::Int32, qsH::Int32, qsB::Int32,
        kbase::Int32, ksL::Int32, ksH::Int32, ksB::Int32,
        vbase::Int32, vsL::Int32, vsH::Int32, vsB::Int32,
        ::Val{E}, ::Val{NW}, ::Val{OUTPERM}, Lq::Int32, Lk::Int32) where {E,NW,OUTPERM}
    # Sizes written from the type parameters, never from a local: Lava
    # miscompiles an `@localmem` sized by a local binding (see `flash.jl`).
    # K rows are padded by 8 halves, which keeps them 16-byte aligned and puts
    # the sixteen columns a K fragment reads in different banks.
    kv = KI.localmemory(FLASHROWS_H4, Val((32 * (E + 8) ÷ 4,)), Val(1))   # K, then V
    sw = KI.localmemory(Float32, Val((NW * 16 * 33,)), Val(2))            # scores, (r, c) at r*33 + c
    pw = KI.localmemory(FLASHROWS_H4, Val((NW * 16 * 32 ÷ 4,)), Val(3))   # P, (r, c) at r*32 + c
    aw = KI.localmemory(Float32, Val((NW * 16,)), Val(4))                 # a row's factor
    KS4 = Int32((E + 8) ÷ 4)
    SS = Int32(33)
    PS4 = Int32(8)
    NT = Int32(NW * 32)
    CPK = Int32(E ÷ 8)                  # 16-byte chunks in a key row
    CPT = E ÷ (NW * 8)                  # chunks a thread stages, per matrix and block

    tid = Int32(KI.get_local_id().x - 1)
    grp = KI.get_group_id()
    qb, h, b = Int32(grp.x - 1), Int32(grp.y - 1), Int32(grp.z - 1)
    w = tid ÷ Int32(32)
    lane = tid % Int32(32)
    # The softmax's view: two lanes a row, sixteen keys each.
    r = lane ÷ Int32(2)
    part = lane % Int32(2)
    # A query extent the tile does not divide slides its last tile back onto the
    # tensor rather than reading past it. The rows it re-covers are computed
    # twice, identically, and stored twice with the same values.
    q0 = min(qb * Int32(16 * NW) + w * Int32(16), Lq - Int32(16))

    @inbounds begin
        qg = qbase + h * qsH + b * qsB + q0 * qsL
        Base.Cartesian.@nexprs 8 et -> if et <= E ÷ 16
            qa_et = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(pointer(q),
                        qg + Int32((et - 1) * 16), qsL, Val(true))
            acco_et = zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
        end

        kh0 = Int(kbase - Int32(1) + h * ksH + b * ksB)
        vh0 = Int(vbase - Int32(1) + h * vsH + b * vsB)
        # exp2 of a pre-scaled score is the hardware's exp.
        sl2 = scale * 1.442695f0
        mrun = -Inf32
        lrun = 0.0f0
        nkb = cld(Lk, Int32(FLASHROWS_BC))
        # Chunk `rr` of this thread is key `key0 + rr * kstep`, eight `e` from `e8`.
        key0 = tid ÷ CPK
        e8 = (tid % CPK) * Int32(8)
        kstep = NT ÷ CPK
        # A block past the key extent re-reads the last key: finite, and its
        # score is masked to -Inf below, so its weight is exactly zero. Nothing
        # past the tensor is read.
        @inline kcl(k0, rr) = min(key0 + Int32(rr) * kstep, Lk - Int32(1) - k0)

        Base.Cartesian.@nexprs 4 rr -> if rr <= CPT
            pk_rr = flashrows_chunk(pointer(k), kh0 + Int(kcl(Int32(0), rr - 1) * ksL + e8))
            kr_rr_1 = Core.Intrinsics.pointerref(pk_rr, 1, 16)
            kr_rr_2 = Core.Intrinsics.pointerref(pk_rr, 2, 8)
        end
        for kb in Int32(0):(nkb - Int32(1))
            k0 = kb * Int32(FLASHROWS_BC)
            ragged = k0 + Int32(FLASHROWS_BC) > Lk
            # K(kb) from registers into shared; V(kb) starts loading.
            Base.Cartesian.@nexprs 4 rr -> if rr <= CPT
                sk_rr = 1 + (key0 + Int32(rr - 1) * kstep) * KS4 + e8 ÷ Int32(4)
                kv[sk_rr] = kr_rr_1
                kv[sk_rr + 1] = kr_rr_2
                pv_rr = flashrows_chunk(pointer(v),
                    vh0 + Int(k0) * Int(vsL) + Int(kcl(k0, rr - 1) * vsL + e8))
                vr_rr_1 = Core.Intrinsics.pointerref(pv_rr, 1, 16)
                vr_rr_2 = Core.Intrinsics.pointerref(pv_rr, 2, 8)
            end
            KI.barrier()

            Base.Cartesian.@nexprs 2 ct -> begin
                s_ct = zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
                Base.Cartesian.@nexprs 8 et -> if et <= E ÷ 16
                    kbm = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(kv,
                        1 + Int32((ct - 1) * 16) * KS4 + Int32((et - 1) * 4), KS4, Val(false))
                    s_ct = muladd(qa_et, kbm, s_ct)
                end
                copyto!(sw, 1 + w * Int32(16) * SS + Int32((ct - 1) * 16), SS, s_ct, Val(true))
            end
            KI.barrier()

            # Every subgroup has its scores, so K's buffer takes V.
            Base.Cartesian.@nexprs 4 rr -> if rr <= CPT
                kv[sk_rr] = vr_rr_1
                kv[sk_rr + 1] = vr_rr_2
            end
            sb = 1 + (w * Int32(16) + r) * SS + part * Int32(16)
            Base.Cartesian.@nexprs 16 j -> x_j = sw[sb + Int32(j - 1)] * sl2
            if ragged
                kfirst = k0 + part * Int32(16)
                Base.Cartesian.@nexprs 16 j -> x_j = kfirst + Int32(j - 1) < Lk ? x_j : -Inf32
            end
            mloc = Base.Cartesian.@ncall 16 max x
            mnew = max(mrun, max(mloc, KI.shfl(mloc, lane ⊻ Int32(1))))
            # `mrun` is -Inf on the first block and `mnew` never is (every block
            # has a key inside the extent), so this is 0 there, not NaN.
            alpha = exp2(mrun - mnew)
            Base.Cartesian.@nexprs 16 j -> p_j = exp2(x_j - mnew)
            lsum = Base.Cartesian.@ncall 16 (+) p
            lrun = lrun * alpha + (lsum + KI.shfl(lsum, lane ⊻ Int32(1)))
            mrun = mnew
            pb = 1 + (w * Int32(16) + r) * PS4 + part * Int32(4)
            Base.Cartesian.@nexprs 4 m -> pw[pb + Int32(m - 1)] =
                (VecElement(Float16(p_{4m - 3})), VecElement(Float16(p_{4m - 2})),
                 VecElement(Float16(p_{4m - 1})), VecElement(Float16(p_{4m})))
            part == Int32(0) && (aw[1 + w * Int32(16) + r] = alpha)
            KI.barrier()

            # K(kb+1) loads while P·V runs.
            if kb + Int32(1) < nkb
                k1 = k0 + Int32(FLASHROWS_BC)
                Base.Cartesian.@nexprs 4 rr -> if rr <= CPT
                    pk_rr = flashrows_chunk(pointer(k),
                        kh0 + Int(k1) * Int(ksL) + Int(kcl(k1, rr - 1) * ksL + e8))
                    kr_rr_1 = Core.Intrinsics.pointerref(pk_rr, 1, 16)
                    kr_rr_2 = Core.Intrinsics.pointerref(pk_rr, 2, 8)
                end
            end
            # O *= alpha row by row: the factors as a matrix whose columns all
            # read the same sixteen values (stride 0).
            fac = Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(aw, 1 + w * Int32(16), 0, Val(false))
            Base.Cartesian.@nexprs 2 kt -> pa_kt =
                Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(pw,
                    1 + w * Int32(16) * PS4 + Int32((kt - 1) * 4), PS4, Val(true))
            Base.Cartesian.@nexprs 8 t -> if t <= E ÷ 16
                acco_t = Mantle.coopmat_mul(acco_t, fac)
                Base.Cartesian.@nexprs 2 kt -> begin
                    vbm = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(kv,
                        1 + Int32((kt - 1) * 16) * KS4 + Int32((t - 1) * 4), KS4, Val(true))
                    acco_t = muladd(pa_kt, vbm, acco_t)
                end
            end
            KI.barrier()
        end

        part == Int32(0) && (aw[1 + w * Int32(16) + r] = 1.0f0 / lrun)
        # Which row and column each accumulator component is, measured: the
        # layout is the implementation's. Here and not before the loop, so none
        # of it is live across the key blocks.
        for idx in tid:NT:Int32(255)
            sw[1 + idx] = Float32(idx)
        end
        KI.barrier()
        idxmat = Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(sw, 1, 16, Val(false))
        fac = Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(aw, 1 + w * Int32(16), 0, Val(false))
        Base.Cartesian.@nexprs 8 i -> begin
            oc_i = unsafe_trunc(Int32, Mantle.coopmat_getcomp(idxmat, Int32(i - 1)))
            orow_i = oc_i % Int32(16)
            ocol_i = oc_i ÷ Int32(16)
        end
        Base.Cartesian.@nexprs 8 t -> if t <= E ÷ 16
            acco_t = Mantle.coopmat_mul(acco_t, fac)
            Base.Cartesian.@nexprs 8 i -> begin
                e_i = 1 + Int32((t - 1) * 16) + ocol_i
                l_i = 1 + q0 + orow_i
                o_i = convert(eltype(out), Mantle.coopmat_getcomp(acco_t, Int32(i - 1)))
                if OUTPERM
                    out[e_i, 1 + h, l_i, 1 + b] = o_i
                else
                    out[e_i, l_i, 1 + h, 1 + b] = o_i
                end
            end
        end
    end
    return nothing
end

"""
    FlashRowsPlan

[`attn_flash_rows!`](@ref) will run this attention. Only the head width varies;
the block is always 64 queries by 32 keys on four subgroups.
"""
struct FlashRowsPlan
    E::Int
end

"""Shared memory [`attn_flash_rows!`](@ref) declares at head width `E`."""
flashrows_shared(E::Int) =
    2 * 32 * (E + 8) + 4 * FLASHROWS_NW * 16 * 33 + 2 * FLASHROWS_NW * 16 * 32 + 4 * FLASHROWS_NW * 16

"""
    flashrows_plan(dev, q, k, v, bias) -> FlashRowsPlan | Decline

Whether [`attn_flash_rows!`](@ref) takes this attention. Every refusal is a
shape or a device the kernel is not written for, and the caller falls through
to [`flashcm_plan`](@ref).
"""
function flashrows_plan(dev::M.DeviceCaps, q, k, v, bias)
    bias === nothing || return Decline(:bias)
    coopmatkernels(dev) || return Decline(:nocoopmat)
    # 16x16 fp16 tiles on 32-lane subgroups: the softmax gives each of a tile's
    # sixteen rows two lanes, and every index in the kernel is written for that.
    (dev.tile == 16 && dev.coopmatsubgroup == 32) || return Decline(:subgroup)
    eltype(q) === Float16 && eltype(k) === Float16 && eltype(v) === Float16 ||
        return Decline(:eltype)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    E in (64, 128) || return Decline(:headdim)
    Lq >= 16 && Lk >= 1 || return Decline(:extent)
    roots = (stridedroot(q), stridedroot(k), stridedroot(v))
    any(isnothing, roots) && return Decline(:wrapped)
    sq, sk, sv = flashstrides(q), flashstrides(k), flashstrides(v)
    (sq[1] == 1 && sk[1] == 1 && sv[1] == 1) || return Decline(:layout)
    # K and V are staged in 16-byte chunks, so every key row has to start on a
    # 16-byte boundary: the operand's base (asked of Mantle — a view of an array
    # carries its offset inside it, where `stridedroot` cannot see it), its
    # offset from that base, and every stride.
    all(s -> s % 8 == 0, (sk[2], sk[3], sk[4], sv[2], sv[3], sv[4])) &&
        all(r -> M.basealignment(r[1]) >= 16 && r[2] % 8 == 0, roots[2:3]) ||
        return Decline(:align)
    # Element offsets are Int32 in the kernel.
    maximum(length(first(r)) for r in roots) < typemax(Int32) || return Decline(:extent)
    FLASHROWS_NW * dev.coopmatsubgroup <= dev.workgrouplimit || return Decline(:workgroup)
    flashrows_shared(E) <= dev.sharedbudget || return Decline(:shared)
    # One workgroup per 64 queries and no split of the key axis: a grid that
    # cannot fill the device is flash-decoding's shape, which `flashcm_plan` has.
    cld(Lq, 16 * FLASHROWS_NW) * H * B >= 2 * dev.cores || return Decline(:grid)
    FlashRowsPlan(E)
end

"""
    flashrows_launches(out, plan, q, k, v, scale; outperm = false) -> Vector

The launch, as [`Mantle.runlaunches!`](@ref) and [`flashrows_dispatch!`](@ref)
take it.
"""
function flashrows_launches(out, plan::FlashRowsPlan, q, k, v, scale; outperm::Bool = false)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    rq, rk, rv = stridedroot(q), stridedroot(k), stridedroot(v)
    sq, sk, sv = flashstrides(q), flashstrides(k), flashstrides(v)
    NT = FLASHROWS_NW * 32
    [(kern = attn_flash_rows!,
      args = (out, flashflat(rq[1]), flashflat(rk[1]), flashflat(rv[1]), Float32(scale),
              Int32(rq[2] + 1), sq[2], sq[3], sq[4],
              Int32(rk[2] + 1), sk[2], sk[3], sk[4],
              Int32(rv[2] + 1), sv[2], sv[3], sv[4],
              Val(E), Val(FLASHROWS_NW), Val(outperm), Int32(Lq), Int32(Lk)),
      ndrange = (NT * cld(Lq, 16 * FLASHROWS_NW), H, B), group = NT)]
end

"""Submit this attention now."""
sdpaflashrows!(ctx, out, plan::FlashRowsPlan, q, k, v, scale) =
    (M.runlaunches!(ctx.backend, flashrows_launches(out, plan, q, k, v, scale)); out)

# Here and not beside the other `sdpa!` methods in `attention.jl`, which is
# included before this file defines the plan type.
function sdpa!(ctx, plan::FlashRowsPlan, out, q, k, v, bias, scale)
    Lq, H, B = size(q, 2), size(q, 3), size(q, 4)
    sdpaflashrows!(ctx, sdpaout(ctx, out, q, v, Lq, H, B), plan, q, k, v, scale)
end

"""DECLARE this attention into a graph."""
function flashrows_dispatch!(g, out, plan::FlashRowsPlan, q, k, v, scale; outperm::Bool = false,
                             name::AbstractString = "sdpa")
    l = only(flashrows_launches(out, plan, q, k, v, scale; outperm))
    M.dispatch!(g, l.kern, l.args, l.ndrange; group = l.group, name)
    return out
end
