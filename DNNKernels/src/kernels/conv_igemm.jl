"""
Implicit-GEMM convolution, macro-free.

The same kernel as `kernels/extern/conv_implicit.jl`'s `conv2d_igemm!`, written
against `KernelInterface`'s intrinsics instead of `@kernel`, so `dispatch!`
takes it as a plain function and Mantle compiles it for whichever backend the
graph is on. Ported from ggml's Vulkan `conv2d_mm.comp` originally; the
arithmetic, the tiling and the split-K scheme are unchanged, and the comments
that explain WHY each of them is shaped that way stay with the code they explain.

The convolution is one GEMM:

    C[K, NPQ] = A[K, CRS] * B[CRS, NPQ]

    K   = Cout                 output channels   (rows)
    CRS = Cin * KH * KW        reduction extent
    NPQ = N * OH * OW          output pixels     (columns)

`B` is never materialised — that is the "implicit" part. Each element is fetched
straight from `x` with the im2col index computed on the fly, so we pay none of
the KH*KW memory blow-up an explicit im2col costs.

## What the macro-free form changes, and what it does not

Three of `@kernel`'s facilities were there for KernelAbstractions' CPU backend,
which runs each barrier-separated region as its own loop over workitems. There is
no such backend behind `KernelInterface`: a workitem is a real thread, its locals
live in registers across a `barrier()`, and none of the three is needed.

  * `@private ACC (TS_K, TS_NPQ)` for the accumulators, "because a plain local is
    not carried over a barrier on the CPU backend". Here they are plain locals,
    minted by `Base.Cartesian.@nexprs` as `acc_i_j`, which is what they compile
    to on a GPU either way.
  * recomputing the local and group ids in every region, "because a derived local
    does not survive a `@synchronize`". Computed once, at the top.
  * `@uniform`, which told the CPU backend it could hoist a value out of its
    per-workitem loop. Every one of them is derived from the `Val` parameters, so
    they are compile-time constants and the annotation had nothing to add.

`@index(Local, NTuple)` was also a workaround: the `Linear` form "has no method
inside a kernel containing `@synchronize`" on that backend. `get_local_id().x` is
the linear id directly, and the workgroup is `(WG, 1)`.

What does NOT change is the +4 padding on each shared tile's minor extent, the
strided register tile, and the guard on the multiply-add rather than only on the
operand loads. Those are about the hardware, and each comment below says which.

Two workgroup buffers, so two `KI.localmemory` ids. That parameter did not exist
when this port was first attempted: `localmemory(T, Val(Dims))` keyed a backend's
workgroup global on `(T, Dims)` alone, and `Ash`/`Bsh` differ in neither on some
shapes, so they would have been ONE buffer with the second write landing on the
first tile. It is in `KernelInterface`'s signature now.
"""

"""
    conv2d_igemm_ki!(out, x, w, bias, Val(ACC), Val(SPLITK), Val(ACT), …) -> nothing

One block tile of the implicit GEMM. See this file's header.

The `Val` parameters are the block and thread tiling, the kernel extents and the
stride/padding/dilation, all of which `convtiles` and the op choose at emit time
so the body specialises on them. `ACC` is the accumulator and shared-memory
element type and is a parameter rather than `accum(eltype(x))` computed in the
body, because workgroup memory needs a static type.
"""
function conv2d_igemm_ki!(out, x, w, bias,
                          ::Val{ACC}, ::Val{SPLITK}, ::Val{ACT},
                          ::Val{BS_K}, ::Val{BS_CRS}, ::Val{BS_NPQ},
                          ::Val{TS_K}, ::Val{TS_NPQ},
                          ::Val{KW}, ::Val{KH},
                          ::Val{SX}, ::Val{SY}, ::Val{PX}, ::Val{PY},
                          ::Val{DX}, ::Val{DY},
                          Cin, Cout, Wid, Hei, OW, OH,
                          NPQ, CRS, NBN) where {ACC,SPLITK,ACT,BS_K,BS_CRS,BS_NPQ,
                                                TS_K,TS_NPQ,KW,KH,SX,SY,PX,PY,DX,DY}
    T = eltype(out)
    A = ACC

    NT_K = BS_K ÷ TS_K
    NT_NPQ = BS_NPQ ÷ TS_NPQ
    WG = NT_K * NT_NPQ
    ArpWg = WG ÷ BS_CRS
    BrpWg = max(WG ÷ BS_NPQ, 1)
    Ash_stride = BS_CRS + 4
    Bsh_stride = BS_NPQ + 4
    nblk = cld(CRS, BS_CRS)
    # Split-K: the reduction is divided over SPLITK workgroups, which is the only
    # way to get both a large thread tile (arithmetic intensity) and enough
    # workgroups to fill the device. This model's dominant convolution is
    # 256 output channels over 120 pixels with a 2304-deep reduction: output
    # tiling alone yields 16 workgroups on 48 SMs, and shrinking the tile to get
    # more drops the MAC:load ratio from 16:8 to 2:3.
    blkper = cld(nblk, SPLITK)

    # +4 on the minor extent staggers rows across banks; without it every thread
    # in a pass hits the same bank on the A load.
    #
    # Ids 1 and 2 because these are two buffers, and a backend keys its
    # workgroup global on what it is handed. `Val` written out rather than left
    # to constant propagation through `localmemory`'s forwarding method: if a
    # size or a type reaches the generator as a value instead of a parameter the
    # failure is silent, the kernel running and writing nothing.
    Ash = KI.localmemory(ACC, Val((BS_K * (BS_CRS + 4),)), Val(1))
    Bsh = KI.localmemory(ACC, Val((BS_CRS * (BS_NPQ + 4),)), Val(2))

    # The workgroup is `(WG, 1)`, so the first component of the local id is the
    # linear one. Computed once: a register survives a barrier.
    tid = KI.get_local_id().x - 1
    bk = KI.get_group_id().x
    gn = KI.get_group_id().y
    # the second grid axis carries both the NPQ block and the split index
    bnpq = (gn - 1) % NBN + 1
    ksplit = (gn - 1) ÷ NBN
    B_idx_K = (bk - 1) * BS_K
    B_idx_NPQ = (bnpq - 1) * BS_NPQ
    Ar = tid ÷ BS_CRS
    Ac = tid % BS_CRS
    Br = tid ÷ BS_NPQ
    Bc = tid % BS_NPQ
    T_y = tid ÷ NT_NPQ
    T_x = tid % NT_NPQ

    # All 64, not only the ones inside the tile: the `if i <= TS_K` guards below
    # fold away at compile time, but the name still has to be bound for the
    # body to lower. The unused ones are dead stores the optimiser drops.
    Base.Cartesian.@nexprs 8 j -> Base.Cartesian.@nexprs 8 i -> acc_i_j = zero(A)

    for bb in 0:(blkper - 1)
        # Blocks past the end make `crs_a`/`crs_b` exceed CRS, so the bounds
        # guards below already stage zeros. The trip count stays uniform, which
        # a workgroup barrier requires: every thread has to reach every one.
        b_crs = ksplit * blkper + bb

        # ── stage A (the kernel), BS_K x BS_CRS ───────────────────────────
        @inbounds begin
            crs_a = b_crs * BS_CRS + Ac
            cin_a = crs_a ÷ (KW * KH)
            rem_a = crs_a % (KW * KH)
            kh_a = rem_a ÷ KW
            kw_a = rem_a % KW
            r = 0
            while r < BS_K
                ky = r + Ar
                kidx = B_idx_K + ky
                Ash[ky * Ash_stride + Ac + 1] = (kidx < Cout && crs_a < CRS) ?
                    A(w[kw_a + 1, kh_a + 1, cin_a + 1, kidx + 1]) : zero(A)
                r += ArpWg
            end

            # ── stage B (the input), BS_CRS x BS_NPQ, gathered by im2col ───
            r = 0
            while r < BS_CRS
                by = r + Br
                npq = B_idx_NPQ + Bc
                n_idx = npq ÷ (OH * OW)
                npqr = npq - n_idx * OH * OW
                oh = npqr ÷ OW
                ow = npqr - oh * OW

                crs_b = b_crs * BS_CRS + by
                cin_b = crs_b ÷ (KW * KH)
                rem_b = crs_b % (KW * KH)
                kh_b = rem_b ÷ KW
                kw_b = rem_b % KW

                ix = ow * SX - PX + kw_b * DX
                iy = oh * SY - PY + kh_b * DY
                inb = (0 <= ix < Wid) && (0 <= iy < Hei) && npq < NPQ && crs_b < CRS
                Bsh[by * Bsh_stride + Bc + 1] =
                    inb ? A(x[ix + 1, iy + 1, cin_b + 1, n_idx + 1]) : zero(A)
                r += BrpWg
            end
        end

        KI.barrier()

        # ── the only hot loop: TS_K + TS_NPQ shared loads per TS_K*TS_NPQ MACs
        #
        # The register tile is *strided*, not blocked: thread T_x owns columns
        # T_x, T_x+NT_NPQ, ... rather than a contiguous run of TS_NPQ. Blocked
        # ownership makes neighbouring threads read TS_NPQ apart, which is a
        # TS_NPQ-way shared-memory bank conflict — measurably fatal at TS_NPQ=8.
        @inbounds for k in 0:(BS_CRS - 1)
            Base.Cartesian.@nexprs 8 i -> a_i = i <= TS_K ?
                Ash[(T_y + (i - 1) * NT_K) * Ash_stride + k + 1] : zero(A)
            Base.Cartesian.@nexprs 8 j -> b_j = j <= TS_NPQ ?
                Bsh[k * Bsh_stride + T_x + (j - 1) * NT_NPQ + 1] : zero(A)
            # The guard belongs on the multiply-add, not just the operand loads.
            # Zeroing a_i/b_j past the tile gives the right answer but still
            # issues all 64 FMAs; TS_K/TS_NPQ are static, so this `if` folds.
            Base.Cartesian.@nexprs 8 j -> Base.Cartesian.@nexprs 8 i -> begin
                if i <= TS_K && j <= TS_NPQ
                    acc_i_j = muladd(a_i, b_j, acc_i_j)
                end
            end
        end

        KI.barrier()
    end

    # ── write back ────────────────────────────────────────────────────────
    @inbounds Base.Cartesian.@nexprs 8 i -> begin
        if i <= TS_K
            kidx = B_idx_K + T_y + (i - 1) * NT_K
            if kidx < Cout
                bv = bias === nothing ? zero(A) : A(bias[kidx + 1])
                Base.Cartesian.@nexprs 8 j -> begin
                    if j <= TS_NPQ
                        npq = B_idx_NPQ + T_x + (j - 1) * NT_NPQ
                        if npq < NPQ
                            n_idx = npq ÷ (OH * OW)
                            npqr = npq - n_idx * OH * OW
                            oh = npqr ÷ OW
                            ow = npqr - oh * OW
                            if SPLITK == 1
                                v = acc_i_j + bv
                                ACT === :relu && (v = max(v, zero(v)))
                                out[ow + 1, oh + 1, kidx + 1, n_idx + 1] = T(v)
                            else
                                # partial sums from each split accumulate; the
                                # graph pre-fills the bias so it is added once
                                Atomix.@atomic out[ow + 1, oh + 1, kidx + 1,
                                                   n_idx + 1] += T(acc_i_j)
                            end
                        end
                    end
                end
            end
        end
    end
    return
end
