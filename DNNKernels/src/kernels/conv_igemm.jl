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
    @nounroll for … end
    @nounroll while … end

The loop with `llvm.loop.unroll.disable` on it: LLVM leaves it rolled, and Lava
writes DontUnroll into its `OpLoopMerge`, so a Vulkan driver leaves it rolled too.
"""
macro nounroll(loop)
    Meta.isexpr(loop, (:for, :while)) ||
        throw(ArgumentError("@nounroll takes a `for` or `while` loop"))
    iter, body = loop.args
    # Last in the body, which is where lowering looks for it.
    hint = Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),))
    return esc(Expr(loop.head, iter, Expr(:block, body, hint)))
end

"""
    conv2d_igemm_ki!(out, x, w, bias, Val(ACC), Val(SPLITK), Val(ACT), …) -> nothing

One block tile of the implicit GEMM. See this file's header.

The `Val` parameters are the block and thread tiling, the kernel extents and the
stride/padding/dilation, all of which `convtiles` and the op choose at emit time
so the body specialises on them. `ACC` is the accumulator and shared-memory
element type and is a parameter rather than `accum(eltype(x))` computed in the
body, because workgroup memory needs a static type.

The index arithmetic runs in the integer type of the extents, and the caller
passes `Int32` whenever every operand fits ([`convindex`](@ref)). See the note in
the body for what `Int64` cost.
"""
function conv2d_igemm_ki!(out, x, w, bias,
                          ::Val{ACC}, ::Val{SPLITK}, ::Val{ACT},
                          ::Val{BS_K}, ::Val{BS_CRS}, ::Val{BS_NPQ},
                          ::Val{TS_K}, ::Val{TS_NPQ},
                          ::Val{KW}, ::Val{KH},
                          ::Val{SX}, ::Val{SY}, ::Val{PX}, ::Val{PY},
                          ::Val{DX}, ::Val{DY},
                          Cin::I, Cout::I, Wid::I, Hei::I, OW::I, OH::I,
                          NPQ::I, CRS::I, NBN::I) where {ACC,SPLITK,ACT,BS_K,BS_CRS,BS_NPQ,
                                                         TS_K,TS_NPQ,KW,KH,SX,SY,PX,PY,DX,DY,
                                                         I<:Integer}
    T = eltype(out)
    A = ACC
    # ── the index type ────────────────────────────────────────────────────
    #
    # Everything below is in `I`, the extents' type, constants included: one
    # `Int64` literal promotes the whole expression back. With `Int64` the Qwen-
    # Image 2.1 VAE's `convolution_37` (3x3, 288 -> 288, 1024x1024, fp32) compiled
    # on RADV to 125,360 instructions and 256 VGPRs at 4 waves per SIMD, and ran
    # in 1585 ms: every `÷` by a runtime extent is a 64-bit division, which the
    # driver expands in software to roughly 900 instructions, and the 64-bit
    # addresses held two registers each across the reduction loop. In `Int32`,
    # with the write-back below decomposing each column once, it is 4,474
    # instructions, 156 VGPRs at 8 waves, and 335 ms. The same change on ROCm,
    # where LLVM already bypasses a 64-bit division whose operands fit in 32 bits,
    # is 363 -> 280 ms. Outputs are bit-identical.
    o = one(I)
    z = zero(I)

    NT_K = I(BS_K ÷ TS_K)
    NT_NPQ = I(BS_NPQ ÷ TS_NPQ)
    WG = NT_K * NT_NPQ
    ArpWg = WG ÷ I(BS_CRS)
    BrpWg = max(WG ÷ I(BS_NPQ), o)
    Ash_stride = I(BS_CRS + 4)
    Bsh_stride = I(BS_NPQ + 4)
    KWKH = I(KW * KH)
    nblk = cld(CRS, I(BS_CRS))
    # Split-K: the reduction is divided over SPLITK workgroups, which is the only
    # way to get both a large thread tile (arithmetic intensity) and enough
    # workgroups to fill the device. This model's dominant convolution is
    # 256 output channels over 120 pixels with a 2304-deep reduction: output
    # tiling alone yields 16 workgroups on 48 SMs, and shrinking the tile to get
    # more drops the MAC:load ratio from 16:8 to 2:3.
    blkper = cld(nblk, I(SPLITK))

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
    tid = (KI.get_local_id().x - 1) % I
    bk = KI.get_group_id().x % I
    gn = KI.get_group_id().y % I
    # the second grid axis carries both the NPQ block and the split index
    bnpq = (gn - o) % NBN + o
    ksplit = (gn - o) ÷ NBN
    B_idx_K = (bk - o) * I(BS_K)
    B_idx_NPQ = (bnpq - o) * I(BS_NPQ)
    Ar = tid ÷ I(BS_CRS)
    Ac = tid % I(BS_CRS)
    Br = tid ÷ I(BS_NPQ)
    Bc = tid % I(BS_NPQ)
    T_y = tid ÷ NT_NPQ
    T_x = tid % NT_NPQ

    # The column a thread stages into B is the same on every block of the
    # reduction, so it is decomposed once, here, and not per staging pass.
    npq_b = B_idx_NPQ + Bc
    n_b = npq_b ÷ (OH * OW)
    r_b = npq_b - n_b * OH * OW
    oh_b = r_b ÷ OW
    ow_b = r_b - oh_b * OW
    ix0 = ow_b * I(SX) - I(PX)
    iy0 = oh_b * I(SY) - I(PY)
    xbase = Wid * Hei * Cin * n_b
    npq_ok = npq_b < NPQ

    # All 64, not only the ones inside the tile: the `if i <= TS_K` guards below
    # fold away at compile time, but the name still has to be bound for the
    # body to lower. The unused ones are dead stores the optimiser drops.
    Base.Cartesian.@nexprs 8 j -> Base.Cartesian.@nexprs 8 i -> acc_i_j = zero(A)

    bb = z
    while bb < blkper
        # Blocks past the end make `crs_a`/`crs_b` exceed CRS, so the bounds
        # guards below already stage zeros. The trip count stays uniform, which
        # a workgroup barrier requires: every thread has to reach every one.
        b_crs = ksplit * blkper + bb

        # ── stage A (the kernel), BS_K x BS_CRS ───────────────────────────
        #
        # Linear offsets rather than four subscripts: an N-d index is widened
        # to `Int` by the array, which is the 64-bit arithmetic this avoids.
        #
        # Both stages LOAD unconditionally and select zero afterwards; an index
        # the guard rejects is replaced by 1, which is always in bounds. Written
        # `ok ? A(w[i]) : zero(A)` the load sits inside a branch, and both LLVM
        # (ROCm) and ACO (RADV) then waited on each gather before starting the
        # next: eight memory latencies in a row per stage, hidden only by
        # whatever other waves were resident. Branch-free, `convolution_37`
        # (3x3, 288 -> 288, 1024x1024) went 273 -> 261 ms on ROCm and
        # 340 -> 324 on RADV. Same values: the zero is still selected.
        @inbounds begin
            crs_a = b_crs * I(BS_CRS) + Ac
            cin_a = crs_a ÷ KWKH
            rem_a = crs_a - cin_a * KWKH
            kh_a = rem_a ÷ I(KW)
            kw_a = rem_a - kh_a * I(KW)
            wlin = kw_a + I(KW) * (kh_a + I(KH) * cin_a)
            r = z
            while r < I(BS_K)
                ky = r + Ar
                kidx = B_idx_K + ky
                ok = kidx < Cout && crs_a < CRS
                v = w[ok ? wlin + KWKH * Cin * kidx + o : o]
                Ash[ky * Ash_stride + Ac + o] = ok ? A(v) : zero(A)
                r += ArpWg
            end

            # ── stage B (the input), BS_CRS x BS_NPQ, gathered by im2col ───
            r = z
            while r < I(BS_CRS)
                by = r + Br
                crs_b = b_crs * I(BS_CRS) + by
                cin_b = crs_b ÷ KWKH
                rem_b = crs_b - cin_b * KWKH
                kh_b = rem_b ÷ I(KW)
                kw_b = rem_b - kh_b * I(KW)
                ix = ix0 + kw_b * I(DX)
                iy = iy0 + kh_b * I(DY)
                inb = (z <= ix < Wid) && (z <= iy < Hei) && npq_ok && crs_b < CRS
                v = x[inb ? xbase + ix + Wid * (iy + Hei * cin_b) + o : o]
                Bsh[by * Bsh_stride + Bc + o] = inb ? A(v) : zero(A)
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
        #
        # Rolled, on purpose. LLVM's AMDGPU backend declines to unroll it, but
        # a Vulkan driver sees a constant trip count and unrolls all BS_CRS
        # steps unless told not to: RADV then held 156 VGPRs, too many for the
        # six workgroups per WGP that shared memory allows, against 120 rolled.
        # `convolution_37` measured 324 ms unrolled and 281 rolled on RADV;
        # rolling the B stage above instead gave 287, and both 309.
        @inbounds @nounroll for k in z:I(BS_CRS - 1)
            Base.Cartesian.@nexprs 8 i -> a_i = i <= TS_K ?
                Ash[(T_y + I(i - 1) * NT_K) * Ash_stride + k + o] : zero(A)
            Base.Cartesian.@nexprs 8 j -> b_j = j <= TS_NPQ ?
                Bsh[k * Bsh_stride + T_x + I(j - 1) * NT_NPQ + o] : zero(A)
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
        bb += o
    end

    # ── write back ────────────────────────────────────────────────────────
    #
    # Columns outside, rows inside: a column's `(n, oh, ow)` is decomposed once
    # and not once per row. The other nesting put the two divisions inside
    # sixty-four separately guarded blocks, which no pass merges.
    OHW = OH * OW
    @inbounds Base.Cartesian.@nexprs 8 j -> begin
        if j <= TS_NPQ
            npq_j = B_idx_NPQ + T_x + I(j - 1) * NT_NPQ
            n_j = npq_j ÷ OHW
            # `ow + OW * oh` is the remainder itself, so it needs no division.
            obase_j = (npq_j - n_j * OHW) + OHW * Cout * n_j
            ok_j = npq_j < NPQ
            Base.Cartesian.@nexprs 8 i -> begin
                if i <= TS_K
                    kidx = B_idx_K + T_y + I(i - 1) * NT_K
                    if kidx < Cout && ok_j
                        bv = bias === nothing ? zero(A) : A(bias[kidx + o])
                        if SPLITK == 1
                            v = acc_i_j + bv
                            ACT === :relu && (v = max(v, zero(v)))
                            out[obase_j + OHW * kidx + o] = T(v)
                        else
                            # partial sums from each split accumulate; the
                            # graph pre-fills the bias so it is added once
                            Atomix.@atomic out[obase_j + OHW * kidx + o] += T(acc_i_j)
                        end
                    end
                end
            end
        end
    end
    return
end
