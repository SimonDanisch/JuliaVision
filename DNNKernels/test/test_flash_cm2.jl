"""
The coopmat2 flash kernel: correct where it runs, and refusing what it should.

`attn_flash_cm2!` is on the `sdpa` path wherever this device has workgroup-scope
cooperative matrices, so this file has two jobs — it is not only "does the kernel
compute attention", it is also "does the PLAN hand back the calls this kernel is
wrong for". The second half is what keeps the decoder's `Lq = 23` on the cm1 path
that beats it by 84%, and it is checkable on a machine with no such device at
all, because every predicate takes a `DeviceCaps`.

`attnref` is the same triple-loop reference `test_flash.jl` uses: no shared code
with either GPU path, and slow enough that the shapes here stay small.
"""

using Test, DNNKernels, Lava, KernelAbstractions

const KA = KernelAbstractions
const DK = DNNKernels
const MT = DNNKernels.M

function attnref_cm2(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float32, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(Float32(qh[e,lq,h,b]) * Float32(kh[e,lk,h,b]) for e in 1:E) * scale
             for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e,lq,h,b] = sum(s[lk] * Float32(vh[e,lk,h,b]) for lk in 1:Lk)
        end
    end
    out
end

@testset "flash coopmat2 (workgroup scope)" begin
    back = LavaBackend()
    ctx = DK.Ctx(back)
    dev = ctx.dev

    # ── The parts that need no such device ────────────────────────────────────
    @testset "granularity comes from the device table" begin
        # An invented device, so the assertions are about the LOOKUP and not
        # about this card. Rows are (invocations, M, N, K).
        d = MT.DeviceCaps(dev; wggran = [(64, 16, 16, 16), (128, 32, 16, 16),
                                         (256, 32, 32, 16)])
        @test DK.cm2granularity(d, 128) == (32, 16, 16)
        @test DK.cm2granularity(d, 512) === nothing      # not a size it reports
        # `EP` takes the coarser of the N and K granularity, because the head
        # extent is the K of `Q·Kᵀ` and the N of `P·V`.
        @test DK.cm2pad(d, 72, 128) == 80
        @test DK.cm2pad(d, 72, 256) == 96
        @test DK.cm2pad(d, 64, 256) == 64
        # …and a tiling is legal only if every extent divides its own granularity.
        @test DK.flashcm2fits(d, 72, 64, 32, 128)
        @test !DK.flashcm2fits(d, 72, 16, 32, 128)       # BR 16 < M granularity 32
        @test !DK.flashcm2fits(d, 72, 64, 16, 256)       # BC 16 < N granularity 32

        # A device without workgroup-scope matrices declines, rather than
        # producing a tiling nothing can run.
        none = MT.DeviceCaps(dev; wggran = NTuple{4,Int}[])
        @test DK.cm2granularity(none, 128) === nothing
        @test !DK.flashcm2fits(none, 72, 64, 32, 128)

        # The workgroup size is chosen by least padding, then `FLASHCM2_NT`
        # (128), then widest — the rule the measurement produced, not the number.
        # `E = 72` pads to 80 at 64 and 128 invocations and to 96 at 256, so the
        # padding term alone picks 128.
        @test DK.flashcm2_workgroup(d, 72) == 128
        # `E = 96` pads NOWHERE, so the tie-break decides alone — and this
        # asserted 256 until 2026-08-12, when Whisper's `E = 64` (which also pads
        # nowhere) measured `64x32/256` at 0.640 ms against `128x32/128`'s 0.282,
        # i.e. the "widest" tie-break was worth **-2.27x** on a real model.
        @test DK.flashcm2_workgroup(d, 96) == 128
        @test DK.flashcm2_workgroup(none, 72) === nothing
        # …and the preference is a PREFERENCE: a device that does not offer 128
        # still gets the widest of what it has, which is the next assertion.
        # A device that only offers the coarse size still gets an answer.
        coarse = MT.DeviceCaps(dev; wggran = [(256, 32, 32, 16)])
        @test DK.flashcm2_workgroup(coarse, 72) == 256
        @test DK.cm2pad(coarse, 72, 256) == 96
    end

    @testset "the plan refuses what it should" begin
        E, L, H, B = 72, 256, 8, 4
        mk() = KA.allocate(back, Float16, E, L, H, B)
        q, k, v = mk(), mk(), mk()
        d = MT.DeviceCaps(dev; wggran = [(128, 32, 16, 16)], cores = 48)

        @test DK.flashcm2_plan(d, q, k, v, nothing) isa DK.FlashCM2Plan
        # A mask has to be added between the product and the softmax, which is
        # inside a matrix here — the same refusal the cm1 path makes.
        @test DK.flashcm2_plan(d, q, k, v, q).reason === :bias
        # No workgroup-scope matrices at all.
        @test DK.flashcm2_plan(MT.DeviceCaps(d; wggran = NTuple{4,Int}[]),
                               q, k, v, nothing).reason === :nowgscope
        # The grid rule: one query block times `H*B` has to fill the device,
        # because this kernel has no key-axis split. SAM 2's decoder cross
        # attention is `Lq = 23, H = 8, B = 1` — 8 workgroups on 48 cores.
        q23 = KA.allocate(back, Float16, E, 23, 8, 1)
        k4k = KA.allocate(back, Float16, E, 4096, 8, 1)
        @test DK.flashcm2_plan(d, q23, k4k, k4k, nothing).reason === :notiling
        # …and it is the GRID that decides, not `Lq`: the same short query axis
        # with enough head-batches is accepted.
        q23b = KA.allocate(back, Float16, E, 23, 8, 64)
        k4kb = KA.allocate(back, Float16, E, 4096, 8, 64)
        @test DK.flashcm2_plan(d, q23b, k4kb, k4kb, nothing) isa DK.FlashCM2Plan
        # fp32 operands go elsewhere.
        f32 = KA.allocate(back, Float32, E, L, H, B)
        @test DK.flashcm2_plan(d, f32, f32, f32, nothing).reason === :eltype
        q = k = v = q23 = k4k = q23b = k4kb = f32 = nothing; GC.gc()
    end

    # ── The kernel itself, if this device can run it ──────────────────────────
    if isempty(dev.wggran)
        @info "no workgroup-scope cooperative matrices here; kernel not exercised" dev
    else
        @testset "correct against the reference" begin
            # `Lq = 100` and `Lk = 200` divide neither tile: the clamping tensor
            # loads pad them and the masked softmax excludes the padding. That is
            # unconditional here, so a regression in it shows up as a wrong
            # answer on this shape and nowhere else.
            for (E, Lq, Lk, H, B) in ((72, 128, 256, 2, 1), (72, 100, 200, 2, 1),
                                      (64, 64, 64, 1, 2))
                qh = Float16.(randn(Float32, E, Lq, H, B) .* 0.2f0)
                kh = Float16.(randn(Float32, E, Lk, H, B) .* 0.2f0)
                vh = Float16.(randn(Float32, E, Lk, H, B) .* 0.2f0)
                q, k, v = DK.toback(back, qh), DK.toback(back, kh), DK.toback(back, vh)
                o = KA.allocate(back, Float32, E, Lq, H, B); fill!(o, Float32(NaN))
                scale = Float32(1 / sqrt(E))
                # An explicit tiling, so this tests the kernel rather than the
                # chooser — the grid rule would otherwise decline these sizes.
                @test DK.sdpaflashcm2!(ctx, o, q, k, v, scale; BR = 64, BC = 64, NT = 128)
                KA.synchronize(back)
                got = Array(o)
                # A kernel that writes nothing matches a zero reference too.
                @test all(isfinite, got)
                @test maximum(abs, got) > 1e-3
                @test maximum(abs, got .- attnref_cm2(qh, kh, vh, scale)) < 3e-3
                q = k = v = o = nothing; GC.gc()
            end
        end

        @testset "the fused softmax agrees, and every ablation runs" begin
            # Two things that are only reachable through diagnostic kwargs and
            # would otherwise rot unnoticed.
            #
            # `osum = :off` is the reduce-based row sum, kept for devices
            # whose `EP` equals `E`. `fused = true` is the matrix-operand
            # formulation. It is NOT the
            # shipped path — it measures +31% — but the docstring's negative
            # result rests on it being *correct*, so an equivalence that stops
            # holding would quietly turn that table into a comparison of two
            # different functions.
            #
            # The `abl` variants compute wrong answers on purpose, so there is
            # nothing to check them against. What can rot is that one stops
            # compiling, or gets folded away because its result is unused — and
            # then the ablation table reports a saving that is really a dead
            # kernel. Finite, non-zero, and different from `:all` is what that
            # can be held to.
            E, Lq, Lk, H, B = 72, 128, 256, 2, 1
            qh = Float16.(randn(Float32, E, Lq, H, B) .* 0.2f0)
            kh = Float16.(randn(Float32, E, Lk, H, B) .* 0.2f0)
            vh = Float16.(randn(Float32, E, Lk, H, B) .* 0.2f0)
            q, k, v = DK.toback(back, qh), DK.toback(back, kh), DK.toback(back, vh)
            scale = Float32(1 / sqrt(E))
            want = attnref_cm2(qh, kh, vh, scale)

            run = function (; kw...)
                o = KA.allocate(back, Float32, E, Lq, H, B); fill!(o, Float32(NaN))
                @test DK.sdpaflashcm2!(ctx, o, q, k, v, scale;
                                       BR = 64, BC = 64, NT = 128, kw...)
                KA.synchronize(back)
                Array(o)
            end

            base = run()
            @test maximum(abs, base .- want) < 3e-3

            # All three ways to get the row sum must agree. `:off` is the
            # fallback for a head dimension whose padding has no spare column
            # (`E = 72` has one, so without this the reduce-based form would
            # only ever be exercised by the `E = 64` shape above); `:pass` is
            # the per-element ones column, which is what `:fill` has to match
            # to be believed, since `:fill` gets its ones from the driver and a
            # clamp value that silently did nothing would look exactly like a
            # kernel that never needed one.
            for m in (:off, :pass)
                g = run(osum = m)
                @test maximum(abs, g .- want) < 3e-3
                @test maximum(abs, g .- base) < 1e-5
            end
            # A mode that is not a mode must be refused rather than producing a
            # kernel that writes no ones column and divides by zero. NOT through
            # `run`: its inner `@test` catches the exception and records it as a
            # failure, so `@test_throws` would never see one thrown.
            let o = KA.allocate(back, Float32, E, Lq, H, B)
                bad(; kw...) = DK.sdpaflashcm2!(ctx, o, q, k, v, scale;
                                                BR = 64, BC = 64, NT = 128, kw...)
                @test_throws ArgumentError bad(osum = :fil)
                @test_throws ArgumentError bad(abl = :norescal)
            end

            # `BC == EP` makes the `Br x EP` smear the identity, so the reduce
            # that resizes `eM` is compiled out entirely. It is a different
            # kernel body from every other tiling here and only `BC = 80`
            # reaches it at `E = 72`.
            let o = KA.allocate(back, Float32, E, Lq, H, B)
                fill!(o, Float32(NaN))
                @test DK.sdpaflashcm2!(ctx, o, q, k, v, scale;
                                       BR = 64, BC = 80, NT = 128)
                KA.synchronize(back)
                g80 = Array(o)
                @test maximum(abs, g80 .- want) < 3e-3
                @test maximum(abs, g80 .- base) < 1e-5
            end

            gf = run(fused = true)
            @test all(isfinite, gf)
            @test maximum(abs, gf .- want) < 3e-3
            # Tighter than the reference tolerance: these are two spellings of
            # one computation, not two approximations of it.
            @test maximum(abs, gf .- base) < 1e-5

            for a in (:noreduce, :noexp, :norescale, :nosoftmax)
                g = run(abl = a)
                @test all(isfinite, g)                      # it ran
                @test maximum(abs, g) > 1e-3                # and wrote something
                # …and computed something ELSE. The bar is what separates "this
                # variant dropped work" from "this variant was folded away and
                # the ablation table is measuring one kernel four times", so it
                # only has to sit above fp32 agreement — the variants that skip
                # a whole reduction land at 1e-1, but `:norescale` on a shape
                # whose row maxima barely move comes in at 9.7e-4, and a 1e-3
                # bar failed it for being too nearly right.
                @test maximum(abs, g .- base) > 1e-5
            end
            q = k = v = nothing; GC.gc()
        end

        @testset "writes the destination it was given, not the one it assumed" begin
            # This is the case that took SAM 2's whole encoder to NaN while every
            # test above passed: the kernel wrote fp32 bytes into the **fp16**
            # slot a graph reserves for attention, because everything synthetic
            # allocates its own fp32 output. A destination is an input.
            E, Lq, Lk, H, B = 72, 128, 256, 2, 1
            qh = Float16.(randn(Float32, E, Lq, H, B) .* 0.2f0)
            kh = Float16.(randn(Float32, E, Lk, H, B) .* 0.2f0)
            vh = Float16.(randn(Float32, E, Lk, H, B) .* 0.2f0)
            q, k, v = DK.toback(back, qh), DK.toback(back, kh), DK.toback(back, vh)
            want = attnref_cm2(qh, kh, vh, Float32(1 / sqrt(E)))
            scale = Float32(1 / sqrt(E))

            o16 = KA.allocate(back, Float16, E, Lq, H, B); fill!(o16, Float16(NaN))
            @test DK.sdpaflashcm2!(ctx, o16, q, k, v, scale; BR = 64, BC = 64, NT = 128)
            KA.synchronize(back)
            g16 = Float32.(Array(o16))
            @test all(isfinite, g16)
            @test maximum(abs, g16 .- want) < 3e-3      # fp16 destination, fp32 math

            # …and a destination that is a VIEW, so the store goes through real
            # strides rather than the packed ones a fresh allocation has.
            big = KA.allocate(back, Float32, E, Lq, H, 2B); fill!(big, Float32(NaN))
            ov = view(big, :, :, :, (B + 1):(2B))
            @test DK.sdpaflashcm2!(ctx, ov, q, k, v, scale; BR = 64, BC = 64, NT = 128)
            KA.synchronize(back)
            gv = Array(big)[:, :, :, (B + 1):(2B)]
            @test all(isfinite, gv)
            @test maximum(abs, gv .- want) < 3e-3
            # The half it must NOT have touched.
            @test all(isnan, Array(big)[:, :, :, 1:B])
            q = k = v = o16 = big = ov = nothing; GC.gc()
        end

        @testset "sdpa routes to it on an encoder-shaped call" begin
            E, Lq, H, B = 72, 256, 8, 4
            mk(L) = DK.toback(back, Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
            q, k, v = mk(Lq), mk(Lq), mk(Lq)
            plan, _, _ = DK.sdpaplan(ctx, q, k, v, nothing)
            @test plan isa DK.FlashCM2Plan
            # …and the kill switch really hands the call back, rather than being
            # a field nothing reads.
            off = DK.Ctx(back; flashcm2 = false)
            planoff, _, _ = DK.sdpaplan(off, q, k, v, nothing)
            @test !(planoff isa DK.FlashCM2Plan)
            # And the result still matches the path it took over from.
            o2 = DK.sdpa(ctx, q, k, v, nothing, Float32(1 / sqrt(E)))
            o3 = DK.sdpa!(ctx, DK.Decline(:threepass), nothing, q, k, v, nothing,
                          Float32(1 / sqrt(E)))
            KA.synchronize(back)
            a2, a3 = Array(o2), Array(o3)
            @test maximum(abs, a2 .- a3) / maximum(abs, a3) < 5e-3
            q = k = v = o2 = o3 = nothing; GC.gc()
        end
    end
end
