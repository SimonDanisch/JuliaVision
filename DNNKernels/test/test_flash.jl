"""
The fused attention kernel: correct where it claims to be, and refusing the
shapes where it is not.

It is not on the `sdpa` path — it measured slower (see the file's docstring) —
so this exists to keep it honest for whoever picks the optimisation back up. A
kernel that is wrong and unused is worse than no kernel: it looks available.
"""

using Test, DNNKernels, Lava, KernelAbstractions
const KA = KernelAbstractions

function attnref(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float32, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(qh[e,lq,h,b] * kh[e,lk,h,b] for e in 1:E) * scale for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e,lq,h,b] = sum(s[lk] * vh[e,lk,h,b] for lk in 1:Lk)
        end
    end
    out
end

@testset "fused attention" begin
    back = LavaBackend()
    # The kernel entry points take a context. `Ctx(backend)` builds one with no
    # graph behind it, which is exactly the direct-call case this file is.
    ctx = DNNKernels.Ctx(back)
    # The tiling chooser and the fit predicates take a `Device` rather than
    # reading one, so they can be asked about hardware this machine is not.
    dev = ctx.dev

    @testset "exact on the validated tiling" begin
        for (E, L, H, B) in ((72, 64, 1, 1), (72, 128, 2, 1), (72, 256, 2, 1))
            qh = randn(Float32,E,L,H,B) .* 0.2f0
            kh = randn(Float32,E,L,H,B) .* 0.2f0
            vh = randn(Float32,E,L,H,B) .* 0.2f0
            q = DNNKernels.toback(back,qh); k = DNNKernels.toback(back,kh); v = DNNKernels.toback(back,vh)
            o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
            scale = Float32(1/sqrt(E))
            @test DNNKernels.sdpaflash!(o, q, k, v, scale; backend = back)
            KA.synchronize(back)
            got = Array(o)
            # A kernel that writes nothing also "matches" a zero reference, so
            # check it produced something before checking what.
            @test maximum(abs, got) > 1e-3
            @test maximum(abs, got .- attnref(qh, kh, vh, scale)) < 1e-5
            q = k = v = o = nothing; GC.gc()
        end
    end

    @testset "exact at every tiling, including the odd slot count" begin
        # `BQ = 32, NT = 256` gives 9 accumulator slots per thread and was wrong
        # from query row 4 on (7.1e-02) until the staging index stopped going
        # through `OpUDiv` — `E = 72` is not a power of two. Varied inputs
        # matter: the broken version was **exact for constant inputs**, which is
        # what disguised it as an accumulator problem for so long.
        E, L, H, B = 72, 128, 2, 2
        qh = randn(Float32,E,L,H,B) .* 0.2f0
        kh = randn(Float32,E,L,H,B) .* 0.2f0
        vh = randn(Float32,E,L,H,B) .* 0.2f0
        q = DNNKernels.toback(back,qh); k = DNNKernels.toback(back,kh); v = DNNKernels.toback(back,vh)
        scale = Float32(1/sqrt(E))
        ref = attnref(qh, kh, vh, scale)
        for (BQ, NT) in ((64, 256), (32, 256), (32, 128), (64, 128))
            @test DNNKernels.flashfits(E, BQ, 32, NT)
            o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
            DNNKernels.attn_flash!(back, NT)(o, q, k, v, scale, Val(BQ), Val(32),
                                             Val(E), Val(NT), Int32(L);
                                             ndrange = (NT * div(L, BQ), H, B))
            KA.synchronize(back)
            got = Array(o)
            @test maximum(abs, got) > 1e-3
            @test maximum(abs, got .- ref) < 1e-4
            o = nothing
        end
        q = k = v = nothing; GC.gc()
    end

    @testset "refuses what it cannot compute" begin
        # Shared memory over budget: this tiling launches and writes zeros.
        @test !DNNKernels.flashfits(72, 64, 64, 256)
        @test DNNKernels.flashshared(72, 64, 64) > DNNKernels.PORTABLE_SHARED_FLOOR
        # The validated one fits, in both senses.
        @test DNNKernels.flashfits(72, 64, 32, 256)
        @test DNNKernels.flashshared(72, 64, 32) <= DNNKernels.PORTABLE_SHARED_FLOOR
        # An odd accumulator-slot count used to be refused, because
        # `BQ = 32, NT = 256` computed wrong results. That was `OpUDiv` in the
        # staging index, not the slot count — see `flashfits` — and `splitidx`
        # fixed it, so both are accepted now and both must be exact.
        @test DNNKernels.flashfits(72, 32, 32, 256)       # BQ*E/NT = 9, odd
        @test DNNKernels.flashfits(72, 32, 32, 128)       # BQ*E/NT = 18
        # …and a sequence the tiling does not divide falls back.
        E, L, H, B = 72, 100, 1, 1              # 100 % 64 != 0
        q = KA.allocate(back, Float32, E,L,H,B); fill!(q, 0.01f0)
        k = KA.allocate(back, Float32, E,L,H,B); fill!(k, 0.01f0)
        v = KA.allocate(back, Float32, E,L,H,B); fill!(v, 0.01f0)
        o = KA.allocate(back, Float32, E,L,H,B)
        @test !DNNKernels.sdpaflash!(o, q, k, v, 0.1f0; backend = back)
        q = k = v = o = nothing; GC.gc()
    end

    # ── the cooperative-matrix form, which IS on the `sdpa` path ─────────────
    if !Lava.coopmat_gemm_available()
        @info "no cooperative-matrix support on this device; skipping the fused path"
    else
        @testset "cooperative-matrix flash: exact at every shipped tiling" begin
            # fp16 operands, because that is what the path requires and what the
            # encoder runs — the tolerance below is the format's, not the
            # algorithm's. Varied inputs for the reason recorded above: the
            # `OpUDiv` failure this kernel's staging indices would otherwise hit
            # is EXACT for constant inputs, so constants prove nothing.
            E, L, H, B = 72, 128, 2, 2
            qh = randn(Float32,E,L,H,B) .* 0.2f0
            kh = randn(Float32,E,L,H,B) .* 0.2f0
            vh = randn(Float32,E,L,H,B) .* 0.2f0
            f16(x) = DNNKernels.toback(back, Float16.(x))
            q, k, v = f16(qh), f16(kh), f16(vh)
            scale = Float32(1/sqrt(E))
            ref = attnref(Float32.(Float16.(qh)), Float32.(Float16.(kh)),
                          Float32.(Float16.(vh)), scale)
            # Both places `O` can live, because `FLASHCM_REGO` is a measured
            # choice and the losing branch still has to be right — a switch whose
            # other side is broken is not a switch.
            for (BR, BC, NW) in DNNKernels.FLASHCM_TILINGS, rego in (false, true),
                lazyrescale in (false, true), held in (false, true)
                NW * 32 <= Lava.WORKGROUP_LIMIT[] || continue
                L % BR == 0 && L % BC == 0 || continue
                o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
                @test DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale;
                                              BR, BC, NW, rego, lazyrescale, held)
                KA.synchronize(back)
                got = Array(o)
                @test maximum(abs, got) > 1e-3          # it wrote something…
                @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3
                o = nothing
            end
            q = k = v = nothing; GC.gc()
        end

        @testset "cooperative-matrix flash: the tiling chooser" begin
            # 64 x 64 wants 66 KB and must never be offered.
            @test !DNNKernels.flashcmfits(dev, 80, 64, 64, 256)
            @test DNNKernels.flashcmfits(dev, 80, 64, 32, 256)
            # The shipped default, on the encoder's two dominant shapes.
            @test DNNKernels.flashcm_tiling(dev, 72, 4096, 4096) == (64, 32, 8)
            @test DNNKernels.flashcm_tiling(dev, 72, 256, 256) == (64, 32, 8)
            # A query count no tiling divides is taken clamped — but only when
            # the padding earns its place. `Lq = 4` would be 94% waste at any
            # tiling, so it still falls back; `Lq = 23` is taken at `BR = 32`
            # (72% occupied) rather than `BR = 64` (36%).
            @test DNNKernels.flashcm_tiling(dev, 72, 4, 16) === nothing
            # Clamping is off by default, so a non-dividing extent is refused
            # until the caller asks for it — which only the decoder does.
            @test DNNKernels.flashcm_tiling(dev, 16, 23, 4096) === nothing
            @test DNNKernels.flashcm_tiling(dev, 16, 23, 4096; clamp=true)[1] == 32
            @test DNNKernels.flashcm_tiling(dev, 16, 23, 23; clamp=true) == (32, 32, 8)
            # …and padding still has to earn its place: 4 queries is 94% waste
            # at any tiling.
            @test DNNKernels.flashcm_tiling(dev, 72, 4, 16; clamp=true) === nothing
            # Every shipped tiling must satisfy the write-out loop's own
            # divisibility, which `flashcmfits` cannot see (it takes the padded
            # head dimension, and the write-out uses the real one).
            for (BR, BC, NW) in DNNKernels.FLASHCM_TILINGS
                @test (BR * 72) % (NW * dev.subgroup) == 0
            end
        end

        @testset "clamped: extents that do not divide the tile" begin
            # The decoder's shapes, which are why CLAMP exists: every one has a
            # 23 in it. Includes Lk = 23 < BC, where the key-block count has to
            # be `cld` — `div` gives ZERO blocks and the loop never runs, which
            # is a silently wrong answer rather than a crash.
            let
                for (E, Lq, Lk, H, B) in ((16, 23, 4096, 8, 1), (32, 23, 23, 8, 1),
                                          (16, 4096, 23, 8, 1), (16, 23, 17, 4, 1))
                    f16r(s, L) = DNNKernels.toback(back,
                                    Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
                    q, k, v = f16r(1, Lq), f16r(2, Lk), f16r(3, Lk)
                    scale = Float32(1/sqrt(E))
                    t = DNNKernels.flashcm_tiling(dev, E, Lq, Lk; clamp=true)
                    @test t !== nothing
                    o = KA.allocate(back, Float32, E, Lq, H, B); fill!(o, 0f0)
                    @test DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; clamp = true,
                                                  BR = t[1], BC = t[2], NW = t[3])
                    KA.synchronize(back)
                    got = Array(o)
                    ref = attnref(Float32.(Array(q)), Float32.(Array(k)), Float32.(Array(v)), scale)
                    @test maximum(abs, got) > 1e-3
                    @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3
                    q = k = v = o = nothing; GC.gc()
                end
            end
        end

        @testset "the one-pass softmax agrees, including past its headroom" begin
            # `onepass` exponentiates against the PREVIOUS block's
            # maximum and defers the correction, which is exact in exact
            # arithmetic but not bit-identical: `ps` is fp16 and now rounds at a
            # different scale, so this is a tolerance and not `==`.
            #
            # The `4096` case is the one that matters. It is 128 key blocks, so
            # the deferred correction is applied 128 times, and `Lk` that long is
            # also where a row is most likely to meet a block hotter than
            # `FLASH_EXP_HEADROOM` and take the two-pass fallback — the branch
            # that would otherwise never run in a test.
            for (E, Lq, Lk, H, B) in ((72, 128, 256, 2, 2), (72, 64, 4096, 2, 1))
                f16r(s, L) = DNNKernels.toback(back,
                                Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
                q, k, v = f16r(1, Lq), f16r(2, Lk), f16r(3, Lk)
                scale = Float32(1/sqrt(E))
                outs = map((false, true)) do op
                    o = KA.allocate(back, Float32, E, Lq, H, B); fill!(o, 0f0)
                    @test DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale; onepass = op)
                    KA.synchronize(back)
                    Array(o)
                end
                @test maximum(abs, outs[1]) > 1e-3
                @test maximum(abs, outs[1] .- outs[2]) / maximum(abs, outs[1]) < 1e-3
                q = k = v = nothing; GC.gc()
            end
        end

        @testset "skipping the rescale is exact, not approximate" begin
            # `lazyrescale` skips `O *= exp(m_old - m_new)` on blocks
            # where no row's max moved, i.e. where the factor is exactly one. If
            # that reasoning is ever wrong the two answers differ, so they are
            # compared to each other rather than to a tolerance.
            E, L, H, B = 72, 256, 2, 2
            f16r(s) = DNNKernels.toback(back, Float16.(randn(Float32,E,L,H,B) .* 0.2f0))
            q, k, v = f16r(1), f16r(2), f16r(3)
            scale = Float32(1/sqrt(E))
            outs = map((false, true)) do lazy
                o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
                DNNKernels.sdpaflashcm!(ctx, o, q, k, v, scale;
                                        lazyrescale = lazy, onepass = false)
                KA.synchronize(back)
                Array(o)
            end
            @test outs[1] == outs[2]
            q = k = v = nothing; GC.gc()
        end

        @testset "cooperative-matrix flash agrees with the path it replaces" begin
            # This guards the ROUTING, not the kernel: `sdpa` must recognise the
            # shape and fuse it, and the fused answer must match the two-GEMM
            # path it displaced. It used to flip `FLASHCM` to get the second
            # answer; that switch was settled and deleted, so the alternative
            # path is now called by name — which also makes the comparison
            # independent of what `sdpa` would fall through to next.
            E, L, H, B = 72, 256, 2, 2
            f16r(s) = DNNKernels.toback(back, Float16.(randn(Float32,E,L,H,B) .* 0.2f0))
            q, k, v = f16r(1), f16r(2), f16r(3)
            wctx = DNNKernels.Ctx(back; ws = DNNKernels.Workspace(back))
            scale = Float32(1/sqrt(E))
            # A plan, not a `Bool`: the same call that decides also carries the
            # tiling, so there is nothing left for a second predicate to disagree
            # with.
            @test DNNKernels.flashcm_plan(dev, q, k, v, nothing) isa DNNKernels.FlashCMPlan
            DNNKernels.reset!(wctx.ws)
            fused = Array(DNNKernels.sdpa(wctx, q, k, v, nothing, scale))
            KA.synchronize(back)
            cmplan = DNNKernels.coopmat_sdpa_plan(dev, q, k, v, nothing)
            @test cmplan isa DNNKernels.CoopMatSDPAPlan
            DNNKernels.reset!(wctx.ws)
            o = KA.allocate(back, Float32, E, L, H, B); fill!(o, 0f0)
            DNNKernels.sdpa_coopmat!(wctx, o, cmplan, q, k, v, scale)
            KA.synchronize(back)
            twogemm = Array(o)
            @test maximum(abs, twogemm) > 1e-3
            @test maximum(abs, fused .- twogemm) / maximum(abs, twogemm) < 5e-3
            q = k = v = o = nothing; GC.gc()
        end
    end
end

@testset "the tiling chooser prefers occupancy when the grid cannot fill the device" begin
    # `FLASHCM_TILINGS` is ordered fastest-first *for a launch that fills the
    # card*. SAM 2's decoder is where none does: `Lq = 23` (the mask prompt's
    # token count) against `Lk = 4096` is one query block, so the grid is
    # `1 * H*B` = 8 workgroups on 48 SMs and the kernel measures 0.10 TFLOP/s.
    # Below one workgroup per shader core the chooser therefore picks for grid
    # size instead.
    dev = DNNKernels.Device(LavaBackend())
    tiling(args...) = DNNKernels.flashcm_tiling(dev, args...; clamp = true)

    # Without a batch count it must behave exactly as it always did.
    @test tiling(16, 23, 4096) == tiling(16, 23, 4096, 0)

    with = tiling(16, 23, 4096, 8)
    @test with !== nothing
    @test cld(23, with[1]) * 8 > cld(23, tiling(16, 23, 4096)[1]) * 8
    # ...and the choice is still a legal tiling for the shape.
    @test 2 * 23 >= with[1]

    # The other two decoder shapes: one is already wide, one is not.
    @test tiling(16, 4096, 23, 8) == tiling(16, 4096, 23)
    @test tiling(32, 23, 23, 8) !== nothing

    @testset "inert on every encoder shape" begin
        # These launch 512 workgroups and must keep the table's own order, or the
        # rule has quietly re-tuned the encoder against a decoder measurement.
        for (E, Lq, Lk, nb) in [(72, 4096, 4096, 8), (72, 256, 256, 128),
                                (72, 512, 512, 32), (72, 1024, 1024, 16)]
            @test tiling(E, Lq, Lk, nb) == tiling(E, Lq, Lk)
        end
    end

    @testset "a grid at the threshold is not re-chosen" begin
        # The core count is a floor, not a target: at or above it the table's
        # order stands. Asked by describing a one-core device rather than by
        # writing to a global and restoring it — which a failing `@test` inside
        # the `try` would have skipped, leaving it flipped for everything after.
        one = DNNKernels.Device(dev.coopmat, dev.tile, dev.subgroup, dev.sharedbudget,
                                dev.workgrouplimit, 1, dev.launchgroup)
        onetiling(args...) = DNNKernels.flashcm_tiling(one, args...; clamp = true)
        @test onetiling(16, 23, 4096, 8) == onetiling(16, 23, 4096)
    end
end

@testset "plans: one decision, and a refusal that says why" begin
    back = LavaBackend()
    ctx = DNNKernels.Ctx(back)
    dev = ctx.dev
    E, L, H, B = 72, 256, 2, 1
    f16(s) = DNNKernels.toback(back, Float16.(randn(Float32, E, L, H, B) .* 0.2f0))
    q, k, v = f16(1), f16(2), f16(3)

    # ── What this replaced could drift; this cannot.
    #
    # `flashcm_applicable` ran `flashcm_tiling` and threw the answer away, `sdpa`
    # ran it again, and `sdpaflashcm!` then re-checked six more conditions and
    # could still decline — after `out` was allocated. The plan is the single
    # answer, and it carries the tiling that decision was made with.
    p = DNNKernels.flashcm_plan(dev, q, k, v, nothing)
    @test p isa DNNKernels.FlashCMPlan
    @test (p.BR, p.BC, p.NW) == DNNKernels.flashcm_tiling(dev, E, L, L, H * B)
    @test p.NT == p.NW * dev.subgroup
    @test p.EP == cld(E, dev.tile) * dev.tile

    # ── Every refusal is named, so a caller can react and a test can assert
    # which rule fired. `nothing` could do neither.
    @test DNNKernels.flashcm_plan(dev, q, k, v, k).reason === :bias
    @test DNNKernels.flashcm_plan(dev, q, k, v, nothing; BR = 64, BC = 64,
                                  NW = 8).reason === :tiling
    f32 = DNNKernels.toback(back, randn(Float32, E, L, H, B))
    @test DNNKernels.flashcm_plan(dev, f32, k, v, nothing).reason === :eltype
    nocm = DNNKernels.Device(false, dev.tile, dev.subgroup, dev.sharedbudget,
                             dev.workgrouplimit, dev.cores, dev.launchgroup)
    @test DNNKernels.flashcm_plan(nocm, q, k, v, nothing).reason === :nocoopmat
    @test DNNKernels.flashcm_plan(dev, q, k, v, nothing; BR = 23).reason === :extent

    # ── A plan is an answer about a device, so it can be asked about one this
    # machine does not have. `NW * 32` was a literal in three places, and on a
    # wave64 part it names half the workgroup the tiling assumes.
    w64 = DNNKernels.Device(true, 16, 64, 65536, 1024, 40, 256)
    p64 = DNNKernels.flashcm_plan(w64, q, k, v, nothing)
    @test p64 isa DNNKernels.FlashCMPlan
    @test p64.NT == p64.NW * 64

    # ── The clamp is per run, not per process. Two contexts, opposite policies,
    # both alive at once — which the `Ref` it replaced could not express, and
    # which is why SAM 2 had to set it and restore it in a `finally`.
    dctx = DNNKernels.Ctx(back; clampattn = true)
    @test !ctx.clampattn && dctx.clampattn
    E2, Lq2 = 16, 23
    g16(s, L) = DNNKernels.toback(back, Float16.(randn(Float32, E2, L, 8, 1) .* 0.2f0))
    q2, k2, v2 = g16(1, Lq2), g16(2, 4096), g16(3, 4096)
    @test DNNKernels.flashcm_plan(ctx.dev, q2, k2, v2, nothing;
                                  clamp = ctx.clampattn) isa DNNKernels.Decline
    @test DNNKernels.flashcm_plan(dctx.dev, q2, k2, v2, nothing;
                                  clamp = dctx.clampattn) isa DNNKernels.FlashCMPlan
end
