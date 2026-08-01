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
        @test DNNKernels.flashshared(72, 64, 64) > DNNKernels.FLASH_SHARED_BUDGET[]
        # The validated one fits, in both senses.
        @test DNNKernels.flashfits(72, 64, 32, 256)
        @test DNNKernels.flashshared(72, 64, 32) <= DNNKernels.FLASH_SHARED_BUDGET[]
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
                @test DNNKernels.sdpaflashcm!(o, q, k, v, scale; backend = back,
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
            @test !DNNKernels.flashcmfits(80, 64, 64, 256)
            @test DNNKernels.flashcmfits(80, 64, 32, 256)
            # The shipped default, on the encoder's two dominant shapes.
            @test DNNKernels.flashcm_tiling(72, 4096, 4096) == (64, 32, 8)
            @test DNNKernels.flashcm_tiling(72, 256, 256) == (64, 32, 8)
            # A query count no tiling divides falls back to the other paths.
            @test DNNKernels.flashcm_tiling(72, 4, 16) === nothing
            # Every shipped tiling must satisfy the write-out loop's own
            # divisibility, which `flashcmfits` cannot see (it takes the padded
            # head dimension, and the write-out uses the real one).
            for (BR, BC, NW) in DNNKernels.FLASHCM_TILINGS
                @test (BR * 72) % (NW * 32) == 0
            end
        end

        @testset "the one-pass softmax agrees, including past its headroom" begin
            # `FLASHCM_ONEPASS` exponentiates against the PREVIOUS block's
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
                    @test DNNKernels.sdpaflashcm!(o, q, k, v, scale; backend = back,
                                                  onepass = op)
                    KA.synchronize(back)
                    Array(o)
                end
                @test maximum(abs, outs[1]) > 1e-3
                @test maximum(abs, outs[1] .- outs[2]) / maximum(abs, outs[1]) < 1e-3
                q = k = v = nothing; GC.gc()
            end
        end

        @testset "skipping the rescale is exact, not approximate" begin
            # `FLASHCM_LAZYRESCALE` skips `O *= exp(m_old - m_new)` on blocks
            # where no row's max moved, i.e. where the factor is exactly one. If
            # that reasoning is ever wrong the two answers differ, so they are
            # compared to each other rather than to a tolerance.
            E, L, H, B = 72, 256, 2, 2
            f16r(s) = DNNKernels.toback(back, Float16.(randn(Float32,E,L,H,B) .* 0.2f0))
            q, k, v = f16r(1), f16r(2), f16r(3)
            scale = Float32(1/sqrt(E))
            outs = map((false, true)) do lazy
                o = KA.allocate(back, Float32, E,L,H,B); fill!(o, 0f0)
                DNNKernels.sdpaflashcm!(o, q, k, v, scale; backend = back,
                                        lazyrescale = lazy, onepass = false)
                KA.synchronize(back)
                Array(o)
            end
            @test outs[1] == outs[2]
            q = k = v = nothing; GC.gc()
        end

        @testset "cooperative-matrix flash agrees with the path it replaces" begin
            # Same call through `sdpa`, switch flipped: this is what guards the
            # routing rather than the kernel.
            E, L, H, B = 72, 256, 2, 2
            f16r(s) = DNNKernels.toback(back, Float16.(randn(Float32,E,L,H,B) .* 0.2f0))
            q, k, v = f16r(1), f16r(2), f16r(3)
            ws = DNNKernels.Workspace(back)
            scale = Float32(1/sqrt(E))
            outs = map((true, false)) do on
                DNNKernels.FLASHCM[] = on
                DNNKernels.reset!(ws)
                o = DNNKernels.sdpa(q, k, v, nothing, scale; backend = back, ws)
                KA.synchronize(back)
                Array(o)
            end
            DNNKernels.FLASHCM[] = true
            @test maximum(abs, outs[1] .- outs[2]) / maximum(abs, outs[2]) < 5e-3
            q = k = v = nothing; GC.gc()
        end
    end
end
